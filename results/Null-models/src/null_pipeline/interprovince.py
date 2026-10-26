from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .time_estimation import TravelTimePredictor


@dataclass
class ProvinceInterAllocator:
    """Allocate inter-province edge counts using gravity + admissibility masks."""

    node_table: pd.DataFrame
    predictor: TravelTimePredictor
    gravity_beta: float = 1.25
    admissible_max_distance_km: float | None = None
    random_seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_seed)
        self._nodes_by_prov = {
            p: g["node_id"].astype(int).tolist()
            for p, g in self.node_table.groupby("province", sort=True)
        }

        province_rows = []
        for p, g in self.node_table.groupby("province", sort=True):
            province_rows.append(
                {
                    "province": str(p),
                    "mass": float(len(g)),
                    "cx": float(g["x"].mean()),
                    "cy": float(g["y"].mean()),
                }
            )
        self._province_df = pd.DataFrame(province_rows)

    def _province_cost(self, p: str, q: str, temporal: bool) -> float:
        rp = self._province_df[self._province_df["province"] == p].iloc[0]
        rq = self._province_df[self._province_df["province"] == q].iloc[0]

        if temporal:
            # Use nearest node to centroid in each province as representative pair.
            np_nodes = self._nodes_by_prov[p]
            nq_nodes = self._nodes_by_prov[q]
            i = int(np_nodes[0])
            j = int(nq_nodes[0])
            return max(self.predictor.predict_pair_travel_time(i, j), 1e-6)

        dist = float(np.hypot(rp.cx - rq.cx, rp.cy - rq.cy))
        return max(dist, 1e-6)

    def _admissible_province_pair(self, p: str, q: str) -> bool:
        if p == q:
            return False
        if self.admissible_max_distance_km is None:
            return True
        rp = self._province_df[self._province_df["province"] == p].iloc[0]
        rq = self._province_df[self._province_df["province"] == q].iloc[0]
        d = float(np.hypot(rp.cx - rq.cx, rp.cy - rq.cy))
        return d <= self.admissible_max_distance_km * 1000.0

    def allocate_inter_province_counts(self, total_edges: int, temporal: bool) -> dict[tuple[str, str], int]:
        """Allocate fixed inter-province edge total using floor + largest remainders."""
        if total_edges <= 0:
            return {}

        provinces = sorted(self._nodes_by_prov)
        pairs = []
        scores = []

        mass = self._province_df.set_index("province")["mass"].to_dict()
        for p, q in combinations(provinces, 2):
            if not self._admissible_province_pair(p, q):
                continue
            c = self._province_cost(p, q, temporal=temporal)
            s = (mass[p] * mass[q]) / (c ** self.gravity_beta)
            if s <= 0 or not np.isfinite(s):
                continue
            pairs.append((p, q))
            scores.append(float(s))

        if not pairs:
            return {}

        scores = np.asarray(scores, dtype=float)
        w = scores / scores.sum()
        raw = w * total_edges
        floor = np.floor(raw).astype(int)
        rem = raw - floor

        counts = floor.copy()
        missing = int(total_edges - counts.sum())
        if missing > 0:
            order = np.argsort(-rem)
            for idx in order[:missing]:
                counts[idx] += 1

        return {pair: int(c) for pair, c in zip(pairs, counts) if c > 0}

    def sample_node_pairs(
        self,
        pair_counts: dict[tuple[str, str], int],
        temporal: bool,
    ) -> list[tuple[int, int]]:
        """Map province-pair allocations to node-level edges.

        Candidate node pairs are biased toward shorter/faster links.
        """
        sampled: list[tuple[int, int]] = []

        for (p, q), m in pair_counts.items():
            p_nodes = self._nodes_by_prov[p]
            q_nodes = self._nodes_by_prov[q]
            if not p_nodes or not q_nodes:
                continue

            # Random candidate pool to avoid full Cartesian explosion.
            n_candidates = min(max(10 * m, 400), len(p_nodes) * len(q_nodes))
            cands = []
            costs = []
            for _ in range(n_candidates):
                i = int(self._rng.choice(p_nodes))
                j = int(self._rng.choice(q_nodes))
                if i == j:
                    continue
                pair = (min(i, j), max(i, j))
                cands.append(pair)
                if temporal:
                    c = self.predictor.predict_pair_travel_time(pair[0], pair[1])
                else:
                    c = self.predictor.pair_distance_m(pair[0], pair[1])
                costs.append(max(float(c), 1e-9))

            if not cands:
                continue

            cands = np.asarray(cands)
            costs = np.asarray(costs, dtype=float)
            # Lower cost gets higher probability.
            w = 1.0 / costs
            w = w / w.sum()

            take = min(m, len(cands))
            idx = self._rng.choice(np.arange(len(cands)), size=take, replace=False, p=w)
            for k in idx:
                sampled.append((int(cands[k][0]), int(cands[k][1])))

        return sampled
