from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import NullFamily
from .interprovince import ProvinceInterAllocator
from .time_estimation import TravelTimePredictor


@dataclass
class TopologyGenerator:
    """Generate topology-only null graphs with fixed node set.

    Historical interpretation of families:
    - empire_spatial: roads emerge from geometric proximity across empire.
    - empire_temporal: roads emerge from predicted travel-time accessibility.
    - province_spatial: local province roads are spatial, with generated inter-province spatial corridors.
    - province_temporal: local province roads follow travel-time accessibility, with generated inter-province temporal corridors.
    """

    node_table: pd.DataFrame
    empirical_graph: nx.Graph
    predictor: TravelTimePredictor
    knn_k: int = 4
    random_seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_seed)
        self._nodes = self.node_table["node_id"].astype(int).tolist()
        self._coords_df = self.node_table.set_index("node_id")[["x", "y"]]
        speeds = []
        for _, _, d in self.empirical_graph.edges(data=True):
            dist = float(d.get("dist_m", np.nan))
            time_s = float(d.get("time_s", np.nan))
            if np.isfinite(dist) and np.isfinite(time_s) and dist > 0 and time_s > 0:
                speeds.append(dist / time_s)
        self._median_speed_mps = float(np.median(speeds)) if speeds else 1.0

    def _empty_graph_with_nodes(self) -> nx.Graph:
        H = nx.Graph()
        for row in self.node_table.itertuples(index=False):
            H.add_node(int(row.node_id), x=float(row.x), y=float(row.y), province=str(row.province))
        return H

    def _knn_edges(self, nodes: list[int], k: int) -> set[tuple[int, int]]:
        if len(nodes) <= 1:
            return set()
        k_eff = max(1, min(k, len(nodes) - 1))

        arr = self._coords_df.loc[nodes].to_numpy(dtype=float)
        tree = cKDTree(arr)
        _, idx = tree.query(arr, k=k_eff + 1)

        out: set[tuple[int, int]] = set()
        for i, nbr_idx in enumerate(np.atleast_2d(idx)):
            u = int(nodes[i])
            for j in nbr_idx[1:]:
                v = int(nodes[int(j)])
                if u == v:
                    continue
                out.add((min(u, v), max(u, v)))
        return out

    def _sample_ranked_edges(
        self,
        candidate_edges: list[tuple[int, int]],
        candidate_costs: np.ndarray,
        target_edges: int,
    ) -> list[tuple[int, int]]:
        """Stochastic ranked sampling that favors low-cost candidate edges."""
        if target_edges <= 0 or not candidate_edges:
            return []

        target_edges = min(target_edges, len(candidate_edges))
        cost = np.asarray(candidate_costs, dtype=float)
        rank = np.argsort(np.argsort(cost))
        # Exponential rank weights keep realizations stochastic while preserving low-cost bias.
        w = np.exp(-rank / max(1.0, 0.2 * len(rank)))
        w = w / w.sum()

        picked = self._rng.choice(np.arange(len(candidate_edges)), size=target_edges, replace=False, p=w)
        return [candidate_edges[int(i)] for i in picked]

    def _empire_candidate_edges(
        self,
        temporal: bool,
        candidate_factor: float,
        temporal_max_hours: float,
        temporal_distance_slack: float,
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        n = len(self._nodes)
        m_emp = self.empirical_graph.number_of_edges()
        pool_target = int(max(m_emp * candidate_factor, m_emp))

        # Candidate pool starts from KNN neighborhood to avoid full O(n^2) generation.
        k = int(max(self.knn_k, np.ceil(candidate_factor * self.knn_k)))
        candidate = list(self._knn_edges(self._nodes, k=k))

        # If still too small, draw random extra non-edges.
        have = set(candidate)
        while len(candidate) < pool_target:
            i = int(self._rng.integers(0, n))
            j = int(self._rng.integers(0, n))
            if i == j:
                continue
            u = int(self._nodes[i])
            v = int(self._nodes[j])
            e = (min(u, v), max(u, v))
            if e in have:
                continue
            have.add(e)
            candidate.append(e)

        costs = []
        kept = []
        max_time_s = float(temporal_max_hours * 3600.0)
        max_dist_for_time = float(self._median_speed_mps * max_time_s * temporal_distance_slack)

        for u, v in candidate:
            if temporal:
                # Fast prefilter: skip clearly impossible temporal links before prediction.
                d = self.predictor.pair_distance_m(u, v)
                if d > max_dist_for_time:
                    continue
                c = self.predictor.predict_pair_travel_time(u, v)
                if c > max_time_s:
                    continue
            else:
                c = self.predictor.pair_distance_m(u, v)
            kept.append((u, v))
            costs.append(float(c))

        if not kept:
            return [], np.asarray([], dtype=float)
        return kept, np.asarray(costs, dtype=float)

    def _build_empire(
        self,
        temporal: bool,
        target_edges: int,
        candidate_factor: float,
        temporal_max_hours: float,
        temporal_distance_slack: float,
    ) -> nx.Graph:
        H = self._empty_graph_with_nodes()
        cands, costs = self._empire_candidate_edges(
            temporal=temporal,
            candidate_factor=candidate_factor,
            temporal_max_hours=temporal_max_hours,
            temporal_distance_slack=temporal_distance_slack,
        )
        edges = self._sample_ranked_edges(cands, costs, target_edges=target_edges)
        H.add_edges_from(edges)
        return H

    def _build_province(
        self,
        temporal: bool,
        target_edges: int,
        gravity_beta: float,
        admissible_max_distance_km: float | None,
        temporal_max_hours: float,
        temporal_distance_slack: float,
    ) -> nx.Graph:
        H = self._empty_graph_with_nodes()
        max_time_s = float(temporal_max_hours * 3600.0)
        max_dist_for_time = float(self._median_speed_mps * max_time_s * temporal_distance_slack)

        # Keep empirical intra/inter split for comparability.
        intra_target = 0
        inter_target = 0
        for u, v in self.empirical_graph.edges():
            pu = H.nodes[u]["province"]
            pv = H.nodes[v]["province"]
            if pu == pv:
                intra_target += 1
            else:
                inter_target += 1

        # Scale split if caller changed total edge target.
        m_emp = max(1, self.empirical_graph.number_of_edges())
        intra_target = int(round(intra_target * target_edges / m_emp))
        inter_target = int(max(0, target_edges - intra_target))

        # Intra-province generation.
        provinces = sorted(self.node_table["province"].astype(str).unique())
        intra_edges: set[tuple[int, int]] = set()
        for p in provinces:
            nodes = self.node_table.loc[self.node_table["province"].astype(str) == p, "node_id"].astype(int).tolist()
            if len(nodes) < 2:
                continue

            # Province-specific target scales with empirical internal edge count.
            m_p_emp = 0
            nodes_set = set(nodes)
            for u, v in self.empirical_graph.edges():
                if u in nodes_set and v in nodes_set:
                    m_p_emp += 1
            if m_p_emp <= 0:
                continue

            if temporal:
                # Candidate pool: KNN + random non-edges; rank by predicted time.
                cands = list(self._knn_edges(nodes, k=max(self.knn_k, 6)))
                want_pool = max(int(3.0 * m_p_emp), len(cands))
                have = set(cands)
                n = len(nodes)
                while len(cands) < want_pool:
                    i = int(self._rng.integers(0, n))
                    j = int(self._rng.integers(0, n))
                    if i == j:
                        continue
                    e = (min(nodes[i], nodes[j]), max(nodes[i], nodes[j]))
                    if e in have:
                        continue
                    have.add(e)
                    cands.append(e)
                cands_t = []
                costs_t = []
                for u, v in cands:
                    d = self.predictor.pair_distance_m(u, v)
                    if d > max_dist_for_time:
                        continue
                    t = self.predictor.predict_pair_travel_time(u, v)
                    if t > max_time_s:
                        continue
                    cands_t.append((u, v))
                    costs_t.append(float(t))
                cands = cands_t
                costs = np.asarray(costs_t, dtype=float)
                picked = self._sample_ranked_edges(cands, costs, m_p_emp)
            else:
                cands = list(self._knn_edges(nodes, k=max(self.knn_k, 6)))
                costs = np.asarray([self.predictor.pair_distance_m(u, v) for u, v in cands], dtype=float)
                picked = self._sample_ranked_edges(cands, costs, m_p_emp)

            intra_edges.update(picked)

        H.add_edges_from(intra_edges)

        # Inter-province generation with gravity allocation + admissibility.
        allocator = ProvinceInterAllocator(
            node_table=self.node_table,
            predictor=self.predictor,
            gravity_beta=gravity_beta,
            admissible_max_distance_km=admissible_max_distance_km,
            random_seed=int(self._rng.integers(0, 2**31 - 1)),
        )
        pair_counts = allocator.allocate_inter_province_counts(
            total_edges=inter_target,
            temporal=temporal,
        )
        inter_edges = allocator.sample_node_pairs(pair_counts=pair_counts, temporal=temporal)
        if temporal:
            inter_edges = [
                (u, v)
                for (u, v) in inter_edges
                if self.predictor.pair_distance_m(u, v) <= max_dist_for_time
                and self.predictor.predict_pair_travel_time(u, v) <= max_time_s
            ]
        H.add_edges_from(inter_edges)

        # If still short due to duplicate sampling, fill from empire candidates with province mismatch.
        if H.number_of_edges() < target_edges:
            need = target_edges - H.number_of_edges()
            cands, costs = self._empire_candidate_edges(
                temporal=temporal,
                candidate_factor=2.5,
                temporal_max_hours=temporal_max_hours,
                temporal_distance_slack=temporal_distance_slack,
            )
            cands_filtered = [
                e for e in cands
                if H.nodes[e[0]]["province"] != H.nodes[e[1]]["province"] and not H.has_edge(*e)
            ]
            costs_filtered = np.asarray([
                self.predictor.predict_pair_travel_time(u, v) if temporal else self.predictor.pair_distance_m(u, v)
                for u, v in cands_filtered
            ], dtype=float)
            fill = self._sample_ranked_edges(cands_filtered, costs_filtered, need)
            H.add_edges_from(fill)

        return H

    def generate(
        self,
        family: NullFamily,
        target_edges: int,
        spatial_candidate_factor: float,
        temporal_candidate_factor: float,
        temporal_max_hours: float,
        temporal_distance_slack: float,
        gravity_beta: float,
        admissible_max_distance_km: float | None,
    ) -> nx.Graph:
        if family == NullFamily.EMPIRE_SPATIAL:
            return self._build_empire(
                temporal=False,
                target_edges=target_edges,
                candidate_factor=spatial_candidate_factor,
                temporal_max_hours=temporal_max_hours,
                temporal_distance_slack=temporal_distance_slack,
            )

        if family == NullFamily.EMPIRE_TEMPORAL:
            return self._build_empire(
                temporal=True,
                target_edges=target_edges,
                candidate_factor=temporal_candidate_factor,
                temporal_max_hours=temporal_max_hours,
                temporal_distance_slack=temporal_distance_slack,
            )

        if family == NullFamily.PROVINCE_SPATIAL:
            return self._build_province(
                temporal=False,
                target_edges=target_edges,
                gravity_beta=gravity_beta,
                admissible_max_distance_km=admissible_max_distance_km,
                temporal_max_hours=temporal_max_hours,
                temporal_distance_slack=temporal_distance_slack,
            )

        if family == NullFamily.PROVINCE_TEMPORAL:
            return self._build_province(
                temporal=True,
                target_edges=target_edges,
                gravity_beta=gravity_beta,
                admissible_max_distance_km=admissible_max_distance_km,
                temporal_max_hours=temporal_max_hours,
                temporal_distance_slack=temporal_distance_slack,
            )

        raise ValueError(f"Unsupported family: {family}")
