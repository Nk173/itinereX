from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass
class TravelTimePredictor:
    """Predict travel time for non-edges using local donor sampling + fallback regression.

    Historical meaning:
    We preserve locally plausible Roman travel conditions by borrowing from nearby
    empirical edges, then use a global distance-time trend only when local evidence
    is unavailable.
    """

    node_table: pd.DataFrame
    edge_table: pd.DataFrame
    donor_k: int = 300
    donor_distance_band: tuple[float, float] = (0.5, 2.0)
    noise_scale: float = 0.08
    random_seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_seed)

        nt = self.node_table.set_index("node_id")
        self._coords = nt[["x", "y"]].to_dict("index")

        edge_mid = []
        edge_dist = []
        edge_time = []
        for row in self.edge_table.itertuples(index=False):
            u = int(row.u_id)
            v = int(row.v_id)
            cu = self._coords[u]
            cv = self._coords[v]
            mx = 0.5 * (cu["x"] + cv["x"])
            my = 0.5 * (cu["y"] + cv["y"])
            edge_mid.append((mx, my))
            edge_dist.append(float(row.dist_m))
            edge_time.append(float(row.time_s))

        self._edge_mid = np.asarray(edge_mid, dtype=float)
        self._edge_dist = np.asarray(edge_dist, dtype=float)
        self._edge_time = np.asarray(edge_time, dtype=float)
        self._edge_tree = cKDTree(self._edge_mid)

        # Global fallback: log(time) = a + b*log(distance)
        valid = (self._edge_dist > 0) & (self._edge_time > 0)
        x = np.log(self._edge_dist[valid])
        y = np.log(self._edge_time[valid])
        self._reg_b, self._reg_a = np.polyfit(x, y, deg=1)

        self._min_time = float(np.nanquantile(self._edge_time[valid], 0.01))
        self._max_time = float(np.nanquantile(self._edge_time[valid], 0.99))

    def pair_distance_m(self, i: int, j: int) -> float:
        ci = self._coords[int(i)]
        cj = self._coords[int(j)]
        return float(np.hypot(ci["x"] - cj["x"], ci["y"] - cj["y"]))

    def predict_pair_travel_time(self, i: int, j: int) -> float:
        """Predict travel time for pair (i, j).

        Steps:
        1. Query local empirical donor edges near pair midpoint.
        2. Prefer donors with similar distance.
        3. If local donor support is weak, fall back to global distance-time regression.
        """
        i = int(i)
        j = int(j)
        ci = self._coords[i]
        cj = self._coords[j]
        pair_dist = float(np.hypot(ci["x"] - cj["x"], ci["y"] - cj["y"]))
        if pair_dist <= 0:
            return self._min_time

        mid = np.array([0.5 * (ci["x"] + cj["x"]), 0.5 * (ci["y"] + cj["y"])], dtype=float)

        k = min(self.donor_k, len(self._edge_time))
        qd, qidx = self._edge_tree.query(mid, k=k)
        qd = np.atleast_1d(qd)
        qidx = np.atleast_1d(qidx).astype(int)

        donor_dist = self._edge_dist[qidx]
        donor_time = self._edge_time[qidx]

        lo, hi = self.donor_distance_band
        ratio = donor_dist / pair_dist
        mask = (ratio >= lo) & (ratio <= hi) & np.isfinite(donor_time) & (donor_time > 0)

        if np.any(mask):
            dd = donor_dist[mask]
            tt = donor_time[mask]
            md = qd[mask]

            # Bias toward nearby-midpoint and distance-matched donor edges.
            w_mid = np.exp(-md / (np.nanpercentile(md, 60) + 1e-9))
            w_dist = np.exp(-np.abs(np.log(dd / pair_dist)))
            w = w_mid * w_dist
            w = np.clip(w, 1e-12, None)
            w = w / w.sum()

            choice = self._rng.choice(np.arange(len(tt)), p=w)
            t = float(tt[choice])

            if self.noise_scale > 0:
                t = t * float(np.exp(self._rng.normal(0.0, self.noise_scale)))

            return float(np.clip(t, self._min_time, self._max_time))

        # Regression fallback.
        pred = float(np.exp(self._reg_a + self._reg_b * np.log(pair_dist)))
        return float(np.clip(pred, self._min_time, self._max_time))
