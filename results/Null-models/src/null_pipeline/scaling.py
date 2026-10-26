from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .time_estimation import TravelTimePredictor


@dataclass(slots=True)
class DecayFitResult:
    family: str
    realization: str
    cost_type: str
    model: str
    exponent: float
    intercept: float
    r2: float
    aic: float


def _sample_pair_dataset(
    G: nx.Graph,
    predictor: TravelTimePredictor,
    sample_non_edges: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    nodes = list(G.nodes())
    n = len(nodes)

    pos = [tuple(sorted((int(u), int(v)))) for u, v in G.edges()]
    pos_set = set(pos)

    neg = set()
    while len(neg) < sample_non_edges:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        u = int(nodes[i])
        v = int(nodes[j])
        e = (min(u, v), max(u, v))
        if e in pos_set or e in neg:
            continue
        neg.add(e)

    rows = []
    for u, v in pos:
        d = predictor.pair_distance_m(u, v)
        t = predictor.predict_pair_travel_time(u, v)
        rows.append({"u": u, "v": v, "is_edge": 1.0, "distance": d, "cost": t})

    for u, v in neg:
        d = predictor.pair_distance_m(u, v)
        t = predictor.predict_pair_travel_time(u, v)
        rows.append({"u": u, "v": v, "is_edge": 0.0, "distance": d, "cost": t})

    return pd.DataFrame(rows)


def _binned_probability(df: pd.DataFrame, xcol: str, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    x = df[xcol].to_numpy(dtype=float)
    y = df["is_edge"].to_numpy(dtype=float)

    q = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.quantile(x, q)
    bins = np.unique(bins)
    if len(bins) < 4:
        return np.array([]), np.array([])

    mids = []
    probs = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (x >= lo) & (x <= hi) if i == len(bins) - 2 else (x >= lo) & (x < hi)
        if mask.sum() < 10:
            continue
        p = float(y[mask].mean())
        if p <= 0:
            continue
        mids.append(float(np.median(x[mask])))
        probs.append(p)

    return np.asarray(mids, dtype=float), np.asarray(probs, dtype=float)


def _fit_power(x: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float]:
    lx = np.log(x)
    ly = np.log(p)
    b1, b0 = np.polyfit(lx, ly, deg=1)  # ly = b0 + b1*lx
    pred = b0 + b1 * lx
    resid = ly - pred
    rss = float(np.sum(resid**2))
    tss = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - rss / max(tss, 1e-12)
    n = len(ly)
    aic = n * np.log(max(rss / max(n, 1), 1e-12)) + 2 * 2
    exponent = -float(b1)
    intercept = float(np.exp(b0))
    return exponent, intercept, r2, aic


def _fit_exp(x: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float]:
    ly = np.log(p)
    b1, b0 = np.polyfit(x, ly, deg=1)  # ly = b0 + b1*x
    pred = b0 + b1 * x
    resid = ly - pred
    rss = float(np.sum(resid**2))
    tss = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - rss / max(tss, 1e-12)
    n = len(ly)
    aic = n * np.log(max(rss / max(n, 1), 1e-12)) + 2 * 2
    exponent = -float(b1)
    intercept = float(np.exp(b0))
    return exponent, intercept, r2, aic


def fit_decay_models_for_graph(
    G: nx.Graph,
    predictor: TravelTimePredictor,
    family: str,
    realization: str,
    sample_non_edges: int,
    n_bins: int,
    random_seed: int,
) -> pd.DataFrame:
    """Fit distance- and cost-decay under power-law and exponential forms."""
    ds = _sample_pair_dataset(
        G=G,
        predictor=predictor,
        sample_non_edges=sample_non_edges,
        random_seed=random_seed,
    )

    out_rows = []
    for xcol, ctype in (("distance", "distance"), ("cost", "cost")):
        x, p = _binned_probability(ds, xcol=xcol, n_bins=n_bins)
        if len(x) < 4:
            continue

        pw = _fit_power(x, p)
        ex = _fit_exp(x, p)
        out_rows.append(
            asdict(
                DecayFitResult(
                    family=family,
                    realization=realization,
                    cost_type=ctype,
                    model="power_law",
                    exponent=pw[0],
                    intercept=pw[1],
                    r2=pw[2],
                    aic=pw[3],
                )
            )
        )
        out_rows.append(
            asdict(
                DecayFitResult(
                    family=family,
                    realization=realization,
                    cost_type=ctype,
                    model="exponential",
                    exponent=ex[0],
                    intercept=ex[1],
                    r2=ex[2],
                    aic=ex[3],
                )
            )
        )

    return pd.DataFrame(out_rows)


def summarize_scaling_closeness(df: pd.DataFrame) -> pd.DataFrame:
    """Report which null family is closest to empirical under same fitting procedure."""
    if df.empty:
        return pd.DataFrame(columns=["family", "cost_type", "model", "abs_exp_gap_to_empirical"])

    # Keep best-AIC fit per realization/cost_type.
    ix = df.groupby(["family", "realization", "cost_type"])["aic"].idxmin()
    best = df.loc[ix].copy()

    emp = best[best["family"] == "empirical"].groupby(["cost_type", "model"])["exponent"].mean().reset_index()
    null = best[best["family"] != "empirical"].copy()

    rows = []
    for r in null.itertuples(index=False):
        m = emp[(emp["cost_type"] == r.cost_type) & (emp["model"] == r.model)]
        if m.empty:
            continue
        gap = abs(float(r.exponent) - float(m.iloc[0]["exponent"]))
        rows.append({
            "family": r.family,
            "cost_type": r.cost_type,
            "model": r.model,
            "abs_exp_gap_to_empirical": gap,
        })

    if not rows:
        return pd.DataFrame(columns=["family", "cost_type", "model", "abs_exp_gap_to_empirical"])

    out = pd.DataFrame(rows)
    out = out.groupby(["family", "cost_type", "model"], as_index=False)["abs_exp_gap_to_empirical"].mean()
    return out.sort_values(["cost_type", "model", "abs_exp_gap_to_empirical"]) 
