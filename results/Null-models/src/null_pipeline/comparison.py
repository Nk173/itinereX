from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
except Exception:  # pragma: no cover
    adjusted_rand_score = None
    normalized_mutual_info_score = None


def edge_jaccard(G1: nx.Graph, G2: nx.Graph) -> float:
    e1 = {tuple(sorted((int(u), int(v)))) for u, v in G1.edges()}
    e2 = {tuple(sorted((int(u), int(v)))) for u, v in G2.edges()}
    union = len(e1 | e2)
    if union == 0:
        return 1.0
    return float(len(e1 & e2) / union)


def cluster_agreement(
    labels_a: dict[int, int],
    labels_b: dict[int, int],
) -> dict[str, float]:
    common = sorted(set(labels_a) & set(labels_b))
    if not common:
        return {"ari": np.nan, "nmi": np.nan}

    y1 = [labels_a[n] for n in common]
    y2 = [labels_b[n] for n in common]

    if adjusted_rand_score is None or normalized_mutual_info_score is None:
        return {"ari": np.nan, "nmi": np.nan}

    return {
        "ari": float(adjusted_rand_score(y1, y2)),
        "nmi": float(normalized_mutual_info_score(y1, y2)),
    }


def pairwise_model_matrices(
    models: list[dict],
    thresholds_s: list[float],
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame], dict[float, pd.DataFrame]]:
    """Compute pairwise edge Jaccard and clustering ARI/NMI matrices."""
    names = [m["name"] for m in models]
    idx = {n: i for i, n in enumerate(names)}

    jac = np.eye(len(names), dtype=float)
    ari = {t: np.eye(len(names), dtype=float) for t in thresholds_s}
    nmi = {t: np.eye(len(names), dtype=float) for t in thresholds_s}

    for a, b in itertools.combinations(models, 2):
        i = idx[a["name"]]
        j = idx[b["name"]]

        ej = edge_jaccard(a["graph"], b["graph"])
        jac[i, j] = jac[j, i] = ej

        for t in thresholds_s:
            agree = cluster_agreement(a["labels"][t], b["labels"][t])
            ari[t][i, j] = ari[t][j, i] = agree["ari"]
            nmi[t][i, j] = nmi[t][j, i] = agree["nmi"]

    jac_df = pd.DataFrame(jac, index=names, columns=names)
    ari_df = {t: pd.DataFrame(v, index=names, columns=names) for t, v in ari.items()}
    nmi_df = {t: pd.DataFrame(v, index=names, columns=names) for t, v in nmi.items()}
    return jac_df, ari_df, nmi_df
