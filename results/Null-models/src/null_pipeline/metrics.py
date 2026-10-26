from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def static_graph_metrics(G: nx.Graph) -> dict[str, float]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    out = {
        "num_nodes": float(n),
        "num_edges": float(m),
        "density": float(nx.density(G)) if n > 1 else 0.0,
        "num_components": float(nx.number_connected_components(G)) if n > 0 else 0.0,
        "avg_clustering": float(nx.average_clustering(G)) if n > 0 else 0.0,
        "mean_degree": float(np.mean([d for _, d in G.degree()])) if n > 0 else 0.0,
    }

    times = [float(d.get("time_s")) for _, _, d in G.edges(data=True) if "time_s" in d]
    dists = [float(d.get("dist_m")) for _, _, d in G.edges(data=True) if "dist_m" in d]
    out["mean_time_s"] = float(np.mean(times)) if times else np.nan
    out["mean_dist_m"] = float(np.mean(dists)) if dists else np.nan
    return out


def province_metrics(G: nx.Graph) -> dict[str, float]:
    provinces = {}
    for n, d in G.nodes(data=True):
        p = str(d.get("province", "UNKNOWN"))
        provinces.setdefault(p, set()).add(n)

    comms = [c for c in provinces.values() if c]
    if not comms:
        return {
            "province_modularity": np.nan,
            "cross_province_edge_frac": np.nan,
            "num_provinces_present": 0.0,
        }

    try:
        mod = float(nx.algorithms.community.quality.modularity(G, comms))
    except Exception:
        mod = np.nan

    cross = 0
    for u, v in G.edges():
        if str(G.nodes[u].get("province")) != str(G.nodes[v].get("province")):
            cross += 1

    return {
        "province_modularity": mod,
        "cross_province_edge_frac": float(cross / max(1, G.number_of_edges())),
        "num_provinces_present": float(len(comms)),
    }


def flatten_percolation(df: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for row in df.itertuples(index=False):
        h = int(round(float(row.threshold_s) / 3600.0))
        out[f"{prefix}clusters_{h}h"] = float(row.n_clusters)
        out[f"{prefix}largest_{h}h"] = float(row.largest_cluster)
    return out
