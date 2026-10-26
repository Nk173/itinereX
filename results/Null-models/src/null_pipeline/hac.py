from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _exact_component_labels(
    Gc: nx.Graph,
    threshold_s: float,
) -> dict[int, int]:
    nodes = list(Gc.nodes())
    n = len(nodes)
    if n == 1:
        return {int(nodes[0]): 1}

    D = nx.floyd_warshall_numpy(Gc, nodelist=nodes, weight="time_s")
    D = np.asarray(D, dtype=float)

    if not np.isfinite(D).all():
        # Component should be connected; if not, fallback to threshold components.
        Ht = nx.Graph()
        Ht.add_nodes_from(nodes)
        Ht.add_edges_from((u, v) for u, v, d in Gc.edges(data=True) if float(d.get("time_s", np.inf)) <= threshold_s)
        labels = {}
        for k, comp in enumerate(nx.connected_components(Ht), start=1):
            for u in comp:
                labels[int(u)] = int(k)
        return labels

    Z = linkage(squareform(D, checks=False), method="average")
    labs = fcluster(Z, t=float(threshold_s), criterion="distance")
    return {int(nodes[i]): int(labs[i]) for i in range(n)}


def percolation_snapshots(
    G: nx.Graph,
    thresholds_s: list[float],
    max_exact_component_size: int = 2200,
) -> tuple[pd.DataFrame, dict[float, dict[int, int]]]:
    """Compute percolation/HAC-like snapshots at fixed thresholds.

    For small components, this uses exact HAC with average linkage over shortest-path distances.
    For very large components, it falls back to thresholded-edge connected components to keep
    memory bounded.
    """
    thresholds_s = [float(t) for t in thresholds_s]
    comps = [set(c) for c in nx.connected_components(G)]

    labels_by_threshold: dict[float, dict[int, int]] = {t: {} for t in thresholds_s}
    method_by_threshold: dict[float, str] = {t: "exact" for t in thresholds_s}

    for comp in comps:
        if len(comp) < 1:
            continue
        Gc = G.subgraph(comp).copy()

        if len(comp) <= max_exact_component_size and len(comp) >= 2:
            for t in thresholds_s:
                comp_labels = _exact_component_labels(Gc, t)
                # Re-index locally unique IDs to avoid collisions by shifting later.
                labels_by_threshold[t].update(comp_labels)
        else:
            if len(comp) > max_exact_component_size:
                warnings.warn(
                    f"Large component of size {len(comp)}: using edge-threshold approximation, not exact HAC"
                )
            for t in thresholds_s:
                method_by_threshold[t] = "approx_edge_threshold"
                Ht = nx.Graph()
                Ht.add_nodes_from(Gc.nodes())
                Ht.add_edges_from((u, v) for u, v, d in Gc.edges(data=True) if float(d.get("time_s", np.inf)) <= t)
                local = {}
                for k, cc in enumerate(nx.connected_components(Ht), start=1):
                    for u in cc:
                        local[int(u)] = int(k)
                labels_by_threshold[t].update(local)

    # Ensure global uniqueness of label IDs per threshold.
    for t in thresholds_s:
        remap: dict[int, int] = {}
        nxt = 1
        for node in sorted(labels_by_threshold[t]):
            old = labels_by_threshold[t][node]
            if old not in remap:
                remap[old] = nxt
                nxt += 1
            labels_by_threshold[t][node] = remap[old]

    rows = []
    for t in thresholds_s:
        labs = labels_by_threshold[t]
        if not labs:
            rows.append({"threshold_s": t, "n_clusters": 0, "largest_cluster": 0, "method": method_by_threshold[t]})
            continue
        vals = list(labs.values())
        vc = pd.Series(vals).value_counts()
        rows.append(
            {
                "threshold_s": t,
                "n_clusters": int(vc.shape[0]),
                "largest_cluster": int(vc.max()),
                "method": method_by_threshold[t],
            }
        )

    return pd.DataFrame(rows), labels_by_threshold
