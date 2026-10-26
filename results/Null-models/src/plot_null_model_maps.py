from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.collections as mc
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]  # itinereX repo root
DEFAULT_EMPIRICAL_PKL = ROOT / "data_cleaning" / "cleaned_data" / "G_clean_weighted_with_ids.pkl"
NULL_GLOB = "*null*.pkl"
OUT_DIR = ROOT / "results" / "Null-models" / "figures" / "null_model_maps"


def _graph_from_tables(node_table: pd.DataFrame, edge_table: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in node_table.iterrows():
        node_id = int(row["node_id"])
        x = float(row["x"]) if pd.notna(row.get("x")) else np.nan
        y = float(row["y"]) if pd.notna(row.get("y")) else np.nan
        G.add_node(node_id, x=x, y=y)

    for _, row in edge_table.iterrows():
        u = int(row["u_id"])
        v = int(row["v_id"])
        G.add_edge(u, v)
    return G


def _node_xy(G: nx.Graph, n: object) -> tuple[float, float] | None:
    attrs = G.nodes[n]
    if "x" in attrs and "y" in attrs:
        x, y = attrs["x"], attrs["y"]
        if pd.notna(x) and pd.notna(y):
            return float(x), float(y)
    if isinstance(n, (tuple, list)) and len(n) >= 2:
        x, y = n[0], n[1]
        if isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number)):
            return float(x), float(y)
    return None


def _positions(G: nx.Graph) -> dict[object, tuple[float, float]]:
    pos = {}
    for n in G.nodes():
        xy = _node_xy(G, n)
        if xy is not None:
            pos[n] = xy
    return pos


def _draw_graph_map(G: nx.Graph, title: str, out_path: Path, province_subgraphs: dict = None) -> None:
    pos = _positions(G)
    if not pos:
        print(f"Skipping {title}: no plottable coordinates")
        return

    H = G.subgraph(pos.keys()).copy()

    fig, ax = plt.subplots(figsize=(11, 8))
    
    # If province data available, separate bridge edges from within-province edges
    if province_subgraphs:
        print(f"  [BRIDGED MODEL] Separating edges using {len(province_subgraphs)} provinces")
        # Build canonical set of within-province edges directly from province subgraphs.
        # This avoids misclassification when boundary nodes appear in multiple provinces.
        within_edge_keys = set()
        for prov_graph in province_subgraphs.values():
            for u, v in prov_graph.edges():
                within_edge_keys.add(frozenset((u, v)))

        # Separate edges in the plotted graph.
        within_prov_edges = []
        bridge_edges = []
        for u, v in H.edges():
            if frozenset((u, v)) in within_edge_keys:
                within_prov_edges.append((u, v))
            else:
                bridge_edges.append((u, v))
        
        print(f"  Within-province edges: {len(within_prov_edges)}, Bridge edges: {len(bridge_edges)}")
        
        # Draw within-province edges as explicit straight segments.
        if within_prov_edges:
            within_lines = [(pos[u], pos[v]) for u, v in within_prov_edges]
            within_lc = mc.LineCollection(
                within_lines,
                colors="#0b3c5d",
                linewidths=0.5,
                alpha=0.8,
                linestyles="solid",
            )
            ax.add_collection(within_lc)
        
        # Draw bridge edges as explicit straight dashed segments.
        if bridge_edges:
            lines = [(pos[u], pos[v]) for u, v in bridge_edges]
            lc = mc.LineCollection(lines, colors="#0b3c5d", linewidths=1.0, alpha=0.8, linestyles="dashed")
            ax.add_collection(lc)
            print(f"  BRIDGES RENDERED: {len(bridge_edges)} dark-blue dashed lines added")
    else:
        # Default: all edges same style
        print(f"  [GLOBAL MODEL] No province data, using default edge coloring")
        nx.draw_networkx_edges(H, pos=pos, ax=ax, width=0.35, alpha=0.65, edge_color="#0b3c5d")
    
    nx.draw_networkx_nodes(H, pos=pos, ax=ax, node_size=1.2, node_color="#b03a2e", alpha=0.55)

    ax.set_title(f"{title}\nNodes={H.number_of_nodes():,}  Edges={H.number_of_edges():,}", fontsize=12)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _load_null_graph(payload: dict) -> nx.Graph | None:
    if "graph" in payload and isinstance(payload["graph"], nx.Graph):
        return payload["graph"]
    if "province_subgraphs" in payload and isinstance(payload["province_subgraphs"], dict):
        gs = [g for g in payload["province_subgraphs"].values() if isinstance(g, nx.Graph)]
        if not gs:
            return None
        return nx.compose_all(gs)
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Empirical map
    if DEFAULT_EMPIRICAL_PKL.exists():
        with open(DEFAULT_EMPIRICAL_PKL, "rb") as f:
            raw = pickle.load(f)
        if "node_table" in raw and "edge_table" in raw:
            G_emp = _graph_from_tables(raw["node_table"], raw["edge_table"])
            _draw_graph_map(G_emp, "Empirical Dataset Network", OUT_DIR / "empirical_dataset_network.png")
        else:
            print(f"Empirical pickle missing expected tables: {DEFAULT_EMPIRICAL_PKL}")
    else:
        print(f"Empirical pickle not found: {DEFAULT_EMPIRICAL_PKL}")

    # Null maps from already-produced pickles
    null_dir = ROOT / "results" / "Null-models" / "data" / "processed"
    pkl_files = sorted(null_dir.glob(NULL_GLOB))
    if not pkl_files:
        print(f"No null-model pickles found in {null_dir}")
        return

    for p in pkl_files:
        # Keep maps focused on selected MST variants:
        # 1) global KNN/Delaunay MST models
        # 2) province-local MST models with explicit cross-province bridges
        keep_global_mst = ("global_spatial_null_" in p.stem and "_mst_" in p.stem)
        keep_province_bridged = p.stem.endswith("_mst_bridged")
        if not (keep_global_mst or keep_province_bridged):
            continue

        with open(p, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            print(f"Skipping {p.name}: unexpected payload type")
            continue

        G = _load_null_graph(payload)
        if G is None:
            print(f"Skipping {p.name}: no graph found")
            continue

        title = p.stem.replace("_", " ").strip()
        out_path = OUT_DIR / f"{p.stem}.png"
        
        # Pass province_subgraphs for bridged models to render bridges subtly
        province_subgraphs = payload.get("province_subgraphs") if isinstance(payload, dict) else None
        _draw_graph_map(G, title, out_path, province_subgraphs=province_subgraphs)

    print("Done.")


if __name__ == "__main__":
    main()
