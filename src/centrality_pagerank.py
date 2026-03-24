"""
centrality_pagerank.py
Compute PageRank on the cleaned Roman road network weighted by travel time.

PageRank treats the road network as a random walk. Edge weights are set to
1/time_s so that faster (shorter) roads attract more traversal probability —
a node connected by quick roads scores higher than one reachable only via
long detours.

Outputs (written to maps/ and data/):
  - maps/pagerank.html        nodes coloured by PageRank score
  - data/pagerank_scores.csv  node_id, pagerank, province

Usage (from repo root):
  python src/centrality_pagerank.py
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

from centrality_utils import load_graph, node_province_map, save_map

OUT_MAP = Path("maps/pagerank.html")
OUT_CSV = Path("data/pagerank_scores.csv")

# ── Load network ──────────────────────────────────────────────────────────────
print("Loading network...")
G, id_to_node, edges_4326, nodes_4326 = load_graph()
print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── Set edge weights: 1/time_s (faster road = higher traversal probability) ──
for u, v, d in G.edges(data=True):
    t = d.get("time_s", 1.0)
    d["pr_weight"] = 1.0 / t if t > 0 else 0.0

# ── Run PageRank ──────────────────────────────────────────────────────────────
print("Computing PageRank...")
pr = nx.pagerank(G, weight="pr_weight", alpha=0.85, max_iter=200, tol=1e-6)
print(f"  Done. Score range: {min(pr.values()):.2e} – {max(pr.values()):.2e}")

# ── Assign province to each node ──────────────────────────────────────────────
print("Assigning provinces to nodes...")
nid_to_province = node_province_map(id_to_node)

# ── Build results table ───────────────────────────────────────────────────────
node_to_id = {v: k for k, v in id_to_node.items()}
rows = []
for xy, score in pr.items():
    nid = G.nodes[xy].get("node_id")
    if nid is None:
        continue
    rows.append({
        "node_id":  int(nid),
        "pagerank": float(score),
        "province": nid_to_province.get(int(nid), "Unknown"),
    })

results = pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)

# ── Save CSV ──────────────────────────────────────────────────────────────────
results.to_csv(OUT_CSV, index=False)
print(f"CSV saved  → {OUT_CSV}  ({len(results)} rows)")
print("\nTop 10 nodes by PageRank:")
print(results.head(10).to_string(index=False))

# ── Save map ──────────────────────────────────────────────────────────────────
print("\nBuilding map...")
nodes_plot = nodes_4326.merge(results[["node_id", "pagerank", "province"]], on="node_id")
save_map(
    nodes_4326=nodes_plot,
    edges_4326=edges_4326,
    score_col="pagerank",
    label="PageRank",
    out_path=OUT_MAP,
    log_scale=True,
)
