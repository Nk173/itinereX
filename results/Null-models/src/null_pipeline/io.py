from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(slots=True)
class EmpiricalData:
    """Empirical graph + tidy node/edge tables used throughout the pipeline."""

    graph: nx.Graph
    node_table: pd.DataFrame
    edge_table: pd.DataFrame


REQUIRED_NODE_COLS = {"node_id", "x", "y"}
REQUIRED_EDGE_COLS = {"u_id", "v_id", "time_s", "dist_m"}


def _infer_node_provinces_from_edge_csv(province_edge_csv: Path) -> dict[int, str]:
    if not province_edge_csv.exists():
        return {}

    df = pd.read_csv(province_edge_csv)
    votes: dict[int, Counter] = {}

    if {"u_id", "v_id", "province_start", "province_end"}.issubset(df.columns):
        for row in df.itertuples(index=False):
            u = int(row.u_id)
            v = int(row.v_id)
            ps = str(getattr(row, "province_start", ""))
            pe = str(getattr(row, "province_end", ""))
            if ps and ps != "nan":
                votes.setdefault(u, Counter())[ps] += 1
            if pe and pe != "nan":
                votes.setdefault(v, Counter())[pe] += 1
    elif {"u_id", "v_id", "province"}.issubset(df.columns):
        for row in df.itertuples(index=False):
            p = str(getattr(row, "province", ""))
            if not p or p == "nan":
                continue
            u = int(row.u_id)
            v = int(row.v_id)
            votes.setdefault(u, Counter())[p] += 1
            votes.setdefault(v, Counter())[p] += 1

    return {nid: cnt.most_common(1)[0][0] for nid, cnt in votes.items() if cnt}


def load_empirical_data(pickle_path: Path, province_edge_csv: Path | None = None) -> EmpiricalData:
    """Load empirical graph pickle expected to include node_table and edge_table.

    Historical meaning:
    The empirical graph is the observed Roman road system with recorded travel times.
    Null models keep this node set fixed and only alter edge organization and weights.
    """
    with open(pickle_path, "rb") as f:
        raw = pickle.load(f)

    if "node_table" not in raw or "edge_table" not in raw:
        raise KeyError("Empirical pickle must include node_table and edge_table")

    node_table = raw["node_table"].copy()
    edge_table = raw["edge_table"].copy()

    missing_nodes = REQUIRED_NODE_COLS - set(node_table.columns)
    if missing_nodes:
        raise KeyError(f"Missing required node columns: {sorted(missing_nodes)}")

    missing_edges = REQUIRED_EDGE_COLS - set(edge_table.columns)
    if missing_edges:
        raise KeyError(f"Missing required edge columns: {sorted(missing_edges)}")

    node_table = node_table[list(dict.fromkeys([*node_table.columns]))]
    edge_table = edge_table[list(dict.fromkeys([*edge_table.columns]))]

    if "province" not in node_table.columns:
        inferred = _infer_node_provinces_from_edge_csv(province_edge_csv) if province_edge_csv else {}
        node_table["province"] = node_table["node_id"].map(inferred).fillna("UNKNOWN")

    # Normalize dtypes.
    node_table["node_id"] = node_table["node_id"].astype(int)
    node_table["x"] = node_table["x"].astype(float)
    node_table["y"] = node_table["y"].astype(float)
    node_table["province"] = node_table["province"].astype(str)

    edge_table["u_id"] = edge_table["u_id"].astype(int)
    edge_table["v_id"] = edge_table["v_id"].astype(int)
    edge_table["time_s"] = edge_table["time_s"].astype(float)
    edge_table["dist_m"] = edge_table["dist_m"].astype(float)

    G = nx.Graph()
    for row in node_table.itertuples(index=False):
        G.add_node(
            int(row.node_id),
            x=float(row.x),
            y=float(row.y),
            province=str(row.province),
        )

    for row in edge_table.itertuples(index=False):
        attrs = {
            "time_s": float(row.time_s),
            "dist_m": float(row.dist_m),
        }
        if hasattr(row, "weight") and pd.notna(row.weight):
            attrs["weight"] = float(row.weight)
        if hasattr(row, "edge_id") and pd.notna(row.edge_id):
            attrs["edge_id"] = int(row.edge_id)
        G.add_edge(int(row.u_id), int(row.v_id), **attrs)

    return EmpiricalData(graph=G, node_table=node_table, edge_table=edge_table)


def ensure_output_layout(base_output: Path) -> dict[str, Path]:
    """Create reproducible output layout and return key directories."""
    dirs = {
        "base": base_output,
        "graphs": base_output / "graphs",
        "metrics": base_output / "metrics",
        "plots": base_output / "plots",
        "summaries": base_output / "summaries",
        "matrices": base_output / "matrices",
        "scaling": base_output / "scaling",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def edge_table_from_graph(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append(
            {
                "u_id": int(u),
                "v_id": int(v),
                "time_s": float(d.get("time_s", np.nan)),
                "dist_m": float(d.get("dist_m", np.nan)),
            }
        )
    return pd.DataFrame(rows)
