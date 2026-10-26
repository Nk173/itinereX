from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from .comparison import pairwise_model_matrices
from .config import NullFamily, PipelineConfig
from .hac import percolation_snapshots
from .io import edge_table_from_graph, ensure_output_layout, load_empirical_data
from .metrics import flatten_percolation, province_metrics, static_graph_metrics
from .scaling import fit_decay_models_for_graph, summarize_scaling_closeness
from .time_estimation import TravelTimePredictor
from .topology import TopologyGenerator
from .weights import assign_predicted_weights


def _save_graph_pickle(path: Path, G: nx.Graph, meta: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump({"graph": G, "meta": meta}, f)


def _plot_percolation_family(metrics_df: pd.DataFrame, family: str, plot_path: Path) -> None:
    sub = metrics_df[metrics_df["family"] == family].copy()
    if sub.empty:
        return

    long_rows = []
    for row in sub.itertuples(index=False):
        for c in [k for k in sub.columns if k.startswith("clusters_") and k.endswith("h")]:
            h = float(c.replace("clusters_", "").replace("h", ""))
            long_rows.append({"threshold_h": h, "clusters": float(getattr(row, c))})

    if not long_rows:
        return

    lf = pd.DataFrame(long_rows)
    agg = lf.groupby("threshold_h", as_index=False)["clusters"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(agg["threshold_h"], agg["mean"], marker="o")
    plt.fill_between(
        agg["threshold_h"],
        agg["mean"] - agg["std"].fillna(0.0),
        agg["mean"] + agg["std"].fillna(0.0),
        alpha=0.2,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Threshold (hours)")
    plt.ylabel("Number of clusters")
    plt.title(f"Percolation summary: {family}")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()


def _plot_scaling_distribution(scaling_df: pd.DataFrame, out_path: Path) -> None:
    if scaling_df.empty:
        return
    use = scaling_df.copy()

    plt.figure(figsize=(9, 5))
    families = sorted(use["family"].unique())
    xpos = {f: i for i, f in enumerate(families)}
    for fam in families:
        y = use.loc[use["family"] == fam, "exponent"].to_numpy()
        x = [xpos[fam]] * len(y)
        plt.scatter(x, y, alpha=0.55, s=14)

    plt.xticks(list(xpos.values()), list(xpos.keys()), rotation=20)
    plt.ylabel("Decay exponent")
    plt.title("Exponent distribution across empirical + null families")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_pipeline(config: PipelineConfig) -> dict[str, Path]:
    """Run full null-model generation + comparison pipeline."""
    dirs = ensure_output_layout(config.output_dir)

    empirical = load_empirical_data(
        config.empirical_pickle,
        province_edge_csv=config.province_edge_csv,
    )
    predictor = TravelTimePredictor(
        node_table=empirical.node_table,
        edge_table=empirical.edge_table,
        donor_k=config.donor_k,
        donor_distance_band=config.donor_distance_band,
        noise_scale=config.predictor_noise_scale,
        random_seed=config.random_seed,
    )

    target_edges = config.target_edge_count or empirical.graph.number_of_edges()

    topo_gen = TopologyGenerator(
        node_table=empirical.node_table,
        empirical_graph=empirical.graph,
        predictor=predictor,
        knn_k=config.knn_k,
        random_seed=config.random_seed,
    )

    metrics_rows = []
    models_for_pairwise = []
    scaling_rows = []

    if not config.skip_metrics:
        # Include empirical baseline in every comparison.
        emp_perc, emp_labels = percolation_snapshots(empirical.graph, config.thresholds_seconds(), max_exact_component_size=config.max_hac_component_size if not config.skip_hac_on_large else 1)
        emp_row = {
            "family": "empirical",
            "realization": "empirical_00",
            **static_graph_metrics(empirical.graph),
            **province_metrics(empirical.graph),
            **flatten_percolation(emp_perc),
        }
        metrics_rows.append(emp_row)
        models_for_pairwise.append({"name": "empirical_00", "graph": empirical.graph, "labels": emp_labels})

        emp_scaling = fit_decay_models_for_graph(
            G=empirical.graph,
            predictor=predictor,
            family="empirical",
            realization="empirical_00",
            sample_non_edges=config.sample_non_edges_for_scaling,
            n_bins=config.scaling_n_bins,
            random_seed=config.random_seed,
        )
        if not emp_scaling.empty:
            scaling_rows.append(emp_scaling)

    for family in config.families:
        family_name = family.value if isinstance(family, NullFamily) else str(family)
        family_dir = dirs["graphs"] / family_name
        family_dir.mkdir(parents=True, exist_ok=True)

        for r in range(config.n_realizations):
            rid = f"{family_name}_{r:02d}"

            G_top = topo_gen.generate(
                family=NullFamily(family_name),
                target_edges=target_edges,
                spatial_candidate_factor=config.spatial_candidate_factor,
                temporal_candidate_factor=config.temporal_candidate_factor,
                temporal_max_hours=config.temporal_max_hours,
                temporal_distance_slack=config.temporal_distance_slack,
                gravity_beta=config.gravity_beta,
                admissible_max_distance_km=config.admissible_max_distance_km,
            )
            G_null = assign_predicted_weights(G_top, predictor=predictor)

            out_pkl = family_dir / f"{rid}.pkl"
            _save_graph_pickle(
                out_pkl,
                G_null,
                meta={
                    "family": family_name,
                    "realization": r,
                    "config": asdict(config),
                },
            )

            edge_table_from_graph(G_null).to_csv(family_dir / f"{rid}_edges.csv", index=False)

            if not config.skip_metrics:
                perc_df, labels = percolation_snapshots(G_null, config.thresholds_seconds(), max_exact_component_size=config.max_hac_component_size if not config.skip_hac_on_large else 1)
                perc_df.to_csv(dirs["metrics"] / f"{rid}_percolation.csv", index=False)

                row = {
                    "family": family_name,
                    "realization": rid,
                    **static_graph_metrics(G_null),
                    **province_metrics(G_null),
                    **flatten_percolation(perc_df),
                }
                metrics_rows.append(row)

                models_for_pairwise.append({"name": rid, "graph": G_null, "labels": labels})

                sc = fit_decay_models_for_graph(
                    G=G_null,
                    predictor=predictor,
                    family=family_name,
                    realization=rid,
                    sample_non_edges=config.sample_non_edges_for_scaling,
                    n_bins=config.scaling_n_bins,
                    random_seed=config.random_seed + r + 17,
                )
                if not sc.empty:
                    scaling_rows.append(sc)

    output = {"output_base": dirs["base"]}
    
    if not config.skip_metrics:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv = dirs["summaries"] / "all_model_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)

        jac_df, ari_mats, nmi_mats = pairwise_model_matrices(
            models=models_for_pairwise,
            thresholds_s=config.thresholds_seconds(),
        )
        jac_path = dirs["matrices"] / "edge_jaccard_matrix.csv"
        jac_df.to_csv(jac_path)
        for t, df in ari_mats.items():
            h = int(round(t / 3600.0))
            df.to_csv(dirs["matrices"] / f"cluster_ari_{h}h.csv")
        for t, df in nmi_mats.items():
            h = int(round(t / 3600.0))
            df.to_csv(dirs["matrices"] / f"cluster_nmi_{h}h.csv")

        if scaling_rows:
            scaling_df = pd.concat(scaling_rows, ignore_index=True)
        else:
            scaling_df = pd.DataFrame()
        scaling_csv = dirs["scaling"] / "decay_fit_results.csv"
        scaling_df.to_csv(scaling_csv, index=False)

        close_df = summarize_scaling_closeness(scaling_df)
        close_csv = dirs["scaling"] / "family_closeness_to_empirical.csv"
        close_df.to_csv(close_csv, index=False)

        # Family-level percolation plots.
        for f in metrics_df["family"].unique():
            _plot_percolation_family(metrics_df, f, dirs["plots"] / f"percolation_{f}.png")

        _plot_scaling_distribution(scaling_df, dirs["plots"] / "scaling_exponent_distribution.png")

        output["metrics_csv"] = metrics_csv
        output["edge_jaccard_matrix"] = jac_path
        output["scaling_results_csv"] = scaling_csv
        output["scaling_closeness_csv"] = close_csv
    else:
        print("Metrics skipped (--skip-metrics flag set). Only graphs and edge tables saved.")

    return output
