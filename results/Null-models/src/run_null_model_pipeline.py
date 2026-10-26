from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from null_pipeline.config import NullFamily, PipelineConfig
from null_pipeline.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Roman-road spatial/temporal null-model pipeline")
    parser.add_argument(
        "--empirical-pickle",
        type=Path,
        default=Path("data_cleaning/cleaned_data/G_clean_weighted_with_ids.pkl"),
        help="Empirical graph pickle with node_table and edge_table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/Null-models/data/processed/null_pipeline"),
        help="Output directory",
    )
    parser.add_argument(
        "--province-edge-csv",
        type=Path,
        default=Path("data_cleaning/cleaned_data/G_clean_weighted_edges_with_province.csv"),
        help="Optional edge CSV with province labels to infer node provinces",
    )
    parser.add_argument(
        "--families",
        type=str,
        nargs="*",
        default=[f.value for f in NullFamily],
        choices=[f.value for f in NullFamily],
        help="Null families to run",
    )
    parser.add_argument("--n-realizations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=4)
    parser.add_argument("--gravity-beta", type=float, default=1.25)
    parser.add_argument("--sample-non-edges", type=int, default=25000)
    parser.add_argument("--temporal-max-hours", type=float, default=1200.0)
    parser.add_argument("--temporal-distance-slack", type=float, default=1.5)
    parser.add_argument("--skip-metrics", action="store_true", help="Skip all HAC/percolation, scaling, and pairwise metrics (only output graphs)")
    parser.add_argument("--skip-hac-on-large", action="store_true", help="Skip exact HAC on large components; use only approximation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PipelineConfig(
        empirical_pickle=args.empirical_pickle,
        province_edge_csv=args.province_edge_csv,
        output_dir=args.output_dir,
        families=[NullFamily(f) for f in args.families],
        n_realizations=args.n_realizations,
        random_seed=args.seed,
        knn_k=args.knn_k,
        gravity_beta=args.gravity_beta,
        sample_non_edges_for_scaling=args.sample_non_edges,
        temporal_max_hours=args.temporal_max_hours,
        temporal_distance_slack=args.temporal_distance_slack,
        skip_metrics=args.skip_metrics,
        skip_hac_on_large=args.skip_hac_on_large,
    )

    outputs = run_pipeline(cfg)

    print("Pipeline complete.")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
