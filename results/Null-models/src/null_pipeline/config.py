from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence


class NullFamily(str, Enum):
    """Four historically interpretable null-model families."""

    EMPIRE_SPATIAL = "empire_spatial"
    EMPIRE_TEMPORAL = "empire_temporal"
    PROVINCE_SPATIAL = "province_spatial"
    PROVINCE_TEMPORAL = "province_temporal"


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration for the null-model pipeline."""

    empirical_pickle: Path
    province_edge_csv: Path | None = None
    output_dir: Path = Path("results/Null-models/data/processed/null_pipeline")
    families: Sequence[NullFamily] = field(default_factory=lambda: list(NullFamily))
    n_realizations: int = 10
    random_seed: int = 42

    # Topology controls
    knn_k: int = 4
    target_edge_count: int | None = None
    temporal_candidate_factor: float = 3.0
    spatial_candidate_factor: float = 3.0
    temporal_max_hours: float = 1200.0
    temporal_distance_slack: float = 1.5

    # Inter-province controls
    gravity_beta: float = 1.25
    inter_total_edges: int | None = None
    admissible_max_distance_km: float | None = None

    # Time prediction controls
    donor_k: int = 300
    donor_distance_band: tuple[float, float] = (0.5, 2.0)
    predictor_noise_scale: float = 0.08

    # Metrics
    hac_threshold_hours: Sequence[float] = (48.0, 168.0, 288.0, 1200.0)
    sample_non_edges_for_scaling: int = 25000
    scaling_n_bins: int = 20
    max_hac_component_size: int = 2200
    
    # Optional skips for faster testing
    skip_metrics: bool = False  # Skip all HAC, scaling, pairwise metrics
    skip_hac_on_large: bool = False  # Use only approx on large components, skip exact

    def thresholds_seconds(self) -> list[float]:
        return [h * 3600.0 for h in self.hac_threshold_hours]
