"""Public API for Stage 8: COLMAP dense stereo depth maps."""

from __future__ import annotations

from .colmap_dense import (
    ColmapDenseConfig,
    ColmapDenseResult,
    collect_colmap_depth_maps,
    convert_colmap_depth_maps_to_dmaps,
    find_colmap_binary,
    read_colmap_depth_map,
    run_colmap_dense_reconstruction,
)
from .depth_range import compute_depth_ranges, print_depth_range_stats
from .dmap import DMap, dmap_from_openmvs_output, read_dmap, write_dmap
from .depth_pipeline import DepthPipelineResult, run_depth_pipeline

__all__ = [
    "run_depth_pipeline",
    "DepthPipelineResult",
    "ColmapDenseConfig",
    "ColmapDenseResult",
    "run_colmap_dense_reconstruction",
    "find_colmap_binary",
    "read_colmap_depth_map",
    "collect_colmap_depth_maps",
    "convert_colmap_depth_maps_to_dmaps",
    "compute_depth_ranges",
    "print_depth_range_stats",
    "DMap",
    "read_dmap",
    "write_dmap",
    "dmap_from_openmvs_output",
]
