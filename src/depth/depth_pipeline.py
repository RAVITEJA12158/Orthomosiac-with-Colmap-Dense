"""
Stage 8 public pipeline: COLMAP dense stereo depth maps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from .colmap_dense import (
    ColmapDenseConfig,
    ColmapDenseResult,
    run_colmap_dense_reconstruction,
)
from .depth_range import compute_depth_ranges, print_depth_range_stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthPipelineResult:
    """
    Stage 8 output contract.

    ``dmap_paths`` keeps the previous DSM handoff stable. ``fused_pointcloud_path``
    is the COLMAP stereo-fusion output used by Stage 9 to build the DSM without
    any OpenMVS scene conversion.
    """

    dmap_paths: List[str]
    fused_pointcloud_path: str
    colmap_depth_map_paths: List[str]
    dense_workspace: str
    manifest_path: str

    def __iter__(self) -> Iterator:
        """
        Backward-compatible tuple unpacking.

        Existing callers that unpack two values receive
        ``(dmap_paths, fused_pointcloud_path)``.
        """
        yield self.dmap_paths
        yield self.fused_pointcloud_path


def run_depth_pipeline(
    reconstruction,
    captures: List,
    output_dir: str,
    colmap_sparse_dir: Optional[str] = None,
    use_gpu: bool = True,
    resolution_level: int = 1,
    num_neighbors: int = 5,
    colmap_bin_dir: str = "",
    max_image_size: Optional[int] = None,
    print_stats: bool = True,
) -> DepthPipelineResult:
    """
    Run Stage 8 with COLMAP 4.x dense reconstruction.

    Produces:
      - native COLMAP depth maps in ``<output_dir>/dense/stereo/depth_maps``
      - compatibility ``.dmap`` files in ``<output_dir>/dmaps``
      - COLMAP fused cloud in ``<output_dir>/dense/fused.ply``
      - ``depth_manifest.json`` for resume and Stage 9 discovery
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "=== Stage 8: COLMAP Dense Stereo === %d images, %d captures",
        len(getattr(reconstruction, "images", [])),
        len(captures),
    )

    logger.info("Step 1/4: Computing sparse SfM depth ranges")
    depth_ranges = compute_depth_ranges(reconstruction, captures)
    if print_stats:
        print_depth_range_stats(depth_ranges)

    logger.info("Step 2/4: Running COLMAP image undistortion and PatchMatch")
    config = ColmapDenseConfig(
        use_gpu=use_gpu,
        resolution_level=resolution_level,
        num_neighbors=num_neighbors,
        colmap_bin_dir=colmap_bin_dir,
        max_image_size=max_image_size,
        geom_consistency=True,
    )
    dense_result: ColmapDenseResult = run_colmap_dense_reconstruction(
        reconstruction=reconstruction,
        captures=captures,
        output_dir=str(output_path),
        colmap_sparse_dir=colmap_sparse_dir,
        depth_ranges=depth_ranges,
        config=config,
    )

    logger.info("Step 3/4: Validating converted .dmap files")
    dmap_paths = _validate_dmap_outputs(dense_result.dmap_paths)

    logger.info("Step 4/4: Validating COLMAP fused point cloud")
    fused_pointcloud = Path(dense_result.fused_pointcloud_path)
    if not fused_pointcloud.is_file() or fused_pointcloud.stat().st_size == 0:
        raise RuntimeError(f"COLMAP fused point cloud is missing or empty: {fused_pointcloud}")

    logger.info(
        "=== Stage 8 complete: %d .dmap files and fused cloud ready for Stage 9 ===",
        len(dmap_paths),
    )
    return DepthPipelineResult(
        dmap_paths=dmap_paths,
        fused_pointcloud_path=str(fused_pointcloud),
        colmap_depth_map_paths=dense_result.colmap_depth_map_paths,
        dense_workspace=dense_result.dense_workspace,
        manifest_path=dense_result.manifest_path,
    )


def _validate_dmap_outputs(dmap_paths: List[str]) -> List[str]:
    valid: List[str] = []
    missing: List[str] = []
    empty: List[str] = []

    for path_string in dmap_paths:
        path = Path(path_string)
        if not path.exists():
            missing.append(path_string)
        elif path.stat().st_size == 0:
            empty.append(path_string)
        else:
            valid.append(str(path))

    if missing:
        logger.warning("%d .dmap files are missing, e.g. %s", len(missing), missing[:3])
    if empty:
        logger.warning("%d .dmap files are empty, e.g. %s", len(empty), empty[:3])
    if not valid:
        raise RuntimeError("Stage 8 produced no valid .dmap files.")

    return sorted(valid)
