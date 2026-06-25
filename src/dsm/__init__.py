"""
Stage 9: DSM generation from COLMAP dense reconstruction.

Call order:
  1. Resolve COLMAP ``fused.ply`` written by Stage 8.
  2. Rasterize fused point cloud to ``dsm_raw.tif``.
  3. Fill gaps into final ``dsm.tif``.
  4. Return ``dsm.tif`` for orthorectification.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .fusion import DEFAULT_NUM_VIEWS_FUSE, resolve_fused_pointcloud, run_fusion
from .interpolate import check_gap_coverage, fill_dsm_gaps
from .rasterize import DEFAULT_TARGET_GSD_M, rasterize_pointcloud

__all__ = ["run_dsm_pipeline", "run_fusion", "rasterize_pointcloud", "fill_dsm_gaps"]


def run_dsm_pipeline(
    dmap_paths: List[str],
    reconstruction,
    output_dir: str,
    target_gsd_m: float = DEFAULT_TARGET_GSD_M,
    num_views_fuse: int = DEFAULT_NUM_VIEWS_FUSE,
    keep_pointcloud: bool = False,
    crs: Optional[str] = None,
    fused_pointcloud_path: Optional[str] = None,
    dense_workspace: Optional[str] = None,
) -> str:
    """
    Run Stage 9 DSM generation.

    ``dmap_paths`` remains the Stage 8 validation boundary. The actual point
    cloud rasterized here is COLMAP's ``stereo_fusion`` output, either supplied
    explicitly or discovered from ``depth_manifest.json``.
    """
    if not dmap_paths:
        raise ValueError(
            "run_dsm_pipeline received an empty dmap_paths list. Stage 8 must "
            "produce at least one .dmap file before Stage 9 can run."
        )

    missing = [path for path in dmap_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} .dmap file(s) from Stage 8 are missing on disk, e.g.: {missing[:3]}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fused_ply = resolve_fused_pointcloud(
        dmap_paths=dmap_paths,
        fused_pointcloud_path=fused_pointcloud_path,
        dense_workspace=dense_workspace,
    )

    dsm_raw_path = str(out_dir / "dsm_raw.tif")
    rasterize_pointcloud(
        ply_path=fused_ply,
        output_path=dsm_raw_path,
        reconstruction=reconstruction,
        target_gsd_m=target_gsd_m,
        crs=crs,
    )

    dsm_path = str(out_dir / "dsm.tif")
    fill_dsm_gaps(dsm_path=dsm_raw_path, output_path=dsm_path)
    check_gap_coverage(dsm_path=dsm_path)

    if not keep_pointcloud:
        local_copy = out_dir / "fused.ply"
        try:
            if Path(fused_ply).resolve() == local_copy.resolve():
                os.remove(fused_ply)
        except OSError:
            pass

    return dsm_path
