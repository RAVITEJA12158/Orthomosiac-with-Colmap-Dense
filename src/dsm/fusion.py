"""
Stage 9a: resolve COLMAP stereo-fusion output for DSM rasterization.

COLMAP dense reconstruction now runs in Stage 8 and writes ``dense/fused.ply``.
Stage 9 consumes that point cloud directly and no longer invokes an external
fusion binary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_NUM_VIEWS_FUSE = 3


def resolve_fused_pointcloud(
    dmap_paths: Iterable[str],
    fused_pointcloud_path: Optional[str] = None,
    dense_workspace: Optional[str] = None,
) -> str:
    """
    Resolve the COLMAP fused point cloud associated with Stage 8 outputs.

    Search order:
      1. Explicit ``fused_pointcloud_path``.
      2. Explicit ``dense_workspace/fused.ply``.
      3. ``depth_manifest.json`` beside the Stage 8 dmap directory.
      4. Conventional ``<depth_dir>/dense/fused.ply`` location.
    """
    candidates: list[Path] = []

    if fused_pointcloud_path:
        candidates.append(Path(fused_pointcloud_path))
    if dense_workspace:
        candidates.append(Path(dense_workspace) / "fused.ply")

    dmap_list = [Path(path) for path in dmap_paths]
    if dmap_list:
        dmap_parent = dmap_list[0].parent
        depth_root = dmap_parent.parent if dmap_parent.name == "dmaps" else dmap_parent
        manifest_candidates = [
            depth_root / "depth_manifest.json",
            dmap_parent / "depth_manifest.json",
        ]
        for manifest_path in manifest_candidates:
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            manifest_ply = manifest.get("fused_pointcloud_path")
            if manifest_ply:
                candidates.append(Path(manifest_ply))

        candidates.extend(
            [
                depth_root / "dense" / "fused.ply",
                dmap_parent / "dense" / "fused.ply",
                depth_root / "fused.ply",
            ]
        )

    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return str(candidate.resolve())

    checked = ", ".join(str(path) for path in candidates) or "<none>"
    raise FileNotFoundError(
        "COLMAP fused point cloud not found. Stage 8 must complete stereo_fusion "
        f"before DSM generation. Checked: {checked}"
    )


def run_fusion(*args, **kwargs) -> str:
    """
    Compatibility wrapper for older imports.

    The fusion step is performed by COLMAP in Stage 8. New callers should use
    ``resolve_fused_pointcloud``.
    """
    if args:
        kwargs.setdefault("dmap_paths", args[0])
    return resolve_fused_pointcloud(**kwargs)

