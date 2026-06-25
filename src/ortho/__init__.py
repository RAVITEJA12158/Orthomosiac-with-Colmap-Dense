"""
src/ortho/__init__.py

Stage 10 — Orthorectification pipeline entry point.

Orchestrates camera_model → dsm_sampler → footprint → backward_project →
tile_writer for every registered image in the reconstruction.

Called by the top-level pipeline runner after Stage 9 (DSM generation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ._xp import GPU_AVAILABLE
from .camera_model import CameraPose, extract_camera_poses
from .dsm_sampler import DSMSampler, load_dsm
from .footprint import BoundingBox, compute_image_footprint
from .backward_project import backward_project_image
from .tile_writer import write_ortho_tile, write_ortho_tile_multispectral


# Multispectral band attribute names on a Capture object
_MS_BANDS = ("green", "red", "reg", "nir")
_MS_BAND_NAMES = ("GRE", "RED", "REG", "NIR")


@dataclass
class OrthoResult:
    rgb_tile_paths: List[str] = field(default_factory=list)
    multi_tile_paths: Dict[str, List[str]] = field(default_factory=dict)
    crs: object = None              # rasterio CRS of all output tiles
    n_skipped: int = 0
    n_processed: int = 0


def _load_rgb(path: str) -> Optional[np.ndarray]:
    """Load an RGB image from disk as uint8 (H, W, 3). Returns None on failure."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _load_band(path: str) -> Optional[np.ndarray]:
    """
    Load a single multispectral band as float32 (H, W).
    Sequoia bands are stored as uint16 GeoTIFFs; normalise to [0, 1]
    reflectance range by dividing by 65535.
    """
    try:
        import rasterio
    except ImportError:
        return None

    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            # If values are uint16-range (> 1.0) normalise to reflectance
            if arr.max() > 1.0:
                arr = arr / 65535.0
            return arr
    except Exception:
        return None


def _find_capture_by_name(captures, image_name: str):
    """
    Match a reconstruction image name (e.g. '000123.jpg') to a Capture object
    by comparing the stem of the image name to capture_id.
    """
    stem = Path(image_name).stem
    for cap in captures:
        if hasattr(cap, "capture_id") and str(cap.capture_id) == stem:
            return cap
        # Fallback: match against the RGB filename stem
        if hasattr(cap, "rgb") and cap.rgb and Path(cap.rgb).stem == stem:
            return cap
    return None


def _ms_output_path(output_dir: str, band_name: str, image_name: str) -> str:
    stem = Path(image_name).stem
    return str(Path(output_dir) / "multi" / band_name / f"{stem}.tif")


def _rgb_output_path(output_dir: str, image_name: str) -> str:
    stem = Path(image_name).stem
    return str(Path(output_dir) / "rgb" / f"{stem}.tif")


def run_ortho_pipeline(
    reconstruction,
    captures,
    dsm_path: str,
    output_dir: str,
    target_gsd_m: float = 0.05,
    process_multispectral: bool = True,
    batch_size: int = 10,             # reserved for future batched GPU upload
) -> OrthoResult:
    """
    Run the full Stage 10 orthorectification pipeline.

    Args:
        reconstruction        : pycolmap.Reconstruction from Stage 6/7
        captures              : List[Capture] from Stage 2 ingestion
        dsm_path              : path to dsm.tif from Stage 9
        output_dir            : root output directory
                                rgb tiles  → <output_dir>/rgb/<name>.tif
                                MS tiles   → <output_dir>/multi/<BAND>/<name>.tif
        target_gsd_m          : output pixel size in metres (default 5 cm)
        process_multispectral : also orthorectify multispectral bands
        batch_size            : (unused in current single-image loop, reserved)

    Returns:
        OrthoResult with paths to all output tiles and the shared CRS.
    """
    print("[ortho] Stage 10 — Orthorectification")
    print(f"[ortho] GPU available: {GPU_AVAILABLE}")
    print(f"[ortho] Target GSD: {target_gsd_m} m/pixel")
    print(f"[ortho] DSM: {dsm_path}")

    # ── 1. Load DSM to GPU once ──────────────────────────────────────────────
    print("[ortho] Loading DSM to GPU...")
    dsm: DSMSampler = load_dsm(dsm_path)
    print(f"[ortho] DSM loaded: {dsm.width}×{dsm.height} px, "
          f"CRS: {dsm.crs.to_epsg()}, mean elevation: {dsm.mean_elevation:.1f} m")

    # ── 2. Extract camera poses from reconstruction ──────────────────────────
    poses: List[CameraPose] = extract_camera_poses(reconstruction)
    print(f"[ortho] {len(poses)} registered images to orthorectify")

    # ── 3. Create output directories ─────────────────────────────────────────
    Path(output_dir, "rgb").mkdir(parents=True, exist_ok=True)
    if process_multispectral:
        for band_name in _MS_BAND_NAMES:
            Path(output_dir, "multi", band_name).mkdir(parents=True, exist_ok=True)

    result = OrthoResult(crs=dsm.crs)
    result.multi_tile_paths = {b: [] for b in _MS_BAND_NAMES}

    t_start = time.perf_counter()
    per_image_times: List[float] = []

    # ── 4. Main loop — one image at a time on GPU ────────────────────────────
    for idx, pose in enumerate(poses):
        t_img = time.perf_counter()

        # ── 4a. Load raw RGB image ────────────────────────────────────────────
        # pose.image_name is a bare filename (e.g. "000123.jpg") relative to
        # the SfM image folder, which is almost never the cwd. Rather than
        # attempting a doomed cv2.imread(bare_name) first and wasting a syscall
        # per image across the whole mission, go straight to the captures
        # lookup which holds the correct absolute path.
        cap = _find_capture_by_name(captures, pose.image_name)
        rgb = None
        if cap is not None and hasattr(cap, "rgb") and cap.rgb:
            rgb = _load_rgb(cap.rgb)
        if rgb is None:
            print(f"[ortho] Warning: cannot load RGB for {pose.image_name} — skipping.")
            result.n_skipped += 1
            continue

        # ── 4b. Compute ground footprint ──────────────────────────────────────
        footprint: Optional[BoundingBox] = compute_image_footprint(
            pose, dsm, target_gsd_m
        )
        if footprint is None:
            print(f"[ortho] Warning: degenerate footprint for {pose.image_name} — skipping.")
            result.n_skipped += 1
            continue

        # ── 4c. Backward project RGB ──────────────────────────────────────────
        ortho_rgb, ortho_mask = backward_project_image(rgb, pose, dsm, footprint)

        # ── 4d. Write RGB GeoTIFF ─────────────────────────────────────────────
        rgb_out = _rgb_output_path(output_dir, pose.image_name)
        write_ortho_tile(ortho_rgb, ortho_mask, footprint, dsm.crs, rgb_out)
        result.rgb_tile_paths.append(rgb_out)

        # ── 4e. Multispectral bands (optional) ────────────────────────────────
        # `cap` was already resolved above in step 4a.
        if process_multispectral:
            if cap is not None:
                for attr, band_name in zip(_MS_BANDS, _MS_BAND_NAMES):
                    band_path = getattr(cap, attr, None)
                    if not band_path:
                        continue
                    band_arr = _load_band(band_path)
                    if band_arr is None:
                        continue

                    ortho_band, band_mask = backward_project_image(
                        band_arr, pose, dsm, footprint
                    )
                    ms_out = _ms_output_path(output_dir, band_name, pose.image_name)
                    write_ortho_tile_multispectral(
                        ortho_band, band_mask, footprint, dsm.crs, ms_out, band_name
                    )
                    result.multi_tile_paths[band_name].append(ms_out)

        result.n_processed += 1
        elapsed_img = time.perf_counter() - t_img
        per_image_times.append(elapsed_img)

        # Progress log every 50 images
        if (idx + 1) % 50 == 0 or (idx + 1) == len(poses):
            elapsed_total = time.perf_counter() - t_start
            avg = sum(per_image_times[-50:]) / len(per_image_times[-50:])
            remaining = avg * (len(poses) - idx - 1)
            pct = 100.0 * (idx + 1) / len(poses)
            print(
                f"[ortho] {idx + 1}/{len(poses)} ({pct:.1f}%)  "
                f"elapsed: {elapsed_total:.0f}s  "
                f"avg: {avg:.2f}s/img  "
                f"ETA: {remaining:.0f}s"
            )

    # ── 5. Summary ────────────────────────────────────────────────────────────
    total = time.perf_counter() - t_start
    print(f"[ortho] Done. Processed: {result.n_processed}  "
          f"Skipped: {result.n_skipped}  "
          f"Total time: {total:.1f}s  "
          f"({total / max(result.n_processed, 1):.2f}s/image)")
    print(f"[ortho] RGB tiles:  {len(result.rgb_tile_paths)}")
    for band_name, paths in result.multi_tile_paths.items():
        if paths:
            print(f"[ortho] {band_name} tiles: {len(paths)}")

    return result