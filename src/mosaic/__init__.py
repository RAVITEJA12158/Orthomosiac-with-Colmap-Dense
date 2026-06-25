"""
src/mosaic/__init__.py

Public entry points for Stage 11 (RGB Mosaicking) and Stage 12
(Multispectral Mosaicking).

    run_rgb_mosaic()  ->  rgb_orthomosaic.tif + seamlines.npz
    run_ms_mosaic()   ->  multispectral_orthomosaic.tif (reuses seamlines)

See mosaicking.md for the full design and the "Data Flow: Stage 10 -> 11 ->
12 -> 13" diagram for how this module fits with src/ortho/ (input) and
src/indices/ (output consumer).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS

from .blender_ms import blend_ms_mosaic, stack_ms_bands
from .blender_rgb import blend_rgb_mosaic
from .gain_compensator import compute_gain_maps
from .radiometric import calibrate_image, compute_mission_panel_factors
from .seam_finder import SeamlineSet, find_seamlines, save_seamlines
from .tile_loader import (
    CanvasInfo,
    compute_mosaic_canvas,
    load_tile_pixels,
    read_tile_infos,
)
from .vignetting import correct_vignetting_band, correct_vignetting_rgb, read_vignetting_coeffs

logger = logging.getLogger(__name__)

_MS_BAND_ORDER = ["GRE", "RED", "REG", "NIR"]
_DEFAULT_KNOWN_REFLECTANCES = {"GRE": 0.47, "RED": 0.47, "REG": 0.47, "NIR": 0.47}


@dataclass
class MosaicResult:
    rgb_mosaic_path: str
    ms_mosaic_path: str
    seamlines_path: str
    crs: CRS


def _read_exif_dict(image_path: str) -> dict:
    """
    Read EXIF/XMP metadata for one image via pyexiftool.

    Imported lazily so that modules which don't need radiometric
    calibration (e.g. RGB-only mosaicking) never require pyexiftool to be
    installed.
    """
    import exiftool

    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata([image_path])
    return metadata[0] if metadata else {}


def run_rgb_mosaic(
    tile_paths: List[str],
    output_path: str,
    seamlines_save_dir: str,
    target_gsd_m: float = 0.05,
) -> Tuple[str, SeamlineSet]:
    """
    Run the full Stage 11 RGB mosaicking pipeline.

    1. read_tile_infos(tile_paths)
    2. compute_mosaic_canvas(tile_infos, gsd)
    3. load + correct_vignetting_rgb() per tile  [all tiles held in memory simultaneously]
    4. compute_gain_maps()                        [second float32 copy created by seam finder]
    5. find_seamlines()
    6. save_seamlines(seamlines, save_dir)        <- CRITICAL, Stage 12 needs this
    7. blend_rgb_mosaic()
    8. write rgb_orthomosaic.tif via Rasterio
    9. return (output_path, seamline_set)

    Memory budget: all tile pixel arrays are loaded into RAM before gain
    compensation and seam-finding, contradicting tile_loader.py's "lazy
    one-at-a-time" docstring. For a 900-tile RGB mission at 5 cm GSD,
    peak RAM usage is typically 4–8 GB. Chunked/tiled canvas processing
    is a future improvement (see mosaicking.md §Memory).
    """
    import time as _time
    t0_mosaic = _time.perf_counter()

    logger.info("run_rgb_mosaic: starting with %d tiles", len(tile_paths))
    print(f"[mosaic/rgb] Stage 11 — RGB mosaicking  ({len(tile_paths)} tiles)")

    print("[mosaic/rgb] Step 1/7  Reading tile metadata...")
    tile_infos = read_tile_infos(tile_paths)
    print(f"[mosaic/rgb] Step 2/7  Computing canvas...  ({_time.perf_counter()-t0_mosaic:.0f}s)")
    canvas = compute_mosaic_canvas(tile_infos, target_gsd_m)
    print(f"[mosaic/rgb]           Canvas: {canvas.width_px}×{canvas.height_px} px")

    print(f"[mosaic/rgb] Step 3/7  Loading + vignetting correction...  ({_time.perf_counter()-t0_mosaic:.0f}s)")
    images: List[np.ndarray] = []
    for ti_idx, ti in enumerate(tile_infos, start=1):
        img = load_tile_pixels(ti)
        try:
            exif_dict = _read_exif_dict(ti.path)
            coeffs = read_vignetting_coeffs(exif_dict)
        except Exception as exc:  # pyexiftool missing / read failure
            logger.warning(
                "run_rgb_mosaic: could not read vignetting EXIF for %s (%s), using identity",
                ti.path, exc,
            )
            coeffs = read_vignetting_coeffs({})
        img = correct_vignetting_rgb(img, coeffs)
        images.append(img)
        if ti_idx % 100 == 0 or ti_idx == len(tile_infos):
            elapsed = _time.perf_counter() - t0_mosaic
            print(f"[mosaic/rgb]   vignetting {ti_idx}/{len(tile_infos)} ({100*ti_idx//len(tile_infos)}%)  elapsed: {elapsed:.0f}s")

    logger.info("run_rgb_mosaic: vignetting-corrected %d tiles", len(images))

    print(f"[mosaic/rgb] Step 4/7  Gain compensation...  ({_time.perf_counter()-t0_mosaic:.0f}s)")
    gain_images = compute_gain_maps(tile_infos, images, canvas)
    logger.info("run_rgb_mosaic: gain compensation done")

    print(f"[mosaic/rgb] Step 5/7  Finding seamlines...  ({_time.perf_counter()-t0_mosaic:.0f}s)")
    seamline_set = find_seamlines(tile_infos, gain_images, canvas)
    seamlines_path = save_seamlines(seamline_set, seamlines_save_dir)
    logger.info("run_rgb_mosaic: seamlines saved to %s", seamlines_path)

    print(f"[mosaic/rgb] Step 6/7  Blending mosaic...  ({_time.perf_counter()-t0_mosaic:.0f}s)")
    mosaic = blend_rgb_mosaic(tile_infos, gain_images, seamline_set, canvas)
    logger.info("run_rgb_mosaic: blending done, writing output")

    print(f"[mosaic/rgb] Step 7/7  Writing GeoTIFF...  ({_time.perf_counter()-t0_mosaic:.0f}s)")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=canvas.height_px,
        width=canvas.width_px,
        count=3,
        dtype="uint8",
        crs=canvas.crs,
        transform=canvas.transform,
        compress="lzw",
    ) as dst:
        for band_idx in range(3):
            dst.write(mosaic[:, :, band_idx], band_idx + 1)

    total_elapsed = _time.perf_counter() - t0_mosaic
    logger.info("run_rgb_mosaic: wrote %s", output_path)
    print(f"[mosaic/rgb] Done. Total time: {total_elapsed:.0f}s  Output: {output_path}")
    return output_path, seamline_set


def run_ms_mosaic(
    multi_tile_paths: Dict[str, List[str]],
    captures,
    seamline_set: SeamlineSet,
    output_path: str,
    known_reflectances: Optional[Dict[str, float]] = None,
    target_gsd_m: float = 0.05,
) -> str:
    """
    Run the full Stage 12 MS mosaicking pipeline.

    1. compute_mission_panel_factors(captures)  [once, all bands]
    2. for each band in [GRE, RED, REG, NIR]:
         a. read_tile_infos(band_paths)
         b. load_tile_pixels() (float32)
         c. correct_vignetting_band() per tile
         d. calibrate_image() per tile
         e. blend_ms_mosaic(tiles, seamline_set)
    3. stack_ms_bands([GRE, RED, REG, NIR])
    4. write multispectral_orthomosaic.tif via Rasterio (band order:
       Green, Red, RedEdge, NIR)

    `target_gsd_m` MUST match the value used in run_rgb_mosaic() — the
    reused seamline_set was computed on that canvas grid.
    """
    import time as _time
    t0_ms = _time.perf_counter()

    missing_bands = [b for b in _MS_BAND_ORDER if b not in multi_tile_paths]
    if missing_bands:
        raise ValueError(f"run_ms_mosaic: missing tile paths for band(s) {missing_bands}")

    known_reflectances = known_reflectances or dict(_DEFAULT_KNOWN_REFLECTANCES)

    print(f"[mosaic/ms] Stage 12 — Multispectral mosaicking  ({len(_MS_BAND_ORDER)} bands)")
    logger.info("run_ms_mosaic: computing panel calibration factors")
    print("[mosaic/ms] Computing panel calibration factors...")
    panel_factors = compute_mission_panel_factors(captures, known_reflectances)

    canvas: Optional[CanvasInfo] = None
    band_mosaics: Dict[str, np.ndarray] = {}

    for band_num, band_name in enumerate(_MS_BAND_ORDER, start=1):
        band_paths = multi_tile_paths[band_name]
        print(f"[mosaic/ms] Band {band_num}/{len(_MS_BAND_ORDER)}: {band_name}  "
              f"({len(band_paths)} tiles)  elapsed: {_time.perf_counter()-t0_ms:.0f}s")
        logger.info("run_ms_mosaic: processing band %s (%d tiles)", band_name, len(band_paths))

        tile_infos = read_tile_infos(band_paths)
        band_canvas = compute_mosaic_canvas(tile_infos, target_gsd_m)
        if canvas is None:
            canvas = band_canvas
        elif (canvas.width_px, canvas.height_px) != (band_canvas.width_px, band_canvas.height_px):
            logger.warning(
                "run_ms_mosaic: band %s canvas size %dx%d differs from reference %dx%d "
                "(check target_gsd_m and tile coverage are consistent across bands)",
                band_name, band_canvas.width_px, band_canvas.height_px,
                canvas.width_px, canvas.height_px,
            )

        calibrated_tiles: List[np.ndarray] = []
        for ti_idx, ti in enumerate(tile_infos, start=1):
            band_img = load_tile_pixels(ti)  # (H, W) float32 raw DN
            try:
                exif_dict = _read_exif_dict(ti.path)
            except Exception as exc:
                if isinstance(exc, ImportError) or "exiftool" in str(exc).lower() or "executable" in str(exc).lower() or "not found" in str(exc).lower():
                    raise RuntimeError(
                        "run_ms_mosaic: pyexiftool or the exiftool executable is required for "
                        "radiometric calibration but failed to run. Please ensure 'pyexiftool' is "
                        "installed via pip and 'exiftool' is available in your PATH."
                    ) from exc
                logger.warning(
                    "run_ms_mosaic: could not read EXIF for %s (%s), using degraded defaults",
                    ti.path, exc,
                )
                exif_dict = {}

            coeffs = read_vignetting_coeffs(exif_dict, band_name)
            band_img = correct_vignetting_band(band_img, coeffs)
            band_img = calibrate_image(band_img, exif_dict, panel_factors, band_name)
            calibrated_tiles.append(band_img)

            if ti_idx % 100 == 0 or ti_idx == len(tile_infos):
                elapsed = _time.perf_counter() - t0_ms
                print(f"[mosaic/ms]   {band_name} calibrated {ti_idx}/{len(tile_infos)} "
                      f"({100*ti_idx//len(tile_infos)}%)  elapsed: {elapsed:.0f}s")

        if len(tile_infos) != len(seamline_set.masks):
            logger.warning(
                "run_ms_mosaic: band %s has %d tiles but seamline_set has %d masks — "
                "tile-to-seamline correspondence relies on identical geographic "
                "footprints/sort order between RGB and every MS band. Mismatches "
                "here mean Stage 11 and Stage 12 were run on different capture sets.",
                band_name, len(tile_infos), len(seamline_set.masks),
            )

        print(f"[mosaic/ms]   {band_name} blending...  elapsed: {_time.perf_counter()-t0_ms:.0f}s")
        band_mosaic = blend_ms_mosaic(tile_infos, calibrated_tiles, seamline_set, canvas)
        band_mosaics[band_name] = band_mosaic
        logger.info("run_ms_mosaic: band %s blended", band_name)

    # canvas is set during the first band's iteration.  If it is still None here
    # every band produced an empty tile list — fail loudly rather than crashing
    # inside rasterio or stack_ms_bands with a confusing error.
    if canvas is None:
        raise RuntimeError(
            "run_ms_mosaic: canvas was never set — all MS bands produced empty tile lists. "
            "Check that run_ortho_pipeline() wrote multispectral tiles before calling run_ms_mosaic()."
        )

    print(f"[mosaic/ms] Stacking bands + writing GeoTIFF...  elapsed: {_time.perf_counter()-t0_ms:.0f}s")
    stacked = stack_ms_bands(band_mosaics, band_order=_MS_BAND_ORDER)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=canvas.height_px,
        width=canvas.width_px,
        count=4,
        dtype="float32",
        crs=canvas.crs,
        transform=canvas.transform,
        nodata=-9999.0,
        compress="lzw",
    ) as dst:
        band_descriptions = ["Green", "Red", "RedEdge", "NIR"]
        for band_idx in range(4):
            dst.write(stacked[band_idx], band_idx + 1)
            dst.set_band_description(band_idx + 1, band_descriptions[band_idx])

    total_elapsed = _time.perf_counter() - t0_ms
    logger.info("run_ms_mosaic: wrote %s", output_path)
    print(f"[mosaic/ms] Done. Total time: {total_elapsed:.0f}s  Output: {output_path}")
    return output_path