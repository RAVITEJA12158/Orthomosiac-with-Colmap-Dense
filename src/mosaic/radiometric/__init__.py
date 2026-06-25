"""
src/mosaic/radiometric/__init__.py

Public entry point for radiometric calibration. Auto-detects the camera
from EXIF Make/Model and dispatches to the matching reader; the
calibration math itself never changes per camera.

Full calibration formula (CLAUDE.md):
    calibrated = (raw - black_level) / vignetting
               / (exposure_time x gain)
               / irradiance
               x panel_factor

The caller (run_ms_mosaic) already divides by `vignetting` via
vignetting.correct_vignetting_band() BEFORE calling calibrate_image(), so
calibrate_image() applies the remaining four terms: black_level
subtraction, exposure/gain normalisation, irradiance correction, and the
panel factor.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from .generic_reader import read_generic_params
from .micasense_reader import read_micasense_params
from .panel_detector import compute_mission_panel_factors  # noqa: F401  (re-exported)
from .params import RadiometricParams  # noqa: F401  (re-exported)
from .sequoia_reader import read_sequoia_params

logger = logging.getLogger(__name__)

_READERS = {
    "sequoia": read_sequoia_params,
    "micasense": read_micasense_params,
}


def detect_camera(exif_dict: dict) -> str:
    """
    Identify the camera from EXIF Make/Model.
    Returns "sequoia" | "micasense" | "generic".
    """
    make = str(exif_dict.get("EXIF:Make", exif_dict.get("XMP:Make", ""))).lower()
    model = str(exif_dict.get("EXIF:Model", exif_dict.get("XMP:Model", ""))).lower()
    combined = f"{make} {model}"

    if "parrot" in combined or "sequoia" in combined:
        return "sequoia"
    if "micasense" in combined or "rededge" in combined:
        return "micasense"

    logger.warning(
        "detect_camera: unrecognised camera (Make=%r, Model=%r) — using generic "
        "fallback reader (degraded radiometric calibration)",
        exif_dict.get("EXIF:Make"), exif_dict.get("EXIF:Model"),
    )
    return "generic"


def calibrate_image(
    band_array: np.ndarray,
    exif_dict: dict,
    panel_factors: Dict[str, float],
    band_name: str,
) -> np.ndarray:
    """
    Convert a vignetting-corrected raw-DN band into absolute reflectance.

    band_array     : (H, W) float32 — raw DN, already vignetting-corrected
    exif_dict       : pyexiftool output for this image
    panel_factors   : {"GRE": factor, "RED": factor, ...} from
                      compute_mission_panel_factors()
    band_name       : "GRE" | "RED" | "REG" | "NIR"

    Returns (H, W) float32, clipped to [0.0, 1.0] (valid reflectance range).
    """
    camera = detect_camera(exif_dict)
    reader = _READERS.get(camera, read_generic_params)
    params: RadiometricParams = reader(exif_dict, band_name)

    denom = params.exposure_time * params.gain
    if denom <= 0:
        logger.warning(
            "calibrate_image: non-positive exposure*gain (%.6g) for band %s, using 1.0",
            denom, band_name,
        )
        denom = 1.0

    irradiance = params.irradiance if params.irradiance > 0 else 1.0
    panel_factor = panel_factors.get(band_name, 1.0)

    calibrated = (band_array.astype(np.float32) - params.black_level) / denom
    calibrated = calibrated / irradiance
    calibrated = calibrated * panel_factor

    calibrated = np.clip(calibrated, 0.0, 1.0).astype(np.float32)
    return calibrated


def build_panel_factors(
    panel_image_paths: Dict[str, str],
    known_reflectances: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper: detect the panel directly in known image paths
    (one per band) and return calibration factors, without needing the
    full mission capture list. Useful for ad-hoc / single-image
    calibration (e.g. testing, or a mission with only one panel shot).
    """
    from .panel_detector import detect_panel

    known_reflectances = known_reflectances or {b: 0.47 for b in panel_image_paths}
    factors: Dict[str, float] = {}
    for band_name, path in panel_image_paths.items():
        known_reflectance = known_reflectances.get(band_name, 0.47)
        factor = detect_panel(path, band_name, known_reflectance)
        factors[band_name] = factor if factor is not None else 1.0
        if factor is None:
            logger.warning(
                "build_panel_factors: no panel detected for band %s in %s — factor=1.0",
                band_name, path,
            )
    return factors
