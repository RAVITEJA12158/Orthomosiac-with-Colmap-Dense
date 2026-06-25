"""
src/mosaic/vignetting.py

Removes lens brightness falloff (vignetting) from orthotiles before any
colour/radiometric processing. Applied to both RGB tiles (Stage 11) and MS
band tiles (Stage 12) — for MS tiles this MUST run before radiometric
calibration, because the calibration math assumes the falloff is already
gone (see CLAUDE.md full formula:
    calibrated = (raw - black_level) / vignetting / (exposure*gain) / irradiance * panel_factor
).

Vignetting model (Sequoia XMP `VignettingPolynomial`):
    vignette(r) = 1 + a1*r^2 + a2*r^4 + a3*r^6 + ...
    r = normalised radial distance from the optical centre
        (0 at centre, 1 at the corner of the sensor)
    pixel_corrected = pixel_raw / vignette(r)

Identity correction (no-op) is coefficients = [0, 0, ...] -> vignette(r) == 1
everywhere. This is used whenever the EXIF field is missing, so the rest of
the pipeline never has to special-case "no vignetting data".
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_COEFFS = np.zeros(3, dtype=np.float32)  # identity: vignette(r) == 1


def _radial_grid(height: int, width: int, image_center: Optional[Tuple[float, float]]) -> np.ndarray:
    """Normalised radial distance grid, 0 at centre, 1 at the corner."""
    cx, cy = image_center if image_center is not None else (width / 2.0, height / 2.0)
    yy, xx = np.indices((height, width), dtype=np.float32)
    r_max = float(np.sqrt(cx ** 2 + cy ** 2))
    if r_max <= 0:
        return np.zeros((height, width), dtype=np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r_max
    return r


def _evaluate_vignette(r: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """vignette(r) = 1 + a1*r^2 + a2*r^4 + a3*r^6 + ..."""
    coeffs = np.asarray(coeffs, dtype=np.float32).ravel()
    vignette = np.ones_like(r, dtype=np.float32)
    for i, a in enumerate(coeffs):
        if a == 0:
            continue
        power = 2 * (i + 1)
        vignette += a * np.power(r, power)
    # Guard against degenerate / extreme coefficients producing <= 0.
    vignette = np.clip(vignette, 1e-3, None)
    return vignette


def correct_vignetting_rgb(
    image: np.ndarray,
    vignetting_coeffs: np.ndarray,
    image_center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Remove vignetting from an RGB tile.

    image             : (H, W, 3) uint8
    vignetting_coeffs  : 1D array of polynomial coefficients [a1, a2, a3, ...]
    image_center       : (cx, cy) in pixels, defaults to image centre

    Returns (H, W, 3) uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"correct_vignetting_rgb expects (H, W, 3), got {image.shape}")

    h, w = image.shape[:2]
    r = _radial_grid(h, w, image_center)
    vignette = _evaluate_vignette(r, vignetting_coeffs)

    corrected = image.astype(np.float32) / vignette[..., None]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def correct_vignetting_band(
    band: np.ndarray,
    vignetting_coeffs: np.ndarray,
    image_center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Remove vignetting from a single multispectral band.

    band              : (H, W) float32 — raw DN (before radiometric calibration)
    vignetting_coeffs : 1D array of polynomial coefficients [a1, a2, a3, ...]

    Returns (H, W) float32. Not clipped to [0, 255] — this is raw DN, not an
    8-bit image, and downstream radiometric calibration expects the true
    (un-clamped) corrected value.
    """
    if band.ndim != 2:
        raise ValueError(f"correct_vignetting_band expects (H, W), got {band.shape}")

    h, w = band.shape
    r = _radial_grid(h, w, image_center)
    vignette = _evaluate_vignette(r, vignetting_coeffs)

    corrected = band.astype(np.float32) / vignette
    return corrected.astype(np.float32)


def read_vignetting_coeffs(exif_dict: dict, band: Optional[str] = None) -> np.ndarray:
    """
    Parse vignetting polynomial coefficients from a pyexiftool output dict.

    Tries, in order:
        XMP:VignettingPolynomial
        XMP:VignettingPolynomial2
        EXIF:VignettingPolynomial   (some tools flatten the group prefix)

    The value is typically a comma/space-separated string, e.g. "0.12,0.05,0.01",
    but pyexiftool may also return it as a list of floats directly — both are
    handled.

    Returns a 1D float32 array [a1, a2, a3, ...].
    Returns zeros (identity — no correction) if no matching key is found, or
    if the value can't be parsed. This is a safe fallback: it means
    "vignetting unknown, don't touch the image" rather than crashing the
    pipeline on one bad/missing EXIF tag.
    """
    candidate_keys = [
        "XMP:VignettingPolynomial",
        "XMP:VignettingPolynomial2",
        "EXIF:VignettingPolynomial",
        "XMP:VignettingPoly",
    ]

    raw_value = None
    for key in candidate_keys:
        if key in exif_dict and exif_dict[key] not in (None, ""):
            raw_value = exif_dict[key]
            break

    if raw_value is None:
        logger.debug("read_vignetting_coeffs: no vignetting field found, using identity")
        return _DEFAULT_COEFFS.copy()

    try:
        if isinstance(raw_value, (list, tuple, np.ndarray)):
            coeffs = np.array([float(v) for v in raw_value], dtype=np.float32)
        else:
            text = str(raw_value).strip()
            parts = text.replace(";", ",").split(",") if "," in text else text.split()
            coeffs = np.array([float(p) for p in parts if p.strip() != ""], dtype=np.float32)

        if coeffs.size == 0:
            raise ValueError("parsed empty coefficient list")
        return coeffs
    except (ValueError, TypeError) as exc:
        logger.warning(
            "read_vignetting_coeffs: could not parse vignetting field %r (%s), using identity",
            raw_value, exc,
        )
        return _DEFAULT_COEFFS.copy()
