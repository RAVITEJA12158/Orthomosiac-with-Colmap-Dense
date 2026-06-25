"""
src/mosaic/radiometric/panel_detector.py

Finds the reflectance calibration panel in mission images and computes the
per-band factor that converts raw DN into absolute reflectance:

    calibration_factor = known_reflectance / mean_DN_inside_panel

The panel has 4 ArUco markers, one at each corner, surrounding a uniform
grey square of known reflectance. Detection works on a single band image
(no colour needed) because ArUco markers are pure geometric patterns.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ARUCO_DICT = cv2.aruco.DICT_4X4_50

# How far to shrink the marker-center quadrilateral toward its centroid
# before sampling DN. Markers themselves (black/white pattern) must be
# excluded from the sample — only the uniform grey panel surface in
# between them should be measured. Tune this against real panel photos if
# the grey square turns out larger/smaller relative to marker spacing.
_PANEL_INSET_FACTOR = 0.55


def _load_band_image(image_path: str) -> Optional[np.ndarray]:
    """Load a single-band (or single-channel-of) image as float32 raw DN."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning("panel_detector: could not read image %s", image_path)
        return None
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)


def _detect_markers(image: np.ndarray):
    """Run ArUco detection on a float image (converted to 8-bit for detection)."""
    img_min, img_max = float(np.min(image)), float(np.max(image))
    if img_max - img_min < 1e-6:
        return [], None
    img_u8 = ((image - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _rejected = detector.detectMarkers(img_u8)
    return corners, ids


def detect_panel(
    image_path: str,
    band_name: str,
    known_reflectance: float,
) -> Optional[float]:
    """
    Detect the calibration panel's 4 ArUco markers in `image_path` and
    compute the calibration factor for `band_name`.

    Returns the calibration factor, or None if fewer than 4 markers were
    found (no usable panel in this image).
    """
    image = _load_band_image(image_path)
    if image is None:
        return None

    corners, ids = _detect_markers(image)
    if ids is None or len(ids) < 4:
        logger.debug(
            "detect_panel: %d/4 ArUco markers found in %s (band %s)",
            0 if ids is None else len(ids), image_path, band_name,
        )
        return None

    # Take the first 4 detected markers' centers (the panel kit places
    # exactly 4 markers around the grey square; extra spurious detections
    # are not expected in practice but are ignored here defensively).
    marker_centers = np.array([c.reshape(-1, 2).mean(axis=0) for c in corners[:4]], dtype=np.float32)
    centroid = marker_centers.mean(axis=0)

    inset_points = centroid + (marker_centers - centroid) * _PANEL_INSET_FACTOR
    inset_points = _order_quadrilateral(inset_points)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [inset_points.astype(np.int32)], 255)

    panel_pixels = image[mask > 0]
    if panel_pixels.size < 16:
        logger.warning(
            "detect_panel: panel mask too small (%d px) in %s, skipping",
            panel_pixels.size, image_path,
        )
        return None

    mean_dn = float(np.mean(panel_pixels))
    if mean_dn <= 0:
        logger.warning("detect_panel: mean panel DN <= 0 in %s, skipping", image_path)
        return None

    calibration_factor = known_reflectance / mean_dn
    logger.info(
        "detect_panel: band=%s mean_DN=%.3f known_reflectance=%.3f -> factor=%.6g (%s)",
        band_name, mean_dn, known_reflectance, calibration_factor, image_path,
    )
    return calibration_factor


def _order_quadrilateral(points: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise starting from top-left, for a clean fillPoly."""
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return points[np.argsort(angles)]


def find_panel_captures(captures: List, band_name: str) -> List[str]:
    """
    Identify which captures likely contain the calibration panel.

    Strategy: the panel is normally shown to the camera just before takeoff
    and/or just after landing, so check the first and last few captures of
    the flight (captures is assumed sorted in flight order, as produced by
    Stage 2 ingestion).
    """
    band_attr = {"GRE": "green", "RED": "red", "REG": "reg", "NIR": "nir"}.get(band_name)
    if band_attr is None:
        raise ValueError(f"find_panel_captures: unknown band_name {band_name!r}")

    n_check = min(5, len(captures))
    candidate_captures = list(captures[:n_check]) + list(captures[-n_check:])

    candidate_paths: List[str] = []
    seen = set()
    for cap in candidate_captures:
        path = getattr(cap, band_attr, None)
        if not path or path in seen:
            continue
        seen.add(path)

        image = _load_band_image(path)
        if image is None:
            continue
        _corners, ids = _detect_markers(image)
        if ids is not None and len(ids) >= 4:
            candidate_paths.append(path)

    logger.info(
        "find_panel_captures: %d candidate panel image(s) found for band %s "
        "(checked %d edge captures)",
        len(candidate_paths), band_name, len(candidate_captures),
    )
    return candidate_paths


def compute_mission_panel_factors(
    captures: List,
    known_reflectances: Dict[str, float],
) -> Dict[str, float]:
    """
    Run detect_panel() across every band/capture combination where the
    panel was found, average the per-band factors, and return them.

    Falls back to a factor of 1.0 (uncalibrated passthrough) for any band
    where no panel was detected at all, logging a WARNING.
    """
    factors: Dict[str, float] = {}

    for band_name, known_reflectance in known_reflectances.items():
        candidate_paths = find_panel_captures(captures, band_name)

        band_factors = []
        for path in candidate_paths:
            factor = detect_panel(path, band_name, known_reflectance)
            if factor is not None:
                band_factors.append(factor)

        if band_factors:
            factors[band_name] = float(np.mean(band_factors))
            logger.info(
                "compute_mission_panel_factors: band=%s factor=%.6g (averaged over %d detections)",
                band_name, factors[band_name], len(band_factors),
            )
        else:
            logger.warning(
                "compute_mission_panel_factors: no panel detected for band %s — "
                "falling back to factor=1.0 (UNCALIBRATED, reflectance values will "
                "be relative DN, not absolute reflectance)",
                band_name,
            )
            factors[band_name] = 1.0

    return factors
