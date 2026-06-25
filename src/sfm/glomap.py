"""
Stage 6, Step 3b — GLOMAP branch with 3-check validation + COLMAP fallback.

GLOMAP is the fast path (5-10 min vs 20-30 min COLMAP).
Only attempted when has_rtk=True — RTK GPS priors are the anchor that
prevents GLOMAP folding the field on repetitive crop textures.

Three validation checks after GLOMAP runs:
  Check 1 — Registration ratio   : >95% of keyframes registered
  Check 2 — GPS reprojection     : mean camera position error < threshold
  Check 3 — Bounding box         : reconstruction covers >90% of GPS field span

Any check fails → log reason → caller falls back to COLMAP incremental.

API surface confirmed against pycolmap 4.0.4 in previous session:
  pycolmap.global_mapping()            → dict[int, Reconstruction]
  pycolmap.GlobalPipelineOptions       → .image_names (list[str])
  reconstruction.num_reg_images        → int
  reconstruction.num_images            → int (total in db, not registered)
  image.projection_center()            → np.ndarray [x, y, z] ECEF
  reconstruction.compute_bounding_box() → (min_xyz, max_xyz)
  pycolmap.GPSTransform().ellipsoid_to_ecef(np.array([[lat, lon, alt]])) → [[x, y, z]]
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..colmap_images import colmap_image_name
from ..ingestion.capture import Capture


# ── Validation thresholds ─────────────────────────────────────────────────────

MIN_REGISTRATION_RATIO   = 0.95   # >95% keyframes must be registered
MAX_GPS_REPROJECTION_M   = 5.0    # mean camera-vs-GPS error in meters
MIN_BBOX_COVERAGE_RATIO  = 0.90   # reconstruction must cover >90% of GPS span


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_num_reg_images(recon) -> int:
    val = recon.num_reg_images
    return int(val() if callable(val) else val)

def _best_reconstruction(reconstructions: dict) -> Optional[object]:
    """
    Pick the largest disconnected component returned by GLOMAP.
    Best = most registered images.
    """
    if not reconstructions:
        return None
    return max(reconstructions.values(), key=_get_num_reg_images)


def _gps_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    """
    Convert WGS84 lat/lon/alt to ECEF XYZ using pycolmap's own transform.
    Identical to what COLMAP uses internally for pose priors, so comparison
    with projection_center() is in the same coordinate frame.
    """
    try:
        import pycolmap
    except ModuleNotFoundError:
        return _wgs84_to_ecef(lat, lon, alt)

    transform = pycolmap.GPSTransform()
    ecef = transform.ellipsoid_to_ecef(np.array([[lat, lon, alt]], dtype="float64"))
    return np.asarray(ecef, dtype="float64")[0]


def _wgs84_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    """Convert WGS84 geodetic coordinates to ECEF without pycolmap."""
    semi_major_axis = 6378137.0
    eccentricity_sq = 6.69437999014e-3

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    prime_vertical_radius = semi_major_axis / math.sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)

    x = (prime_vertical_radius + alt) * cos_lat * cos_lon
    y = (prime_vertical_radius + alt) * cos_lat * sin_lon
    z = (prime_vertical_radius * (1.0 - eccentricity_sq) + alt) * sin_lat

    return np.array([x, y, z], dtype="float64")


def _camera_gps_pairs(reconstruction, captures: List[Capture]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return matched camera centers and GPS ECEF positions for registered images.

    The two returned arrays have shape (N, 3): computed camera centers first,
    expected GPS positions second.
    """
    cap_by_id = {c.capture_id: c for c in captures}
    computed = []
    expected = []

    for _, image in reconstruction.images.items():
        capture_id = Path(image.name).stem
        cap = cap_by_id.get(capture_id)
        if cap is None or cap.latitude is None or cap.longitude is None:
            continue
        if getattr(image, "is_registered", lambda: True)():
            try:
                center = image.projection_center()
            except AttributeError:
                if hasattr(image, "cam_from_world"):
                    cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
                    center = -cfw.rotation.matrix().T @ cfw.translation
                else:
                    center = -image.rotation_matrix().T @ image.tvec
            computed.append(np.asarray(center, dtype="float64"))
            expected.append(_gps_to_ecef(
                cap.latitude,
                cap.longitude,
                cap.altitude or 0.0,
            ))

    if not computed:
        return np.empty((0, 3)), np.empty((0, 3))

    return np.vstack(computed), np.vstack(expected)


def _align_points_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Align source points to target points with a least-squares similarity.

    This makes validation robust when GLOMAP returns a reconstruction in an
    arbitrary local frame before the final GPS alignment stage. Folding still
    fails because the shape cannot fit the GPS positions with low residuals.
    """
    if len(source) < 3:
        return source

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = (target_centered.T @ source_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)

    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1

    rotation = u @ np.diag(sign) @ vt
    variance = np.mean(np.sum(source_centered ** 2, axis=1))
    if variance <= 0:
        return source

    scale = float(np.sum(singular_values * sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)

    return (scale * (rotation @ source.T)).T + translation


def _bbox_area_two_largest(points: np.ndarray) -> float:
    """Area of the two largest dimensions of a point cloud bounding box."""
    if len(points) < 2:
        return 0.0
    span = np.ptp(points, axis=0)
    two_largest = sorted(span)[-2:]
    return float(two_largest[0] * two_largest[1])


def _set_option_if_present(options, name: str, value) -> bool:
    """Set a pycolmap option only when this binding version exposes it."""
    try:
        getattr(options, name)
    except AttributeError:
        return False

    try:
        setattr(options, name, value)
        return True
    except Exception:
        return False


def _gps_field_span_m(captures: List[Capture]) -> Tuple[float, float]:
    """
    Compute expected field bounding box size in meters from GPS coordinates.
    Returns (lat_span_m, lon_span_m).
    Used for Check 3 — bounding box coverage.
    """
    gps = [c for c in captures if c.latitude is not None]
    if len(gps) < 2:
        return 0.0, 0.0

    lats = [c.latitude for c in gps]
    lons = [c.longitude for c in gps]
    alts = [c.altitude or 0.0 for c in gps]
    mean_alt = sum(alts) / len(alts)

    # Convert corners to ECEF and measure span
    min_pt = np.array(_gps_to_ecef(min(lats), min(lons), mean_alt))
    max_pt = np.array(_gps_to_ecef(max(lats), max(lons), mean_alt))
    span   = np.abs(max_pt - min_pt)

    return float(span[0]), float(span[1])  # approximate X and Y span in meters


# ── Validation ────────────────────────────────────────────────────────────────

def validate_glomap_reconstruction(
    reconstruction,
    keyframes         : List[Capture],
    has_rtk           : bool = True,
    min_reg_ratio     : float = MIN_REGISTRATION_RATIO,
    max_gps_error_m   : float = MAX_GPS_REPROJECTION_M,
    min_bbox_coverage : float = MIN_BBOX_COVERAGE_RATIO,
) -> Tuple[bool, str]:
    """
    Run 3 checks on a GLOMAP reconstruction.

    Returns:
        (passed: bool, reason: str)
        reason is empty string on pass, describes failure on fail.

    Check 1 — Registration ratio
        Are enough keyframes actually in the reconstruction?
        GLOMAP failing to register >5% of keyframes means something went wrong.

    Check 2 — GPS reprojection error (RTK only)
        For each registered image, compare COLMAP's computed camera position
        (projection_center, in ECEF via use_prior_position=True) against
        the expected ECEF position from the capture's GPS coordinates.
        Mean error > threshold = reconstruction is spatially wrong.
        Only meaningful when has_rtk=True (cm-accuracy priors).
        With normal GPS (3-5m accuracy), even a correct reconstruction
        will show 3-5m error — skip this check to avoid false positives.

    Check 3 — Bounding box coverage
        Does the reconstruction's spatial extent cover >=90% of the GPS
        bounding box of the input captures?
        A folded field would show ~50% coverage — both halves overlap.
    """
    n_reg   = _get_num_reg_images(reconstruction)
    n_total = len(keyframes)

    # ── Check 1: Registration ratio ───────────────────────────────────────────
    reg_ratio = n_reg / n_total if n_total > 0 else 0.0
    if reg_ratio < min_reg_ratio:
        return False, (
            f"Check 1 FAIL: only {n_reg}/{n_total} keyframes registered "
            f"({reg_ratio*100:.1f}% < required {min_reg_ratio*100:.1f}%)"
        )
    print(f"[glomap_validate] Check 1 PASS: {n_reg}/{n_total} registered "
          f"({reg_ratio*100:.1f}%)")

    # ── Check 2: GPS reprojection error (RTK only) ────────────────────────────
    camera_points, gps_points = _camera_gps_pairs(reconstruction, keyframes)
    aligned_camera_points = (
        _align_points_similarity(camera_points, gps_points)
        if len(camera_points) >= 3
        else camera_points
    )

    if has_rtk:
        if len(camera_points) == 0:
            return False, "Check 2 FAIL: no registered images with GPS to compare."

        errors = np.linalg.norm(aligned_camera_points - gps_points, axis=1)
        mean_error = float(np.mean(errors))
        if mean_error > max_gps_error_m:
            return False, (
                f"Check 2 FAIL: mean GPS reprojection error {mean_error:.2f}m "
                f"> threshold {max_gps_error_m}m "
                f"(checked {len(errors)} images)"
            )
        print(f"[glomap_validate] Check 2 PASS: mean GPS error "
              f"{mean_error:.2f}m <= {max_gps_error_m}m "
              f"({len(errors)} images)")
    else:
        print("[glomap_validate] Check 2 SKIP: not RTK (normal GPS error too large for this check).")

    # ── Check 3: Bounding box coverage ────────────────────────────────────────
    if len(camera_points) >= 3:
        recon_area = _bbox_area_two_largest(aligned_camera_points)
        gps_area = _bbox_area_two_largest(gps_points)

        if gps_area > 0:
            # Use the two larger dimensions (field is flat — Z span is small)
            coverage = recon_area / gps_area

            if coverage < min_bbox_coverage:
                return False, (
                    f"Check 3 FAIL: reconstruction covers {coverage*100:.1f}% "
                    f"of expected field area (< {min_bbox_coverage*100:.1f}%). "
                    f"Possible folding detected."
                )
            print(f"[glomap_validate] Check 3 PASS: coverage {coverage*100:.1f}% "
                  f">= {min_bbox_coverage*100:.1f}%")
        else:
            print("[glomap_validate] Check 3 SKIP: GPS span too small to compare.")
    else:
        # Fewer than 3 camera-GPS pairs means we cannot construct a meaningful
        # bounding box in GPS space.  The COLMAP bounding box (in reconstruction
        # frame) and the GPS span (in metres) use different coordinate origins
        # and cannot be compared directly — doing so produces unreliable
        # coverage ratios.  Skip the check rather than risk a false positive.
        print(
            f"[glomap_validate] Check 3 SKIP: only {len(camera_points)} camera-GPS "
            "pair(s) available (need >= 3 for a reliable area comparison)."
        )


    return True, ""


# ── GLOMAP runner ─────────────────────────────────────────────────────────────

def run_glomap(
    database_path   : str,
    image_dir       : str,
    output_dir      : str,
    keyframes       : List[Capture],
    has_rtk         : bool = True,
) -> Optional[object]:
    """
    Step 3b — Run GLOMAP (global SfM) on keyframes only.
    Validates result with 3 checks.

    Args:
        database_path : COLMAP .db with features, matches, pose priors
        output_dir    : where to write sparse/ model
        keyframes     : keyframe Capture list (for validation)
        has_rtk       : controls GPS check severity in validation

    Returns:
        Reconstruction if all 3 checks pass, None otherwise.
        Caller falls back to COLMAP incremental on None.
    """
    import pycolmap

    output_path = Path(output_dir) / "glomap"
    output_path.mkdir(parents=True, exist_ok=True)

    # Build keyframe image name list for GLOMAP
    # GLOMAP processes only these images, not the full database
    image_names = [colmap_image_name(cap) for cap in keyframes]

    print(f"[glomap] Running GLOMAP on {len(keyframes)} keyframes...")

    options = pycolmap.GlobalPipelineOptions()
    try:
        options.image_names = image_names
    except Exception as e:
        print(f"[glomap] Cannot restrict GLOMAP to keyframes: {e}. "
              "Falling back to COLMAP incremental.")
        return None

    if has_rtk:
        if _set_option_if_present(options, "use_prior_position", True):
            print("[glomap] Enabled use_prior_position for RTK priors.")
        else:
            print("[glomap] Warning: this pycolmap GlobalPipelineOptions build "
                  "does not expose use_prior_position; validation will guard fallback.")

    try:
        reconstructions = pycolmap.global_mapping(
            database_path = str(database_path),
            image_path    = str(image_dir),
            output_path   = str(output_path),
            options       = options,
        )
    except Exception as e:
        print(f"[glomap] GLOMAP crashed: {e}")
        return None

    recon = _best_reconstruction(reconstructions)
    if recon is None:
        print("[glomap] GLOMAP produced no reconstruction.")
        return None

    print(f"[glomap] GLOMAP registered {_get_num_reg_images(recon)} images. Validating...")

    passed, reason = validate_glomap_reconstruction(
        recon, keyframes, has_rtk=has_rtk
    )

    if not passed:
        print(f"[glomap] Validation FAILED — {reason}")
        return None

    print("[glomap] Validation PASSED. Using GLOMAP reconstruction.")
    return recon
