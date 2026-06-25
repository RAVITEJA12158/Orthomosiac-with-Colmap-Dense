"""
depth_range.py
==============
Compute per-image tight depth ranges from the SfM sparse point cloud.

Each image gets its own (depth_min, depth_max) derived from the actual
3D points visible in that image â€” not a fixed global range.

This is the key optimization for Stage 8:
  Generic search window  : 20-35 min
  Percentile depth ranges:  8-15 min  (40-60% faster)

Algorithm
---------
For each image in the COLMAP reconstruction:
  1. Collect all 2D keypoints that have an associated 3D point.
  2. Project each 3D point through the camera pose (R, t) to get depth.
  3. Take the 2nd and 98th percentile of those depths (clip outliers).
  4. Add a 10% buffer each side.
  5. Clamp depth_min >= 0.1 m (never zero or negative).

If fewer than MIN_SPARSE_POINTS visible points exist for an image,
fall back to the configured fallback range (80m, 200m) â€” conservative
bounds that cover normal agricultural drone altitudes.

Usage
-----
    from src.depth.depth_range import compute_depth_ranges

    depth_ranges = compute_depth_ranges(reconstruction, captures)
    # {"000.jpg": (94.2, 148.7), "001.jpg": (91.5, 151.3), ...}
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    # pycolmap.Reconstruction is only available at runtime
    pass


# ---------------------------------------------------------------------------
# pycolmap API compatibility helpers
# ---------------------------------------------------------------------------
# pycolmap 4.x moved rotation/translation into `image.cam_from_world`.
# Older builds (and the legacy API) expose `image.rotmat()` / `image.tvec`.
# These helpers try the modern API first and fall back gracefully.

def _get_R(image) -> np.ndarray:
    """Return the 3Ã—3 world-to-camera rotation matrix for a pycolmap.Image."""
    if hasattr(image, "cam_from_world"):
        cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        rot = cfw.rotation
        if hasattr(rot, "matrix"):
            return np.asarray(rot.matrix(), dtype=np.float64)
        # Some builds expose a quaternion (.quat is xyzw)
        q = np.asarray(rot.quat, dtype=np.float64)
        x, y, z, w = q
        s = 2.0 / (x*x + y*y + z*z + w*w)
        return np.array([
            [1 - s*(y*y+z*z),  s*(x*y - z*w),   s*(x*z + y*w)],
            [s*(x*y + z*w),    1 - s*(x*x+z*z),  s*(y*z - x*w)],
            [s*(x*z - y*w),    s*(y*z + x*w),    1 - s*(x*x+y*y)],
        ], dtype=np.float64)
    # Legacy pycolmap: image.rotmat() returns a 3Ã—3 list/array
    if hasattr(image, "rotmat"):
        return np.asarray(image.rotmat(), dtype=np.float64)
    raise AttributeError(
        f"Cannot extract rotation from pycolmap.Image of type {type(image)!r}. "
        "Expected cam_from_world.rotation or rotmat()."
    )


def _get_t(image) -> np.ndarray:
    """Return the (3,) world-to-camera translation vector for a pycolmap.Image."""
    if hasattr(image, "cam_from_world"):
        cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        return np.asarray(cfw.translation, dtype=np.float64)
    if hasattr(image, "tvec"):
        return np.asarray(image.tvec, dtype=np.float64)
    raise AttributeError(
        f"Cannot extract translation from pycolmap.Image of type {type(image)!r}. "
        "Expected cam_from_world.translation or tvec."
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERCENTILE_LOW: int = 2
"""Exclude the closest 2% of depths â€” likely noise or mismatched points."""

PERCENTILE_HIGH: int = 98
"""Exclude the farthest 2% of depths â€” likely bad triangulations."""

BUFFER_FRACTION: float = 0.10
"""Add 10% of the range as a margin on each side."""

MIN_SPARSE_POINTS: int = 10
"""Minimum number of visible sparse points required to use percentile method.
Images below this threshold fall back to FALLBACK range."""

FALLBACK_MIN_M: float = 80.0
"""Fallback depth_min (metres) when too few sparse points are available.
80 m covers most agricultural nadir flights at 100-120 m AGL."""

FALLBACK_MAX_M: float = 200.0
"""Fallback depth_max (metres). 200 m is a generous upper bound for
agricultural drone flights."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_depth_ranges(
    reconstruction,
    captures: List,
    percentile_low: int = PERCENTILE_LOW,
    percentile_high: int = PERCENTILE_HIGH,
    buffer: float = BUFFER_FRACTION,
    fallback_min: float = FALLBACK_MIN_M,
    fallback_max: float = FALLBACK_MAX_M,
) -> dict[str, tuple[float, float]]:
    """
    Compute per-image depth ranges from SfM sparse points.

    Parameters
    ----------
    reconstruction : pycolmap.Reconstruction
        The sparse 3D reconstruction output from Stage 6/7 (SfM +
        Georeferencing). Must contain .images and .points3D attributes.
    captures : List[Capture]
        Capture list from the ingestion stage. Not used for computation
        here â€” depth ranges are derived from sparse points alone â€” but
        kept in the signature for API consistency and future use
        (e.g. per-capture logging).
    percentile_low : int
        Lower percentile for depth clipping (default 2).
    percentile_high : int
        Upper percentile for depth clipping (default 98).
    buffer : float
        Fractional buffer added to each side of the range (default 0.10
        = 10%).
    fallback_min : float
        depth_min used when an image has fewer than MIN_SPARSE_POINTS
        visible 3D points (metres).
    fallback_max : float
        depth_max used in the same fallback case (metres).

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from image name (as stored in the reconstruction) to
        (depth_min_metres, depth_max_metres).

        Example::

            {
                "000.jpg": (94.2, 148.7),
                "001.jpg": (91.5, 151.3),
                ...
            }
    """
    depth_ranges: dict[str, tuple[float, float]] = {}
    fallback_count = 0

    for image_id, image in reconstruction.images.items():
        depths = _collect_depths_for_image(image, reconstruction)

        if len(depths) < MIN_SPARSE_POINTS:
            logger.warning(
                "Image '%s' has only %d visible sparse points (< %d). "
                "Using fallback depth range (%.1f m, %.1f m).",
                image.name,
                len(depths),
                MIN_SPARSE_POINTS,
                fallback_min,
                fallback_max,
            )
            depth_ranges[image.name] = (fallback_min, fallback_max)
            fallback_count += 1
            continue

        d_min, d_max = _percentile_range_with_buffer(
            depths,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            buffer=buffer,
        )

        depth_ranges[image.name] = (d_min, d_max)
        logger.debug(
            "Image '%s': %d points â†’ depth range (%.2f m, %.2f m)",
            image.name,
            len(depths),
            d_min,
            d_max,
        )

    total = len(reconstruction.images)
    logger.info(
        "Depth ranges computed: %d images, %d used percentile method, "
        "%d used fallback.",
        total,
        total - fallback_count,
        fallback_count,
    )

    return depth_ranges


def print_depth_range_stats(depth_ranges: dict[str, tuple[float, float]]) -> None:
    """
    Print a human-readable summary of computed depth ranges.

    Useful for sanity-checking before launching dense stereo â€” lets you catch
    obviously wrong values (e.g. depths of 0 m or 5000 m) before they
    waste GPU time.

    Parameters
    ----------
    depth_ranges : dict[str, tuple[float, float]]
        Output of :func:`compute_depth_ranges`.

    Example output::

        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚  Depth Range Statistics (123 images)        â”‚
        â”‚  depth_min  â†’ mean: 92.4 m  range: [80.0, 101.3] m
        â”‚  depth_max  â†’ mean: 149.8 m range: [131.2, 200.0] m
        â”‚  range_widthâ†’ mean: 57.4 m  range: [36.1, 120.0] m
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    """
    if not depth_ranges:
        print("  [depth range stats] No depth ranges to display.")
        return

    mins = np.array([v[0] for v in depth_ranges.values()])
    maxs = np.array([v[1] for v in depth_ranges.values()])
    widths = maxs - mins
    n = len(depth_ranges)

    print(f"\n{'â”€' * 57}")
    print(f"  Depth Range Statistics ({n} images)")
    print(
        f"  depth_min   â†’ mean: {mins.mean():.1f} m  "
        f"range: [{mins.min():.1f}, {mins.max():.1f}] m"
    )
    print(
        f"  depth_max   â†’ mean: {maxs.mean():.1f} m  "
        f"range: [{maxs.min():.1f}, {maxs.max():.1f}] m"
    )
    print(
        f"  range_width â†’ mean: {widths.mean():.1f} m  "
        f"range: [{widths.min():.1f}, {widths.max():.1f}] m"
    )
    print(f"{'â”€' * 57}\n")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _collect_depths_for_image(image, reconstruction) -> list[float]:
    """
    Return a list of positive depth values (metres) for all 3D points
    visible in *image*.

    A depth is the Z coordinate of a world point after transformation
    into camera space:

        cam_coords = R @ xyz + t
        depth      = cam_coords[2]          # Z axis points out of lens

    Points behind the camera (depth <= 0) are discarded â€” they indicate
    a wrong match or a numerical issue and must not influence the range.

    Parameters
    ----------
    image : pycolmap.Image
    reconstruction : pycolmap.Reconstruction

    Returns
    -------
    list[float]
        Positive depth values in metres. May be empty.
    """
    # Use compat helpers â€” pycolmap 4.x removed .rotmat() and .tvec in favour
    # of image.cam_from_world.rotation / .translation.
    R = _get_R(image)   # (3, 3) world-to-camera rotation
    t = _get_t(image)   # (3,)   world-to-camera translation

    depths: list[float] = []

    for point2d in image.points2D:
        if not point2d.has_point3D():
            continue

        point3d_id = point2d.point3D_id
        if point3d_id not in reconstruction.points3D:
            # Defensive: skip if point was filtered during BA
            continue

        xyz = reconstruction.points3D[point3d_id].xyz  # world coords

        # Transform to camera frame
        cam_coords = R @ xyz + t
        depth = float(cam_coords[2])

        if depth > 0.0:
            depths.append(depth)

    return depths


def _percentile_range_with_buffer(
    depths: list[float],
    percentile_low: int,
    percentile_high: int,
    buffer: float,
) -> tuple[float, float]:
    """
    Compute a buffered depth range from a list of depth values.

    Steps:
      1. p_low  = percentile(depths, percentile_low)
      2. p_high = percentile(depths, percentile_high)
      3. buf    = (p_high - p_low) * buffer
      4. d_min  = max(0.1, p_low  - buf)
      5. d_max  = p_high + buf

    The 0.1 m clamp on d_min prevents zero or negative depth_min being
    passed to dense stereo, which would cause undefined behaviour.

    Parameters
    ----------
    depths : list[float]
        Positive depth values. Must have at least MIN_SPARSE_POINTS
        entries (caller's responsibility).
    percentile_low : int
        Lower clip percentile (e.g. 2).
    percentile_high : int
        Upper clip percentile (e.g. 98).
    buffer : float
        Fractional buffer (e.g. 0.10 for 10%).

    Returns
    -------
    tuple[float, float]
        (depth_min_metres, depth_max_metres)
    """
    arr = np.asarray(depths, dtype=np.float64)

    p_low = float(np.percentile(arr, percentile_low))
    p_high = float(np.percentile(arr, percentile_high))

    buf = (p_high - p_low) * buffer
    d_min = max(0.1, p_low - buf)
    d_max = p_high + buf

    return (d_min, d_max)

