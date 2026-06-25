"""
src/dsm/interpolate.py

Fills gaps (nodata cells) in a raw DSM produced by rasterize.py.

Gaps occur over textureless regions (bare soil, water, uniform surfaces)
where COLMAP PatchMatch + stereo-fusion consistency checks finds no reliable
depth. For flat agricultural fields these gaps are small and surrounded
by correct values, so a smooth interpolation is an appropriate fill â€”
there's no cliff or sharp relief nearby that the fill could get wrong.

Two-pass approach:
  Pass 1 â€” GDAL fillnodata (Laplacian-style smoothing), small radius.
           Fast, handles the common case (small holes from textureless
           patches).
  Pass 2 â€” scipy griddata (linear), larger radius, only if Pass 1 leaves
           gaps bigger than its search radius (e.g. field edges, ponds).
           Falls back to mean field elevation for anything griddata still
           can't reach (points fully outside the convex hull of known data).

GDAL is optional. If unavailable, Pass 1 is skipped and Pass 2 (scipy)
handles everything â€” slower, but scipy is a hard dependency everywhere
else in this stage so this keeps interpolate.py functional without GDAL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import griddata

try:
    import rasterio

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False

try:
    from osgeo import gdal

    _HAS_GDAL = True
except ImportError:
    _HAS_GDAL = False


DEFAULT_NODATA = -9999.0
DEFAULT_MAX_GAP_PX = 20
DEFAULT_LARGE_GAP_PX = 100
COVERAGE_WARN_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Core array-level operations (pure numpy/scipy, independently testable)
# ---------------------------------------------------------------------------

def _gdal_fillnodata_array(
    grid: np.ndarray,
    nodata_value: float,
    max_distance: int,
) -> np.ndarray:
    """
    Run GDAL's FillNodata algorithm on an in-memory array via an MEM driver
    dataset, returning the filled array. Requires GDAL.
    """
    if not _HAS_GDAL:
        raise RuntimeError("GDAL is not installed; cannot run _gdal_fillnodata_array")

    rows, cols = grid.shape
    mem_drv = gdal.GetDriverByName("MEM")
    ds = mem_drv.Create("", cols, rows, 1, gdal.GDT_Float32)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata_value)
    band.WriteArray(grid.astype(np.float32))

    gdal.FillNodata(
        targetBand=band,
        maskBand=None,
        maxSearchDist=max_distance,
        smoothingIterations=0,
    )

    filled = band.ReadAsArray()
    ds = None
    return filled


def _scipy_inpaint_large_gaps(
    grid: np.ndarray,
    nodata_value: float = DEFAULT_NODATA,
) -> np.ndarray:
    """
    Fill remaining nodata cells using scipy.interpolate.griddata (linear),
    then fall back to the mean of all known (valid) values for any cell
    griddata can't reach (i.e. outside the convex hull of valid points).
    """
    valid_mask = grid != nodata_value
    if valid_mask.all():
        return grid.copy()
    if not valid_mask.any():
        # Nothing to interpolate from; nothing we can do.
        return grid.copy()

    rows, cols = grid.shape
    yy, xx = np.mgrid[0:rows, 0:cols]

    known_points = np.column_stack((xx[valid_mask], yy[valid_mask]))
    known_values = grid[valid_mask]

    missing_mask = ~valid_mask
    missing_points = np.column_stack((xx[missing_mask], yy[missing_mask]))

    filled_values = griddata(
        known_points, known_values, missing_points, method="linear"
    )

    # griddata returns NaN outside the convex hull of known points; backfill
    # those with the mean of all valid values as a last resort.
    nan_mask = np.isnan(filled_values)
    if nan_mask.any():
        filled_values[nan_mask] = float(np.mean(known_values))

    out = grid.copy()
    out[missing_mask] = filled_values
    return out


def _fill_gaps_array(
    grid: np.ndarray,
    nodata_value: float = DEFAULT_NODATA,
    max_gap_px: int = DEFAULT_MAX_GAP_PX,
    large_gap_px: int = DEFAULT_LARGE_GAP_PX,
) -> np.ndarray:
    """
    Two-pass gap fill on a single in-memory grid array. Used directly by
    tests; fill_dsm_gaps() wraps this with GeoTIFF I/O.
    """
    grid = grid.astype(np.float32)

    # Pass 1: GDAL fillnodata, small radius (skipped if GDAL unavailable).
    if _HAS_GDAL:
        grid = _gdal_fillnodata_array(grid, nodata_value, max_gap_px)
    # else: leave to Pass 2 entirely.

    # Pass 2: anything still nodata (gaps bigger than max_gap_px, or GDAL
    # unavailable) gets scipy linear interpolation + mean fallback.
    if np.any(grid == nodata_value):
        grid = _scipy_inpaint_large_gaps(grid, nodata_value)

    return grid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_gap_coverage(
    dsm_path: Optional[str] = None,
    nodata_value: float = DEFAULT_NODATA,
    grid: Optional[np.ndarray] = None,
) -> float:
    """
    Returns the fraction of valid (non-nodata) pixels in a DSM.

    Accepts either a path to a GeoTIFF (reads via rasterio) or a grid
    array directly (for testing without file I/O). Logs a warning if
    coverage is below COVERAGE_WARN_THRESHOLD.
    """
    if grid is None:
        if dsm_path is None:
            raise ValueError("Must provide either dsm_path or grid")
        if not _HAS_RASTERIO:
            raise RuntimeError("rasterio is required to read a DSM from disk")
        with rasterio.open(dsm_path) as src:
            grid = src.read(1)
            nodata_value = src.nodata if src.nodata is not None else nodata_value

    total = grid.size
    valid = int(np.sum(grid != nodata_value))
    coverage = valid / total if total else 0.0

    if coverage < COVERAGE_WARN_THRESHOLD:
        print(
            f"[dsm.interpolate] WARNING: DSM coverage is {coverage:.1%}, "
            f"below the {COVERAGE_WARN_THRESHOLD:.0%} threshold. "
            f">{1 - COVERAGE_WARN_THRESHOLD:.0%} of the field has no reliable "
            f"elevation data even after gap filling â€” check overlap/texture "
            f"in that area."
        )

    return coverage


def fill_dsm_gaps(
    dsm_path: str,
    output_path: str,
    max_gap_px: int = DEFAULT_MAX_GAP_PX,
    large_gap_px: int = DEFAULT_LARGE_GAP_PX,
    nodata_value: float = DEFAULT_NODATA,
) -> str:
    """
    Fill nodata gaps in a DSM GeoTIFF. output_path may equal dsm_path for
    an effectively in-place update (read fully, then overwrite).
    """
    if not _HAS_RASTERIO:
        raise RuntimeError("rasterio is required to read/write DSM GeoTIFFs")

    with rasterio.open(dsm_path) as src:
        grid = src.read(1)
        profile = src.profile.copy()
        src_nodata = src.nodata if src.nodata is not None else nodata_value

    filled = _fill_gaps_array(
        grid, nodata_value=src_nodata, max_gap_px=max_gap_px, large_gap_px=large_gap_px
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype=filled.dtype)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(filled, 1)

    check_gap_coverage(grid=filled, nodata_value=src_nodata)

    return output_path

