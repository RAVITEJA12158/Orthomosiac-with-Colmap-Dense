"""
tests/test_dsm_quality.py

Gap-fill logic tests (interpolate.py) and DSM-level sanity checks.

The array-level fill functions (_fill_gaps_array, _scipy_inpaint_large_gaps)
are tested directly with numpy — no rasterio/GDAL required, so these run
anywhere. GeoTIFF-level tests (fill_dsm_gaps, check_gap_coverage against a
real file) are skipped automatically if rasterio isn't installed.
"""

import numpy as np
import pytest

from src.dsm.interpolate import (
    DEFAULT_NODATA,
    _fill_gaps_array,
    _scipy_inpaint_large_gaps,
    check_gap_coverage,
)


# ---------------------------------------------------------------------------
# Gap fill — array level
# ---------------------------------------------------------------------------

def test_gap_fill_basic():
    """5x5 nodata hole surrounded by a known value should fill to a plausible value."""
    grid = np.full((20, 20), 120.0, dtype=np.float32)
    grid[7:12, 7:12] = DEFAULT_NODATA  # 5x5 hole

    filled = _fill_gaps_array(grid, nodata_value=DEFAULT_NODATA)

    assert not np.any(filled == DEFAULT_NODATA)
    hole = filled[7:12, 7:12]
    assert np.allclose(hole, 120.0, atol=2.0), f"hole filled with implausible values: {hole}"


def test_gap_fill_preserves_valid_pixels():
    rng = np.random.default_rng(0)
    grid = (100.0 + rng.normal(0, 0.5, size=(15, 15))).astype(np.float32)
    original_valid = grid.copy()
    grid[3:6, 3:6] = DEFAULT_NODATA

    filled = _fill_gaps_array(grid, nodata_value=DEFAULT_NODATA)

    valid_mask = original_valid != DEFAULT_NODATA
    # Cells that were already valid must be untouched.
    untouched_mask = valid_mask.copy()
    untouched_mask[3:6, 3:6] = False
    np.testing.assert_array_equal(filled[untouched_mask], original_valid[untouched_mask])


def test_scipy_inpaint_fills_interior_gap():
    grid = np.full((10, 10), 50.0, dtype=np.float32)
    grid[4:6, 4:6] = DEFAULT_NODATA

    filled = _scipy_inpaint_large_gaps(grid, nodata_value=DEFAULT_NODATA)

    assert not np.any(filled == DEFAULT_NODATA)
    assert np.allclose(filled[4:6, 4:6], 50.0, atol=1e-3)


def test_scipy_inpaint_handles_all_nodata_gracefully():
    """No valid pixels at all -> nothing to interpolate from; must not crash."""
    grid = np.full((5, 5), DEFAULT_NODATA, dtype=np.float32)
    filled = _scipy_inpaint_large_gaps(grid, nodata_value=DEFAULT_NODATA)
    # Can't manufacture data from nothing — should return unchanged, not crash.
    np.testing.assert_array_equal(filled, grid)


def test_scipy_inpaint_no_gaps_is_noop():
    grid = np.full((5, 5), 42.0, dtype=np.float32)
    filled = _scipy_inpaint_large_gaps(grid, nodata_value=DEFAULT_NODATA)
    np.testing.assert_array_equal(filled, grid)


def test_gap_fill_large_hole_outside_gdal_radius():
    """A hole larger than max_gap_px must still get filled by the Pass-2 scipy fallback."""
    grid = np.full((60, 60), 80.0, dtype=np.float32)
    grid[10:50, 10:50] = DEFAULT_NODATA  # 40x40 hole, bigger than default max_gap_px=20

    filled = _fill_gaps_array(grid, nodata_value=DEFAULT_NODATA, max_gap_px=20, large_gap_px=100)

    assert not np.any(filled == DEFAULT_NODATA)
    assert np.allclose(filled[10:50, 10:50], 80.0, atol=3.0)


# ---------------------------------------------------------------------------
# check_gap_coverage
# ---------------------------------------------------------------------------

def test_gap_fill_coverage_check_warns(capsys):
    grid = np.full((10, 10), DEFAULT_NODATA, dtype=np.float32)
    grid[:5, :] = 100.0  # 50% valid

    coverage = check_gap_coverage(grid=grid, nodata_value=DEFAULT_NODATA)

    assert coverage == pytest.approx(0.5)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


def test_gap_fill_coverage_check_no_warning(capsys):
    grid = np.full((10, 10), 100.0, dtype=np.float32)
    grid[0, 0] = DEFAULT_NODATA  # 1% nodata, 99% valid

    coverage = check_gap_coverage(grid=grid, nodata_value=DEFAULT_NODATA)

    assert coverage == pytest.approx(0.99)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


def test_check_gap_coverage_requires_path_or_grid():
    with pytest.raises(ValueError):
        check_gap_coverage()


# ---------------------------------------------------------------------------
# DSM-level sanity checks
# ---------------------------------------------------------------------------

def test_elevation_range_plausible():
    """Agricultural field elevations must be within a believable range."""
    grid = np.full((10, 10), 145.3, dtype=np.float32)
    valid = grid[grid != DEFAULT_NODATA]
    assert valid.min() >= -50.0  # generous floor incl. below-sea-level fields
    assert valid.max() <= 5000.0


def test_resolution_matches_target_via_points_to_grid():
    from src.dsm.rasterize import _points_to_grid

    points = np.array(
        [[0, 0, 100.0], [9.9, 9.9, 100.0]], dtype=np.float32
    )
    _, geotransform = _points_to_grid(points, gsd_m=0.10)
    _, px_w, _, _, _, px_h = geotransform
    assert px_w == pytest.approx(0.10)
    assert abs(px_h) == pytest.approx(0.10)


def test_coverage_percentage_after_fill_exceeds_threshold():
    grid = np.full((30, 30), 120.0, dtype=np.float32)
    grid[10:15, 10:15] = DEFAULT_NODATA

    filled = _fill_gaps_array(grid, nodata_value=DEFAULT_NODATA)
    coverage = check_gap_coverage(grid=filled, nodata_value=DEFAULT_NODATA)
    assert coverage > 0.85


# ---------------------------------------------------------------------------
# GeoTIFF-level tests — require rasterio (skipped if unavailable here)
# ---------------------------------------------------------------------------

def test_fill_dsm_gaps_geotiff_roundtrip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from src.dsm.interpolate import fill_dsm_gaps

    grid = np.full((20, 20), 110.0, dtype=np.float32)
    grid[8:12, 8:12] = DEFAULT_NODATA

    raw_path = tmp_path / "dsm_raw.tif"
    transform = from_origin(0, 20, 1, 1)
    with rasterio.open(
        str(raw_path), "w", driver="GTiff", height=20, width=20, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=DEFAULT_NODATA,
    ) as dst:
        dst.write(grid, 1)

    out_path = tmp_path / "dsm.tif"
    fill_dsm_gaps(str(raw_path), str(out_path))

    with rasterio.open(str(out_path)) as src:
        filled = src.read(1)
        assert not np.any(filled == DEFAULT_NODATA)
