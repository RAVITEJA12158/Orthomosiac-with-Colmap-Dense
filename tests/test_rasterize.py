"""
tests/test_rasterize.py

Tests for src/dsm/rasterize.py.

Tests against the pure numpy/struct logic (_points_to_grid, PLY parsing)
run anywhere. Tests that need rasterio to actually write/read a GeoTIFF
are skipped automatically (pytest.importorskip) if rasterio isn't
installed in the current environment — they will run on the project
machine where the full stack (per CLAUDE.md install requirements) is
present.
"""

import struct

import numpy as np
import pytest

from src.dsm.rasterize import (
    NODATA_VALUE,
    _manual_read_ply_xyz,
    _points_to_grid,
    _read_ply_xyz,
)


# ---------------------------------------------------------------------------
# _points_to_grid
# ---------------------------------------------------------------------------

def test_rasterize_basic():
    """100 points on a flat plane at Z=120m -> valid cells ~120m, nodata elsewhere."""
    rng = np.random.default_rng(42)
    xs = rng.uniform(0, 5, size=100)
    ys = rng.uniform(0, 5, size=100)
    zs = np.full(100, 120.0)
    points = np.column_stack([xs, ys, zs]).astype(np.float32)

    grid, geotransform = _points_to_grid(points, gsd_m=0.10)

    valid = grid[grid != NODATA_VALUE]
    assert valid.size > 0
    assert np.allclose(valid, 120.0, atol=1e-3)
    assert np.any(grid == NODATA_VALUE), "sparse 100 points over 5x5m at 10cm GSD should leave gaps"


def test_rasterize_uses_median():
    """One outlier in a cell with 5 good points: median rejects it, mean would not."""
    good = np.array([
        [0.01, 0.01, 120.0],
        [0.02, 0.02, 120.0],
        [0.03, 0.03, 120.0],
        [0.04, 0.04, 120.0],
        [0.01, 0.04, 120.0],
    ])
    outlier = np.array([[0.02, 0.03, 500.0]])
    points = np.vstack([good, outlier]).astype(np.float32)

    # All 6 points fall in the same single grid cell at a coarse GSD.
    grid, _ = _points_to_grid(points, gsd_m=1.0)

    assert grid.shape == (1, 1)
    median_val = grid[0, 0]
    mean_val = points[:, 2].mean()

    assert np.isclose(median_val, 120.0), f"expected median ~120, got {median_val}"
    assert not np.isclose(median_val, mean_val), (
        "median should differ from (outlier-dragged) mean"
    )
    assert mean_val > 150  # sanity check the outlier actually drags the mean


def test_geotransform_shape_and_origin():
    points = np.array(
        [[0.0, 0.0, 100.0], [10.0, 10.0, 100.0], [5.0, 5.0, 100.0]], dtype=np.float32
    )
    grid, geotransform = _points_to_grid(points, gsd_m=1.0)
    x_min, px_w, _, y_max, _, px_h = geotransform
    assert x_min == 0.0
    assert y_max == 10.0
    assert px_w == 1.0
    assert px_h == -1.0
    assert grid.shape[0] >= 10 and grid.shape[1] >= 10


def test_points_to_grid_rejects_empty():
    with pytest.raises(ValueError):
        _points_to_grid(np.zeros((0, 3), dtype=np.float32), gsd_m=0.1)


# ---------------------------------------------------------------------------
# PLY parsing (manual fallback parser — exercised directly, no open3d needed)
# ---------------------------------------------------------------------------

def _write_ascii_ply(path, points):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")


def _write_binary_ply(path, points, with_normals=False):
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(points)}\n".encode("ascii"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if with_normals:
            f.write(b"property float nx\n")
            f.write(b"property float ny\n")
            f.write(b"property float nz\n")
        f.write(b"end_header\n")
        for pt in points:
            x, y, z = pt[:3]
            if with_normals:
                f.write(struct.pack("<ffffff", x, y, z, 0.0, 0.0, 1.0))
            else:
                f.write(struct.pack("<fff", x, y, z))


def test_manual_ply_ascii_roundtrip(tmp_path):
    points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (-1.5, 0.0, 100.25)]
    ply_path = tmp_path / "ascii.ply"
    _write_ascii_ply(ply_path, points)

    result = _manual_read_ply_xyz(str(ply_path))
    expected = np.array(points, dtype=np.float32)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_manual_ply_ascii_crlf_header_roundtrip(tmp_path):
    points = [(1.0, 2.0, 3.0), (-1.5, 0.0, 100.25)]
    ply_path = tmp_path / "ascii_crlf.ply"
    with open(ply_path, "wb") as f:
        f.write(b"ply\r\n")
        f.write(b"format ascii 1.0\r\n")
        f.write(f"element vertex {len(points)}\r\n".encode("ascii"))
        f.write(b"property float x\r\n")
        f.write(b"property float y\r\n")
        f.write(b"property float z\r\n")
        f.write(b"end_header\r\n")
        for x, y, z in points:
            f.write(f"{x} {y} {z}\r\n".encode("ascii"))

    result = _manual_read_ply_xyz(str(ply_path))
    np.testing.assert_allclose(result, np.array(points, dtype=np.float32), atol=1e-4)


def test_manual_ply_binary_roundtrip(tmp_path):
    points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (-1.5, 0.0, 100.25)]
    ply_path = tmp_path / "binary.ply"
    _write_binary_ply(ply_path, points, with_normals=False)

    result = _manual_read_ply_xyz(str(ply_path))
    expected = np.array(points, dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_manual_ply_binary_with_extra_properties(tmp_path):
    """OpenMVS .ply files typically also carry normals/colors — must be skipped, not break x,y,z parsing."""
    points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    ply_path = tmp_path / "binary_normals.ply"
    _write_binary_ply(ply_path, points, with_normals=True)

    result = _manual_read_ply_xyz(str(ply_path))
    expected = np.array(points, dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_read_ply_xyz_dispatches_without_open3d(tmp_path, monkeypatch):
    """_read_ply_xyz must work via the manual fallback when open3d is absent."""
    import src.dsm.rasterize as rasterize_mod

    monkeypatch.setattr(rasterize_mod, "_HAS_OPEN3D", False)
    points = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
    ply_path = tmp_path / "test.ply"
    _write_ascii_ply(ply_path, points)

    result = _read_ply_xyz(str(ply_path))
    np.testing.assert_allclose(result, np.array(points, dtype=np.float32))


# ---------------------------------------------------------------------------
# GeoTIFF writing — requires rasterio, skipped if not installed here
# ---------------------------------------------------------------------------

def test_geotiff_has_crs_and_transform(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from src.dsm.rasterize import rasterize_pointcloud

    points = [(i * 0.5, j * 0.5, 100.0) for i in range(10) for j in range(10)]
    ply_path = tmp_path / "flat.ply"
    _write_ascii_ply(ply_path, points)

    out_path = tmp_path / "dsm_raw.tif"
    rasterize_pointcloud(str(ply_path), str(out_path), target_gsd_m=0.5)

    with rasterio.open(str(out_path)) as src:
        assert src.crs is not None
        assert src.transform is not None and not src.transform.is_identity
        assert src.nodata == NODATA_VALUE


def test_nodata_value_set_in_output(tmp_path):
    pytest.importorskip("rasterio")
    import rasterio as rio
    from src.dsm.rasterize import rasterize_pointcloud

    # Sparse points -> guaranteed gaps at fine GSD.
    points = [(0.0, 0.0, 100.0), (5.0, 5.0, 100.0)]
    ply_path = tmp_path / "sparse.ply"
    _write_ascii_ply(ply_path, points)

    out_path = tmp_path / "dsm_raw.tif"
    rasterize_pointcloud(str(ply_path), str(out_path), target_gsd_m=0.1)

    with rio.open(str(out_path)) as src:
        grid = src.read(1)
        assert np.any(grid == NODATA_VALUE)
