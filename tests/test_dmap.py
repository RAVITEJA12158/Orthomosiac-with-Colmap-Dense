"""
tests/test_dmap.py
==================
Tests for src/depth/dmap.py

Covers .dmap binary write + read round-trips, the DMap dataclass helper
methods, and correct handling of the content_type flag (depth-only vs
depth+normals+confidence).

No OpenMVS installation required — all tests use synthetic DMap objects
produced by write_dmap() and verified by read_dmap().

Run with:
    pytest tests/test_dmap.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.depth.dmap import (
    CONTENT_DEPTH_NORMALS_CONFIDENCE,
    CONTENT_DEPTH_ONLY,
    DMap,
    dmap_from_openmvs_output,
    read_dmap,
    write_dmap,
)


# ---------------------------------------------------------------------------
# Synthetic DMap factory
# ---------------------------------------------------------------------------

def _make_dmap(
    width: int = 100,
    height: int = 80,
    image_name: str = "000.jpg",
    depth_min: float = 90.0,
    depth_max: float = 150.0,
    depth_values: np.ndarray | None = None,
    include_normals: bool = False,
) -> DMap:
    """
    Create a synthetic DMap for testing.

    Parameters
    ----------
    width, height : int
        Pixel dimensions.
    image_name : str
        Stored in the binary header.
    depth_min, depth_max : float
        Header depth range values.
    depth_values : np.ndarray or None
        If None, filled with a gradient between depth_min and depth_max.
    include_normals : bool
        If True, generate synthetic normal and confidence arrays.

    Returns
    -------
    DMap
    """
    if depth_values is None:
        # Gradient depth: increases from depth_min to depth_max across rows
        row_depths = np.linspace(depth_min, depth_max, height, dtype=np.float32)
        depth = np.tile(row_depths[:, None], (1, width))
    else:
        depth = depth_values.astype(np.float32)

    K = np.array([[800.0, 0.0, width / 2],
                  [0.0,   800.0, height / 2],
                  [0.0,   0.0,   1.0]], dtype=np.float64)

    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)

    normal = None
    confidence = None
    if include_normals:
        # All normals pointing straight up in camera space
        normal = np.zeros((height, width, 3), dtype=np.float32)
        normal[:, :, 2] = 1.0   # Z component = 1
        confidence = np.full((height, width), 0.9, dtype=np.float32)

    return DMap(
        image_name=image_name,
        width=width,
        height=height,
        depth_min=depth_min,
        depth_max=depth_max,
        depth=depth,
        normal=normal,
        confidence=confidence,
        K=K,
        R=R,
        t=t,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDmapWriteReadRoundtrip:
    """write_dmap → read_dmap must recover identical data."""

    def test_roundtrip_depth_only(self, tmp_path):
        """Depth-only .dmap survives a write → read cycle."""
        original = _make_dmap(width=64, height=48, include_normals=False)
        out_path = str(tmp_path / "test.dmap")

        write_dmap(original, out_path)
        recovered = read_dmap(out_path)

        assert recovered.image_name == original.image_name
        assert recovered.width == original.width
        assert recovered.height == original.height
        assert recovered.depth_min == pytest.approx(original.depth_min, abs=1e-4)
        assert recovered.depth_max == pytest.approx(original.depth_max, abs=1e-4)
        np.testing.assert_array_almost_equal(recovered.depth, original.depth, decimal=4)
        assert recovered.normal is None
        assert recovered.confidence is None

    def test_roundtrip_with_normals(self, tmp_path):
        """DMap with normals + confidence survives write → read."""
        original = _make_dmap(width=32, height=24, include_normals=True)
        out_path = str(tmp_path / "test_normals.dmap")

        write_dmap(original, out_path)
        recovered = read_dmap(out_path)

        np.testing.assert_array_almost_equal(recovered.depth, original.depth, decimal=4)
        assert recovered.normal is not None
        np.testing.assert_array_almost_equal(recovered.normal, original.normal, decimal=4)
        assert recovered.confidence is not None
        np.testing.assert_array_almost_equal(
            recovered.confidence, original.confidence, decimal=4
        )

    def test_roundtrip_camera_matrices(self, tmp_path):
        """K, R, t must be exactly preserved through the binary format."""
        original = _make_dmap()
        # Set non-trivial K and R
        original.K = np.array([[1200.0, 0.0, 960.0],
                                [0.0, 1200.0, 540.0],
                                [0.0, 0.0, 1.0]])
        original.R = np.array([[0.999, -0.01,  0.02],
                                [0.01,   0.999,  0.005],
                                [-0.02, -0.004,  0.999]])
        original.t = np.array([1.23, -0.45, 0.67])

        out_path = str(tmp_path / "cam.dmap")
        write_dmap(original, out_path)
        recovered = read_dmap(out_path)

        # float32 storage → tolerance of 1e-4 on float64 values
        np.testing.assert_array_almost_equal(recovered.K, original.K, decimal=3)
        np.testing.assert_array_almost_equal(recovered.R, original.R, decimal=3)
        np.testing.assert_array_almost_equal(recovered.t, original.t, decimal=4)

    def test_roundtrip_image_name_truncation(self, tmp_path):
        """Image names longer than 63 chars are truncated to fit 64-byte field."""
        long_name = "A" * 100 + ".jpg"
        original = _make_dmap(image_name=long_name)
        out_path = str(tmp_path / "truncated.dmap")

        write_dmap(original, out_path)
        recovered = read_dmap(out_path)

        # Must survive the round-trip (truncated is fine)
        assert len(recovered.image_name) <= 64

    def test_dmap_from_openmvs_output_alias(self, tmp_path):
        """dmap_from_openmvs_output is a semantic alias for read_dmap."""
        original = _make_dmap()
        out_path = str(tmp_path / "alias.dmap")
        write_dmap(original, out_path)

        via_alias  = dmap_from_openmvs_output(out_path)
        via_direct = read_dmap(out_path)

        np.testing.assert_array_equal(via_alias.depth, via_direct.depth)
        assert via_alias.image_name == via_direct.image_name


class TestDmapValidPixelRatio:
    """DMap.valid_pixel_ratio() must count non-zero pixels correctly."""

    def test_all_valid(self):
        """All pixels with depth > 0 → ratio = 1.0."""
        dmap = _make_dmap(depth_min=90.0, depth_max=150.0)
        # Default gradient has no zeros
        assert dmap.valid_pixel_ratio() == pytest.approx(1.0, abs=1e-6)

    def test_half_invalid(self):
        """Half the pixels set to 0 → ratio ≈ 0.5."""
        depth = np.ones((80, 100), dtype=np.float32) * 120.0
        depth[:40, :] = 0.0   # top half invalid

        dmap = _make_dmap(depth_values=depth)
        ratio = dmap.valid_pixel_ratio()

        assert ratio == pytest.approx(0.5, abs=1e-4)

    def test_all_invalid(self):
        """All-zero depth map → ratio = 0.0."""
        depth = np.zeros((80, 100), dtype=np.float32)
        dmap = _make_dmap(depth_values=depth)
        assert dmap.valid_pixel_ratio() == pytest.approx(0.0, abs=1e-6)

    def test_empty_depth_map(self):
        """Zero-size depth map (edge case) must return 0.0, not crash."""
        depth = np.zeros((0, 0), dtype=np.float32)
        dmap = _make_dmap(depth_values=depth)
        dmap.width = 0
        dmap.height = 0
        assert dmap.valid_pixel_ratio() == pytest.approx(0.0, abs=1e-6)


class TestDmapDepthStats:
    """DMap.depth_stats() must return correct statistics for valid pixels."""

    def test_known_values(self):
        """depth_stats on a uniform depth map."""
        depth = np.full((50, 50), 120.0, dtype=np.float32)
        dmap = _make_dmap(depth_values=depth)

        stats = dmap.depth_stats()

        assert stats["min"]  == pytest.approx(120.0, abs=1e-3)
        assert stats["max"]  == pytest.approx(120.0, abs=1e-3)
        assert stats["mean"] == pytest.approx(120.0, abs=1e-3)
        assert stats["std"]  == pytest.approx(0.0,   abs=1e-3)
        assert stats["valid_count"] == 50 * 50
        assert stats["total_pixels"] == 50 * 50

    def test_zeros_excluded_from_stats(self):
        """Zero pixels (invalid) must be excluded from stats computation."""
        depth = np.zeros((10, 10), dtype=np.float32)
        depth[5:, :] = 100.0   # bottom half valid at 100 m

        dmap = _make_dmap(depth_values=depth)
        stats = dmap.depth_stats()

        assert stats["min"]  == pytest.approx(100.0, abs=1e-3)
        assert stats["max"]  == pytest.approx(100.0, abs=1e-3)
        assert stats["valid_count"] == 50
        assert stats["total_pixels"] == 100

    def test_all_zeros_returns_nan(self):
        """All-zero depth map → stats values are NaN, no crash."""
        depth = np.zeros((20, 20), dtype=np.float32)
        dmap = _make_dmap(depth_values=depth)
        stats = dmap.depth_stats()

        import math
        assert math.isnan(stats["min"])
        assert math.isnan(stats["max"])
        assert math.isnan(stats["mean"])
        assert stats["valid_count"] == 0

    def test_stats_with_gradient(self):
        """Gradient depth map — verify mean is midpoint of range."""
        depth = np.linspace(90.0, 150.0, 100 * 80, dtype=np.float32).reshape(80, 100)
        dmap = _make_dmap(depth_values=depth)
        stats = dmap.depth_stats()

        assert stats["min"]  == pytest.approx(90.0,  abs=0.1)
        assert stats["max"]  == pytest.approx(150.0, abs=0.1)
        assert stats["mean"] == pytest.approx(120.0, abs=0.5)


class TestDmapWithNormals:
    """Content_type=3 path: normals and confidence arrays round-trip correctly."""

    def test_normals_shape(self, tmp_path):
        """Written + read normal array must have shape (H, W, 3)."""
        original = _make_dmap(width=40, height=30, include_normals=True)
        out_path = str(tmp_path / "normals.dmap")
        write_dmap(original, out_path)
        recovered = read_dmap(out_path)

        assert recovered.normal is not None
        assert recovered.normal.shape == (30, 40, 3)

    def test_normals_values(self, tmp_path):
        """Normal values survive the binary round-trip."""
        original = _make_dmap(width=20, height=15, include_normals=True)
        out_path = str(tmp_path / "norm_vals.dmap")
        write_dmap(original, out_path)
        recovered = read_dmap(out_path)

        # All normals should be [0, 0, 1] (pointing along camera Z)
        np.testing.assert_array_almost_equal(
            recovered.normal[:, :, 2],
            np.ones((15, 20), dtype=np.float32),
            decimal=4,
        )


class TestDmapWithoutNormals:
    """Content_type=1 path: normal and confidence must be None after read."""

    def test_no_normals_after_roundtrip(self, tmp_path):
        """Depth-only DMap → normal and confidence must be None on read-back."""
        original = _make_dmap(include_normals=False)
        out_path = str(tmp_path / "depth_only.dmap")
        write_dmap(original, out_path)
        recovered = read_dmap(out_path)

        assert recovered.normal is None
        assert recovered.confidence is None


class TestDmapZeroInvalidHandling:
    """Confirm that depth==0 is consistently treated as 'invalid' everywhere."""

    def test_zero_invalid_excluded_from_ratio(self):
        """valid_pixel_ratio treats 0 as invalid."""
        depth = np.array([[0.0, 100.0],
                          [100.0, 0.0]], dtype=np.float32)
        dmap = _make_dmap(depth_values=depth)
        assert dmap.valid_pixel_ratio() == pytest.approx(0.5, abs=1e-6)

    def test_zero_invalid_excluded_from_stats(self):
        """depth_stats excludes zeros from all statistical measures."""
        depth = np.array([[0.0, 0.0,   120.0],
                          [0.0, 130.0,  0.0 ]], dtype=np.float32)
        dmap = _make_dmap(depth_values=depth)
        stats = dmap.depth_stats()

        assert stats["valid_count"] == 2
        assert stats["min"]  == pytest.approx(120.0, abs=1e-3)
        assert stats["max"]  == pytest.approx(130.0, abs=1e-3)
        assert stats["mean"] == pytest.approx(125.0, abs=1e-3)

    def test_zero_survives_roundtrip_as_zero(self, tmp_path):
        """Zeros written to .dmap must come back as zeros (not some other sentinel)."""
        depth = np.zeros((10, 10), dtype=np.float32)
        depth[5, 5] = 120.0  # single valid pixel

        dmap = _make_dmap(width=10, height=10, depth_values=depth)
        out_path = str(tmp_path / "zeros.dmap")
        write_dmap(dmap, out_path)
        recovered = read_dmap(out_path)

        # All except [5,5] must still be zero
        mask = np.ones((10, 10), dtype=bool)
        mask[5, 5] = False
        assert (recovered.depth[mask] == 0.0).all()
        assert recovered.depth[5, 5] == pytest.approx(120.0, abs=1e-4)


class TestReadDmapErrors:
    """read_dmap raises informative errors on bad input."""

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=".dmap file not found"):
            read_dmap(str(tmp_path / "nonexistent.dmap"))

    def test_file_too_small(self, tmp_path):
        """A file smaller than the header raises ValueError."""
        bad_path = tmp_path / "bad.dmap"
        bad_path.write_bytes(b"\x00" * 10)  # too short for header

        with pytest.raises(ValueError, match="too small"):
            read_dmap(str(bad_path))
