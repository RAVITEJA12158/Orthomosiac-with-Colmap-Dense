"""
tests/test_depth_range.py
=========================
Tests for src/depth/depth_range.py

All tests use lightweight synthetic objects that mimic the pycolmap
Reconstruction / Image / Point2D / Point3D API.  No real mission data
or actual pycolmap installation required to run these tests.

Run with:
    pytest tests/test_depth_range.py -v
"""

from __future__ import annotations

import math
import io
import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pytest

from src.depth.depth_range import (
    FALLBACK_MAX_M,
    FALLBACK_MIN_M,
    MIN_SPARSE_POINTS,
    compute_depth_ranges,
    print_depth_range_stats,
    _collect_depths_for_image,
    _percentile_range_with_buffer,
)


# ---------------------------------------------------------------------------
# Synthetic pycolmap-like stubs
# ---------------------------------------------------------------------------

@dataclass
class FakePoint3D:
    xyz: np.ndarray


@dataclass
class FakePoint2D:
    _point3D_id: Optional[int] = None

    def has_point3D(self) -> bool:
        return self._point3D_id is not None

    @property
    def point3D_id(self) -> int:
        assert self._point3D_id is not None
        return self._point3D_id


@dataclass
class FakeImage:
    """Mimics pycolmap.Image API used by depth_range.py."""
    name: str
    _R: np.ndarray                          # (3, 3) world-to-camera rotation
    tvec: np.ndarray                        # (3,) world-to-camera translation
    points2D: List[FakePoint2D] = field(default_factory=list)

    def rotmat(self) -> np.ndarray:
        return self._R.copy()


class FakeReconstruction:
    """Mimics pycolmap.Reconstruction API used by depth_range.py."""

    def __init__(self):
        self.images: Dict[int, FakeImage] = {}
        self.points3D: Dict[int, FakePoint3D] = {}


# ---------------------------------------------------------------------------
# Helpers to build synthetic scenes
# ---------------------------------------------------------------------------

def _identity_pose() -> tuple[np.ndarray, np.ndarray]:
    """Camera at origin looking along +Z axis (trivial pose)."""
    R = np.eye(3)
    t = np.zeros(3)
    return R, t


def _make_reconstruction_with_depths(
    depths: List[float],
    image_name: str = "000.jpg",
) -> FakeReconstruction:
    """
    Build a FakeReconstruction where one image sees 3D points at the
    given depths.

    With identity pose (R=I, t=0), depth of a world point [0, 0, d] is
    simply d: cam_coords = I @ [0,0,d] + 0 = [0,0,d], Z = d.
    """
    recon = FakeReconstruction()
    R, t = _identity_pose()
    image = FakeImage(name=image_name, _R=R, tvec=t)

    for i, d in enumerate(depths):
        xyz = np.array([0.0, 0.0, d])  # world point at depth d
        recon.points3D[i] = FakePoint3D(xyz=xyz)
        image.points2D.append(FakePoint2D(_point3D_id=i))

    recon.images[0] = image
    return recon


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicDepthRange:
    """compute_depth_ranges: basic correctness with known depths."""

    def test_basic_depth_range(self):
        """
        10 points at depths 90–99 m.
        Percentile 2nd/98th on 10 values should return approximately
        the min and max, and the output must be a sensible range.
        """
        depths = list(range(90, 100))  # [90, 91, ..., 99]
        recon = _make_reconstruction_with_depths(depths)

        result = compute_depth_ranges(recon, captures=[])

        assert "000.jpg" in result
        d_min, d_max = result["000.jpg"]

        # With 10% buffer applied: should bracket [90, 99]
        assert d_min < 90.0
        assert d_max > 99.0
        # But should not be wildly off
        assert d_min > 80.0
        assert d_max < 110.0

    def test_output_type(self):
        """Return value must be dict[str, tuple[float, float]]."""
        recon = _make_reconstruction_with_depths(list(range(90, 105)))
        result = compute_depth_ranges(recon, captures=[])
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, str)
            assert isinstance(val, tuple)
            assert len(val) == 2
            assert all(isinstance(v, float) for v in val)


class TestPercentileExcludesOutliers:
    """Extreme outlier depths must be clipped by the 2nd/98th percentile."""

    def test_outliers_excluded(self):
        """
        100 points at ~120 m plus one at 5 m and one at 2000 m.
        The output range must not include the extremes.
        """
        depths = [120.0] * 100
        depths[0] = 5.0      # noise / wrong match
        depths[-1] = 2000.0  # bad triangulation artifact

        recon = _make_reconstruction_with_depths(depths)
        result = compute_depth_ranges(recon, captures=[])

        d_min, d_max = result["000.jpg"]

        # 5 m outlier must be excluded
        assert d_min > 10.0, f"d_min={d_min} includes the 5 m outlier"
        # 2000 m outlier must be excluded
        assert d_max < 500.0, f"d_max={d_max} includes the 2000 m outlier"


class TestFallbackWhenFewPoints:
    """Images with fewer than MIN_SPARSE_POINTS get the fallback range."""

    def test_fallback_triggered(self):
        """Fewer than MIN_SPARSE_POINTS → fallback (FALLBACK_MIN, FALLBACK_MAX)."""
        depths = [100.0] * (MIN_SPARSE_POINTS - 1)   # one short
        recon = _make_reconstruction_with_depths(depths)

        result = compute_depth_ranges(recon, captures=[])
        d_min, d_max = result["000.jpg"]

        assert d_min == FALLBACK_MIN_M
        assert d_max == FALLBACK_MAX_M

    def test_exactly_min_points_not_fallback(self):
        """Exactly MIN_SPARSE_POINTS points must use percentile method, not fallback."""
        depths = [100.0] * MIN_SPARSE_POINTS
        recon = _make_reconstruction_with_depths(depths)

        result = compute_depth_ranges(recon, captures=[])
        d_min, d_max = result["000.jpg"]

        # Constant array → percentile range is (100, 100), width = 0,
        # buffer = 10% of 0 = 0.
        # d_min = max(0.1, 100.0 - 0.0) = 100.0
        # d_max = 100.0 + 0.0 = 100.0
        # Key assertion: we must NOT get the fallback values (80, 200).
        assert d_min != FALLBACK_MIN_M
        assert d_max != FALLBACK_MAX_M
        # And the values must be close to the actual depth
        assert d_min == pytest.approx(100.0, abs=1.0)
        assert d_max == pytest.approx(100.0, abs=1.0)

    def test_exactly_min_points_constant_depth(self):
        """Constant depth array with MIN_SPARSE_POINTS → d_min and d_max near the constant."""
        val = 120.0
        depths = [val] * MIN_SPARSE_POINTS
        recon = _make_reconstruction_with_depths(depths)

        result = compute_depth_ranges(recon, captures=[])
        d_min, d_max = result["000.jpg"]

        # Range is zero, buffer is 10% of zero = 0
        # d_min = max(0.1, 120.0 - 0) = 120.0
        # d_max = 120.0 + 0 = 120.0
        assert d_min == pytest.approx(120.0, abs=1e-3)
        assert d_max == pytest.approx(120.0, abs=1e-3)


class TestBufferApplied:
    """Verify that the 10% buffer is correctly applied to each side."""

    def test_buffer_applied_correctly(self):
        """
        Points at 100 m and 140 m → percentile range (100, 140) →
        buffer = (140 - 100) * 0.10 = 4 m →
        expected: d_min ≈ 96 m, d_max ≈ 144 m.
        """
        # Use a large uniform block to make percentiles exactly 100 and 140
        depths = [100.0] * 50 + [140.0] * 50
        d_min, d_max = _percentile_range_with_buffer(
            depths,
            percentile_low=2,
            percentile_high=98,
            buffer=0.10,
        )

        assert d_min == pytest.approx(96.0, abs=1.0)
        assert d_max == pytest.approx(144.0, abs=1.0)

    def test_buffer_fraction_configurable(self):
        """A 20% buffer should produce a wider range than a 10% buffer."""
        depths = list(range(100, 141))  # 100..140

        d_min_10, d_max_10 = _percentile_range_with_buffer(
            depths, 2, 98, buffer=0.10
        )
        d_min_20, d_max_20 = _percentile_range_with_buffer(
            depths, 2, 98, buffer=0.20
        )

        assert d_min_20 < d_min_10
        assert d_max_20 > d_max_10


class TestAllDepthsPositive:
    """Points behind the camera (depth ≤ 0) must be excluded."""

    def test_negative_depths_excluded(self):
        """
        Mix positive and negative depths.  Negatives must not influence
        the range calculation.
        """
        recon = FakeReconstruction()
        R, t = _identity_pose()
        image = FakeImage(name="000.jpg", _R=R, tvec=t)

        # 5 points behind camera (Z < 0 in camera frame)
        for i in range(5):
            # To get cam_coord Z = -50: world point at [0, 0, -50]
            recon.points3D[i] = FakePoint3D(xyz=np.array([0.0, 0.0, -50.0]))
            image.points2D.append(FakePoint2D(_point3D_id=i))

        # 20 points in front at 120 m
        for i in range(5, 25):
            recon.points3D[i] = FakePoint3D(xyz=np.array([0.0, 0.0, 120.0]))
            image.points2D.append(FakePoint2D(_point3D_id=i))

        recon.images[0] = image

        result = compute_depth_ranges(recon, captures=[])
        d_min, d_max = result["000.jpg"]

        # Range must be around 120 m, not anywhere near -50 m
        assert d_min > 0.0
        assert d_max < 200.0
        # All valid points are exactly at 120 m → constant range,
        # so d_min == d_max == 120.0 (buffer of zero-width range = 0)
        assert d_min == pytest.approx(120.0, abs=1.0)
        assert d_max == pytest.approx(120.0, abs=1.0)

    def test_zero_depth_excluded(self):
        """Depth == 0.0 (camera origin) must not enter the calculation."""
        depths_positive = [100.0] * 20
        depths_mixed = [0.0] * 5 + depths_positive  # 5 zeros prepended

        # Build manually so we can mix zeros (which won't be at Z=0 in world)
        # Instead, use _collect_depths_for_image and patch a point at world [0,0,0]
        recon = FakeReconstruction()
        R, t = _identity_pose()
        image = FakeImage(name="000.jpg", _R=R, tvec=t)

        # 5 points at world origin → cam Z = 0 → must be excluded
        for i in range(5):
            recon.points3D[i] = FakePoint3D(xyz=np.array([0.0, 0.0, 0.0]))
            image.points2D.append(FakePoint2D(_point3D_id=i))

        # 20 points at 100 m
        for i in range(5, 25):
            recon.points3D[i] = FakePoint3D(xyz=np.array([0.0, 0.0, 100.0]))
            image.points2D.append(FakePoint2D(_point3D_id=i))

        recon.images[0] = image
        collected = _collect_depths_for_image(image, recon)

        # Only the 20 positive-depth points should be collected
        assert len(collected) == 20
        assert all(d == 100.0 for d in collected)


class TestDepthRangeStatsPrints:
    """print_depth_range_stats() must not crash on valid input."""

    def test_stats_prints_without_error(self):
        """Smoke test: function runs and produces output."""
        depth_ranges = {
            "000.jpg": (90.0, 140.0),
            "001.jpg": (95.0, 145.0),
            "002.jpg": (88.0, 138.0),
        }

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_depth_range_stats(depth_ranges)

        output = buf.getvalue()
        assert len(output) > 0
        assert "Depth Range Statistics" in output

    def test_stats_empty_dict_no_crash(self):
        """Empty input must not raise."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_depth_range_stats({})

        output = buf.getvalue()
        assert "No depth ranges" in output


class TestNoGpsCapturesHandled:
    """Depth ranges are computed from sparse points, not GPS.  GPS absence is irrelevant."""

    def test_no_gps_captures(self):
        """
        Captures without GPS coordinates should still produce depth ranges,
        because compute_depth_ranges() derives everything from 3D point
        depths, not from GPS.
        """
        depths = list(range(90, 115))
        recon = _make_reconstruction_with_depths(depths)

        # Simulate captures with no GPS (latitude / longitude = None)
        class CaptureNoGPS:
            capture_id = "000"
            rgb = "dummy.jpg"
            latitude = None
            longitude = None
            altitude = None

        result = compute_depth_ranges(recon, captures=[CaptureNoGPS()])

        assert "000.jpg" in result
        d_min, d_max = result["000.jpg"]
        assert d_min > 0.0
        assert d_max > d_min


class TestNonVisiblePoints:
    """Points that are in points3D but not visible in an image are skipped."""

    def test_unlinked_points_ignored(self):
        """
        Points in reconstruction.points3D but not referenced by any
        point2D in the image must not affect the depth range.
        """
        recon = FakeReconstruction()
        R, t = _identity_pose()
        image = FakeImage(name="000.jpg", _R=R, tvec=t)

        # 20 visible points at 120 m
        for i in range(20):
            recon.points3D[i] = FakePoint3D(xyz=np.array([0.0, 0.0, 120.0]))
            image.points2D.append(FakePoint2D(_point3D_id=i))

        # 100 unlinked points at 5 m (not referenced by any point2D)
        for i in range(100, 200):
            recon.points3D[i] = FakePoint3D(xyz=np.array([0.0, 0.0, 5.0]))
        # (no corresponding point2D entries in image)

        recon.images[0] = image
        collected = _collect_depths_for_image(image, recon)

        # Only the 20 linked points should be collected
        assert len(collected) == 20
        assert all(abs(d - 120.0) < 1e-6 for d in collected)
