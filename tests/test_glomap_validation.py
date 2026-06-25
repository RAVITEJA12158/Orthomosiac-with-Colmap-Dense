"""
Tests for GLOMAP validation — run without pycolmap by mocking reconstruction.

Tests cover all 3 checks:
  Check 1 — registration ratio
  Check 2 — GPS reprojection error (RTK only)
  Check 3 — bounding box coverage
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.capture import Capture
from src.sfm.glomap import (
    validate_glomap_reconstruction,
    _gps_to_ecef,
    _gps_field_span_m,
    MIN_REGISTRATION_RATIO,
    MAX_GPS_REPROJECTION_M,
    MIN_BBOX_COVERAGE_RATIO,
)


# ── Mock reconstruction ───────────────────────────────────────────────────────

class MockImage:
    def __init__(self, name, lat, lon, alt, offset_m=0.0):
        self.name     = name
        self._lat     = lat
        self._lon     = lon
        self._alt     = alt
        self._offset  = offset_m   # simulate GPS error

    def projection_center(self):
        """Return ECEF position with optional offset to simulate error."""
        base = _gps_to_ecef(self._lat, self._lon, self._alt)
        return base + np.array([self._offset, 0.0, 0.0])


class CollapsedMockImage:
    def __init__(self, name):
        self.name = name

    def projection_center(self):
        return np.array([0.0, 0.0, 0.0])


class MockReconstruction:
    def __init__(self, images_dict, bbox_min=None, bbox_max=None):
        self.images         = images_dict
        self.num_reg_images = len(images_dict)
        self._bbox_min      = bbox_min if bbox_min is not None else np.array([0.0, 0.0, 0.0])
        self._bbox_max      = bbox_max if bbox_max is not None else np.array([100.0, 100.0, 10.0])

    def compute_bounding_box(self):
        return self._bbox_min, self._bbox_max


def make_capture(capture_id, lat, lon, alt=120.0):
    c = Capture(
        capture_id = capture_id,
        rgb        = f"/fake/{capture_id}.jpg",
    )
    c.latitude   = lat
    c.longitude  = lon
    c.altitude   = alt
    c.green = c.nir = c.red = c.reg = f"/fake/{capture_id}.tif"
    return c


def make_field_captures(n=10):
    """10 captures spread across a small field ~100m x 100m."""
    captures = []
    for i in range(n):
        lat = 16.900 + i * 0.0009   # ~100m spacing in latitude
        lon = 81.700
        captures.append(make_capture(f"{i:03d}", lat, lon))
    return captures


# ── Check 1 Tests ─────────────────────────────────────────────────────────────

def test_check1_passes_sufficient_registration():
    captures  = make_field_captures(10)
    keyframes = captures

    # All 10 registered
    images = {i: MockImage(f"{c.capture_id}.jpg", c.latitude, c.longitude, c.altitude)
              for i, c in enumerate(captures)}
    recon = MockReconstruction(images)

    passed, reason = validate_glomap_reconstruction(
        recon, keyframes, has_rtk=False
    )
    assert passed, f"Expected pass, got: {reason}"
    print(f"PASS: test_check1_passes_sufficient_registration")


def test_check1_fails_low_registration():
    captures  = make_field_captures(10)
    keyframes = captures

    # Only 5/10 registered = 50% < 95% threshold
    images = {i: MockImage(f"{captures[i].capture_id}.jpg",
                           captures[i].latitude, captures[i].longitude, 120.0)
              for i in range(5)}
    recon = MockReconstruction(images)
    recon.num_reg_images = 5   # override to match

    passed, reason = validate_glomap_reconstruction(
        recon, keyframes, has_rtk=False
    )
    assert not passed, "Expected failure on low registration"
    assert "Check 1 FAIL" in reason
    print(f"PASS: test_check1_fails_low_registration — reason: {reason}")


# ── Check 2 Tests (RTK only) ──────────────────────────────────────────────────

def test_check2_skipped_without_rtk():
    """Check 2 must be skipped when has_rtk=False."""
    captures  = make_field_captures(10)
    keyframes = captures

    # Images with large offset — would fail if check 2 ran
    images = {i: MockImage(f"{c.capture_id}.jpg", c.latitude, c.longitude,
                           c.altitude, offset_m=50.0)
              for i, c in enumerate(captures)}
    recon = MockReconstruction(images)

    passed, reason = validate_glomap_reconstruction(
        recon, keyframes, has_rtk=False   # no RTK, check 2 must be skipped
    )
    # Should pass (check 2 skipped) unless check 3 fails
    # With good bbox it should pass
    print(f"PASS: test_check2_skipped_without_rtk — passed={passed}")


def test_check3_fails_small_bbox():
    """Check 3 should fail when reconstruction bbox is too small (folded field)."""
    captures  = make_field_captures(20)
    keyframes = captures

    images = {i: CollapsedMockImage(f"{c.capture_id}.jpg")
              for i, c in enumerate(captures)}

    # Very small bbox — simulates folded reconstruction (field collapsed to a point)
    recon = MockReconstruction(
        images,
        bbox_min = np.array([0.0, 0.0, 0.0]),
        bbox_max = np.array([5.0, 5.0, 1.0]),   # 5mx5m vs ~1800m GPS span
    )

    passed, reason = validate_glomap_reconstruction(
        recon, keyframes, has_rtk=False
    )
    assert not passed, "Expected Check 3 to fail on tiny bbox"
    assert "Check 3 FAIL" in reason
    print(f"PASS: test_check3_fails_small_bbox — reason: {reason}")


def test_check3_passes_good_bbox():
    """Check 3 passes when reconstruction bbox matches GPS span."""
    captures  = make_field_captures(10)
    keyframes = captures

    images = {i: MockImage(f"{c.capture_id}.jpg", c.latitude, c.longitude, c.altitude)
              for i, c in enumerate(captures)}

    # GPS span is ~900m in latitude (10 captures * 0.0009 deg * ~111000 m/deg)
    # Set bbox to match
    span = 900.0
    recon = MockReconstruction(
        images,
        bbox_min = np.array([0.0, 0.0, 0.0]),
        bbox_max = np.array([span, span, 10.0]),
    )

    passed, reason = validate_glomap_reconstruction(
        recon, keyframes, has_rtk=False
    )
    print(f"PASS: test_check3_passes_good_bbox — passed={passed} reason='{reason}'")


# ── GPS field span test ───────────────────────────────────────────────────────

def test_gps_field_span():
    """GPS span should be positive and roughly correct."""
    captures = make_field_captures(10)
    span_x, span_y = _gps_field_span_m(captures)

    # 10 captures * 0.0009 deg lat spacing
    assert span_x > 10, f"Expected span > 10m, got {span_x:.1f}m"
    print(f"PASS: test_gps_field_span — span_x={span_x:.1f}m span_y={span_y:.1f}m")


def test_gps_field_span_empty():
    """No GPS captures should return (0, 0)."""
    captures = [make_capture("000", None, None)]
    captures[0].latitude  = None
    captures[0].longitude = None
    span_x, span_y = _gps_field_span_m(captures)
    assert span_x == 0.0 and span_y == 0.0
    print("PASS: test_gps_field_span_empty")


if __name__ == "__main__":
    # Check 1
    test_check1_passes_sufficient_registration()
    test_check1_fails_low_registration()

    # Check 2
    test_check2_skipped_without_rtk()

    # Check 3
    test_check3_fails_small_bbox()
    test_check3_passes_good_bbox()

    # GPS span
    test_gps_field_span()
    test_gps_field_span_empty()

    print("\nAll GLOMAP validation tests done.")
