"""
Tests for Stage 6, Steps 1-2 — keyframe selection + GPS-guided init pair.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.capture import Capture
from src.sfm.keyframes import (
    select_keyframes,
    write_keyframe_list,
    find_gps_guided_init_pair,
)


def make_capture(capture_id: str, lat: float, lon: float) -> Capture:
    c = Capture(capture_id=capture_id, rgb=f"/fake/{capture_id}.jpg")
    c.latitude = lat
    c.longitude = lon
    c.altitude = 120.0
    return c


def make_grid_mission(rows=10, cols=10, spacing_deg=0.0001):
    """
    Simulate a boustrophedon (lawnmower) drone flight path: row 0 west->east,
    row 1 east->west, etc. — the realistic flight pattern for ag missions.
    capture_id assigned in true flight-path order, NOT spatial row-major order,
    to make sure keyframe selection orders by GPS, not by filename/id.
    """
    captures = []
    idx = 0
    for row in range(rows):
        col_range = range(cols) if row % 2 == 0 else reversed(range(cols))
        for col in col_range:
            cap = make_capture(
                str(idx).zfill(4),
                lat=16.900 + row * spacing_deg,
                lon=81.700 + col * spacing_deg,
            )
            captures.append(cap)
            idx += 1
    return captures


# ── Step 1: keyframe selection ────────────────────────────────────────────

def test_select_keyframes_interval_ratio():
    captures = make_grid_mission(rows=5, cols=6)  # 30 captures
    keyframes, non_keyframes = select_keyframes(captures, interval=3)

    assert len(keyframes) + len(non_keyframes) == len(captures)
    # interval=3 -> roughly 1/3 are keyframes
    expected_keyframe_count = (len(captures) + 2) // 3
    assert len(keyframes) == expected_keyframe_count


def test_select_keyframes_excludes_no_gps():
    captures = make_grid_mission(rows=3, cols=3)
    captures[0].latitude = None
    captures[0].longitude = None

    keyframes, non_keyframes = select_keyframes(captures, interval=3)
    total = len(keyframes) + len(non_keyframes)
    assert total == len(captures) - 1  # the no-GPS capture excluded entirely


def test_select_keyframes_interval_one_means_all_keyframes():
    captures = make_grid_mission(rows=3, cols=3)
    keyframes, non_keyframes = select_keyframes(captures, interval=1)
    assert len(keyframes) == len(captures)
    assert len(non_keyframes) == 0


def test_select_keyframes_invalid_interval_raises():
    captures = make_grid_mission(rows=2, cols=2)
    try:
        select_keyframes(captures, interval=0)
        assert False, "Expected ValueError for interval=0"
    except ValueError:
        pass


def test_select_keyframes_no_coverage_gaps():
    """
    Every non-keyframe capture's nearest keyframe (by GPS) should be close
    (within ~2x grid spacing), proving keyframe selection doesn't leave large
    spatial gaps in coverage — the core safety claim behind Optimization 1.
    """
    import math

    captures = make_grid_mission(rows=8, cols=8, spacing_deg=0.0001)
    keyframes, non_keyframes = select_keyframes(captures, interval=3)

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    grid_spacing_m = haversine(16.900, 81.700, 16.900, 81.7001)  # ~ one grid step
    max_gap_allowed_m = grid_spacing_m * 3  # generous margin

    for nk in non_keyframes:
        nearest_dist = min(
            haversine(nk.latitude, nk.longitude, kf.latitude, kf.longitude)
            for kf in keyframes
        )
        assert nearest_dist <= max_gap_allowed_m, (
            f"Non-keyframe {nk.capture_id} is {nearest_dist:.1f}m from nearest "
            f"keyframe, exceeding {max_gap_allowed_m:.1f}m coverage margin."
        )


def test_write_keyframe_list(tmp_path):
    captures = make_grid_mission(rows=3, cols=3)
    keyframes, _ = select_keyframes(captures, interval=3)

    out_path = write_keyframe_list(keyframes, str(tmp_path / "keyframes.txt"))
    lines = open(out_path).read().strip().split("\n")

    assert len(lines) == len(keyframes)
    for line, kf in zip(lines, keyframes):
        assert line == f"{kf.capture_id}.jpg"


# ── Step 2: GPS-guided init pair ──────────────────────────────────────────

def test_find_gps_guided_init_pair_returns_valid_pair():
    captures = make_grid_mission(rows=10, cols=10, spacing_deg=0.0001)
    keyframes, _ = select_keyframes(captures, interval=3)

    result = find_gps_guided_init_pair(keyframes)
    assert result is not None
    cap_a, cap_b = result
    assert cap_a.capture_id != cap_b.capture_id


def test_find_gps_guided_init_pair_baseline_within_window():
    import math

    captures = make_grid_mission(rows=10, cols=10, spacing_deg=0.0001)
    keyframes, _ = select_keyframes(captures, interval=3)

    min_baseline, max_baseline = 5.0, 40.0
    result = find_gps_guided_init_pair(keyframes, min_baseline, max_baseline)
    assert result is not None
    cap_a, cap_b = result

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    dist = haversine(cap_a.latitude, cap_a.longitude, cap_b.latitude, cap_b.longitude)
    assert min_baseline <= dist <= max_baseline


def test_find_gps_guided_init_pair_no_candidates_returns_none():
    """If the baseline window can't be satisfied, return None (caller falls back)."""
    captures = [make_capture("000", 16.900, 81.700), make_capture("001", 16.9001, 81.700)]
    # baseline between these two is tiny (~11m); ask for an impossible window
    result = find_gps_guided_init_pair(captures, min_baseline_m=10000.0, max_baseline_m=20000.0)
    assert result is None


def test_find_gps_guided_init_pair_insufficient_keyframes():
    captures = [make_capture("000", 16.900, 81.700)]
    result = find_gps_guided_init_pair(captures)
    assert result is None
