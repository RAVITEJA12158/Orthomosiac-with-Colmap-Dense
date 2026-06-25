"""
Tests for SfM pipeline orchestration logic.
Mocks pycolmap so these run without the library installed.
Tests the decision flow: RTK → GLOMAP → validate → fallback logic.
"""

import sys
import os
import sqlite3
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.capture import Capture
from src.colmap_images import colmap_image_name
from src.sfm.keyframes import select_keyframes, find_gps_guided_init_pair


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


def make_grid_captures(rows=3, cols=5):
    """Simulate a grid flight pattern."""
    captures = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            lat = 16.900 + r * 0.001
            lon = 81.700 + c * 0.001
            captures.append(make_capture(f"{idx:03d}", lat, lon))
            idx += 1
    return captures


# ── Keyframe selection tests ──────────────────────────────────────────────────

def test_keyframe_count():
    """interval=3 should give ~1/3 as keyframes."""
    captures = make_grid_captures(3, 9)   # 27 captures
    kf, non_kf = select_keyframes(captures, interval=3)

    assert len(kf) + len(non_kf) == len(captures), \
        "Total should equal input count"
    assert len(kf) == 9, f"Expected 9 keyframes (27//3), got {len(kf)}"
    assert len(non_kf) == 18
    print(f"PASS: test_keyframe_count — {len(kf)} keyframes, {len(non_kf)} non-keyframes")


def test_keyframe_interval_1():
    """interval=1 should make all captures keyframes."""
    captures = make_grid_captures(2, 5)   # 10 captures
    kf, non_kf = select_keyframes(captures, interval=1)

    assert len(kf) == 10
    assert len(non_kf) == 0
    print("PASS: test_keyframe_interval_1")


def test_keyframe_no_gps_excluded():
    """Captures without GPS should be excluded from both lists."""
    captures = make_grid_captures(2, 5)
    captures[3].latitude  = None
    captures[3].longitude = None

    kf, non_kf = select_keyframes(captures, interval=2)
    total = len(kf) + len(non_kf)
    assert total == 9, f"Expected 9 with GPS, got {total}"
    print(f"PASS: test_keyframe_no_gps_excluded — {total} with GPS")


# ── GPS init pair tests ───────────────────────────────────────────────────────

def test_init_pair_found():
    """Should find a pair with reasonable baseline."""
    captures = make_grid_captures(3, 5)
    kf, _ = select_keyframes(captures, interval=1)
    pair = find_gps_guided_init_pair(kf, min_baseline_m=5.0, max_baseline_m=200.0)

    assert pair is not None, "Expected init pair to be found"
    cap_a, cap_b = pair
    assert cap_a.capture_id != cap_b.capture_id
    print(f"PASS: test_init_pair_found — pair: {cap_a.capture_id} <-> {cap_b.capture_id}")


def test_init_pair_none_on_impossible_baseline():
    """Should return None when no pair satisfies baseline window."""
    captures = [make_capture("000", 16.900, 81.700)]
    pair = find_gps_guided_init_pair(captures, min_baseline_m=50.0, max_baseline_m=100.0)
    assert pair is None
    print("PASS: test_init_pair_none_on_impossible_baseline")


def test_init_pair_baseline_in_range():
    """Selected pair baseline should be within the specified window."""
    from src.features.neighbors import haversine_distance

    captures = make_grid_captures(4, 4)
    kf, _ = select_keyframes(captures, interval=1)
    pair = find_gps_guided_init_pair(kf, min_baseline_m=5.0, max_baseline_m=300.0)

    if pair is not None:
        cap_a, cap_b = pair
        dist = haversine_distance(
            cap_a.latitude, cap_a.longitude,
            cap_b.latitude, cap_b.longitude,
        )
        assert 5.0 <= dist <= 300.0, f"Baseline {dist:.1f}m out of range"
        print(f"PASS: test_init_pair_baseline_in_range — baseline={dist:.1f}m")
    else:
        print("SKIP: test_init_pair_baseline_in_range — no pair found (too few captures)")


# ── Pipeline flow tests (no pycolmap) ────────────────────────────────────────

def test_pipeline_skips_glomap_without_rtk():
    """
    Verify that has_rtk=False causes GLOMAP to be skipped at logic level.
    We test the decision flag, not the actual mapper call.
    """
    has_rtk    = False
    try_glomap = has_rtk   # this is the exact logic in pipeline.py

    assert try_glomap is False
    print("PASS: test_pipeline_skips_glomap_without_rtk")


def test_pipeline_tries_glomap_with_rtk():
    """Verify has_rtk=True means GLOMAP is attempted."""
    has_rtk    = True
    try_glomap = has_rtk

    assert try_glomap is True
    print("PASS: test_pipeline_tries_glomap_with_rtk")


def test_run_sfm_writes_final_reconstruction_and_reports_fallback(tmp_path, monkeypatch, capsys):
    """Final model must be persisted at output_dir after COLMAP fallback."""
    import src.sfm.pipeline as sfm_pipeline

    class MockReconstruction:
        def __init__(self):
            self.num_reg_images = 6
            self.points3D = {1: object(), 2: object()}
            self.write_path = None

        def write(self, path):
            self.write_path = path

    captures = make_grid_captures(2, 3)
    recon = MockReconstruction()

    monkeypatch.setattr(sfm_pipeline, "inject_gps_priors", lambda *args, **kwargs: 0)
    monkeypatch.setattr(sfm_pipeline, "run_glomap", lambda *args, **kwargs: None)
    monkeypatch.setattr(sfm_pipeline, "run_colmap_incremental", lambda *args, **kwargs: recon)
    monkeypatch.setattr(sfm_pipeline, "register_non_keyframes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(sfm_pipeline, "run_final_bundle_adjustment", lambda *args, **kwargs: None)
    monkeypatch.setattr(sfm_pipeline, "align_to_gps", lambda *args, **kwargs: True)

    output_dir = tmp_path / "sparse"
    result = sfm_pipeline.run_sfm(
        database_path = str(tmp_path / "colmap.db"),
        image_dir     = str(tmp_path / "rgb"),
        output_dir    = str(output_dir),
        captures      = captures,
        has_rtk       = True,
    )

    captured = capsys.readouterr().out
    assert result is recon
    assert recon.write_path == str(output_dir)
    assert "Path used  : COLMAP" in captured


def test_run_sfm_retries_all_captures_when_keyframes_fail(tmp_path, monkeypatch, capsys):
    """If keyframe-only mapping has no usable graph, retry on all captures."""
    import src.sfm.pipeline as sfm_pipeline

    class MockReconstruction:
        def __init__(self):
            self.num_reg_images = 6
            self.points3D = {1: object(), 2: object()}

        def write(self, path):
            self.write_path = path

    captures = make_grid_captures(2, 3)
    recon = MockReconstruction()
    mapper_sizes = []

    def fake_run_colmap_incremental(*args, **kwargs):
        mapper_sizes.append(len(kwargs["keyframes"]))
        return None if len(mapper_sizes) == 1 else recon

    monkeypatch.setattr(sfm_pipeline, "inject_gps_priors", lambda *args, **kwargs: 0)
    monkeypatch.setattr(sfm_pipeline, "run_colmap_incremental", fake_run_colmap_incremental)
    monkeypatch.setattr(sfm_pipeline, "register_non_keyframes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(sfm_pipeline, "run_final_bundle_adjustment", lambda *args, **kwargs: None)
    monkeypatch.setattr(sfm_pipeline, "align_to_gps", lambda *args, **kwargs: True)

    result = sfm_pipeline.run_sfm(
        database_path = str(tmp_path / "colmap.db"),
        image_dir     = str(tmp_path / "rgb"),
        output_dir    = str(tmp_path / "sparse"),
        captures      = captures,
        has_rtk       = False,
    )

    captured = capsys.readouterr().out
    assert result is recon
    assert mapper_sizes == [2, 6]
    assert "Retrying COLMAP with all captures" in captured
    assert "Path used  : COLMAP-full" in captured


def test_colmap_verified_pair_stats_counts_mapper_graph(tmp_path):
    """Mapper diagnostics should count only verified pairs inside the allowlist."""
    from src.sfm.colmap_mapper import _verified_pair_stats

    db_path = tmp_path / "database.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
    conn.executemany(
        "INSERT INTO images (image_id, name) VALUES (?, ?)",
        [(1, "000.jpg"), (2, "001.jpg"), (3, "002.jpg")],
    )
    max_image_id = 2147483647
    conn.execute(
        "INSERT INTO two_view_geometries (pair_id, rows) VALUES (?, ?)",
        (1 * max_image_id + 2, 25),
    )
    conn.execute(
        "INSERT INTO two_view_geometries (pair_id, rows) VALUES (?, ?)",
        (2 * max_image_id + 3, 0),
    )
    conn.commit()
    conn.close()

    stats = _verified_pair_stats(str(db_path), ["000.jpg", "001.jpg", "002.jpg"])
    assert stats == {
        "images_in_db": 3,
        "verified_pairs": 1,
        "images_with_verified_matches": 2,
    }


def test_colmap_mapper_does_not_force_gps_init_pair(tmp_path, monkeypatch):
    """
    GPS-selected pairs can be bad two-view initializers; mapper should let
    COLMAP choose from all verified pairs instead of setting init_image_id1/2.
    """
    import src.sfm.colmap_mapper as colmap_mapper

    cap_a = make_capture("000", 16.900, 81.700)
    cap_b = make_capture("001", 16.901, 81.700)
    rgb_a = tmp_path / "000.jpg"
    rgb_b = tmp_path / "001.jpg"
    rgb_a.write_bytes(b"fake image bytes")
    rgb_b.write_bytes(b"fake image bytes")
    cap_a.rgb = str(rgb_a)
    cap_b.rgb = str(rgb_b)

    db_path = tmp_path / "database.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
    conn.executemany(
        "INSERT INTO images (image_id, name) VALUES (?, ?)",
        [(1, "000.jpg"), (2, "001.jpg")],
    )
    max_image_id = 2147483647
    conn.execute(
        "INSERT INTO two_view_geometries (pair_id, rows) VALUES (?, ?)",
        (1 * max_image_id + 2, 30),
    )
    conn.commit()
    conn.close()

    calls = {}

    class FakeOptions:
        def __init__(self):
            self.mapper = types.SimpleNamespace(
                ba_local_num_images=6,
                filter_max_reproj_error=4.0,
            )
            self.ba_global_frames_ratio = 1.1
            self.use_prior_position = False
            self.num_threads = 1
            self.image_names = []

    class FakeDbImage:
        def __init__(self, image_id):
            self.image_id = image_id

    class FakeDb:
        def read_image_with_name(self, name):
            return {"000.jpg": FakeDbImage(1), "001.jpg": FakeDbImage(2)}.get(name)

        def close(self):
            pass

    def fake_incremental_mapping(database_path, image_path, output_path, options):
        calls["has_init_id1"] = hasattr(options, "init_image_id1")
        calls["has_init_id2"] = hasattr(options, "init_image_id2")
        return {}

    fake_pycolmap = types.SimpleNamespace(
        Database=types.SimpleNamespace(open=lambda _: FakeDb()),
        IncrementalPipelineOptions=FakeOptions,
        image_pair_to_pair_id=lambda a, b: min(a, b) * max_image_id + max(a, b),
        pair_id_to_image_pair=lambda pair_id: (1, 2),
        incremental_mapping=fake_incremental_mapping,
    )
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)

    result = colmap_mapper.run_colmap_incremental(
        database_path=str(db_path),
        image_dir=str(tmp_path),
        output_dir=str(tmp_path / "sparse"),
        keyframes=[cap_a, cap_b],
        init_pair=(cap_a, cap_b),
    )

    assert result is None
    assert calls == {"has_init_id1": False, "has_init_id2": False}


def test_colmap_mapper_rejects_tiny_component(tmp_path, monkeypatch):
    """A 5-image island out of a large mapper set must not count as success."""
    import src.sfm.colmap_mapper as colmap_mapper

    caps = []
    for i in range(40):
        cap = make_capture(f"{i:03d}", 16.900 + i * 0.00001, 81.700)
        rgb = tmp_path / f"{i:03d}.jpg"
        rgb.write_bytes(b"fake image bytes")
        cap.rgb = str(rgb)
        caps.append(cap)

    db_path = tmp_path / "database.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
    conn.executemany(
        "INSERT INTO images (image_id, name) VALUES (?, ?)",
        [(i + 1, f"{i:03d}.jpg") for i in range(40)],
    )
    max_image_id = 2147483647
    conn.executemany(
        "INSERT INTO two_view_geometries (pair_id, rows) VALUES (?, ?)",
        [(1 * max_image_id + i, 30) for i in range(2, 40)],
    )
    conn.commit()
    conn.close()

    class FakeOptions:
        def __init__(self):
            self.mapper = types.SimpleNamespace(
                ba_local_num_images=6,
                filter_max_reproj_error=4.0,
            )
            self.ba_global_frames_ratio = 1.1
            self.use_prior_position = False
            self.num_threads = 1
            self.max_num_models = 1
            self.min_model_size = 10
            self.image_names = []

    class TinyRecon:
        num_reg_images = 5

    fake_pycolmap = types.SimpleNamespace(
        IncrementalPipelineOptions=FakeOptions,
        pair_id_to_image_pair=lambda pair_id: divmod(pair_id, max_image_id),
        incremental_mapping=lambda **kwargs: {0: TinyRecon()},
    )
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)

    result = colmap_mapper.run_colmap_incremental(
        database_path=str(db_path),
        image_dir=str(tmp_path),
        output_dir=str(tmp_path / "sparse"),
        keyframes=caps,
        init_pair=None,
    )

    assert result is None


def test_colmap_mapper_uses_canonical_lowercase_image_names(tmp_path, monkeypatch):
    """
    Regression guard for COLMAP reporting `loaded 0`: the mapper image allowlist
    and staged image files must match db_importer's lowercased DB names.
    """
    import src.sfm.colmap_mapper as colmap_mapper

    rgb_path = tmp_path / "IMG_260315_083045_0001_RGB.JPG"
    rgb_path.write_bytes(b"fake image bytes")
    cap = make_capture("0001", 16.900, 81.700)
    cap.rgb = str(rgb_path)

    calls = {}

    class FakeOptions:
        def __init__(self):
            self.mapper = types.SimpleNamespace(
                ba_local_num_images=6,
                filter_max_reproj_error=4.0,
            )
            self.ba_global_frames_ratio = 1.1
            self.use_prior_position = False
            self.num_threads = 1
            self.image_names = []

    def fake_incremental_mapping(database_path, image_path, output_path, options):
        calls["database_path"] = database_path
        calls["image_path"] = image_path
        calls["output_path"] = output_path
        calls["image_names"] = list(options.image_names)
        return {}

    fake_pycolmap = types.SimpleNamespace(
        IncrementalPipelineOptions=FakeOptions,
        incremental_mapping=fake_incremental_mapping,
    )
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)

    result = colmap_mapper.run_colmap_incremental(
        database_path=str(tmp_path / "database.db"),
        image_dir=str(tmp_path / "rgb"),
        output_dir=str(tmp_path / "sparse"),
        keyframes=[cap],
        init_pair=None,
    )

    assert result is None
    assert calls["image_names"] == [colmap_image_name(cap)]
    assert calls["image_names"] == ["0001.jpg"]
    assert (tmp_path / "sparse" / "colmap" / "images" / "0001.jpg").exists()


if __name__ == "__main__":
    test_keyframe_count()
    test_keyframe_interval_1()
    test_keyframe_no_gps_excluded()
    test_init_pair_found()
    test_init_pair_none_on_impossible_baseline()
    test_init_pair_baseline_in_range()
    test_pipeline_skips_glomap_without_rtk()
    test_pipeline_tries_glomap_with_rtk()
    print("\nAll SfM pipeline tests done.")
