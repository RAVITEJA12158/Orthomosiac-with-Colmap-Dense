"""
Tests for Stage 3 (ALIKED extraction), Stage 4 (LightGlue matching),
and Stage 5 bridge (COLMAP database import).

All tests are fully offline — no real images, no real models.
ALIKED and LightGlue are monkey-patched with minimal fakes.
"""

import os
import sys
import sqlite3
import struct
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

h5py = pytest.importorskip("h5py")

from src.ingestion.capture import Capture
from src.features.neighbors import build_neighbor_pairs
from src.features.rgb_only import load_rgb_captures

WORK_DIR = Path(__file__).parent / "_work" / "features"


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_capture(capture_id: str, lat: float, lon: float) -> Capture:
    c = Capture(capture_id=capture_id, rgb=f"/fake/{capture_id}.jpg")
    c.latitude = lat
    c.longitude = lon
    c.altitude = 120.0
    c.green = c.nir = c.red = c.reg = f"/fake/{capture_id}.tif"
    return c


def make_feature_file(features_dir: Path, capture_id: str, n_kpts: int = 64) -> Path:
    """Write a minimal valid ALIKED .h5 feature file."""
    h5_path = features_dir / f"{capture_id}.h5"
    features_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("keypoints",   data=np.random.rand(n_kpts, 2).astype("float32"))
        f.create_dataset("descriptors", data=np.random.rand(n_kpts, 256).astype("float32"))
        f.create_dataset("image_size",  data=np.array([4000, 3000], dtype="int32"))
    return h5_path


def make_matches_file(matches_path: Path, pairs: list) -> None:
    """Write a minimal valid matches.h5 file."""
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(matches_path, "w") as f:
        for id_a, id_b, n_matches in pairs:
            grp = f.require_group(id_a)
            sub = grp.create_group(id_b)
            matches = np.stack([
                np.arange(n_matches, dtype="int32"),
                np.arange(n_matches, dtype="int32"),
            ], axis=1)
            sub.create_dataset("matches0", data=matches)


# ── neighbors.py (unchanged, just re-verified) ───────────────────────────────

def test_neighbor_pair_uniqueness_and_sorting():
    captures = [make_capture(str(i).zfill(3), 16.9 + i * 0.001, 81.7) for i in range(10)]
    pairs = build_neighbor_pairs(captures, n_neighbors=3)
    assert len(pairs) == len(set(pairs)), "Duplicate pairs"
    for a, b in pairs:
        assert a < b, f"Pair not sorted: ({a}, {b})"


def test_neighbor_pair_reduction():
    n = 50
    captures = [make_capture(str(i).zfill(3), 16.9 + i * 0.001, 81.7) for i in range(n)]
    pairs = build_neighbor_pairs(captures, n_neighbors=8)
    exhaustive = n * (n - 1) // 2
    assert len(pairs) < exhaustive


def test_neighbor_missing_gps_excluded():
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
    ]
    captures[1].latitude = None
    captures[1].longitude = None
    pairs = build_neighbor_pairs(captures, n_neighbors=8)
    assert pairs == []


def test_neighbor_distance_ordering():
    center = make_capture("000", 16.900, 81.700)
    near   = make_capture("001", 16.901, 81.700)
    far    = make_capture("002", 16.950, 81.700)
    pairs  = build_neighbor_pairs([center, near, far], n_neighbors=1)
    assert ("000", "001") in pairs


# ── extractor.py ─────────────────────────────────────────────────────────────

class FakeALIKEDResult:
    def __init__(self, n_kpts=64):
        import torch
        self._kpts = torch.rand(1, n_kpts, 2)
        self._desc = torch.rand(1, n_kpts, 256)

    def __getitem__(self, key):
        if key == "keypoints":   return self._kpts
        if key == "descriptors": return self._desc
        raise KeyError(key)


class FakeALIKED:
    """Minimal ALIKED drop-in that returns random tensors."""
    def __init__(self, model=None, max_num_keypoints=None):
        self._n = max_num_keypoints or 64

    def eval(self):      return self
    def to(self, dev):   return self

    def extract(self, img):
        return FakeALIKEDResult(n_kpts=self._n)


def _fake_load_image(path, resize=None):
    import torch
    h = 300 if resize is None else resize
    w = 400 if resize is None else int(resize * 4 / 3)
    return torch.rand(3, h, w)


def _fake_pil_open(path):
    class FakeImg:
        size = (4000, 3000)
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return FakeImg()


def test_extract_features_writes_h5(tmp_path, monkeypatch):
    """extract_features() writes one .h5 per capture with correct keys."""
    captures = [make_capture("000", 16.9, 81.7), make_capture("001", 16.91, 81.7)]

    # Patch lightglue imports
    fake_lg_module = types.ModuleType("lightglue")
    fake_lg_module.ALIKED = FakeALIKED
    fake_utils = types.ModuleType("lightglue.utils")
    fake_utils.load_image = _fake_load_image
    monkeypatch.setitem(sys.modules, "lightglue", fake_lg_module)
    monkeypatch.setitem(sys.modules, "lightglue.utils", fake_utils)

    import PIL.Image
    monkeypatch.setattr(PIL.Image, "open", lambda p: _fake_pil_open(p))

    from src.features.extractor import extract_features
    features_dir = extract_features(captures, str(tmp_path), use_gpu=False, max_keypoints=64)

    for cap in captures:
        h5 = Path(features_dir) / f"{cap.capture_id}.h5"
        assert h5.exists(), f"Missing feature file: {h5}"
        with h5py.File(h5, "r") as f:
            assert "keypoints"   in f
            assert "descriptors" in f
            assert "image_size"  in f
            assert f["keypoints"].shape[1]   == 2
            assert f["descriptors"].shape[1] == 256


def test_extract_features_skips_existing(tmp_path, monkeypatch):
    """extract_features() does not overwrite an existing .h5 file."""
    captures = [make_capture("000", 16.9, 81.7)]

    # Pre-create the feature file with a sentinel value
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    existing = feat_dir / "000.h5"
    with h5py.File(existing, "w") as f:
        f.create_dataset("sentinel", data=np.array([42]))

    fake_lg = types.ModuleType("lightglue")
    fake_lg.ALIKED = FakeALIKED
    fake_utils = types.ModuleType("lightglue.utils")
    fake_utils.load_image = _fake_load_image
    monkeypatch.setitem(sys.modules, "lightglue", fake_lg)
    monkeypatch.setitem(sys.modules, "lightglue.utils", fake_utils)

    from src.features.extractor import extract_features
    extract_features(captures, str(tmp_path), use_gpu=False)

    # Sentinel must still be there — file was not overwritten
    with h5py.File(existing, "r") as f:
        assert "sentinel" in f


# ── matcher.py ───────────────────────────────────────────────────────────────

class FakeLightGlueResult:
    def __init__(self, n_kpts=64, n_matches=20):
        import torch
        # matches0: (1, N) — first n_matches entries matched, rest -1
        m = torch.full((1, n_kpts), -1, dtype=torch.long)
        m[0, :n_matches] = torch.arange(n_matches, dtype=torch.long)
        self._data = {"matches0": m}

    def __getitem__(self, key):
        return self._data[key]


class FakeLightGlue:
    def __init__(self, features=None):
        pass
    def eval(self):       return self
    def to(self, dev):    return self
    def __call__(self, d): return FakeLightGlueResult(n_kpts=64, n_matches=20)


def test_match_features_writes_matches_h5(tmp_path, monkeypatch):
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
        make_capture("002", 16.902, 81.700),
    ]

    feat_dir = tmp_path / "features"
    for cap in captures:
        make_feature_file(feat_dir, cap.capture_id, n_kpts=64)

    fake_lg = types.ModuleType("lightglue")
    fake_lg.LightGlue = FakeLightGlue
    monkeypatch.setitem(sys.modules, "lightglue", fake_lg)

    from src.features.matcher import match_features
    matches_path = match_features(captures, str(tmp_path), n_neighbors=2, use_gpu=False)

    assert Path(matches_path).exists()
    with h5py.File(matches_path, "r") as f:
        # At least one pair should be present
        assert len(f.keys()) > 0
        for id_a in f.keys():
            for id_b in f[id_a].keys():
                m = f[id_a][id_b]["matches0"][:]
                assert m.ndim == 2
                assert m.shape[1] == 2


def test_match_features_pair_order(tmp_path, monkeypatch):
    """Pair (id_a, id_b) in matches.h5 must have id_a < id_b (sorted)."""
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
    ]

    feat_dir = tmp_path / "features"
    for cap in captures:
        make_feature_file(feat_dir, cap.capture_id)

    fake_lg = types.ModuleType("lightglue")
    fake_lg.LightGlue = FakeLightGlue
    monkeypatch.setitem(sys.modules, "lightglue", fake_lg)

    from src.features.matcher import match_features
    matches_path = match_features(captures, str(tmp_path), n_neighbors=1, use_gpu=False)

    with h5py.File(matches_path, "r") as f:
        for id_a in f.keys():
            for id_b in f[id_a].keys():
                assert id_a <= id_b, f"Pair not sorted: ({id_a}, {id_b})"


# ── db_importer.py ────────────────────────────────────────────────────────────

def test_import_creates_database(tmp_path):
    """import_to_colmap() creates a valid SQLite database with all tables."""
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
    ]

    feat_dir = tmp_path / "features"
    for cap in captures:
        make_feature_file(feat_dir, cap.capture_id)

    make_matches_file(tmp_path / "matches.h5", [("000", "001", 10)])

    from src.features.db_importer import import_to_colmap
    db_path = import_to_colmap(
        captures,
        str(tmp_path),
        focal_length_px=3500.0,
        run_geometric_verification=False,  # skip pycolmap in unit tests
    )

    assert Path(db_path).exists()

    conn = sqlite3.connect(db_path)
    for table in ("cameras", "images", "keypoints", "descriptors", "matches"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert count > 0, f"Table '{table}' is empty"
    conn.close()


def test_import_single_camera(tmp_path):
    """All images share the same camera_id (SINGLE camera mode)."""
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
        make_capture("002", 16.902, 81.700),
    ]

    feat_dir = tmp_path / "features"
    for cap in captures:
        make_feature_file(feat_dir, cap.capture_id)

    make_matches_file(tmp_path / "matches.h5", [])

    from src.features.db_importer import import_to_colmap
    db_path = import_to_colmap(captures, str(tmp_path),
                                focal_length_px=3500.0,
                                run_geometric_verification=False)

    conn = sqlite3.connect(db_path)
    cam_ids = [r[0] for r in conn.execute("SELECT DISTINCT camera_id FROM images")]
    assert len(cam_ids) == 1, f"Expected 1 camera, got {cam_ids}"
    conn.close()


def test_import_keypoint_counts(tmp_path):
    """Keypoint row count in DB matches .h5 file."""
    captures = [make_capture("000", 16.9, 81.7)]
    N = 128
    feat_dir = tmp_path / "features"
    make_feature_file(feat_dir, "000", n_kpts=N)
    make_matches_file(tmp_path / "matches.h5", [])

    from src.features.db_importer import import_to_colmap
    db_path = import_to_colmap(captures, str(tmp_path),
                                focal_length_px=3500.0,
                                run_geometric_verification=False)

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT rows FROM keypoints WHERE image_id=1").fetchone()[0]
    assert rows == N, f"Expected {N} keypoints, got {rows}"
    conn.close()


def test_import_match_pair_id_encoding(tmp_path):
    """
    Verify the pair_id encoding matches COLMAP's formula.
    pair_id = min_id * 2^20 + max_id
    """
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
    ]
    feat_dir = tmp_path / "features"
    for cap in captures:
        make_feature_file(feat_dir, cap.capture_id)

    make_matches_file(tmp_path / "matches.h5", [("000", "001", 5)])

    from src.features.db_importer import import_to_colmap, _image_pair_id
    db_path = import_to_colmap(captures, str(tmp_path),
                                focal_length_px=3500.0,
                                run_geometric_verification=False)

    conn = sqlite3.connect(db_path)
    img_id_1 = conn.execute("SELECT image_id FROM images WHERE name='000.jpg'").fetchone()[0]
    img_id_2 = conn.execute("SELECT image_id FROM images WHERE name='001.jpg'").fetchone()[0]
    expected_pair_id = _image_pair_id(img_id_1, img_id_2)

    pair_ids = [r[0] for r in conn.execute("SELECT pair_id FROM matches")]
    assert expected_pair_id in pair_ids, f"pair_id {expected_pair_id} not in DB {pair_ids}"
    conn.close()


def test_import_missing_features_file_tolerates(tmp_path):
    """import_to_colmap() tolerates captures with no .h5 — they get 0 keypoints."""
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),  # no feature file for this one
    ]
    feat_dir = tmp_path / "features"
    make_feature_file(feat_dir, "000")
    make_matches_file(tmp_path / "matches.h5", [])

    from src.features.db_importer import import_to_colmap
    db_path = import_to_colmap(captures, str(tmp_path),
                                focal_length_px=3500.0,
                                run_geometric_verification=False)

    conn = sqlite3.connect(db_path)
    img_count = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    assert img_count == 2, "Both images should be registered even if one has no features"
    conn.close()


# ── rgb_only.py ───────────────────────────────────────────────────────────────

def test_load_rgb_captures_parses_capture_id(tmp_path, monkeypatch):
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    (rgb_dir / "IMG_0001_123_RGB.jpg").write_bytes(b"\xff\xd8\xff")

    monkeypatch.setattr("src.features.rgb_only.read_gps", lambda p: (16.9, 81.7, 120.0))

    captures = load_rgb_captures(str(rgb_dir))
    assert len(captures) == 1
    assert captures[0].capture_id == "123"
    assert captures[0].latitude == 16.9
