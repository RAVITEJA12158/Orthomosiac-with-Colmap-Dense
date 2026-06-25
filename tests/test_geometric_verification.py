"""
Tests for Stage 5b — PoseLib geometric verification.

Offline only — poselib's actual RANSAC solvers are exercised with synthetic
2D correspondences generated from a known camera + relative pose, so the
real C++ solver runs (no mocking of poselib itself), but no real images
or COLMAP installation is required.
"""

import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.capture import Capture
from src.features.db_importer import import_to_colmap, OPENCV_MODEL_ID

poselib = pytest.importorskip("poselib")


def make_capture(capture_id: str, lat: float, lon: float) -> Capture:
    c = Capture(capture_id=capture_id, rgb=f"/fake/{capture_id}.jpg")
    c.latitude = lat
    c.longitude = lon
    c.altitude = 120.0
    return c


def make_feature_file(features_dir: Path, capture_id: str, kpts: np.ndarray) -> None:
    import h5py
    features_dir.mkdir(parents=True, exist_ok=True)
    n = len(kpts)
    desc = np.random.rand(n, 256).astype("float32")
    with h5py.File(features_dir / f"{capture_id}.h5", "w") as f:
        f.create_dataset("keypoints",   data=kpts.astype("float32"))
        f.create_dataset("descriptors", data=desc)
        f.create_dataset("image_size",  data=np.array([4000, 3000], dtype="int32"))


def make_matches_file(matches_path: Path, id_a: str, id_b: str, matches: np.ndarray) -> None:
    import h5py
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(matches_path, "w") as f:
        grp = f.require_group(id_a)
        sub = grp.create_group(id_b)
        sub.create_dataset("matches0", data=matches.astype("int32"))


def _synthetic_correspondences(n=40, width=4000, height=3000, focal=4800.0, seed=0):
    """
    Generate a synthetic pair of nadir-like views of a random 3D planar-ish
    point cloud, with a known small relative translation/rotation, and
    project them through a pinhole camera to get pixel correspondences.
    This exercises poselib's real solver end-to-end without needing real images.
    """
    rng = np.random.default_rng(seed)

    # 3D points roughly on a plane (z ~ -50, nadir scene) with small relief
    X = rng.uniform(-20, 20, size=(n, 1))
    Y = rng.uniform(-15, 15, size=(n, 1))
    Z = -50 + rng.uniform(-2, 2, size=(n, 1))
    points3d = np.hstack([X, Y, Z])

    cx, cy = width / 2.0, height / 2.0
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])

    # Camera A at origin, identity rotation
    pts_cam_a = points3d
    pix_a = (K @ pts_cam_a.T).T
    pix_a = pix_a[:, :2] / pix_a[:, 2:3]

    # Camera B: small translation + small rotation (typical of overlapping drone frames)
    theta = np.deg2rad(3.0)
    Rb = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    tb = np.array([5.0, 0.5, 0.0])
    pts_cam_b = (Rb @ points3d.T).T + tb
    pix_b = (K @ pts_cam_b.T).T
    pix_b = pix_b[:, :2] / pix_b[:, 2:3]

    valid = (
        (pix_a[:, 0] > 0) & (pix_a[:, 0] < width) & (pix_a[:, 1] > 0) & (pix_a[:, 1] < height) &
        (pix_b[:, 0] > 0) & (pix_b[:, 0] < width) & (pix_b[:, 1] > 0) & (pix_b[:, 1] < height)
    )
    return pix_a[valid], pix_b[valid]


def test_verify_matches_poselib_populates_two_view_geometries(tmp_path):
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
    ]

    pix_a, pix_b = _synthetic_correspondences(n=60)
    n = len(pix_a)

    feat_dir = tmp_path / "features"
    make_feature_file(feat_dir, "000", pix_a)
    make_feature_file(feat_dir, "001", pix_b)

    match_indices = np.stack([np.arange(n), np.arange(n)], axis=1)
    make_matches_file(tmp_path / "matches.h5", "000", "001", match_indices)

    db_path = import_to_colmap(
        captures, str(tmp_path),
        focal_length_px=4800.0,
        run_geometric_verification=False,
    )

    from src.features.geometric_verification import verify_matches_poselib
    stats = verify_matches_poselib(db_path)

    assert stats["pairs_verified"] == 1
    assert stats["total_inliers"] > 0

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    assert rows == 1

    config, qvec, tvec = conn.execute(
        "SELECT config, qvec, tvec FROM two_view_geometries"
    ).fetchone()
    assert config == 2  # CONFIG_CALIBRATED
    assert qvec is not None and len(qvec) == 32  # 4 float64
    assert tvec is not None and len(tvec) == 24  # 3 float64
    conn.close()


def test_verify_matches_poselib_rejects_too_few_matches(tmp_path):
    """Pairs with fewer than 5 matches must be skipped, not crash the solver."""
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
    ]

    kpts_a = np.array([[100.0, 100.0], [200.0, 200.0]], dtype="float32")
    kpts_b = np.array([[110.0, 105.0], [210.0, 205.0]], dtype="float32")

    feat_dir = tmp_path / "features"
    make_feature_file(feat_dir, "000", kpts_a)
    make_feature_file(feat_dir, "001", kpts_b)

    match_indices = np.array([[0, 0], [1, 1]])
    make_matches_file(tmp_path / "matches.h5", "000", "001", match_indices)

    db_path = import_to_colmap(
        captures, str(tmp_path),
        focal_length_px=4800.0,
        run_geometric_verification=False,
    )

    from src.features.geometric_verification import verify_matches_poselib
    stats = verify_matches_poselib(db_path)

    assert stats["pairs_verified"] == 0
    assert stats["pairs_rejected"] == 1


def test_verify_matches_poselib_rejects_random_noise(tmp_path):
    """
    Pure random (non-geometrically-consistent) correspondences should fail
    PoseLib's inlier threshold and be rejected, proving RANSAC is actually
    discriminating rather than accepting everything.
    """
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
    ]

    rng = np.random.default_rng(1)
    kpts_a = rng.uniform(0, 4000, size=(30, 2)).astype("float32")
    kpts_b = rng.uniform(0, 4000, size=(30, 2)).astype("float32")  # unrelated to a

    feat_dir = tmp_path / "features"
    make_feature_file(feat_dir, "000", kpts_a)
    make_feature_file(feat_dir, "001", kpts_b)

    match_indices = np.stack([np.arange(30), np.arange(30)], axis=1)
    make_matches_file(tmp_path / "matches.h5", "000", "001", match_indices)

    db_path = import_to_colmap(
        captures, str(tmp_path),
        focal_length_px=4800.0,
        run_geometric_verification=False,
    )

    from src.features.geometric_verification import verify_matches_poselib
    stats = verify_matches_poselib(db_path)

    # Random noise should yield very few (if any) geometric inliers
    assert stats["pairs_verified"] == 0
    assert stats["pairs_rejected"] == 1


def test_import_to_colmap_full_pipeline_with_poselib(tmp_path):
    """End-to-end: import_to_colmap(run_geometric_verification=True) using PoseLib."""
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
    ]

    pix_a, pix_b = _synthetic_correspondences(n=50)
    n = len(pix_a)

    feat_dir = tmp_path / "features"
    make_feature_file(feat_dir, "000", pix_a)
    make_feature_file(feat_dir, "001", pix_b)

    match_indices = np.stack([np.arange(n), np.arange(n)], axis=1)
    make_matches_file(tmp_path / "matches.h5", "000", "001", match_indices)

    db_path = import_to_colmap(
        captures, str(tmp_path),
        focal_length_px=4800.0,
        run_geometric_verification=True,  # triggers PoseLib internally
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    assert count == 1
    conn.close()
