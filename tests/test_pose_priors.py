"""
Tests for Stage 6, Step 3 — GPS/RTK pose prior injection.

Builds a real COLMAP database via db_importer.import_to_colmap() (not mocked)
to exercise the actual rig/frame/data_t chain that inject_gps_priors() depends
on, since this is exactly the kind of pycolmap version-specific wiring that
silently breaks if assumed rather than verified end-to-end.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

h5py = pytest.importorskip("h5py")
pycolmap = pytest.importorskip("pycolmap")

from src.ingestion.capture import Capture
from src.features.db_importer import import_to_colmap
from src.sfm.pose_priors import (
    inject_gps_priors,
    RTK_PRIOR_WEIGHT,
    GPS_PRIOR_WEIGHT,
)


def make_capture(capture_id: str, lat, lon, alt=120.0) -> Capture:
    c = Capture(capture_id=capture_id, rgb=f"/fake/{capture_id}.jpg")
    c.latitude = lat
    c.longitude = lon
    c.altitude = alt
    return c


def make_feature_file(features_dir: Path, capture_id: str, n_kpts: int = 20) -> None:
    features_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(features_dir / f"{capture_id}.h5", "w") as f:
        f.create_dataset("keypoints",   data=np.random.rand(n_kpts, 2).astype("float32"))
        f.create_dataset("descriptors", data=np.random.rand(n_kpts, 256).astype("float32"))
        f.create_dataset("image_size",  data=np.array([4000, 3000], dtype="int32"))


def build_test_db(tmp_path, captures):
    feat_dir = tmp_path / "features"
    for cap in captures:
        make_feature_file(feat_dir, cap.capture_id)
    with h5py.File(tmp_path / "matches.h5", "w") as f:
        pass  # no matches needed for pose prior tests
    return import_to_colmap(
        captures, str(tmp_path),
        focal_length_px=3500.0,
        run_geometric_verification=False,
    )


def _read_all_priors(db_path):
    """Read back all pose priors with their associated image, by corr_data_id."""
    db = pycolmap.Database.open(str(db_path))
    images_by_id = {img.image_id: img for img in db.read_all_images()}
    priors = []
    for pp in db.read_all_pose_priors():
        priors.append(pp)
    db.close()
    return priors, images_by_id


def test_inject_rtk_priors_uses_strong_weight(tmp_path):
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
    ]
    db_path = build_test_db(tmp_path, captures)

    written = inject_gps_priors(str(db_path), captures, has_rtk=True)
    assert written == 2

    priors, _ = _read_all_priors(db_path)
    assert len(priors) == 2
    for p in priors:
        expected_var = 1.0 / RTK_PRIOR_WEIGHT
        assert np.allclose(np.diag(p.position_covariance), expected_var)
        assert p.coordinate_system == pycolmap.PosePriorCoordinateSystem.WGS84


def test_inject_gps_priors_uses_weak_weight(tmp_path):
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
    ]
    db_path = build_test_db(tmp_path, captures)

    written = inject_gps_priors(str(db_path), captures, has_rtk=False)
    assert written == 2

    priors, _ = _read_all_priors(db_path)
    for p in priors:
        expected_var = 1.0 / GPS_PRIOR_WEIGHT
        assert np.allclose(np.diag(p.position_covariance), expected_var)


def test_inject_priors_position_matches_capture_gps(tmp_path):
    captures = [make_capture("000", 16.912345, 81.723456, alt=137.5)]
    db_path = build_test_db(tmp_path, captures)

    inject_gps_priors(str(db_path), captures, has_rtk=True)

    priors, _ = _read_all_priors(db_path)
    assert len(priors) == 1
    pos = priors[0].position
    assert np.allclose(pos, [16.912345, 81.723456, 137.5])


def test_inject_priors_skips_captures_without_gps(tmp_path):
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", None, None),
    ]
    db_path = build_test_db(tmp_path, captures)

    written = inject_gps_priors(str(db_path), captures, has_rtk=False)
    assert written == 1


def test_inject_priors_correct_data_id_association(tmp_path):
    """
    Each prior's corr_data_id must point to the correct image's camera_id +
    image_id — not just any valid data_id. This guards against a regression
    where all priors silently get associated with the same (e.g. first) image.
    """
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.701),
    ]
    db_path = build_test_db(tmp_path, captures)
    inject_gps_priors(str(db_path), captures, has_rtk=True)

    db = pycolmap.Database.open(str(db_path))
    images_by_name = {img.name: img for img in db.read_all_images()}
    priors = db.read_all_pose_priors()
    db.close()

    assert len(priors) == 2

    # Build expected position per image_id from captures
    expected_by_image_id = {}
    for cap in captures:
        img = images_by_name[f"{cap.capture_id}.jpg"]
        expected_by_image_id[img.image_id] = (cap.latitude, cap.longitude, cap.altitude)

    for p in priors:
        image_id = p.corr_data_id.id
        expected = expected_by_image_id[image_id]
        assert np.allclose(p.position, expected)
