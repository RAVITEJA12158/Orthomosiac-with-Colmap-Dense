"""
Stage 6, Step 3 (prior injection) — Inject GPS/RTK position priors

RTK path : weight 1e6  (high confidence — RTK accuracy is centimeter-level)
Non-RTK  : weight 1e2  (weak prior — consumer GPS accuracy is meter-level,
                        used only as a loose regularizer, not ground truth)

pycolmap.PosePrior stores position + position_covariance, keyed to a frame's
sensor data via `corr_data_id` (a pycolmap.data_t combining sensor + image id)
rather than directly to an image_id. This is a COLMAP 4.x change: pose priors
moved from being associated with images to being associated with generic
sensor measurement data within a frame. For our single-camera-per-image
trivial-frame setup, corr_data_id is built as:

    data_t(sensor_t(SensorType.CAMERA, image.camera_id), image.image_id)

This was verified empirically against a database built by db_importer.py
(see test_pose_priors.py) rather than assumed from documentation, since the
naive "prior keyed directly by image_id" approach raises a RuntimeError on
this pycolmap version ("PosePrior API has changed: pose priors are now
associated with frames, not images").

COLMAP's BA reads the covariance as an inverse-weight: smaller covariance
(diagonal values) -> prior trusted more strongly.

weight is converted into covariance as:  covariance = I * (1 / weight)
    weight=1e6 -> covariance diagonal = 1e-6   (tight prior, trusted heavily)
    weight=1e2 -> covariance diagonal = 1e-2   (loose prior, mostly ignored
                                                 except as a coarse anchor)
"""

from __future__ import annotations

import sqlite3
from typing import List

import numpy as np

from ..colmap_images import colmap_image_name
from ..ingestion.capture import Capture

RTK_PRIOR_WEIGHT = 1e6
GPS_PRIOR_WEIGHT = 1e2


def _weight_to_covariance(weight: float) -> np.ndarray:
    """Diagonal 3x3 covariance matrix from a scalar confidence weight."""
    variance = 1.0 / weight
    return np.eye(3, dtype="float64") * variance


def inject_gps_priors(
    db_path: str,
    captures: List[Capture],
    has_rtk: bool,
) -> int:
    """
    Write a PosePrior (WGS84 lat/lon/alt + covariance) into the COLMAP
    database for every capture that has GPS and a matching `images` row.

    Args:
        db_path  : path to database.db (already populated by db_importer,
                   which creates the trivial rig/frame/image chain this
                   function depends on)
        captures : Capture list with latitude/longitude/altitude set.
                   Same list used throughout the pipeline — RTK-corrected
                   coordinates are expected to already be populated into
                   these same fields upstream when has_rtk=True.
        has_rtk  : selects prior weight.
                   True  -> RTK_PRIOR_WEIGHT (1e6, strong)
                   False -> GPS_PRIOR_WEIGHT (1e2, weak)

    Returns:
        Number of pose priors written.
    """
    import pycolmap

    weight = RTK_PRIOR_WEIGHT if has_rtk else GPS_PRIOR_WEIGHT
    covariance = _weight_to_covariance(weight)
    label = "RTK" if has_rtk else "GPS"

    print(f"[pose_priors] Injecting {label} priors  (weight={weight:.0e}, "
          f"covariance_diag={1.0/weight:.0e})")

    # Remove any pose priors written by a previous run.  The pose_priors table
    # has a UNIQUE constraint on corr_data_id, so re-inserting without clearing
    # first raises a SQLite "constraint failed" error when the pipeline is
    # restarted from Stage 6 with an existing database.db.
    with sqlite3.connect(str(db_path)) as _conn:
        deleted = _conn.execute("DELETE FROM pose_priors").rowcount
        _conn.commit()
    if deleted:
        print(f"[pose_priors] Cleared {deleted} stale pose prior(s) from previous run.")

    db = pycolmap.Database.open(str(db_path))

    # Map image name -> Image (need camera_id + image_id to build corr_data_id),
    # same canonical naming convention as db_importer
    name_to_image = {img.name: img for img in db.read_all_images()}

    written = 0
    skipped_no_gps = 0
    skipped_no_image_row = 0

    for cap in captures:
        if cap.latitude is None or cap.longitude is None:
            skipped_no_gps += 1
            continue

        name = colmap_image_name(cap)

        image = name_to_image.get(name)
        if image is None:
            skipped_no_image_row += 1
            continue

        altitude = cap.altitude if cap.altitude is not None else 0.0

        # corr_data_id associates this prior with the image's measurement
        # data within its frame (sensor=this image's camera, id=image_id).
        sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, image.camera_id)
        data_id = pycolmap.data_t(sensor, image.image_id)

        prior = pycolmap.PosePrior()
        prior.position = np.array([cap.latitude, cap.longitude, altitude], dtype="float64")
        prior.position_covariance = covariance
        prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.WGS84
        prior.corr_data_id = data_id

        db.write_pose_prior(prior, use_pose_prior_id=False)
        written += 1

    db.close()

    if skipped_no_gps:
        print(f"[pose_priors] Skipped {skipped_no_gps} captures with no GPS.")
    if skipped_no_image_row:
        print(f"[pose_priors] Skipped {skipped_no_image_row} captures with no matching "
              f"image row in database (not imported by db_importer).")

    print(f"[pose_priors] Wrote {written} {label} pose priors.")
    return written
