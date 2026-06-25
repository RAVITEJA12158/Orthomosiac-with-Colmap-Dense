"""
Stage 5 (bridge) — Import ALIKED + LightGlue results into a COLMAP database

COLMAP's incremental mapper (Stage 6) reads from a .db SQLite file.
This module bridges the gap:

    features/               <- written by extractor.py (ALIKED .h5 files)
    matches.h5              <- written by matcher.py   (LightGlue matches)
         |
         v
    database.db             <- COLMAP SQLite database
         |
         v
    COLMAP RANSAC geometric verification  (run inside this module via pycolmap)
         |
         v
    Stage 6: pycolmap incremental mapper reads database.db

Flow inside this module:
    1. Create COLMAP database (via pycolmap.Database — required so rigs/
       frames are created correctly; COLMAP 4.x requires every image to
       belong to a frame, even for the trivial single-camera-per-image case)
    2. Add ONE shared camera (OPENCV model, intrinsics from EXIF or defaults)
    3. Add a trivial rig (one sensor = the shared camera)
    4. Add all images, each with its own trivial frame referencing the rig
    5. Import ALIKED keypoints per image
    6. Import LightGlue match indices per pair
    7. Run PoseLib geometric verification on all imported pairs
       (two_view_geometries table populated — mapper requires this)

After this module, database.db is indistinguishable from one produced by
COLMAP's own feature_extractor + exhaustive_matcher pipeline, except that the
keypoints are ALIKED (better) and the matches are LightGlue (fewer, cleaner).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..colmap_images import colmap_image_name
from ..ingestion.capture import Capture


# ── COLMAP camera model IDs (from colmap/src/colmap/sensor/models.h) ────────
OPENCV_MODEL_ID = 4   # fx, fy, cx, cy, k1, k2, p1, p2

# Default focal length multiplier when EXIF focal length is unavailable.
# 1.2 × sensor_diagonal is a reasonable prior for drone cameras.
FOCAL_MULTIPLIER_DEFAULT = 1.2


def _derive_focal_length_from_exif(
    captures: List[Capture],
    img_w: int,
) -> "Tuple[Optional[float], int]":
    """
    Derive a robust focal length (in pixels, at the working image width
    `img_w`) from per-capture EXIF focal-plane data written by grouper.py.

    Each capture's focal_length_px was computed at that capture's own
    EXIF image width (focal_length_ref_width), which may differ from
    img_w if extractor.py resized images before feature extraction.
    Each sample is rescaled to img_w before taking the median, so the
    result is directly usable as this camera's calibrated focal length:

        focal_px_at_w = focal_px_exif * (img_w / ref_width)

    Returns None if fewer than a handful of captures have usable EXIF
    focal data — a single or a few EXIF reads are not enough to trust
    as a calibrated prior for the *entire* shared camera.
    """
    samples = []
    for cap in captures:
        if cap.focal_length_px is None or not cap.focal_length_ref_width:
            continue
        scale = img_w / float(cap.focal_length_ref_width)
        samples.append(cap.focal_length_px * scale)

    # Require a reasonable sample size before trusting EXIF as a prior —
    # a couple of misread tags shouldn't poison the whole mission.
    min_samples = max(5, int(0.5 * len(captures))) if captures else 5
    if len(samples) < min_samples:
        return None, len(samples)

    samples.sort()
    n = len(samples)
    median = samples[n // 2] if n % 2 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])
    return float(median), len(samples)


# ── COLMAP database helpers (pycolmap.Database — required for rig/frame) ────

def _open_database(db_path: Path):
    """
    Create/open a COLMAP database via pycolmap.Database.open().

    pycolmap.Database.open() creates the standard schema (cameras, rigs,
    frames, images, keypoints, descriptors, matches, two_view_geometries,
    pose_priors) if the file doesn't exist yet — including the rigs/frames
    tables that raw sqlite3 table creation would miss. COLMAP 4.x requires
    every image to belong to a frame (even a trivial one-image, one-sensor
    frame); skipping this causes the incremental mapper to silently fail
    to register any image.
    """
    import pycolmap

    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return pycolmap.Database.open(str(db_path))


def _float32_descriptors_to_uint8(desc: np.ndarray) -> np.ndarray:
    """
    pycolmap.Database.write_descriptors() requires a pycolmap.FeatureDescriptors
    object, which stores raw uint8 bytes (the `type` enum field tells readers
    how to reinterpret them). ALIKED descriptors are float32, so each row's
    bytes are viewed as uint8 without copying/converting values — this is a
    reinterpretation of the same bits, not a numeric cast.
    """
    desc = np.ascontiguousarray(desc, dtype="float32")
    return desc.view("uint8").reshape(desc.shape[0], -1)


def _image_pair_id(image_id1: int, image_id2: int) -> int:
    """
    Compute COLMAP's canonical pair_id for two image IDs.

    This is the same order-independent encoding used internally by
    db.write_matches() (db.write_matches() does this work for us when
    importing — this wrapper exists so callers/tests can independently
    compute the pair_id to look matches up by it later).
    """
    import pycolmap
    return pycolmap.image_pair_to_pair_id(int(image_id1), int(image_id2))


# ── public API ───────────────────────────────────────────────────────────────

def import_to_colmap(
    captures: List[Capture],
    output_dir: str,
    focal_length_px: Optional[float] = None,
    run_geometric_verification: bool = True,
) -> Path:
    """
    Import ALIKED features + LightGlue matches into a COLMAP database.

    Args:
        captures                  : Capture list from ingestion
        output_dir                : same root dir as extract_features / match_features
        focal_length_px           : known focal length in pixels.
                                    If None, estimated from image size using
                                    FOCAL_MULTIPLIER_DEFAULT heuristic.
        run_geometric_verification: run PoseLib RANSAC to populate
                                    two_view_geometries (required for mapper).
                                    Set False to skip during testing.

    Returns:
        Path to database.db
    """
    import h5py  # lazy import: only required at runtime, not at module load
    import pycolmap

    out = Path(output_dir)
    features_dir = out / "features"
    matches_path = out / "matches.h5"
    db_path = out / "database.db"

    if not features_dir.exists():
        raise FileNotFoundError(f"Features not found: {features_dir}. Run extract_features() first.")
    if not matches_path.exists():
        raise FileNotFoundError(f"Matches not found: {matches_path}. Run match_features() first.")

    t0 = time.perf_counter()
    print(f"[db_importer] Importing into COLMAP database: {db_path}")

    db = _open_database(db_path)

    # ── 1. Read image dimensions from the first valid feature file ──────────
    # All RGB images share the same camera (SINGLE camera mode).
    sample_cap = next((c for c in captures if c.rgb), None)
    if sample_cap is None:
        raise ValueError("No captures with RGB images.")

    sample_h5 = features_dir / f"{sample_cap.capture_id}.h5"
    with h5py.File(sample_h5, "r") as f:
        img_w, img_h = int(f["image_size"][0]), int(f["image_size"][1])

    # ── 2. Camera — single shared OPENCV model ──────────────────────────────
    has_prior_focal_length = False
    if focal_length_px is not None:
        # Caller explicitly supplied a known/calibrated value.
        has_prior_focal_length = True
        print(f"[db_importer] focal_length_px: {focal_length_px:.1f} px  "
              f"(source: explicit override, has_prior_focal_length=True)")
    else:
        derived_focal_px, n_samples = _derive_focal_length_from_exif(captures, img_w)
        if derived_focal_px is not None:
            focal_length_px = derived_focal_px
            has_prior_focal_length = True
            print(f"[db_importer] focal_length_px: {focal_length_px:.1f} px  "
                  f"(source: EXIF median over {n_samples} captures, "
                  f"has_prior_focal_length=True)")
        else:
            # Heuristic prior: focal ≈ 1.2 × max(width, height)
            focal_length_px = FOCAL_MULTIPLIER_DEFAULT * max(img_w, img_h)
            has_prior_focal_length = False
            print(f"[db_importer] focal_length_px not found in EXIF "
                  f"(only {n_samples} usable captures) — using heuristic: "
                  f"{focal_length_px:.1f} px  ({img_w}×{img_h} image, "
                  f"has_prior_focal_length=False)")
            print("[db_importer] WARNING: without a calibrated focal length, "
                  "GLOMAP's global positioner and the incremental mapper's "
                  "self-calibration are both far less stable, especially for "
                  "near-planar nadir drone scenes. If reconstructions are "
                  "fragmenting or GLOMAP keeps failing validation, pass a "
                  "known focal_length_px explicitly or check why EXIF "
                  "FocalLength/FocalPlaneXResolution tags are missing.")

    cx, cy = img_w / 2.0, img_h / 2.0
    # OPENCV params: fx, fy, cx, cy, k1, k2, p1, p2  (all distortion = 0 initially)
    params = [focal_length_px, focal_length_px, cx, cy, 0.0, 0.0, 0.0, 0.0]

    camera = pycolmap.Camera(model="OPENCV", params=params, width=img_w, height=img_h)
    camera.has_prior_focal_length = has_prior_focal_length
    camera_id = db.write_camera(camera)

    # ── 3. Trivial rig — one sensor (the shared camera), identity pose ──────
    # COLMAP 4.x requires every image to belong to a frame, which in turn
    # belongs to a rig — even for the single-camera-per-image case. This is
    # the "trivial rig" pattern: one rig per camera, one frame per image.
    sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id)
    rig = pycolmap.Rig()
    rig.add_ref_sensor(sensor)
    rig_id = db.write_rig(rig)

    # ── 4. Images + trivial frames ────────────────────────────────────────
    # Map capture_id → COLMAP image_id for pair-ID calculation later.
    cap_id_to_image_id: dict[str, int] = {}

    for cap in captures:
        # Image name = what the mapper will look for in the image folder.
        # We reuse <capture_id>.jpg  (same naming as the feature .h5 files).
        name = colmap_image_name(cap)

        # read_image_with_name() returns None when the image does not exist yet.
        # We avoid exists_image() because it is not present in all pycolmap 4.x builds.
        existing = db.read_image_with_name(name)
        if existing is not None:
            cap_id_to_image_id[cap.capture_id] = existing.image_id
            continue

        image = pycolmap.Image(name=name, camera_id=camera_id)
        image_id = db.write_image(image)

        # Frame: one image per frame (trivial frame), references the rig.
        frame = pycolmap.Frame()
        frame.rig_id = rig_id
        data_id = pycolmap.data_t(sensor, image_id)
        frame.add_data_id(data_id)
        frame_id = db.write_frame(frame)

        # Back-fill the image's frame_id now that the frame exists.
        image.image_id = image_id
        image.frame_id = frame_id
        db.update_image(image)

        cap_id_to_image_id[cap.capture_id] = image_id

    print(f"[db_importer] Registered {len(captures)} images  "
          f"(camera_id={camera_id}, rig_id={rig_id})")

    # ── 5. Keypoints & descriptors — native pycolmap.Database methods ────────
    # write_keypoints/write_descriptors handle the schema (including the
    # descriptors.type column) correctly without us needing to track COLMAP's
    # internal blob layout by hand.
    kp_count_total = 0
    missing_feature_files = 0

    for cap in captures:
        h5_path = features_dir / f"{cap.capture_id}.h5"
        if not h5_path.exists():
            missing_feature_files += 1
            continue

        with h5py.File(h5_path, "r") as f:
            kpts = f["keypoints"][:]      # (N, 2) float32
            desc = f["descriptors"][:]    # (N, D) float32

        image_id = cap_id_to_image_id[cap.capture_id]

        db.write_keypoints(image_id, np.ascontiguousarray(kpts, dtype="float32"))

        desc_uint8 = _float32_descriptors_to_uint8(desc)

        # pycolmap.FeatureExtractorType.ALIKED_N16ROT was added in a specific
        # pycolmap release.  We probe for it at import time and fall back to
        # CUSTOM (an always-present enum value) or a bare descriptor if needed.
        _et = getattr(pycolmap, "FeatureExtractorType", None)
        _aliked_type = (
            getattr(_et, "ALIKED_N16ROT", None)
            or getattr(_et, "CUSTOM", None)
        ) if _et is not None else None

        if _aliked_type is not None:
            feature_descriptors = pycolmap.FeatureDescriptors(_aliked_type, desc_uint8)
        else:
            # Oldest pycolmap builds accept just the raw uint8 array.
            feature_descriptors = pycolmap.FeatureDescriptors(desc_uint8)
        db.write_descriptors(image_id, feature_descriptors)

        kp_count_total += len(kpts)

    print(f"[db_importer] Keypoints imported: {kp_count_total}  "
          f"(missing feature files: {missing_feature_files})")

    # ── 6. Matches — native write_matches (handles pair_id + column order) ──
    match_pair_count = 0

    with h5py.File(matches_path, "r") as mf:
        for id_a in mf.keys():
            if id_a not in cap_id_to_image_id:
                continue
            for id_b in mf[id_a].keys():
                if id_b not in cap_id_to_image_id:
                    continue

                matches = mf[id_a][id_b]["matches0"][:]   # (M, 2)  int32
                if len(matches) == 0:
                    continue

                img_id_a = cap_id_to_image_id[id_a]
                img_id_b = cap_id_to_image_id[id_b]

                if img_id_a > img_id_b:
                    matches = matches[:, ::-1].copy()
                    img_id_a, img_id_b = img_id_b, img_id_a

                # write_matches handles pair_id encoding internally.
                # By enforcing img_id_a < img_id_b, we avoid any ambiguity in PyCOLMAP's
                # python bindings regarding automatic column swapping.
                db.write_matches(img_id_a, img_id_b, matches.astype("uint32"))
                match_pair_count += 1

    print(f"[db_importer] Match pairs imported: {match_pair_count}")

    # ── 7. Geometric verification (PoseLib RANSAC) ───────────────────────────
    db.close()
    if run_geometric_verification:
        _run_geometric_verification(db_path)
    else:
        print("[db_importer] Geometric verification skipped (run_geometric_verification=False).")

    elapsed = time.perf_counter() - t0
    print(f"[db_importer] Done. Database: {db_path}  Time: {elapsed:.1f}s")
    return db_path


def _run_geometric_verification(db_path: Path) -> None:
    """
    Run PoseLib RANSAC on all match pairs to populate two_view_geometries.

    PoseLib is used instead of COLMAP's built-in RANSAC because its
    LO-RANSAC + non-linear refinement solvers are more numerically stable,
    which matters most for nadir drone imagery over flat farmland — a
    near-planar scene that is close to the degenerate case for 5-point
    essential matrix estimation.
    """
    from .geometric_verification import verify_matches_poselib

    print("[db_importer] Running PoseLib geometric verification...")
    t = time.perf_counter()
    verify_matches_poselib(db_path)
    print(f"[db_importer] Geometric verification done. ({time.perf_counter() - t:.1f}s)")
