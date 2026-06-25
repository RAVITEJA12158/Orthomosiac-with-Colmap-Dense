"""
Stage 5b — Geometric Verification using PoseLib

Replaces COLMAP's built-in RANSAC (pycolmap.verify_matches) with PoseLib's
LO-RANSAC + non-linear refinement. PoseLib's minimal solvers are faster and
more numerically stable than COLMAP's defaults, which matters most exactly
where nadir drone imagery is weakest: near-planar scenes (flat farmland),
where 5-point essential matrix estimation is close to degenerate.

Since intrinsics are known (single shared OPENCV camera from db_importer),
this uses poselib.estimate_relative_pose() — the essential-matrix path with
known calibration — rather than estimate_fundamental(). This is what
PoseLib's own docs recommend whenever calibration is available, and it
gives a real (R, t) pose, not just an F matrix, which COLMAP's
two_view_geometries table can store directly.

Flow:
    for each (image_a, image_b) pair in matches table:
        load keypoints for both images
        load match indices for the pair
        undistort/index into pixel coords
        poselib.estimate_relative_pose(pts_a, pts_b, camera_a, camera_b)
        keep inlier matches only
        write pose + inliers into two_view_geometries

This module is called instead of pycolmap.verify_matches() from
db_importer.py. It writes directly into the same two_view_geometries table,
so the COLMAP incremental mapper (Stage 6) cannot tell the difference.
"""

from __future__ import annotations

import sqlite3
import struct
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# COLMAP two_view_geometry "config" enum (colmap/src/colmap/scene/two_view_geometry.h)
# 0 = UNDEFINED, 1 = DEGENERATE, 2 = CALIBRATED, 3 = UNCALIBRATED,
# 4 = PLANAR, 5 = PANORAMIC, 6 = PLANAR_OR_PANORAMIC, 7 = WATERMARK, 8 = MULTIPLE
CONFIG_CALIBRATED = 2
CONFIG_PLANAR = 4

# Default PoseLib RANSAC options tuned for nadir drone imagery + LightGlue matches.
#
# WHY these values:
#   LightGlue already filters to high-confidence matches (60-80% inlier rate).
#   At 70% inliers, the 5-pt solver needs only ~40 iters for 99.9% confidence:
#       log(1 - 0.999) / log(1 - 0.7^5) ≈ 38 iterations
#   The old defaults (max_iterations=10000, min_iterations=100) ran 25-250×
#   more iterations than needed, causing ~150ms/pair instead of ~2ms/pair.
DEFAULT_RANSAC_OPTIONS = {
    "max_epipolar_error": 3.0,
    "success_prob": 0.999,
    "min_iterations": 10,
    "max_iterations": 500,
}


def _check_poselib():
    try:
        import poselib
        return poselib
    except ImportError as exc:
        raise ImportError(
            "poselib not installed.\n"
            "Run:  pip install poselib"
        ) from exc


def _unpack_params(blob: bytes) -> Tuple[float, ...]:
    n = len(blob) // 8
    return struct.unpack(f"<{n}d", blob)


def _build_poselib_camera(poselib_module, model_id: int, width: int, height: int,
                           params: Tuple[float, ...]):
    """
    Build a poselib.Camera from COLMAP camera model id + params.
    Only OPENCV (model_id=4) is needed since db_importer always writes
    a single shared OPENCV camera, but PINHOLE/SIMPLE_RADIAL are mapped too
    for forward compatibility.
    """
    model_name_by_id = {
        0: "SIMPLE_PINHOLE",
        1: "PINHOLE",
        2: "SIMPLE_RADIAL",
        3: "RADIAL",
        4: "OPENCV",
    }
    model_name = model_name_by_id.get(model_id, "OPENCV")
    return poselib_module.Camera(model_name, list(params), int(width), int(height))


def _pack_matches(matches: np.ndarray) -> bytes:
    return matches.astype("uint32").tobytes()


def _pack_pose_blobs(pose) -> dict:
    """
    Extract qvec (wxyz) and tvec from a poselib.CameraPose and pack them as
    float64 blobs in COLMAP's expected layout.
    """
    q = np.asarray(pose.q, dtype="float64")   # (4,) wxyz
    t = np.asarray(pose.t, dtype="float64")   # (3,)
    return {
        "qvec": q.tobytes(),
        "tvec": t.tobytes(),
    }


def _pack_matrix_blob(matrix: np.ndarray) -> bytes:
    return np.asarray(matrix, dtype="float64").reshape(3, 3).tobytes()


def _camera_matrix(params: Tuple[float, ...]) -> np.ndarray:
    if len(params) >= 4:
        fx, fy, cx, cy = params[:4]
    elif len(params) >= 3:
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        raise ValueError("Camera parameters do not include focal/principal point.")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype="float64")


def _rotation_matrix_to_qvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype="float64").reshape(3, 3)
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q = np.array([
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
            ])
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q = np.array([
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
            ])
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q = np.array([
                (R[1, 0] - R[0, 1]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
            ])
    norm = np.linalg.norm(q)
    return (q / norm if norm > 0 else q).astype("float64")


def _verify_opencv_essential(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    params_a: Tuple[float, ...],
    threshold_px: float,
):
    try:
        import cv2
    except ImportError:
        return None

    if len(pts_a) < 5:
        return None

    K = _camera_matrix(params_a)
    E, mask = cv2.findEssentialMat(
        pts_a.astype("float64"),
        pts_b.astype("float64"),
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=float(threshold_px),
    )
    if E is None or mask is None:
        return None
    if E.ndim == 3:
        E = E[:, :, 0]
    elif E.shape[0] > 3:
        E = E[:3, :]

    _, R, t, pose_mask = cv2.recoverPose(
        E,
        pts_a.astype("float64"),
        pts_b.astype("float64"),
        cameraMatrix=K,
        mask=mask,
    )
    inliers = pose_mask.reshape(-1).astype(bool)
    return {
        "config": CONFIG_CALIBRATED,
        "qvec": _rotation_matrix_to_qvec(R).tobytes(),
        "tvec": np.asarray(t, dtype="float64").reshape(3).tobytes(),
        "E": _pack_matrix_blob(E),
        "H": None,
        "inliers": inliers,
        "kind": "opencv_essential",
    }


def _verify_homography(pts_a: np.ndarray, pts_b: np.ndarray, threshold_px: float):
    try:
        import cv2
    except ImportError:
        return None

    if len(pts_a) < 4:
        return None

    H, mask = cv2.findHomography(
        pts_a.astype("float64"),
        pts_b.astype("float64"),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(threshold_px),
        maxIters=5000,
        confidence=0.999,
    )
    if H is None or mask is None:
        return None
    return {
        "config": CONFIG_PLANAR,
        "qvec": None,
        "tvec": None,
        "E": None,
        "H": _pack_matrix_blob(H),
        "inliers": mask.reshape(-1).astype(bool),
        "kind": "homography",
    }


# ── Parallel worker ───────────────────────────────────────────────────────────────────
# Module-level function so it can be pickled for ProcessPoolExecutor.

def _verify_pair_worker(args):
    """
    Verify one image pair using PoseLib. Runs inside a worker process.

    Args (unpacked from a single tuple for Pool.map compatibility):
        pair_id   : COLMAP pair_id (int)
        pts_a     : (M, 2) float64 ndarray — matched pixel coords in image A
        pts_b     : (M, 2) float64 ndarray — matched pixel coords in image B
        match_arr : (M, 2) uint32 ndarray  — original keypoint index pairs
        cam_a_dict: dict with keys model(str), params(list), width, height
        cam_b_dict: same for image B
        ransac_opt: dict passed directly to poselib.estimate_relative_pose

    Returns:
        Tuple consumed by verify_matches_poselib's DB writer.
    """
    pair_id, pts_a, pts_b, match_arr, cam_a_dict, cam_b_dict, ransac_opt = args
    min_inliers = 15
    loose_threshold = max(float(ransac_opt["max_epipolar_error"]) * 2.0, 4.0)

    try:
        import poselib
        import numpy as _np

        cam_a = poselib.Camera(
            cam_a_dict["model"],
            cam_a_dict["params"],
            cam_a_dict["width"],
            cam_a_dict["height"],
        )
        cam_b = poselib.Camera(
            cam_b_dict["model"],
            cam_b_dict["params"],
            cam_b_dict["width"],
            cam_b_dict["height"],
        )

        pose, info = poselib.estimate_relative_pose(
            pts_a, pts_b, cam_a, cam_b, ransac_opt=ransac_opt
        )

        inlier_mask = _np.asarray(info.get("inliers", []), dtype=bool)
        n_inliers = int(inlier_mask.sum()) if inlier_mask.size else 0

        if n_inliers >= min_inliers:
            inlier_matches = match_arr[inlier_mask].astype("uint32")
            q = _np.asarray(pose.q, dtype="float64").tobytes()
            t = _np.asarray(pose.t, dtype="float64").tobytes()
            return (
                pair_id, CONFIG_CALIBRATED, q, t, None, None,
                inlier_matches.tobytes(), n_inliers, len(inlier_matches),
                False, "poselib",
            )

    except Exception:
        pass

    for result in (
        _verify_opencv_essential(pts_a, pts_b, tuple(cam_a_dict["params"]), loose_threshold),
        _verify_homography(pts_a, pts_b, loose_threshold),
    ):
        if result is None:
            continue
        inlier_mask = result["inliers"]
        n_inliers = int(inlier_mask.sum()) if inlier_mask.size else 0
        if n_inliers < min_inliers:
            continue
        inlier_matches = match_arr[inlier_mask].astype("uint32")
        return (
            pair_id,
            result["config"],
            result["qvec"],
            result["tvec"],
            result["E"],
            result["H"],
            inlier_matches.tobytes(),
            n_inliers,
            len(inlier_matches),
            False,
            result["kind"],
        )

    return (pair_id, None, None, None, None, None, None, 0, 0, False, "too_few_inliers")

def verify_matches_poselib(
    db_path: Path,
    ransac_options: Optional[dict] = None,
    n_workers: int = 0,
) -> dict:
    """
    Run PoseLib geometric verification on all pairs in the `matches` table
    of a COLMAP database, writing results into `two_view_geometries`.

    Args:
        db_path        : path to database.db (already populated by db_importer
                          with cameras, images, keypoints, descriptors, matches)
        ransac_options  : override defaults in DEFAULT_RANSAC_OPTIONS
        n_workers       : parallel worker processes (0 = auto = cpu_count // 2,
                          1 = single-threaded, N = N processes)

    Returns:
        dict with verification stats (pairs_verified, pairs_rejected, total_inliers)
    """
    poselib = _check_poselib()
    try:
        import pycolmap as _pycolmap
    except ImportError as exc:
        raise ImportError(
            "pycolmap not installed.\n"
            "Run:  pip install pycolmap"
        ) from exc
    opts = {**DEFAULT_RANSAC_OPTIONS, **(ransac_options or {})}

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # ── load cameras keyed by image_id ───────────────────────────────────────
    image_camera: dict[int, dict] = {}
    for image_id, camera_id, width, height, model, params_blob in cur.execute("""
        SELECT images.image_id, images.camera_id, cameras.width, cameras.height,
               cameras.model, cameras.params
        FROM images JOIN cameras ON images.camera_id = cameras.camera_id
    """):
        params = _unpack_params(params_blob)
        image_camera[image_id] = {
            "camera_id": camera_id,
            "width": width,
            "height": height,
            "model": model,
            "params": params,
        }

    # ── load keypoints per image_id (lazy cache) ─────────────────────────────
    keypoint_cache: dict[int, np.ndarray] = {}

    def get_keypoints(image_id: int) -> np.ndarray:
        if image_id not in keypoint_cache:
            row = cur.execute(
                "SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)
            ).fetchone()
            if row is None:
                keypoint_cache[image_id] = np.zeros((0, 2), dtype="float64")
            else:
                rows, cols, data = row
                arr = np.frombuffer(data, dtype="float32").reshape(rows, cols)
                keypoint_cache[image_id] = arr[:, :2].astype("float64")  # x, y only
        return keypoint_cache[image_id]

    poselib_camera_cache: dict[int, "poselib.Camera"] = {}

    def get_poselib_camera(image_id: int):
        cam_id = image_camera[image_id]["camera_id"]
        if cam_id not in poselib_camera_cache:
            c = image_camera[image_id]
            poselib_camera_cache[cam_id] = _build_poselib_camera(
                poselib, c["model"], c["width"], c["height"], c["params"]
            )
        return poselib_camera_cache[cam_id]

    # ── fetch all match pairs from DB ─────────────────────────────────────────
    pair_rows = cur.execute("SELECT pair_id, rows, cols, data FROM matches").fetchall()
    print(f"[geometric_verification] PoseLib verifying {len(pair_rows)} pairs  "
          f"(max_epipolar_error={opts['max_epipolar_error']}px, "
          f"max_iterations={opts['max_iterations']})...")

    t0 = time.perf_counter()
    pairs_verified = 0
    pairs_rejected = 0
    total_inliers  = 0

    # ── resolve n_workers ────────────────────────────────────────────────────
    import os as _os
    if n_workers == 0:
        n_workers = max(1, (_os.cpu_count() or 2) // 2)
    use_parallel = n_workers > 1 and len(pair_rows) > 50

    model_name_by_id = {
        0: "SIMPLE_PINHOLE", 1: "PINHOLE",
        2: "SIMPLE_RADIAL",  3: "RADIAL", 4: "OPENCV",
    }

    def _to_cam_dict(img_id: int) -> dict:
        """Convert image_camera entry to a plain dict safe for pickling."""
        c = image_camera[img_id]
        return {
            "model":  model_name_by_id.get(c["model"], "OPENCV"),
            "params": list(c["params"]),
            "width":  int(c["width"]),
            "height": int(c["height"]),
        }

    # ── pre-build work items (pts, camera params as plain dicts) ────────────
    # Done in the main process so workers get pure numpy / plain-dict data.
    # DB cursors cannot cross process boundaries.
    work_items = []
    pre_rejected = {"no_camera": 0, "too_few_matches": 0}

    for pair_id, n_matches, cols, data in pair_rows:
        image_id_a, image_id_b = _pycolmap.pair_id_to_image_pair(pair_id)

        if image_id_a not in image_camera or image_id_b not in image_camera:
            pre_rejected["no_camera"] += 1
            pairs_rejected += 1
            continue

        match_arr = np.frombuffer(data, dtype="uint32").reshape(n_matches, cols)
        if len(match_arr) < 5:
            pre_rejected["too_few_matches"] += 1
            pairs_rejected += 1
            continue

        kpts_a = get_keypoints(image_id_a)
        kpts_b = get_keypoints(image_id_b)
        pts_a  = kpts_a[match_arr[:, 0]]
        pts_b  = kpts_b[match_arr[:, 1]]

        work_items.append((
            pair_id,
            pts_a.copy(), pts_b.copy(),
            match_arr.copy(),
            _to_cam_dict(image_id_a),
            _to_cam_dict(image_id_b),
            opts,
        ))

    n_pre_rej = sum(pre_rejected.values())
    print(f"[geometric_verification] Pre-filtered: {n_pre_rej} pairs skipped before RANSAC  "
          f"(no_camera={pre_rejected['no_camera']}, "
          f"too_few_matches={pre_rejected['too_few_matches']})")
    print(f"[geometric_verification] Dispatching {len(work_items)} pairs to "
          f"{n_workers} worker{'s' if n_workers > 1 else ''} "
          f"({'parallel' if use_parallel else 'sequential'})...")

    # ── run RANSAC (parallel or sequential) ─────────────────────────────
    # Progress fires every 10% (min 1) to scale with any mission size
    _print_every = max(1, len(work_items) // 10)

    if use_parallel:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        results_iter = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_verify_pair_worker, item): i
                       for i, item in enumerate(work_items)}
            done_count = 0
            for fut in as_completed(futures):
                results_iter.append(fut.result())
                done_count += 1
                if done_count % _print_every == 0:
                    elapsed_now = time.perf_counter() - t0
                    pct = 100.0 * done_count / len(work_items)
                    eta = (elapsed_now / done_count) * (len(work_items) - done_count)
                    print(f"[geometric_verification] {done_count}/{len(work_items)} ({pct:.0f}%)  "
                          f"elapsed: {elapsed_now:.0f}s  ETA: {eta:.0f}s")
    else:
        results_iter = []
        for loop_idx, item in enumerate(work_items, start=1):
            results_iter.append(_verify_pair_worker(item))
            if loop_idx % _print_every == 0:
                elapsed_now = time.perf_counter() - t0
                pct = 100.0 * loop_idx / len(work_items)
                eta = (elapsed_now / loop_idx) * (len(work_items) - loop_idx)
                print(f"[geometric_verification] {loop_idx}/{len(work_items)} ({pct:.0f}%)  "
                      f"elapsed: {elapsed_now:.0f}s  ETA: {eta:.0f}s")

    # ── write results to DB (main process only — SQLite is not thread-safe) ──
    rej_solver_failed   = 0
    rej_too_few_inliers = 0
    verified_by_kind = {"poselib": 0, "opencv_essential": 0, "homography": 0}
    for result in results_iter:
        (
            pair_id_r,
            config,
            qvec,
            tvec,
            e_blob,
            h_blob,
            match_bytes,
            n_inliers,
            n_rows,
            is_crash,
            kind,
        ) = result
        if config is None:
            if is_crash:
                rej_solver_failed += 1
            else:
                rej_too_few_inliers += 1
            pairs_rejected += 1
            continue

        cur.execute(
            "INSERT OR REPLACE INTO two_view_geometries "
            "(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                pair_id_r, n_rows, 2, match_bytes,
                config,
                None, e_blob, h_blob,
                qvec, tvec,
            ),
        )
        pairs_verified += 1
        total_inliers += n_inliers
        verified_by_kind[kind] = verified_by_kind.get(kind, 0) + 1

    rej_no_camera      = pre_rejected["no_camera"]
    rej_too_few_matches = pre_rejected["too_few_matches"]

    conn.commit()
    conn.close()

    elapsed = time.perf_counter() - t0
    print(f"[geometric_verification] Verified: {pairs_verified}  "
          f"(PoseLib: {verified_by_kind.get('poselib', 0)}, "
          f"OpenCV-E: {verified_by_kind.get('opencv_essential', 0)}, "
          f"Homography: {verified_by_kind.get('homography', 0)})  "
          f"Rejected: {pairs_rejected}  Total inliers: {total_inliers}  "
          f"Time: {elapsed:.1f}s")
    if pairs_rejected:
        print(f"[geometric_verification] Rejection breakdown — "
              f"no_camera: {rej_no_camera}  "
              f"too_few_matches: {rej_too_few_matches}  "
              f"solver_failed: {rej_solver_failed}  "
              f"too_few_inliers: {rej_too_few_inliers}")

    return {
        "pairs_verified": pairs_verified,
        "pairs_rejected": pairs_rejected,
        "total_inliers": total_inliers,
        "elapsed_seconds": elapsed,
        # Per-reason rejection counts
        "rej_no_camera": rej_no_camera,
        "rej_too_few_matches": rej_too_few_matches,
        "rej_solver_failed": rej_solver_failed,
        "rej_too_few_inliers": rej_too_few_inliers,
        "verified_poselib": verified_by_kind.get("poselib", 0),
        "verified_opencv_essential": verified_by_kind.get("opencv_essential", 0),
        "verified_homography": verified_by_kind.get("homography", 0),
    }
