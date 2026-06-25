"""
Stage 6, Step 4 — COLMAP incremental SfM with agricultural optimizations.

This is the safe path — always runs when has_rtk=False,
and as fallback when GLOMAP fails validation.

5 risk-free optimizations applied:

  1. Keyframe-only mapping
     Only keyframes go through full incremental SfM.
     Non-keyframes registered separately via fast PnP (Step 5).
     Safe at 80% overlap — every ground point still in 8+ keyframes.

  2. GPS-guided init pair
     Precomputed in keyframes.py. Avoids COLMAP scoring all pairs.
     Safe because GPS baseline is physically meaningful.

  3. Bundle adjustment frequency tuning
     ba_local_num_images    = 12  (default 6)
     opt.ba_global_frames_ratio = 1.3  (default 1.1)
     Run BA less often but with more context each time.
     Safe for flat terrain + clean LightGlue matches.

  4. Tight reprojection filter
     mapper.filter_max_reproj_error = 2.0  (default 4.0)
     Safe because ALIKED+LightGlue produces clean matches.
     Would be risky with noisy SIFT matches — not risky here.

  5. All CPU cores
     num_threads = -1
     No risk. Linear speedup with core count.

GPS prior weight is set by pose_priors.py before this runs:
  RTK  → 1e6 covariance diagonal → strong anchor
  GPS  → 1e2 covariance diagonal → loose regularizer

use_prior_position = True is required to activate the priors.
This was confirmed necessary — default is False and priors are ignored
without it.

API surface confirmed against pycolmap 4.0.4:
  opt.mapper.ba_local_num_images       → int
  opt.mapper.filter_max_reproj_error   → float
  opt.ba_global_frames_ratio           → float  (renamed from ba_global_images_ratio)
  opt.init_image_id1 / init_image_id2  → int
  opt.use_prior_position               → bool
  opt.num_threads                      → int
  pycolmap.incremental_mapping()       → dict[int, Reconstruction]
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from ..colmap_images import colmap_image_name
from ..ingestion.capture import Capture


def _get_num_reg_images(recon) -> int:
    val = recon.num_reg_images
    return int(val() if callable(val) else val)


def _best_reconstruction(reconstructions: dict) -> Optional[object]:
    if not reconstructions:
        return None
    return max(reconstructions.values(), key=_get_num_reg_images)


def _set_opt(obj, name: str, value) -> bool:
    """
    Safely set a pycolmap option attribute only if it exists.

    pycolmap's IncrementalPipelineOptions (and sub-objects) occasionally rename
    or add fields across minor releases.  Using setattr without checking first
    would either raise AttributeError (hard crash) or silently create a new
    Python-only attribute that COLMAP never reads (silent misconfiguration).
    This helper avoids both failure modes.

    Returns True if the attribute was found and set, False if it was skipped.
    """
    if not hasattr(obj, name):
        return False
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


def _resolve_image_id(db, capture: Capture) -> Optional[int]:
    """
    Look up the COLMAP image_id for a Capture by its name convention.
    db_importer names images as <capture_id><ext>.
    """
    name = colmap_image_name(capture)
    img  = db.read_image_with_name(name)
    return img.image_id if img is not None else None


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _prepare_mapper_image_dir(keyframes: List[Capture], output_path: Path) -> Path:
    """
    Create a flat image directory whose filenames match the COLMAP DB names.

    Mission RGB files can have names like IMG_..._RGB.JPG while the database
    stores canonical names like 0000.jpg. COLMAP's mapper needs the on-disk
    files to be addressable by those database names.
    """
    image_path = output_path / "images"
    image_path.mkdir(parents=True, exist_ok=True)

    linked = 0
    missing = 0
    for cap in keyframes:
        if not cap.rgb:
            missing += 1
            continue

        src = Path(cap.rgb)
        if not src.exists():
            missing += 1
            continue

        dst = image_path / colmap_image_name(cap)
        if not dst.exists():
            _symlink_or_copy(src, dst)
            linked += 1

    if linked:
        print(f"[colmap] Prepared {linked} canonical image links in {image_path}")
    if missing:
        print(f"[colmap] Warning: {missing} keyframe RGB files were missing; "
              "COLMAP may not be able to load those images.")

    return image_path


def _decode_pair_id(pair_id: int) -> Tuple[int, int]:
    """
    Decode COLMAP's order-independent pair_id into image IDs.
    """
    try:
        import pycolmap
        return tuple(map(int, pycolmap.pair_id_to_image_pair(int(pair_id))))
    except Exception:
        pass

    max_image_id = 2147483647
    image_id2 = int(pair_id) % max_image_id
    image_id1 = (int(pair_id) - image_id2) // max_image_id
    return image_id1, image_id2


def _image_pair_id(image_id1: int, image_id2: int) -> int:
    """
    Compute COLMAP's pair_id using pycolmap when available.
    """
    try:
        import pycolmap
        return int(pycolmap.image_pair_to_pair_id(int(image_id1), int(image_id2)))
    except Exception:
        pass

    max_image_id = 2147483647
    a, b = sorted((int(image_id1), int(image_id2)))
    return a * max_image_id + b


def _has_verified_pair(database_path: str, image_id1: int, image_id2: int) -> bool:
    """
    Return True when two images have a verified two-view geometry with inliers.
    """
    try:
        conn = sqlite3.connect(str(database_path))
        row = conn.execute(
            "SELECT rows FROM two_view_geometries WHERE pair_id=?",
            (_image_pair_id(image_id1, image_id2),),
        ).fetchone()
        conn.close()
    except sqlite3.Error:
        return False

    return row is not None and int(row[0] or 0) > 0


def _verified_pair_stats(database_path: str, image_names: List[str]) -> Optional[dict]:
    """
    Summarize verified two-view geometry connectivity for a mapper image list.

    COLMAP's mapper consumes the two_view_geometries table, not the raw matches
    table. If keyframe filtering leaves no verified keyframe-to-keyframe edges,
    incremental mapping will report "No images with matches".
    """
    db_path = Path(database_path)
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT image_id, name FROM images").fetchall()
        wanted_names = set(image_names)
        wanted_ids = {
            int(image_id)
            for image_id, name in rows
            if name in wanted_names
        }

        if not wanted_ids:
            conn.close()
            return {
                "images_in_db": 0,
                "verified_pairs": 0,
                "images_with_verified_matches": 0,
            }

        verified_pairs = 0
        images_with_matches: set[int] = set()
        for pair_id, n_rows in conn.execute("SELECT pair_id, rows FROM two_view_geometries"):
            if int(n_rows or 0) <= 0:
                continue
            image_id1, image_id2 = _decode_pair_id(pair_id)
            if image_id1 in wanted_ids and image_id2 in wanted_ids:
                verified_pairs += 1
                images_with_matches.update((image_id1, image_id2))

        conn.close()
        return {
            "images_in_db": len(wanted_ids),
            "verified_pairs": verified_pairs,
            "images_with_verified_matches": len(images_with_matches),
        }
    except sqlite3.Error:
        return None


def run_colmap_incremental(
    database_path  : str,
    image_dir      : str,
    output_dir     : str,
    keyframes      : List[Capture],
    init_pair      : Optional[Tuple[Capture, Capture]] = None,
    use_prior_position: bool = True,
    use_default_colmap: bool = False,
) -> Optional[object]:
    """
    Step 4 — COLMAP incremental SfM on keyframes only.

    Args:
        database_path : COLMAP .db with features, matches, pose priors
        image_dir     : directory containing actual image files
                        (COLMAP requires real files on disk matching db names)
        output_dir    : where to write the sparse/ model
        keyframes     : keyframe Capture list for SfM
        init_pair     : (cap_a, cap_b) from find_gps_guided_init_pair()
                        or None to let COLMAP choose

    Returns:
        Best Reconstruction, or None if mapper produced nothing.
    """
    import pycolmap

    output_path = Path(output_dir) / "colmap"
    output_path.mkdir(parents=True, exist_ok=True)
    mapper_image_path = _prepare_mapper_image_dir(keyframes, output_path)

    # ── Pre-flight connectivity check ─────────────────────────────────────────
    image_names = [colmap_image_name(cap) for cap in keyframes]
    stats = _verified_pair_stats(database_path, image_names)

    # pct_matched must be initialised here — it's read by _force_init logic
    # below regardless of whether the stats block runs.
    pct_matched = 0.0
    avg_pairs   = 0.0
    if stats is not None:
        n_kf       = len(image_names)
        n_matched  = stats["images_with_verified_matches"]
        n_pairs    = stats["verified_pairs"]
        n_isolated = n_kf - n_matched
        avg_pairs  = n_pairs / max(n_matched, 1)
        pct_matched = 100.0 * n_matched / max(n_kf, 1)

        print("[colmap] ─── Connectivity check ────────────────────────────────")
        print(f"[colmap]  Keyframes for SfM     : {n_kf}")
        print(f"[colmap]  With verified matches  : {n_matched}/{n_kf} ({pct_matched:.1f}%)")
        print(f"[colmap]  Isolated keyframes     : {n_isolated}")
        print(f"[colmap]  Verified pairs total   : {n_pairs}")
        print(f"[colmap]  Avg pairs / keyframe   : {avg_pairs:.1f}")

        if n_pairs == 0:
            print("[colmap] FATAL: No verified pairs among keyframes.")
            print("[colmap] → Run from Stage 4 with --n-neighbors 20 or higher.")
            return None

        # Diagnose the real root cause when avg_pairs is low despite
        # high n_neighbors. This is the most common failure mode:
        # feature matching found candidates but PoseLib RANSAC rejected them
        # (too few inliers, or max_epipolar_error=1.5px was too tight).
        if avg_pairs < 3.0 and pct_matched > 80.0:
            print("[colmap] WARNING: High connectivity but LOW avg pairs/keyframe.")
            print(f"[colmap]   {avg_pairs:.1f} pairs/image means most match candidates")
            print("[colmap]   were rejected by PoseLib geometric verification.")
            print("[colmap] LIKELY CAUSE: max_epipolar_error=1.5px too tight for")
            print("[colmap]   this camera/altitude combination. The database must")
            print("[colmap]   be rebuilt with a looser threshold.")
            print("[colmap] FIX: In src/features/geometric_verification.py:")
            print("[colmap]   DEFAULT_RANSAC_OPTIONS['max_epipolar_error'] = 3.0")
            print("[colmap]   Then re-run from Stage 5 (geometric verification).")
            print("[colmap]   Do NOT set keyframe_interval=1 — that is not the cause.")

        if pct_matched < 80.0:
            print(f"[colmap] WARNING: Only {pct_matched:.0f}% of keyframes have verified matches.")
            print(f"[colmap]   {n_isolated} isolated keyframes will never be registered.")
            print(f"[colmap]   Expect {max(1, round(n_isolated/60))}+ disconnected sub-reconstructions.")
            print("[colmap] RECOMMENDATION: Stop now and re-run from Stage 4:")
            print("[colmap]   --n-neighbors 20   (current is likely 8)")
            if avg_pairs < 3.0:
                print("[colmap]   avg pairs/keyframe is very low — also check")
                print("[colmap]   max_epipolar_error in geometric_verification.py")
        elif pct_matched < 95.0:
            print(f"[colmap] OK — {pct_matched:.0f}% connected. A few isolated images is normal.")
            print("[colmap]   Expect 1-2 sub-reconstructions at most.")
        else:
            print(f"[colmap] GOOD — {pct_matched:.0f}% of keyframes are connected.")
            print("[colmap]   Single-component reconstruction expected. ✓")
        print("[colmap] ────────────────────────────────────────────────────────")

    # ── Options ───────────────────────────────────────────────────────────────
    opt = pycolmap.IncrementalPipelineOptions()

    # Optimization 1 — keyframe image list (image_names filters what mapper sees)
    # We always set this, even in default colmap mode, because if default_colmap is used, 
    # keyframes contains ALL images.
    opt.image_names = image_names

    if not use_default_colmap:
        # Optimization 3 — BA frequency tuning (safe for flat terrain + clean matches)
        _set_opt(opt.mapper, "ba_local_num_images", 12)   # default 6
        _set_opt(opt, "ba_global_frames_ratio", 1.3)       # default 1.1

        # Optimization 4 — tight reprojection filter (safe with ALIKED+LightGlue)
        _set_opt(opt.mapper, "filter_max_reproj_error", 2.0)  # default 4.0

    # Optimization 5 — all CPU cores
    opt.num_threads = -1

    # Keep this low. If COLMAP starts repeatedly discarding components, letting
    # it try dozens of seeds can burn tens of minutes without improving the
    # final model. Three attempts are enough to distinguish a bad first seed
    # from a genuinely fragmented verified graph.
    _set_opt(opt, "max_num_models", 3)
    _set_opt(opt, "min_model_size", 10)

    # min_num_matches: keep at COLMAP default (15 verified inliers minimum).
    # Do NOT lower this — borderline pairs hurt bundle adjustment accuracy.
    # The fix for disconnected graphs is more neighbors (--n-neighbors 20),
    # not weaker pair quality thresholds.

    # GPS/RTK priors — activate prior position constraints.
    # use_prior_position tells BA to use the pose priors injected by pose_priors.py.
    # Use _set_opt: the attribute is absent in some pycolmap debug builds.
    if not _set_opt(opt, "use_prior_position", use_prior_position):
        print("[colmap] Warning: use_prior_position not available in this pycolmap build; "
              "GPS pose priors will not be enforced during incremental mapping.")

    # GPS-guided init pair is diagnostic-only. A GPS baseline can be physically
    # reasonable while still being a poor two-view initializer over planar crops.
    # Leave init_image_id1/2 unset so COLMAP can rank all verified pairs.
    if init_pair is not None:
        db = pycolmap.Database.open(str(database_path))
        id1 = _resolve_image_id(db, init_pair[0])
        id2 = _resolve_image_id(db, init_pair[1])
        db.close()

        if id1 is not None and id2 is not None and _has_verified_pair(database_path, id1, id2):
            print(f"[colmap] GPS-guided init pair verified but not forced: "
                  f"{init_pair[0].capture_id} (id={id1}) <-> "
                  f"{init_pair[1].capture_id} (id={id2}). "
                  "COLMAP will select the best init pair.")
        elif id1 is not None and id2 is not None:
            print("[colmap] Warning: GPS-guided init pair has no verified geometry. "
                  "COLMAP will select its own init pair.")
        else:
            print("[colmap] Warning: could not resolve init pair image ids. "
                  "COLMAP will select its own init pair.")

    # ── Run mapper ────────────────────────────────────────────────────────────
    print(f"[colmap] Running incremental SfM on {len(keyframes)} keyframes...")
    ba_local  = getattr(getattr(opt, "mapper", opt), "ba_local_num_images", "(default)")
    ba_global = getattr(opt, "ba_global_frames_ratio", "(default)")
    reproj    = getattr(getattr(opt, "mapper", opt), "filter_max_reproj_error", "(default)")
    use_prior = getattr(opt, "use_prior_position", "(default)")
    print(f"[colmap] Options: ba_local_num_images={ba_local}, "
          f"ba_global_frames_ratio={ba_global}, "
          f"filter_max_reproj_error={reproj}, "
          f"use_prior_position={use_prior}")

    try:
        reconstructions = pycolmap.incremental_mapping(
            database_path = str(database_path),
            image_path    = str(mapper_image_path),
            output_path   = str(output_path),
            options       = opt,
        )
    except Exception as e:
        print(f"[colmap] incremental_mapping failed: {e}")
        return None

    recon = _best_reconstruction(reconstructions)

    if recon is None:
        print("[colmap] Mapper produced no reconstruction.")
        return None

    # Show all sub-reconstruction sizes so the user knows if the graph was disconnected
    if len(reconstructions) > 1:
        print(f"[colmap] WARNING: mapper produced {len(reconstructions)} sub-reconstructions "
              f"(disconnected graph). Sizes:")
        for idx, r in sorted(reconstructions.items(), key=lambda kv: -_get_num_reg_images(kv[1])):
            print(f"[colmap]   component {idx}: {_get_num_reg_images(r)} images registered")
        print(f"[colmap] Using the largest: component with {_get_num_reg_images(recon)} images.")
        print("[colmap] Diagnosis: verified geometry is fragmented into tiny islands.")
        print("[colmap]   This is usually a geometric-verification problem, not just a")
        print("[colmap]   neighbor-count problem. The mapper will reject tiny islands and")
        print("[colmap]   let the pipeline retry with a denser image set when possible.")

    n_reg   = _get_num_reg_images(recon)
    n_total = len(keyframes)
    min_usable = max(10, int(0.25 * n_total))
    print(f"[colmap] Registered {n_reg}/{n_total} keyframes "
          f"({n_reg/n_total*100:.1f}%)")

    if n_reg < min_usable:
        print(f"[colmap] Largest component is too small to use ({n_reg}/{n_total}; "
              f"minimum usable is {min_usable}).")
        print("[colmap] Treating this COLMAP attempt as failed so the pipeline can retry "
              "with a denser image set.")
        return None

    if n_reg < n_total * 0.70:
        print("[colmap] Warning: fewer than 70% of keyframes registered. "
              "Check feature matching quality or increase --n-neighbors.")

    return recon
