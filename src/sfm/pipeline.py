"""
Stage 6 — SfM pipeline orchestrator.

Runs Steps 1-7 in order. Single entry point for the SfM stage.

Flow:

  Step 1  select_keyframes()           GPS-ordered, every 3rd image
  Step 2  find_gps_guided_init_pair()  field-centroid pair, good baseline
  Step 3a inject_gps_priors()          RTK weight=1e6 / GPS weight=1e2
  Step 3b run_glomap()                 fast path, RTK only, 3-check validation
          → if fails → fallback
  Step 4  run_colmap_incremental()     safe path, 5 optimizations
  Step 5  register_non_keyframes()     PnP for skipped images
  Step 6  run_final_bundle_adjustment() refine all poses together
  Step 7  align_to_gps()               WGS84 alignment, seconds

Decision logic:

  has_rtk=True
    → try GLOMAP (fast, 5-10 min)
    → validate (3 checks)
    → if pass: Step 5 → 6 → 7 → done
    → if fail: Step 4 (COLMAP) → 5 → 6 → 7 → done

  has_rtk=False
    → skip GLOMAP entirely
    → Step 4 (COLMAP) → 5 → 6 → 7 → done
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..ingestion.capture import Capture
from .keyframes       import select_keyframes, write_keyframe_list, find_gps_guided_init_pair
from .pose_priors     import inject_gps_priors
from .glomap          import run_glomap
from .colmap_mapper   import run_colmap_incremental
from .pnp_registration import register_non_keyframes
from .final_ba        import run_final_bundle_adjustment, align_to_gps


def _write_final_reconstruction(reconstruction, output_path: Path) -> bool:
    """
    Persist the final in-memory model after PnP, final BA, and GPS alignment.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        if hasattr(reconstruction, "write"):
            reconstruction.write(str(output_path))
            print(f"[sfm] Wrote final sparse model to {output_path}")
            return True
        if hasattr(reconstruction, "write_binary"):
            reconstruction.write_binary(str(output_path))
            print(f"[sfm] Wrote final sparse model to {output_path}")
            return True
    except Exception as e:
        print(f"[sfm] Warning: failed to write final sparse model: {e}")
        return False

    print("[sfm] Warning: reconstruction object has no write/write_binary method. "
          "Final model was not persisted to disk.")
    return False


def run_sfm(
    database_path  : str,
    image_dir      : str,
    output_dir     : str,
    captures       : List[Capture],
    has_rtk        : bool = False,
    keyframe_interval: int = 3,
    use_prior_position: bool = False,
    use_default_colmap: bool = False,
) -> Optional[object]:
    """
    Full SfM pipeline — Steps 1-7.

    Args:
        database_path    : COLMAP .db with features + verified matches
        image_dir        : directory with actual image files (COLMAP needs this)
        output_dir       : root output directory for sparse model
        captures         : full Capture list from ingestion + quality filter
        has_rtk          : True  → try GLOMAP first, strong GPS prior
                           False → COLMAP directly, weak GPS prior
        keyframe_interval: 1-in-N images used as keyframes for SfM
                           3 = safe at 80% overlap (default)
                           2 = more conservative, slower SfM
        use_prior_position:
                          Enable COLMAP's pose-prior constraints during mapping.
                          Use only for RTK-quality GPS. Ordinary GPS is aligned
                          after mapping in Step 7.

    Returns:
        pycolmap.Reconstruction with all cameras registered,
        or None if both GLOMAP and COLMAP fail.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Keyframe selection ────────────────────────────────────────────
    print("\n[sfm] ── Step 1: Keyframe selection ──")
    keyframes, non_keyframes = select_keyframes(captures, interval=keyframe_interval)

    keyframe_list_path = output_path / "keyframes.txt"
    write_keyframe_list(keyframes, str(keyframe_list_path))

    # ── Step 2: GPS-guided init pair ──────────────────────────────────────────
    print("\n[sfm] ── Step 2: GPS-guided init pair ──")
    init_pair = find_gps_guided_init_pair(keyframes)
    # None is fine — COLMAP falls back to its own search

    # ── Step 3a: GPS/RTK prior injection ─────────────────────────────────────
    print("\n[sfm] ── Step 3a: GPS prior injection ──")
    inject_gps_priors(database_path, captures, has_rtk=has_rtk)

    # ── Step 3b: GLOMAP (fast path, RTK only) ────────────────────────────────
    reconstruction = None
    path_used = None

    if has_rtk and not use_default_colmap:
        print("\n[sfm] ── Step 3b: GLOMAP (fast path) ──")
        reconstruction = run_glomap(
            database_path = database_path,
            image_dir     = str(image_dir),
            output_dir    = str(output_path),
            keyframes     = keyframes,
            has_rtk       = has_rtk,
        )
        if reconstruction is None:
            print("[sfm] GLOMAP failed or did not pass validation. "
                  "Falling back to COLMAP incremental.")
        else:
            path_used = "GLOMAP"
    else:
        print("\n[sfm] ── Step 3b: GLOMAP skipped (no RTK) ──")

    # ── Step 4: COLMAP incremental (safe path / fallback) ────────────────────
    if reconstruction is None:
        print("\n[sfm] ── Step 4: COLMAP incremental ──")
        reconstruction = run_colmap_incremental(
            database_path = database_path,
            image_dir     = image_dir,
            output_dir    = str(output_path),
            keyframes     = keyframes,
            init_pair     = init_pair,
            use_prior_position = use_prior_position,
            use_default_colmap = use_default_colmap,
        )
        if reconstruction is not None:
            path_used = "COLMAP"
        elif len(keyframes) < len(captures):
            print("[sfm] Keyframe-only COLMAP produced no reconstruction. "
                  "Retrying COLMAP with all captures so the mapper can use the full match graph.")
            reconstruction = run_colmap_incremental(
                database_path = database_path,
                image_dir     = image_dir,
                output_dir    = str(output_path),
                keyframes     = captures,
                init_pair     = init_pair,
                use_prior_position = use_prior_position,
                use_default_colmap = use_default_colmap,
            )
            if reconstruction is not None:
                path_used = "COLMAP-full"

    if reconstruction is None:
        print("[sfm] CRITICAL: Both GLOMAP and COLMAP produced no reconstruction. "
              "Check feature matching output.")
        return None

    # ── Step 5: PnP registration of non-keyframes ────────────────────────────
    print("\n[sfm] ── Step 5: PnP non-keyframe registration ──")
    register_non_keyframes(
        reconstruction = reconstruction,
        database_path  = database_path,
        non_keyframes  = non_keyframes,
    )

    # ── Step 6: Final bundle adjustment ──────────────────────────────────────
    print("\n[sfm] ── Step 6: Final bundle adjustment ──")
    run_final_bundle_adjustment(reconstruction)

    # ── Step 7: GPS alignment ─────────────────────────────────────────────────
    print("\n[sfm] ── Step 7: GPS alignment ──")
    align_to_gps(reconstruction, captures)

    # Final persisted model lives at output_dir itself:
    # sparse/cameras.bin, sparse/images.bin, sparse/points3D.bin.
    _write_final_reconstruction(reconstruction, output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_reg_val = reconstruction.num_reg_images
    n_reg = int(n_reg_val() if callable(n_reg_val) else n_reg_val)
    n_total  = len(captures)
    n_points_val = getattr(reconstruction, "num_points3D", None) or getattr(reconstruction, "points3D", None)
    if callable(n_points_val):
        n_points = len(n_points_val()) if n_points_val.__name__ == "points3D" else int(n_points_val())
    elif n_points_val is not None:
        n_points = int(n_points_val) if not hasattr(n_points_val, '__len__') else len(n_points_val)
    else:
        n_points = 0

    print("\n[sfm] ── Complete ──")
    print(f"[sfm] Registered : {n_reg}/{n_total} images "
          f"({n_reg/n_total*100:.1f}%)")
    print(f"[sfm] 3D points  : {n_points}")
    print(f"[sfm] Path used  : {path_used or 'unknown'}")

    return reconstruction
