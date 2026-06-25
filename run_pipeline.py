"""
run_pipeline.py  â€”  Full Agri Orthomosaic Engine pipeline
Usage: python run_pipeline.py --mission /path/to/your_mission --output /path/to/outputs
"""

import argparse
import math
import os
import ssl
import time
from pathlib import Path

# Stage 3 (ALIKED) and Stage 4 (LightGlue) pull pretrained weights over HTTPS
# the first time they run. Some corporate/CI networks sit behind a TLS-
# intercepting proxy with a certificate chain Python doesn't trust, which
# makes that download fail. Rather than disabling certificate verification
# for the *entire* process (which would also weaken every other HTTPS call
# the pipeline makes), only do it when the operator opts in explicitly.
if os.environ.get("AGRI_ENGINE_INSECURE_SSL") == "1":
    print("[pipeline] WARNING: AGRI_ENGINE_INSECURE_SSL=1 — TLS certificate "
          "verification is DISABLED for this run. Only use this on a trusted "
          "network to work around a broken local certificate chain.")
    ssl._create_default_https_context = ssl._create_unverified_context

from src.ingestion import load_mission
from src.quality import filter_quality
from src.features import extract_features, match_features, import_to_colmap
from src.sfm import run_sfm
from src.depth import run_depth_pipeline
from src.dsm import run_dsm_pipeline
from src.ortho import run_ortho_pipeline
from src.mosaic import run_rgb_mosaic, run_ms_mosaic

def main():
    parser = argparse.ArgumentParser(description="Agri Orthomosaic Engine")
    parser.add_argument("--mission", required=True,
                        help="Path to mission folder (contains rgb/ and multi/)")
    parser.add_argument("--output",  required=True,
                        help="Output directory for all results")
    parser.add_argument("--gsd",     type=float, default=0.05,
                        help="Target ground sampling distance in metres/pixel (default 0.05 = 5cm)")
    parser.add_argument("--rtk",     action="store_true",
                        help="Force RTK mode (strong priors + GLOMAP). "
                             "By default the pipeline auto-detects RTK from "
                             "image EXIF/XMP (drone-dji:RtkFlag or GPS GPSDifferential).")
    parser.add_argument("--no-rtk",  action="store_true",
                        help="Force non-RTK mode even if EXIF/XMP indicates RTK.")
    parser.add_argument("--no-gpu",  action="store_true",
                        help="Disable GPU (run on CPU only â€” much slower)")
    parser.add_argument("--n-neighbors", type=int, default=20,
                        help="GPS neighbors per image for feature matching (default 20)")
    parser.add_argument("--default-colmap", action="store_true",
                        help="Disable SfM optimisations (keyframe skipping, BA tuning, tight filters) and run with pure COLMAP defaults.")
    parser.add_argument(
        "--start-from-stage", type=int, default=1,
        choices=range(1, 13), metavar="N",
        help=(
            "Resume pipeline from stage N (1-12). "
            "Stages 1-2 (ingestion) always run â€” they are fast and required. "
            "Stages that were skipped must have already written their outputs to --output. "
            "Example: --start-from-stage 5  resumes from geometric verification."
        ),
    )
    args = parser.parse_args()

    start = args.start_from_stage
    mission_dir = args.mission
    output_dir  = Path(args.output)
    use_gpu     = not args.no_gpu

    if start > 1:
        print(f"\n[pipeline] Resuming from stage {start}. "
              f"Stages 1-{start - 1} will be skipped (outputs must exist in {output_dir}).")

    total_start = time.time()
    
    # Helper to print elapsed time
    def print_stage_time(stage_name, start_time):
        elapsed = time.time() - start_time
        print(f"  [Time] {stage_name} took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
        return time.time()

    # â”€â”€ Stages 1+2: Ingest + quality filter (always runs â€” fast, needed everywhere) â”€â”€
    print("\n=== Stage 1+2: Ingestion & Quality Filter ===")
    stage_start = time.time()
    captures = load_mission(mission_dir)

    # â”€â”€ RTK detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if args.rtk and args.no_rtk:
        print("[pipeline] ERROR: --rtk and --no-rtk are mutually exclusive.")
        return

    if args.rtk:
        has_rtk = True
        print("[pipeline] RTK mode: FORCED ON via --rtk flag.")
    elif args.no_rtk:
        has_rtk = False
        print("[pipeline] RTK mode: FORCED OFF via --no-rtk flag.")
    else:
        # Auto-detect from per-capture EXIF/XMP written during ingestion.
        # Require a supermajority (â‰¥ 80 %) of captures to show RTK Fixed before
        # treating the whole mission as RTK â€” a few non-RTK frames at the start
        # of a flight (while the rover is acquiring a fix) should not downgrade
        # an otherwise RTK mission.
        n_total  = len(captures)
        n_rtk    = sum(1 for c in captures if c.has_rtk)
        rtk_frac = n_rtk / n_total if n_total else 0.0
        has_rtk  = rtk_frac >= 0.80

        print("\n[pipeline] â”€â”€ RTK auto-detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        print(f"[pipeline]  Captures with RTK Fixed EXIF/XMP : {n_rtk}/{n_total} "
              f"({rtk_frac*100:.0f}%)")
        if has_rtk:
            print("[pipeline]  Decision : RTK ON  "
                  "(â‰¥ 80 % threshold met â€” using GLOMAP + strong priors)")
        else:
            print("[pipeline]  Decision : RTK OFF "
                  "(< 80 % threshold â€” using COLMAP + weak GPS priors)")
        print("[pipeline]  Override : pass --rtk or --no-rtk to force.")
        print("[pipeline] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")

    # â”€â”€ Overlap estimation + dynamic SfM parameter selection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    has_any_gps = any(c.latitude is not None and c.longitude is not None for c in captures)

    if not has_any_gps:
        # No GPS in EXIF â€” generate a synthetic grid so the rest of the pipeline
        # can run in sequential / relative mode.
        print("\n[pipeline] WARNING: No GPS in EXIF. Generating synthetic coordinates for relative reconstruction.")
        for i, cap in enumerate(captures):
            cap.latitude  = 45.0 + i * 0.000045   # ~5 m spacing
            cap.longitude = 9.0
            cap.altitude  = 120.0
        sfm_keyframe_interval = 1
        args.n_neighbors = len(captures)   # exhaustive matching (no GPS filter)
        print(f"[pipeline] No-GPS mode: using all {args.n_neighbors} captures as neighbors.")
    else:
        from src.features.neighbors import estimate_overlap
        ov = estimate_overlap(captures)

        if ov.footprint_m > 0:
            # Use the recommended values from the overlap estimator, but
            # never go below what the user explicitly requested on the CLI.
            sfm_keyframe_interval = ov.keyframe_interval
            args.n_neighbors      = max(args.n_neighbors, ov.n_neighbors)

            print("\n[pipeline] â”€â”€ Flight overlap estimate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
            print(f"[pipeline]  Altitude (median)   : {ov.footprint_m / (2.0 * math.tan(math.radians(40))):.0f} m")
            print(f"[pipeline]  Image footprint     : ~{ov.footprint_m:.0f} m")
            print(f"[pipeline]  Forward spacing     : {ov.forward_spacing_m:.1f} m  ->  {ov.forward_overlap*100:.0f}% forward overlap")
            print(f"[pipeline]  Side spacing        : {ov.side_spacing_m:.1f} m  ->  {ov.side_overlap*100:.0f}% side overlap")
            print(f"[pipeline]  Keyframe interval   : {sfm_keyframe_interval}  (every {sfm_keyframe_interval}{'rd' if sfm_keyframe_interval==3 else 'nd' if sfm_keyframe_interval==2 else 'st'} image used for SfM)")
            print(f"[pipeline]  n_neighbors         : {args.n_neighbors}")
            print("[pipeline] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        else:
            # Not enough GPS captures to estimate (< 20) â€” use safe defaults
            sfm_keyframe_interval = 1
            print("[pipeline] Too few GPS captures to estimate overlap. Using interval=1, n_neighbors=20.")

    if args.default_colmap:
        print("\n[pipeline] â”€â”€ Default COLMAP mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        print("[pipeline]  --default-colmap passed. Disabling keyframe skipping.")
        sfm_keyframe_interval = 1

    captures = filter_quality(captures)
    print(f"  {len(captures)} valid captures ready")
    stage_start = print_stage_time("Stage 1+2", stage_start)

    # â”€â”€ Stage 3: Feature extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if start <= 3:
        print("\n=== Stage 3: Feature Extraction (ALIKED) ===")
        extract_features(captures, output_dir=str(output_dir), use_gpu=use_gpu)
        stage_start = print_stage_time("Stage 3", stage_start)
    else:
        print("\n=== Stage 3: Feature Extraction â€” SKIPPED ===")

    # â”€â”€ Stage 4: Feature matching â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if start <= 4:
        print(f"\n=== Stage 4: Feature Matching (LightGlue, n_neighbors={args.n_neighbors}) ===")
        match_features(captures, output_dir=str(output_dir), n_neighbors=args.n_neighbors, use_gpu=use_gpu)
        stage_start = print_stage_time("Stage 4", stage_start)
    else:
        print("\n=== Stage 4: Feature Matching â€” SKIPPED ===")

    # â”€â”€ Stage 5: DB import + Geometric verification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if start <= 5:
        print("\n=== Stage 5: DB Import + Geometric Verification (PoseLib) ===")
        db_path = import_to_colmap(captures, output_dir=str(output_dir))
        stage_start = print_stage_time("Stage 5", stage_start)
    else:
        print("\n=== Stage 5: DB Import + Geometric Verification â€” SKIPPED ===")
        db_path = output_dir / "database.db"
        if not db_path.exists():
            print(f"[pipeline] ERROR: database.db not found at {db_path}. Run from stage 5 or earlier.")
            return
        print(f"[pipeline] Using existing database: {db_path}")

    # â”€â”€ Stage 6+7: SfM + Georeferencing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if start <= 7:
        print(f"\n=== Stage 6+7: SfM Mapping + Georeferencing (keyframe_interval={sfm_keyframe_interval}) ===")
        reconstruction = run_sfm(
            database_path      = str(db_path),
            image_dir          = str(Path(mission_dir) / "rgb"),
            output_dir         = str(output_dir / "sparse"),
            captures           = captures,
            has_rtk            = has_rtk,
            keyframe_interval  = sfm_keyframe_interval,
            # COLMAP's use_prior_position is brittle with ordinary GPS and can
            # repeatedly discard otherwise valid components when prior alignment
            # fails. Use it only for RTK; normal GPS is applied in final alignment.
            use_prior_position = has_rtk,
            use_default_colmap = args.default_colmap,
        )
        if reconstruction is None:
            print("[pipeline] ERROR: SfM failed. Check GPS metadata and image overlap.")
            return
        stage_start = print_stage_time("Stage 6+7", stage_start)
    else:
        print(f"\n=== Stage 6+7: SfM â€” SKIPPED (start-from-stage={start}) ===")
        reconstruction = _load_reconstruction(output_dir / "sparse")
        if reconstruction is None:
            print(f"ERROR: Could not load reconstruction from {output_dir / 'sparse'}. Run from stage 6 or earlier first.")
            return
        print(f"  Loaded reconstruction: {reconstruction.num_reg_images} registered images")

    # â”€â”€ Stage 8: Depth maps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    depth_dir = output_dir / "depth"
    if start <= 8:
        print("\n=== Stage 8: Depth Maps (COLMAP Dense Stereo) ===")
        depth_result = run_depth_pipeline(
            reconstruction    = reconstruction,
            captures          = captures,
            output_dir        = str(depth_dir),
            use_gpu           = use_gpu,
        )
        dmap_paths = depth_result.dmap_paths
        fused_pointcloud = depth_result.fused_pointcloud_path
        stage_start = print_stage_time("Stage 8", stage_start)
    else:
        print(f"\n=== Stage 8: Depth Maps - SKIPPED (start-from-stage={start}) ===")
        dmap_paths = sorted(str(p) for p in (depth_dir / "dmaps").glob("*.dmap") if p.stat().st_size > 0)
        if not dmap_paths:
            dmap_paths = sorted(str(p) for p in depth_dir.glob("*.dmap") if p.stat().st_size > 0)
        fused_pointcloud = str(depth_dir / "dense" / "fused.ply")
        if not dmap_paths:
            print(f"ERROR: No .dmap files found in {depth_dir}. Run from stage 8 or earlier first.")
            return
        if not Path(fused_pointcloud).exists():
            print(f"ERROR: COLMAP fused point cloud not found at {fused_pointcloud}. Run from stage 8 or earlier first.")
            return
        print(f"  Found {len(dmap_paths)} existing .dmap files + COLMAP fused point cloud")

    # â”€â”€ Stage 9: DSM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    dsm_dir = output_dir / "dsm"
    if start <= 9:
        print("\n=== Stage 9: DSM Generation (COLMAP fused cloud) ===")
        dsm_path = run_dsm_pipeline(
            dmap_paths            = dmap_paths,
            fused_pointcloud_path = fused_pointcloud,
            reconstruction        = reconstruction,
            output_dir            = str(dsm_dir),
            target_gsd_m          = args.gsd,
        )
        print(f"  DSM written to: {dsm_path}")
        stage_start = print_stage_time("Stage 9", stage_start)
    else:
        print(f"\n=== Stage 9: DSM Generation â€” SKIPPED (start-from-stage={start}) ===")
        dsm_path = str(dsm_dir / "dsm.tif")
        if not Path(dsm_path).exists():
            print(f"ERROR: dsm.tif not found at {dsm_path}. Run from stage 9 or earlier first.")
            return
        print(f"  Using existing DSM: {dsm_path}")

    # â”€â”€ Stage 10: Orthorectification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ortho_dir = output_dir / "ortho"
    if start <= 10:
        print("\n=== Stage 10: Orthorectification (CuPy) ===")
        ortho_result = run_ortho_pipeline(
            reconstruction        = reconstruction,
            captures              = captures,
            dsm_path              = dsm_path,
            output_dir            = str(ortho_dir),
            target_gsd_m          = args.gsd,
            process_multispectral = True,
        )
        print(f"  {len(ortho_result.rgb_tile_paths)} RGB tiles written")
        stage_start = print_stage_time("Stage 10", stage_start)
    else:
        print(f"\n=== Stage 10: Orthorectification â€” SKIPPED (start-from-stage={start}) ===")
        ortho_result = _load_ortho_result(ortho_dir)
        if not ortho_result.rgb_tile_paths:
            print(f"ERROR: No ortho RGB tiles found under {ortho_dir / 'rgb'}. Run from stage 10 or earlier first.")
            return
        print(f"  Found {len(ortho_result.rgb_tile_paths)} existing RGB tiles")

    # â”€â”€ Stage 11: RGB Mosaicking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    seamlines_dir = str(output_dir / "seamlines")
    if start <= 11:
        print("\n=== Stage 11: RGB Mosaicking (OpenCV) ===")
        rgb_mosaic_path, seamlines = run_rgb_mosaic(
            tile_paths        = ortho_result.rgb_tile_paths,
            output_path       = str(output_dir / "rgb_orthomosaic.tif"),
            seamlines_save_dir= seamlines_dir,
            target_gsd_m      = args.gsd,
        )
        print(f"  RGB mosaic: {rgb_mosaic_path}")
        stage_start = print_stage_time("Stage 11", stage_start)
    else:
        print(f"\n=== Stage 11: RGB Mosaicking â€” SKIPPED (start-from-stage={start}) ===")
        seamlines = _load_seamlines(seamlines_dir)
        if seamlines is None:
            print(f"ERROR: Seamlines not found in {seamlines_dir}. Run from stage 11 or earlier first.")
            return
        rgb_mosaic_path = str(output_dir / "rgb_orthomosaic.tif")
        print(f"  Loaded existing seamlines, RGB mosaic: {rgb_mosaic_path}")

    # â”€â”€ Stage 12: Multispectral Mosaicking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n=== Stage 12: Multispectral Mosaicking (NumPy) ===")
    ms_mosaic_path = run_ms_mosaic(
        multi_tile_paths = ortho_result.multi_tile_paths,
        captures         = captures,
        seamline_set     = seamlines,
        output_path      = str(output_dir / "multispectral_orthomosaic.tif"),
        target_gsd_m     = args.gsd,
    )
    print(f"  MS mosaic: {ms_mosaic_path}")
    stage_start = print_stage_time("Stage 12", stage_start)

    # â”€â”€ Done â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    total_elapsed = time.time() - total_start
    print("\n=== Pipeline Complete ===")
    print(f"  [Time] TOTAL PIPELINE TIME: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes).")
    print(f"  RGB orthomosaic:            {output_dir}/rgb_orthomosaic.tif")
    print(f"  Multispectral orthomosaic:  {output_dir}/multispectral_orthomosaic.tif")
    print(f"  DSM:                        {output_dir}/dsm/dsm.tif")


# â”€â”€ Resume helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_reconstruction(sparse_path: Path):
    """Load a pycolmap Reconstruction from a binary sparse model on disk."""
    try:
        import pycolmap
        recon = pycolmap.Reconstruction()
        recon.read(str(sparse_path))
        return recon
    except Exception as e:
        print(f"[pipeline] Could not load reconstruction from {sparse_path}: {e}")
        return None


def _load_ortho_result(ortho_dir: Path):
    """Reconstruct an OrthoResult by globbing the tile directories."""
    from src.ortho import OrthoResult
    result = OrthoResult()
    rgb_dir = ortho_dir / "rgb"
    if rgb_dir.exists():
        result.rgb_tile_paths = sorted(str(p) for p in rgb_dir.glob("*.tif"))
    for band in ("GRE", "RED", "REG", "NIR"):
        band_dir = ortho_dir / "multi" / band
        if band_dir.exists():
            result.multi_tile_paths[band] = sorted(str(p) for p in band_dir.glob("*.tif"))
        else:
            result.multi_tile_paths[band] = []
    return result


def _load_seamlines(seamlines_dir: str):
    """Load a saved SeamlineSet from the seamlines directory."""
    try:
        from src.mosaic.seam_finder import SeamlineSet
        import numpy as np
        seamlines_path = Path(seamlines_dir) / "seamlines.npz"
        if not seamlines_path.exists():
            # Try any .npz in the directory
            candidates = list(Path(seamlines_dir).glob("*.npz"))
            if not candidates:
                return None
            seamlines_path = candidates[0]
        data = np.load(seamlines_path, allow_pickle=True)
        return SeamlineSet(masks=list(data["masks"]))
    except Exception as e:
        print(f"[pipeline] Could not load seamlines from {seamlines_dir}: {e}")
        return None


if __name__ == "__main__":
    main()



