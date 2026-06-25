"""
Benchmark — Stage 3 + 4 + 5 (ALIKED + LightGlue + COLMAP import)

Usage:
    python benchmarks/feature_stage.py --rgb-dir /path/to/mission/rgb
    python benchmarks/feature_stage.py --rgb-dir /path/to/rgb --output-dir output/bench --cpu
    python benchmarks/feature_stage.py --rgb-dir /path/to/rgb --skip-matching
    python benchmarks/feature_stage.py --rgb-dir /path/to/rgb --check-gps-only
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import gps_summary, load_rgb_captures
from src.features.extractor import extract_features
from src.features.matcher import match_features
from src.features.db_importer import import_to_colmap


def _db_summary(db_path: Path) -> None:
    if not db_path.exists():
        print("[benchmark] database.db not found — skipping summary")
        return
    print("\n[benchmark] COLMAP database summary")
    conn = sqlite3.connect(db_path)
    for table in ("cameras", "images", "keypoints", "descriptors",
                  "matches", "two_view_geometries"):
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<25} {count}")
        except sqlite3.OperationalError:
            print(f"  {table:<25} (table not found)")
    conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ALIKED + LightGlue feature pipeline.")
    p.add_argument("--rgb-dir",      required=True,              help="Folder with RGB images.")
    p.add_argument("--output-dir",   default="output/feature_benchmark",
                   help="Root output folder.")
    p.add_argument("--n-neighbors",  type=int, default=8,        help="GPS neighbors per image.")
    p.add_argument("--max-keypoints",type=int, default=8192,     help="ALIKED keypoints per image.")
    p.add_argument("--resize",       type=int, default=1600,
                   help="Cap longest image dimension before extraction. 0 = no resize.")
    p.add_argument("--focal-px",     type=float, default=None,
                   help="Known focal length in pixels. Omit to use heuristic.")
    p.add_argument("--cpu",          action="store_true",         help="Disable GPU.")
    p.add_argument("--skip-matching",action="store_true",         help="Run extraction only.")
    p.add_argument("--skip-import",  action="store_true",
                   help="Skip COLMAP database import (extraction + matching only).")
    p.add_argument("--no-verify",    action="store_true",
                   help="Skip COLMAP RANSAC geometric verification in import step.")
    p.add_argument("--check-gps-only",action="store_true",        help="Print GPS status and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    use_gpu    = not args.cpu
    resize     = args.resize if args.resize > 0 else None

    # ── load captures ─────────────────────────────────────────────────────────
    captures = load_rgb_captures(args.rgb_dir)
    with_gps, without_gps = gps_summary(captures)

    print(f"[benchmark] RGB images     : {len(captures)}")
    print(f"[benchmark] With GPS       : {with_gps}  |  Without GPS: {without_gps}")
    print(f"[benchmark] GPU            : {use_gpu}")
    print(f"[benchmark] Max keypoints  : {args.max_keypoints}")
    print(f"[benchmark] Resize cap     : {resize}")
    print(f"[benchmark] GPS neighbors  : {args.n_neighbors}")

    if args.check_gps_only:
        for cap in captures:
            tag = "GPS" if cap.latitude is not None else "NO GPS"
            print(f"  {cap.capture_id}: {tag} ({cap.latitude}, {cap.longitude}, {cap.altitude})")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    # ── Stage 3: ALIKED extraction ────────────────────────────────────────────
    print("\n[benchmark] ── Stage 3: ALIKED extraction ──")
    t = time.perf_counter()
    extract_features(
        captures=captures,
        output_dir=str(output_dir),
        use_gpu=use_gpu,
        max_keypoints=args.max_keypoints,
        resize=resize,
    )
    t_extract = time.perf_counter() - t
    print(f"[benchmark] Extraction time: {t_extract:.2f}s  "
          f"({t_extract / len(captures):.2f}s/image)")

    if args.skip_matching:
        print(f"\n[benchmark] Total: {time.perf_counter() - t_total:.2f}s")
        return 0

    # ── Stage 4: LightGlue matching ───────────────────────────────────────────
    print("\n[benchmark] ── Stage 4: LightGlue matching ──")
    t = time.perf_counter()
    match_features(
        captures=captures,
        output_dir=str(output_dir),
        n_neighbors=args.n_neighbors,
        use_gpu=use_gpu,
    )
    t_match = time.perf_counter() - t
    print(f"[benchmark] Matching time  : {t_match:.2f}s")

    if args.skip_import:
        print(f"\n[benchmark] Total (extract+match): {time.perf_counter() - t_total:.2f}s")
        return 0

    # ── Stage 5: COLMAP database import ──────────────────────────────────────
    print("\n[benchmark] ── Stage 5: COLMAP database import ──")
    t = time.perf_counter()
    db_path = import_to_colmap(
        captures=captures,
        output_dir=str(output_dir),
        focal_length_px=args.focal_px,
        run_geometric_verification=not args.no_verify,
    )
    t_import = time.perf_counter() - t
    print(f"[benchmark] Import time    : {t_import:.2f}s")

    _db_summary(Path(db_path))

    # ── summary ───────────────────────────────────────────────────────────────
    t_pipeline = time.perf_counter() - t_total
    print(f"\n[benchmark] ── Stage summary ──")
    print(f"  Extraction   : {t_extract:.2f}s")
    print(f"  Matching     : {t_match:.2f}s")
    print(f"  Import+RANSAC: {t_import:.2f}s")
    print(f"  TOTAL        : {t_pipeline:.2f}s")
    print(f"\n[benchmark] Outputs")
    print(f"  features/    : {output_dir / 'features'}")
    print(f"  matches.h5   : {output_dir / 'matches.h5'}")
    print(f"  database.db  : {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
