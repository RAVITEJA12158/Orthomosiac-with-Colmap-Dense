"""
Feature pipeline: Stage 3 (ALIKED extraction) + Stage 4 (LightGlue matching)
+ Stage 5 bridge (COLMAP database import + RANSAC geometric verification).

Entry point:
    from src.features import run_feature_pipeline
    db_path = run_feature_pipeline(captures, output_dir)
"""

from typing import Optional

# Lightweight sub-modules with no heavy ML dependencies — safe to import eagerly.
from .neighbors import build_neighbor_pairs
from .rgb_only import gps_summary, load_rgb_captures


def run_feature_pipeline(
    captures,
    output_dir: str,
    use_gpu: bool = True,
    n_neighbors: int = 8,
    max_keypoints: int = 8192,
    resize: int = 1600,
    focal_length_px: Optional[float] = None,
    run_geometric_verification: bool = True,
) -> str:
    """
    Full feature pipeline: ALIKED → LightGlue → COLMAP database.

    Stage 3 — ALIKED extraction
        One image at a time. No OOM risk. Writes features/<capture_id>.h5.

    Stage 4 — LightGlue matching
        GPS-filtered pairs only (~7,200 from 810,000 exhaustive).
        Writes matches.h5.

    Stage 5 bridge — COLMAP import + RANSAC
        Imports .h5 files into COLMAP database.db.
        Runs COLMAP RANSAC for geometric verification (two_view_geometries).
        After this, database.db is ready for pycolmap incremental mapper (Stage 6).

    Args:
        captures                  : List[Capture] from ingestion.load_mission()
        output_dir                : root output directory
                                    features/ and matches.h5 written here
        use_gpu                   : use CUDA for ALIKED + LightGlue
        n_neighbors               : GPS neighbors per image for pair filtering
        max_keypoints             : ALIKED keypoints per image
        resize                    : cap longest image dimension before extraction
                                    (1600 safe for 16 GB VRAM; None = no resize)
        focal_length_px           : known focal length in pixels for COLMAP camera
                                    None = use heuristic (1.2 × max image dim)
        run_geometric_verification: run COLMAP RANSAC before returning
                                    (required for Stage 6 mapper)

    Returns:
        str path to database.db
    """
    # Lazy imports: torch/h5py only required at runtime, not at import time.
    from .extractor import extract_features
    from .matcher import match_features
    from .db_importer import import_to_colmap

    # Stage 3 — ALIKED
    extract_features(
        captures=captures,
        output_dir=output_dir,
        use_gpu=use_gpu,
        max_keypoints=max_keypoints,
        resize=resize,
    )

    # Stage 4 — LightGlue
    match_features(
        captures=captures,
        output_dir=output_dir,
        n_neighbors=n_neighbors,
        use_gpu=use_gpu,
    )

    # Stage 5 bridge — import into COLMAP DB + RANSAC
    db_path = import_to_colmap(
        captures=captures,
        output_dir=output_dir,
        focal_length_px=focal_length_px,
        run_geometric_verification=run_geometric_verification,
    )

    return str(db_path)


def __getattr__(name: str):
    """Lazy attribute access for heavy sub-module symbols."""
    _lazy = {
        "extract_features": (".extractor", "extract_features"),
        "features_exist":   (".extractor", "features_exist"),
        "match_features":   (".matcher",   "match_features"),
        "import_to_colmap": (".db_importer", "import_to_colmap"),
    }
    if name in _lazy:
        from importlib import import_module
        module_name, attr_name = _lazy[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "run_feature_pipeline",
    "extract_features",
    "features_exist",
    "match_features",
    "import_to_colmap",
    "build_neighbor_pairs",
    "load_rgb_captures",
    "gps_summary",
]