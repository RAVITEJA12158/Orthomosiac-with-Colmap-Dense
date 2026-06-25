"""
Stage 6 — Structure-from-Motion pipeline.

Entry point:
    from src.sfm import run_sfm
    reconstruction = run_sfm(
        database_path = "mission/colmap.db",
        image_dir     = "mission/rgb_images/",
        output_dir    = "mission/sparse/",
        captures      = captures,
        has_rtk       = False,
    )

Internal steps:
    keyframes.py        Steps 1-2  keyframe selection + GPS init pair
    pose_priors.py      Step 3a    GPS/RTK prior injection
    glomap.py           Step 3b    GLOMAP fast path + 3-check validation
    colmap_mapper.py    Step 4     COLMAP incremental + 5 optimizations
    pnp_registration.py Step 5     PnP non-keyframe registration
    final_ba.py         Steps 6-7  final BA + GPS alignment
    pipeline.py                    orchestrator
"""

_EXPORTS = {
    "run_sfm": (".pipeline", "run_sfm"),
    "select_keyframes": (".keyframes", "select_keyframes"),
    "write_keyframe_list": (".keyframes", "write_keyframe_list"),
    "find_gps_guided_init_pair": (".keyframes", "find_gps_guided_init_pair"),
    "inject_gps_priors": (".pose_priors", "inject_gps_priors"),
    "RTK_PRIOR_WEIGHT": (".pose_priors", "RTK_PRIOR_WEIGHT"),
    "GPS_PRIOR_WEIGHT": (".pose_priors", "GPS_PRIOR_WEIGHT"),
    "run_glomap": (".glomap", "run_glomap"),
    "validate_glomap_reconstruction": (".glomap", "validate_glomap_reconstruction"),
    "run_colmap_incremental": (".colmap_mapper", "run_colmap_incremental"),
    "register_non_keyframes": (".pnp_registration", "register_non_keyframes"),
    "run_final_bundle_adjustment": (".final_ba", "run_final_bundle_adjustment"),
    "align_to_gps": (".final_ba", "align_to_gps"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value

__all__ = [
    "run_sfm",
    "select_keyframes",
    "write_keyframe_list",
    "find_gps_guided_init_pair",
    "inject_gps_priors",
    "RTK_PRIOR_WEIGHT",
    "GPS_PRIOR_WEIGHT",
    "run_glomap",
    "validate_glomap_reconstruction",
    "run_colmap_incremental",
    "register_non_keyframes",
    "run_final_bundle_adjustment",
    "align_to_gps",
]
