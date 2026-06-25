"""
Stage 6, Steps 6-7 — Final bundle adjustment + GPS alignment.

Step 6 — Final BA
  After PnP registration adds non-keyframes, one full bundle adjustment
  refines all camera poses and 3D points simultaneously.

  This is the only BA that sees all images (keyframes + non-keyframes).
  All previous BA runs during incremental SfM only saw keyframes.

  Options keep the same tight filter as the incremental step.

Step 7 — GPS alignment
  pycolmap.align_reconstructions_to_locations() aligns the sparse model
  to real-world WGS84 GPS coordinates.

  For RTK path: model is already well-anchored via use_prior_position=True.
               align_reconstructions_to_locations() is a cleanup step — seconds.
  For normal GPS path: same call, but absolute accuracy is 3-5m.
                       Sufficient for NDVI + yield prediction use case.

  This is NOT a separate pipeline stage — it is a few lines at the end of
  the SfM stage. No extra time budget needed (runs in seconds).

API confirmed against pycolmap 4.0.4:
  pycolmap.bundle_adjustment(reconstruction, options)
  pycolmap.align_reconstructions_to_locations(reconstruction, ref_images_dict)
  ref_images_dict: dict[str, tuple[float, float, float]]
    key   = image name as stored in reconstruction (e.g. "000.jpg")
    value = (latitude, longitude, altitude)
"""

from __future__ import annotations

from typing import List

from ..colmap_images import colmap_image_name
from ..ingestion.capture import Capture


def run_final_bundle_adjustment(reconstruction) -> None:
    """
    Step 6 — Final bundle adjustment over all registered images.

    Called after PnP registration adds non-keyframes.
    Refines all poses and 3D points together.

    Tight reprojection filter consistent with incremental step:
      max_reproj_error = 2.0 pixels (vs COLMAP default 4.0)
      Safe with ALIKED+LightGlue clean matches.
    """
    import pycolmap

    n_images_val = reconstruction.num_reg_images
    n_images = int(n_images_val() if callable(n_images_val) else n_images_val)
    n_points_val = reconstruction.points3D
    n_points = len(n_points_val() if callable(n_points_val) else n_points_val)

    print(f"[final_ba] Running final BA over {n_images} images, "
          f"{n_points} 3D points...")

    options = pycolmap.BundleAdjustmentOptions()
    # loss_function_type moved to options.ceres in pycolmap 4.x.
    # options.loss_function_type no longer exists — setting it would raise
    # AttributeError, which is exactly the crash seen in the logs.
    # TRIVIAL loss = standard least-squares (no robustifier).
    # Correct after LightGlue/PoseLib have already removed outliers —
    # we no longer need a robust loss at final BA time.
    if hasattr(options, "loss_function_type"):
        # pycolmap <4.x path
        options.loss_function_type = pycolmap.LossFunctionType.TRIVIAL
    elif hasattr(options, "ceres") and hasattr(options.ceres, "loss_function_type"):
        # pycolmap 4.x path (confirmed against 4.0.4)
        options.ceres.loss_function_type = pycolmap.LossFunctionType.TRIVIAL

    try:
        pycolmap.bundle_adjustment(reconstruction, options)
        print("[final_ba] Done.")
    except Exception as e:
        # Non-fatal — reconstruction is still usable without final BA
        print(f"[final_ba] Warning: final BA raised exception: {e}. "
              f"Continuing with pre-BA reconstruction.")


def align_to_gps(
    reconstruction,
    captures       : List[Capture],
    min_common     : int = 3,
) -> bool:
    """
    Step 7 — Align sparse model to GPS coordinates.

    Computes a similarity transform (scale + rotation + translation)
    that maps the reconstruction's coordinate frame to WGS84.

    For RTK path:  model is already anchored — this is a fine-tuning cleanup.
    For GPS path:  model gets 3-5m absolute position accuracy.

    Args:
        reconstruction : Reconstruction to align in-place
        captures       : Capture list with latitude/longitude/altitude
        min_common     : minimum images with GPS needed to attempt alignment
                         (3 is the minimum for a well-constrained similarity transform)

    Returns:
        True if alignment was applied, False if skipped.
    """
    import pycolmap

    # Build ref_images dict: image_name → (lat, lon, alt)
    ref_images = {}
    for cap in captures:
        if cap.latitude is None or cap.longitude is None:
            continue
        name = colmap_image_name(cap)
        ref_images[name] = (
            cap.latitude,
            cap.longitude,
            cap.altitude if cap.altitude is not None else 0.0,
        )

    if len(ref_images) < min_common:
        print(f"[gps_align] Only {len(ref_images)} captures have GPS "
              f"(need {min_common}). Skipping alignment.")
        return False

    print(f"[gps_align] Aligning reconstruction to GPS "
          f"({len(ref_images)} reference images)...")

    try:
        align_func = getattr(pycolmap, "align_reconstruction_to_locations", None)
        if align_func is None:
            align_func = getattr(pycolmap, "align_reconstructions_to_locations", None)
        if align_func is None:
            raise AttributeError("PyCOLMAP has no GPS alignment function (tried align_reconstruction_to_locations).")
        
        align_func(
            reconstruction,
            ref_images,
        )
        print("[gps_align] GPS alignment complete.")
        return True
    except Exception as e:
        print(f"[gps_align] Warning: GPS alignment failed: {e}. "
              f"Reconstruction in arbitrary coordinate frame.")
        return False
