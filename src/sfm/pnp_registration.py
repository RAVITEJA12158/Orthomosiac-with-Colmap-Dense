"""
Stage 6, Step 5 — PnP registration of non-keyframe images.

After keyframe SfM produces a sparse model, the remaining 2-in-3 images
(non-keyframes) are registered against it using PnP (Perspective-n-Point).

PnP registration is cheap:
  - No new 3D points triangulated
  - No bundle adjustment
  - Just: find 2D-3D correspondences → solve camera pose → accept/reject
  - ~100-200ms per image vs full SfM cost

Why this is safe:
  80% overlap means every non-keyframe shares matches with multiple keyframes.
  The sparse model already covers the full field from keyframes.
  PnP just locates the non-keyframe camera within the existing model.
  NDVI and orthomosaic quality benefit from denser camera coverage.

API surface confirmed against pycolmap 4.0.4:
  pycolmap.register_image(reconstruction, database, image_id)
  returns PnPResult with .success bool
  reconstruction.register_image(image_id) adds it to the model

Failures are non-fatal — skip and continue.
A non-keyframe failing to register leaves a small gap in camera density
but does not break the reconstruction.
"""

from __future__ import annotations

from typing import List

from ..colmap_images import colmap_image_name
from ..ingestion.capture import Capture


def _registration_succeeded(result, reconstruction, image_id: int) -> bool:
    """Normalize pycolmap registration return shapes across versions."""
    if result is not None and hasattr(result, "success"):
        return bool(result.success)
    if isinstance(result, bool):
        return result
    if result is not None:
        return True
    if hasattr(reconstruction, "is_image_registered"):
        try:
            return bool(reconstruction.is_image_registered(image_id))
        except Exception:
            return False
    return False


def _register_image_compat(pycolmap, reconstruction, db, database_path: str, image_id: int):
    """
    Try known pycolmap.register_image signatures.

    Some bindings accept an open Database object, while older local notes used a
    database path string. Keeping both attempts makes Step 5 robust without
    changing behavior on the installed version.
    """
    errors = []
    for database_arg in (db, str(database_path)):
        try:
            return pycolmap.register_image(reconstruction, database_arg, image_id)
        except TypeError as e:
            errors.append(e)
            continue

    if errors:
        raise errors[-1]
    return None


def register_non_keyframes(
    reconstruction,
    database_path  : str,
    non_keyframes  : List[Capture],
) -> int:
    """
    Step 5 — Register non-keyframe captures against the sparse model via PnP.

    Args:
        reconstruction : existing Reconstruction from keyframe SfM (Step 4)
        database_path  : COLMAP .db (has features + matches for all images)
        non_keyframes  : captures not used in keyframe SfM

    Returns:
        Number of non-keyframes successfully registered.
    """
    import pycolmap

    if not non_keyframes:
        print("[pnp] No non-keyframes to register.")
        return 0

    if not hasattr(pycolmap, "register_image"):
        print("[pnp] Warning: pycolmap.register_image is unavailable in this build. "
              "Skipping non-keyframe registration.")
        return 0

    db = pycolmap.Database.open(str(database_path))

    try:
        # Build name → image_id map from database
        name_to_id = {
            img.name: img.image_id
            for img in db.read_all_images()
        }
        registered = 0
        failed     = 0

        print(f"[pnp] Registering {len(non_keyframes)} non-keyframes via PnP...")

        for cap in non_keyframes:
            name = colmap_image_name(cap)

            image_id = name_to_id.get(name)
            if image_id is None:
                print(f"[pnp] Warning: no database row for {name} — skipping.")
                failed += 1
                continue

            try:
                result = _register_image_compat(
                    pycolmap,
                    reconstruction,
                    db,
                    database_path,
                    image_id,
                )
                if _registration_succeeded(result, reconstruction, image_id):
                    registered += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[pnp] Warning: PnP failed for {cap.capture_id}: {e}")
                failed += 1

        total = len(non_keyframes)
        print(f"[pnp] Registered {registered}/{total} non-keyframes "
              f"({registered/total*100:.1f}%)")

        if failed > total * 0.20:
            print(f"[pnp] Warning: {failed} non-keyframes failed PnP registration. "
                  f"Check feature matching quality or reduce keyframe interval.")

    finally:
        # Always close the DB — on Windows an unclosed handle locks the .db file
        # and prevents every subsequent pipeline stage from opening it.
        db.close()

    return registered
