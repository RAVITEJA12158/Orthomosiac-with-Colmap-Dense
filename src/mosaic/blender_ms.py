"""
src/mosaic/blender_ms.py

Blends multispectral tiles using a distance-weighted average — NOT
multi-band (Laplacian pyramid) blending. Multi-band blending recombines
frequency bands in a way that changes the actual pixel value at every
point near a seam; for an 8-bit RGB photo that's invisible and desirable
(it's literally the point). For a calibrated reflectance value, any change
to the pixel value is a measurement error — it directly corrupts NDVI/NDRE.

Weighted average preserves the per-pixel measurement exactly wherever only
one tile contributes (the normalisation `canvas / weight_canvas` cancels
the weight in that case), and blends two overlapping measurements smoothly
near a seam without any pyramid-style alteration of the underlying values.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

if TYPE_CHECKING:
    from .seam_finder import SeamlineSet
    from .tile_loader import CanvasInfo, TileInfo

logger = logging.getLogger(__name__)

MS_NODATA = -9999.0


def blend_ms_mosaic(
    tile_infos: List["TileInfo"],
    band_images: List[np.ndarray],
    seamline_set: "SeamlineSet",
    canvas: "CanvasInfo",
) -> np.ndarray:
    """
    Distance-weighted average blend of one calibrated MS band over the full
    canvas, reusing the RGB SeamlineSet for tile placement and ownership.

    tile_infos    : metadata for each tile (length must match band_images)
    band_images   : calibrated (H, W) float32 reflectance per tile
    seamline_set  : SeamlineSet produced by Stage 11 (RGB) — REUSED, not
                    recomputed
    canvas        : full mosaic canvas geometry (same target_gsd_m as RGB)

    Algorithm:
    1. canvas_sum, canvas_weight <- zeros
    2. for each tile:
         - seam mask says which canvas pixels this tile owns / could own
         - weight = sqrt(distance to nearest invalid/seam pixel) -> tile
           centres dominate, tile edges fade out
         - canvas_sum[region]    += tile * weight
         - canvas_weight[region] += weight
    3. output = canvas_sum / canvas_weight wherever weight > 0
    4. pixels with zero total weight -> nodata

    Returns (canvas.height_px, canvas.width_px) float32, nodata = -9999.0.
    """
    if len(band_images) != len(seamline_set.masks):
        raise ValueError(
            f"band_images ({len(band_images)}) and seamline masks "
            f"({len(seamline_set.masks)}) length mismatch"
        )

    h_canvas, w_canvas = canvas.height_px, canvas.width_px
    canvas_sum = np.zeros((h_canvas, w_canvas), dtype=np.float64)
    canvas_weight = np.zeros((h_canvas, w_canvas), dtype=np.float64)

    for tile_img, mask, corner in zip(band_images, seamline_set.masks, seamline_set.corners):
        th, tw = tile_img.shape[:2]
        mh, mw = mask.shape[:2]

        if (mh, mw) != (th, tw):
            # MS tiles must be generated at the same target_gsd_m as the RGB
            # tiles for the reused seamline to apply pixel-for-pixel. A
            # mismatch here means run_ms_mosaic() was called with a
            # different target_gsd_m than run_rgb_mosaic() — resize as a
            # defensive fallback, but this should not happen in normal use.
            logger.warning(
                "blend_ms_mosaic: tile shape %s != seam mask shape %s, resizing mask "
                "(check target_gsd_m matches between RGB and MS mosaicking)",
                tile_img.shape, mask.shape,
            )
            mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        cx, cy = corner
        x0, y0 = max(cx, 0), max(cy, 0)
        x1, y1 = min(cx + tw, w_canvas), min(cy + th, h_canvas)
        if x1 <= x0 or y1 <= y0:
            continue  # tile falls entirely outside the canvas — shouldn't happen

        src_x0, src_y0 = x0 - cx, y0 - cy
        src_x1, src_y1 = src_x0 + (x1 - x0), src_y0 + (y1 - y0)

        tile_crop = tile_img[src_y0:src_y1, src_x0:src_x1]
        mask_crop = mask[src_y0:src_y1, src_x0:src_x1]

        valid_pixels = (mask_crop > 0) & np.isfinite(tile_crop)
        distance = distance_transform_edt(mask_crop > 0)
        weight = np.sqrt(distance).astype(np.float64)
        weight[~valid_pixels] = 0.0

        tile_clean = np.where(valid_pixels, tile_crop, 0.0).astype(np.float64)

        canvas_sum[y0:y1, x0:x1] += tile_clean * weight
        canvas_weight[y0:y1, x0:x1] += weight

    output = np.full((h_canvas, w_canvas), MS_NODATA, dtype=np.float32)
    valid = canvas_weight > 0
    output[valid] = (canvas_sum[valid] / canvas_weight[valid]).astype(np.float32)

    logger.info(
        "blend_ms_mosaic: blended %d tiles, %.1f%% of canvas has data",
        len(band_images), 100.0 * np.count_nonzero(valid) / valid.size,
    )
    return output


def stack_ms_bands(
    band_mosaics: Dict[str, np.ndarray],
    band_order: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Stack the 4 single-band float32 mosaics into one (4, H, W) array in the
    canonical output band order: Green, Red, RedEdge, NIR.
    """
    if band_order is None:
        band_order = ["GRE", "RED", "REG", "NIR"]
    
    missing = [b for b in band_order if b not in band_mosaics]
    if missing:
        raise ValueError(f"stack_ms_bands: missing band(s) {missing} in band_mosaics")

    shapes = {b: band_mosaics[b].shape for b in band_order}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"stack_ms_bands: band shapes don't match: {shapes}")

    stacked = np.stack([band_mosaics[b] for b in band_order], axis=0).astype(np.float32)
    return stacked