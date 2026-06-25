"""
src/mosaic/blender_rgb.py

Blends gain-compensated, seam-cut RGB tiles into the final seamless mosaic
using Laplacian pyramid (multi-band) blending. RGB only.

Why multi-band blending works:
    - Each tile is decomposed into a Laplacian pyramid (a stack of
      increasingly blurred/downsampled frequency bands).
    - Low frequencies (overall colour/brightness) are blended over a WIDE
      zone around the seam — this is what removes visible brightness
      jumps.
    - High frequencies (texture, crop rows, edges) are blended over a
      NARROW zone — this is what avoids ghosting/doubling of detail.
    - Reconstructing from the blended pyramid gives a seamless mosaic with
      no visible joins and no blurring of real detail.

This is the same method used by Metashape, Hugin (Enblend), and
PTGui. NOT used for multispectral — see blender_ms.py for why.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import cv2
import numpy as np

if TYPE_CHECKING:
    from .seam_finder import SeamlineSet
    from .tile_loader import CanvasInfo, TileInfo

logger = logging.getLogger(__name__)


def blend_rgb_mosaic(
    tile_infos: List["TileInfo"],
    images: List[np.ndarray],
    seamline_set: "SeamlineSet",
    canvas: "CanvasInfo",
    num_bands: int = 5,
) -> np.ndarray:
    """
    Blend all gain-corrected RGB tiles into the final mosaic using
    cv2.detail_MultiBandBlender.

    tile_infos    : metadata for each tile (length must match images)
    images        : gain-corrected (H, W, 3) uint8 per tile
    seamline_set  : seam masks + corners from find_seamlines()
    canvas        : full mosaic canvas geometry
    num_bands     : Laplacian pyramid levels (5 = Metashape-equivalent default)

    Returns (canvas.height_px, canvas.width_px, 3) uint8 — the full mosaic.
    """
    if len(images) != len(seamline_set.masks):
        raise ValueError(
            f"images ({len(images)}) and seamline masks ({len(seamline_set.masks)}) length mismatch"
        )

    dst_roi = (0, 0, canvas.width_px, canvas.height_px)

    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(num_bands)
    blender.prepare(dst_roi)

    for img, mask, corner in zip(images, seamline_set.masks, seamline_set.corners):
        blender.feed(img.astype(np.int16), mask, corner)

    result, _result_mask = blender.blend(None, None)
    if isinstance(result, cv2.UMat):
        result = result.get()

    result = np.clip(result, 0, 255).astype(np.uint8)

    logger.info(
        "blend_rgb_mosaic: blended %d tiles into %dx%d canvas (num_bands=%d)",
        len(images), canvas.width_px, canvas.height_px, num_bands,
    )
    return result
