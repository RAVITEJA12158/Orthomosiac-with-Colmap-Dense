"""
src/mosaic/gain_compensator.py

Normalizes brightness across all RGB tiles so two tiles captured under
slightly different lighting don't show a visible brightness jump at their
shared border. RGB only — NEVER applied to multispectral tiles, because MS
pixel values are absolute reflectance after Stage 12 calibration, and
gain compensation would silently rescale that reflectance and corrupt NDVI.

Uses OpenCV's stitching-detail GainCompensator: a global least-squares
solver that looks at every pair of overlapping tiles simultaneously and
finds the per-tile brightness scale factors that minimise total brightness
difference in all overlap regions. This is the same approach Metashape
uses for colour balancing across a photo set.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import cv2
import numpy as np

from .tile_loader import tile_corner_px, valid_mask

if TYPE_CHECKING:
    from .tile_loader import CanvasInfo, TileInfo

logger = logging.getLogger(__name__)


def compute_gain_maps(
    tile_infos: List["TileInfo"],
    images: List[np.ndarray],
    canvas: "CanvasInfo",
) -> List[np.ndarray]:
    """
    Run cv2.detail_GainCompensator over all tiles and return gain-corrected
    copies.

    tile_infos : metadata for each tile (for canvas placement)
    images     : (H, W, 3) uint8 per tile, already vignetting-corrected
    canvas     : full mosaic canvas geometry

    Returns a list of gain-corrected (H, W, 3) uint8 images, same order and
    length as `images`.
    """
    if len(tile_infos) != len(images):
        raise ValueError(
            f"tile_infos ({len(tile_infos)}) and images ({len(images)}) length mismatch"
        )
    if len(images) < 2:
        logger.info("compute_gain_maps: fewer than 2 tiles, skipping gain compensation")
        return list(images)

    corners = [tile_corner_px(ti, canvas) for ti in tile_infos]
    masks = [valid_mask(img) for img in images]

    compensator = cv2.detail_GainCompensator()
    compensator.feed(corners, images, masks)

    gain_corrected: List[np.ndarray] = []
    for idx, (corner, img, mask) in enumerate(zip(corners, images, masks)):
        out = compensator.apply(idx, corner, img, mask)
        if isinstance(out, cv2.UMat):
            out = out.get()
        gain_corrected.append(out.astype(np.uint8))

    logger.info("compute_gain_maps: gain-compensated %d tiles", len(gain_corrected))
    return gain_corrected
