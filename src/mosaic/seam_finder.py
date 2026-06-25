"""
src/mosaic/seam_finder.py

Finds the optimal cut line between overlapping RGB tiles using OpenCV's
GraphCut seam finder, and saves the result to disk so Stage 12 (MS
mosaicking) can reuse it at zero extra cost — same camera geometry means
the same seamlines apply pixel-for-pixel to the multispectral bands.

GraphCut algorithm summary:
    - Tiles are graph nodes, overlap pixels are edges.
    - Edge weight = colour difference between the two tiles at that pixel.
    - The minimum cut through the graph routes the seam through pixels
      where the two tiles already agree in colour — minimising visible
      discontinuity before blending even starts.
    - Output: each overlap pixel is assigned exclusively to one tile (a
      hard boundary). Multi-band blending (blender_rgb.py) then smooths
      across that boundary.

Saving seamlines is NOT optional — Stage 12 has no other way to guarantee
pixel-perfect spatial alignment between the RGB and MS mosaics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import cv2
import numpy as np

from .tile_loader import tile_corner_px, valid_mask

if TYPE_CHECKING:
    from .tile_loader import CanvasInfo, TileInfo

logger = logging.getLogger(__name__)


@dataclass
class SeamlineSet:
    """Per-tile seam mask + canvas placement, reused by Stage 12."""

    masks: List[np.ndarray]  # per tile: uint8 mask, 255 = this tile owns the pixel
    corners: List[Tuple[int, int]]  # canvas pixel coords of each tile's top-left


def _to_ndarray(x) -> np.ndarray:
    """cv2's stitching-detail bindings sometimes return cv2.UMat instead of
    a plain ndarray. Normalise to ndarray either way."""
    if isinstance(x, cv2.UMat):
        return x.get()
    return np.asarray(x)


def find_seamlines(
    tile_infos: List["TileInfo"],
    images: List[np.ndarray],
    canvas: "CanvasInfo",
) -> SeamlineSet:
    """
    Run cv2.detail_GraphCutSeamFinder("COST_COLOR") over all gain-corrected
    RGB tiles.

    tile_infos : metadata for each tile (for canvas placement)
    images     : gain-corrected (H, W, 3) uint8 per tile
    canvas     : full mosaic canvas geometry

    Returns a SeamlineSet with one mask per tile, same order as `images`.
    """
    if len(tile_infos) != len(images):
        raise ValueError(
            f"tile_infos ({len(tile_infos)}) and images ({len(images)}) length mismatch"
        )

    corners = [tile_corner_px(ti, canvas) for ti in tile_infos]

    if len(images) < 2:
        logger.info("find_seamlines: fewer than 2 tiles, using full-validity masks (no seam needed)")
        masks = [valid_mask(img) for img in images]
        return SeamlineSet(masks=masks, corners=corners)

    masks_in = [valid_mask(img) for img in images]
    images_f32 = [img.astype(np.float32) for img in images]

    seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
    result = seam_finder.find(images_f32, corners, [m.copy() for m in masks_in])

    seam_masks = [_to_ndarray(m).astype(np.uint8) for m in result]

    logger.info("find_seamlines: computed seam masks for %d tiles", len(seam_masks))
    return SeamlineSet(masks=seam_masks, corners=corners)


def save_seamlines(seamline_set: SeamlineSet, output_dir: str) -> str:
    """
    Serialise a SeamlineSet to <output_dir>/seamlines.npz.

    Tiles aren't guaranteed to all be the same shape, so each mask is saved
    under its own key (mask_0, mask_1, ...) rather than stacked into one
    array.

    Returns the path to the saved file.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "seamlines.npz")

    save_kwargs = {f"mask_{i}": m for i, m in enumerate(seamline_set.masks)}
    save_kwargs["corners"] = np.array(seamline_set.corners, dtype=np.int64)
    save_kwargs["num_tiles"] = np.array(len(seamline_set.masks), dtype=np.int64)

    np.savez_compressed(path, **save_kwargs)
    logger.info("save_seamlines: wrote %d tile masks to %s", len(seamline_set.masks), path)
    return path


def load_seamlines(path: str) -> SeamlineSet:
    """Deserialise a SeamlineSet written by save_seamlines()."""
    data = np.load(path)
    num_tiles = int(data["num_tiles"])
    masks = [data[f"mask_{i}"] for i in range(num_tiles)]
    corners = [tuple(c) for c in data["corners"]]

    logger.info("load_seamlines: loaded %d tile masks from %s", num_tiles, path)
    return SeamlineSet(masks=masks, corners=corners)
