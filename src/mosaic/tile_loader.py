"""
src/mosaic/tile_loader.py

Reads orthotiles produced by Stage 10 (orthorectification) and builds the
geometry needed for the mosaic canvas: union bounding box, output raster
size, and the Affine transform of the full mosaic.

Only metadata is read here (transform / CRS / bounds / size). Pixel data is
loaded lazily, one tile at a time, by load_tile_pixels(). NOTE: the caller
(run_rgb_mosaic in mosaic/__init__.py) currently loads all tiles into memory
before gain compensation and seam-finding. Chunked canvas processing is a
planned improvement — see mosaicking.md §Memory.

IMPORTANT — ordering guarantee:
    read_tile_infos() sorts tiles by geographic position (top-left first,
    then left-to-right). This sort order is what guarantees RGB and MS
    tiles end up in the same order, capture-for-capture, even though they
    come from independent tile_paths lists (different bands/folders) and
    were never explicitly matched by capture_id here. Two ortho tiles that
    cover the same ground footprint have (numerically) the same bounds, so
    they sort identically. This is the mechanism that lets Stage 12 reuse
    the Stage 11 SeamlineSet index-for-index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS

logger = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """Metadata for a single orthotile. No pixel data."""

    path: str
    bounds: BoundingBox  # left, bottom, right, top — CRS units
    transform: Affine
    crs: CRS
    width: int
    height: int
    dtype: str  # "uint8" or "float32"


@dataclass
class CanvasInfo:
    """Geometry of the full mosaic canvas (union of all tile bounds)."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width_px: int
    height_px: int
    transform: Affine
    crs: CRS


def read_tile_infos(tile_paths: List[str]) -> List[TileInfo]:
    """
    Open each GeoTIFF with Rasterio, read transform + CRS + bounds + size.
    Does NOT load pixel data.

    Returns list of TileInfo sorted by geographic position:
    top-left to bottom-right (north-to-south primary key, west-to-east
    secondary key), with path as a final deterministic tie-break.
    """
    if not tile_paths:
        raise ValueError("read_tile_infos() got an empty tile_paths list")

    infos: List[TileInfo] = []
    for path in tile_paths:
        with rasterio.open(path) as src:
            infos.append(
                TileInfo(
                    path=path,
                    bounds=src.bounds,
                    transform=src.transform,
                    crs=src.crs,
                    width=src.width,
                    height=src.height,
                    dtype=str(src.dtypes[0]),
                )
            )

    infos.sort(key=lambda ti: (-round(ti.bounds.top, 3), round(ti.bounds.left, 3), ti.path))

    logger.info("read_tile_infos: loaded metadata for %d tiles", len(infos))
    return infos


def compute_mosaic_canvas(tile_infos: List[TileInfo], target_gsd_m: float) -> CanvasInfo:
    """
    Compute the union bounding box of all tiles and the output raster grid
    at target_gsd_m resolution (square pixels, north-up).
    """
    if not tile_infos:
        raise ValueError("compute_mosaic_canvas() got an empty tile_infos list")
    if target_gsd_m <= 0:
        raise ValueError(f"target_gsd_m must be positive, got {target_gsd_m}")

    min_x = min(ti.bounds.left for ti in tile_infos)
    max_x = max(ti.bounds.right for ti in tile_infos)
    min_y = min(ti.bounds.bottom for ti in tile_infos)
    max_y = max(ti.bounds.top for ti in tile_infos)

    crs = tile_infos[0].crs
    for ti in tile_infos[1:]:
        if ti.crs != crs:
            raise ValueError(
                f"Tile {ti.path} has CRS {ti.crs}, expected {crs}. "
                "All tiles must share the same CRS (guaranteed by Stage 10)."
            )

    width_px = max(1, int(np.round((max_x - min_x) / target_gsd_m)))
    height_px = max(1, int(np.round((max_y - min_y) / target_gsd_m)))

    transform = Affine(target_gsd_m, 0.0, min_x, 0.0, -target_gsd_m, max_y)

    canvas = CanvasInfo(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        width_px=width_px,
        height_px=height_px,
        transform=transform,
        crs=crs,
    )
    logger.info(
        "compute_mosaic_canvas: %dx%d px at %.3fm GSD, extent (%.2f, %.2f) -> (%.2f, %.2f)",
        width_px, height_px, target_gsd_m, min_x, min_y, max_x, max_y,
    )
    return canvas


def load_tile_pixels(tile_info: TileInfo) -> np.ndarray:
    """
    Load one tile's pixel data.
    RGB tile (3-band)  -> (H, W, 3) uint8
    MS tile (1-band)   -> (H, W)    float32
    """
    with rasterio.open(tile_info.path) as src:
        if src.count >= 3:
            arr = src.read([1, 2, 3])  # (3, H, W)
            arr = np.moveaxis(arr, 0, -1)  # (H, W, 3)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
        elif src.count == 1:
            arr = src.read(1).astype(np.float32)  # (H, W)
            return arr
        else:
            raise ValueError(
                f"Tile {tile_info.path} has unexpected band count {src.count} "
                "(expected 1 for MS or 3 for RGB)"
            )


def tile_corner_px(tile_info: "TileInfo", canvas: "CanvasInfo") -> Tuple[int, int]:
    """
    Top-left corner of a tile in full-canvas pixel coordinates.
    Used by gain_compensator / seam_finder / blenders to place tiles on the
    shared canvas.
    """
    col, row = ~canvas.transform * (tile_info.bounds.left, tile_info.bounds.top)
    return int(round(col)), int(round(row))


def valid_mask(image: np.ndarray, nodata_value: Optional[float] = None) -> np.ndarray:
    """
    Derive a uint8 validity mask (255 = valid, 0 = nodata) for a tile.

    Ortho tiles (Stage 10) write 0 (RGB) / NaN-or-0 (MS) outside the
    projected image footprint. If nodata_value is given, pixels equal to it
    (within float tolerance) are treated as invalid; otherwise an
    all-channels-zero pixel is treated as invalid.
    """
    if nodata_value is not None:
        if image.ndim == 3:
            invalid = np.all(np.isclose(image, nodata_value), axis=-1)
        else:
            invalid = np.isclose(image, nodata_value)
    else:
        if image.ndim == 3:
            invalid = np.all(image == 0, axis=-1)
        else:
            invalid = (image == 0) | ~np.isfinite(image)

    mask = np.where(invalid, 0, 255).astype(np.uint8)
    return mask
