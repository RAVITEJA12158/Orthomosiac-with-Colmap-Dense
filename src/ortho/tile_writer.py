"""
src/ortho/tile_writer.py

Writes orthoimage arrays (NumPy, downloaded from GPU by backward_project.py)
to cloud-optimised GeoTIFFs using Rasterio.

RGB  → uint8, 3-band, DEFLATE compression, nodata=0
MS   → float32, 1-band, DEFLATE compression, nodata=-9999.0

Both use 256×256 internal tiling so Stage 11 mosaicking can read tiles in
random-access order without decompressing entire files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError as exc:  # pragma: no cover
    raise ImportError("rasterio is required for tile_writer") from exc

from .footprint import BoundingBox


def write_ortho_tile(
    ortho_rgb: np.ndarray,       # (H_out, W_out, 3) uint8
    ortho_mask: np.ndarray,      # (H_out, W_out) bool — True = valid
    footprint: BoundingBox,
    crs,                         # rasterio CRS from DSM
    output_path: str,
    nodata: int = 0,
    compress: str = "deflate",
    tiled: bool = True,
) -> None:
    """
    Write an RGB orthoimage tile as a cloud-optimised GeoTIFF.

    Invalid pixels (ortho_mask == False) are set to the nodata value (0)
    before writing, so Stage 11 mosaickers know where the footprint ends.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Apply mask: invalid pixels → nodata
    out = ortho_rgb.copy()
    out[~ortho_mask] = nodata

    H_out, W_out = ortho_rgb.shape[:2]
    transform = footprint.to_affine()

    profile = {
        "driver":    "GTiff",
        "count":     3,
        "dtype":     "uint8",
        "crs":       crs,
        "transform": transform,
        "width":     W_out,
        "height":    H_out,
        "nodata":    nodata,
        "compress":  compress,
    }
    if tiled:
        profile.update({"tiled": True, "blockxsize": 256, "blockysize": 256})

    with rasterio.open(output_path, "w", **profile) as dst:
        # Rasterio band order: (band, row, col)
        dst.write(out[:, :, 0], 1)
        dst.write(out[:, :, 1], 2)
        dst.write(out[:, :, 2], 3)


def write_ortho_tile_multispectral(
    ortho_band: np.ndarray,      # (H_out, W_out) float32
    ortho_mask: np.ndarray,      # (H_out, W_out) bool
    footprint: BoundingBox,
    crs,
    output_path: str,
    band_name: str,              # "GRE" | "RED" | "REG" | "NIR"
    nodata: float = -9999.0,
    compress: str = "deflate",
    tiled: bool = True,
) -> None:
    """
    Write a single multispectral band orthotile as float32 GeoTIFF.

    float32 preserves absolute reflectance values from Stage 12 radiometric
    calibration — lossy conversions to uint8 or uint16 would corrupt NDVI.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    out = ortho_band.astype(np.float32).copy()
    out[~ortho_mask] = nodata

    H_out, W_out = ortho_band.shape[:2]
    transform = footprint.to_affine()

    profile = {
        "driver":    "GTiff",
        "count":     1,
        "dtype":     "float32",
        "crs":       crs,
        "transform": transform,
        "width":     W_out,
        "height":    H_out,
        "nodata":    nodata,
        "compress":  compress,
    }
    if tiled:
        profile.update({"tiled": True, "blockxsize": 256, "blockysize": 256})

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.update_tags(BAND_NAME=band_name)