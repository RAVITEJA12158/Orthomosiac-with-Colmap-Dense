"""
src/ortho/dsm_sampler.py

Loads the Stage 9 DSM GeoTIFF once, uploads it to GPU memory (resident for
the whole Stage 10 run), and provides a CPU-side single-point lookup used
only by footprint.py (which needs just a handful of lookups per image, not
a full grid — not worth a GPU round-trip).

DSM must be in a projected CRS (metres) for the backward-projection math
(world coordinates in metres, intrinsics in pixels) to be dimensionally
consistent. If Stage 9 produced a WGS84 (degree-unit) DSM, load_dsm()
reprojects it to the appropriate UTM zone before anything downstream sees it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._xp import xp

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False


@dataclass
class DSMSampler:
    data_gpu: "xp.ndarray"   # (H, W) float32, on GPU if available, NaN = nodata
    transform: "object"      # rasterio Affine: pixel -> CRS coordinate
    crs: "object"            # rasterio CRS
    nodata: float
    origin_x: float          # top-left X in CRS units
    origin_y: float          # top-left Y in CRS units
    pixel_width: float       # metres/pixel, positive
    pixel_height: float      # metres/pixel, positive (Y axis flips downward)
    height: int
    width: int
    mean_elevation: float    # cached nanmean, used by footprint.py


def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """Return the EPSG code of the appropriate UTM zone for a WGS84 point."""
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    if lat >= 0:
        return 32600 + zone  # UTM North
    return 32700 + zone  # UTM South


def _is_geographic_crs(crs) -> bool:
    if crs is None:
        return False
    try:
        return crs.is_geographic
    except AttributeError:
        return False


def load_dsm(dsm_path: str) -> DSMSampler:
    """
    Read dsm.tif, auto-reproject to UTM if it's in a geographic CRS, replace
    nodata with NaN, upload to GPU, and return a DSMSampler.
    """
    if not _HAS_RASTERIO:
        raise RuntimeError("rasterio is required to load the DSM")

    with rasterio.open(dsm_path) as src:
        if _is_geographic_crs(src.crs):
            src_array = src.read(1)
            src_nodata = src.nodata
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2.0
            center_lat = (bounds.top + bounds.bottom) / 2.0
            dst_epsg = _utm_epsg_for_lonlat(center_lon, center_lat)
            dst_crs = CRS.from_epsg(dst_epsg)

            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

            array = np.empty((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=src_array,
                destination=array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                src_nodata=src_nodata,
                dst_nodata=src_nodata,
                resampling=Resampling.bilinear,
            )
            transform = dst_transform
            crs = dst_crs
            nodata = src_nodata if src_nodata is not None else -9999.0
        else:
            array = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata if src.nodata is not None else -9999.0

    array = np.where(array == nodata, np.nan, array)

    height, width = array.shape
    origin_x = transform.c
    origin_y = transform.f
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)
    mean_elevation = float(np.nanmean(array)) if np.any(np.isfinite(array)) else 0.0

    data_gpu = xp.asarray(array)

    return DSMSampler(
        data_gpu=data_gpu,
        transform=transform,
        crs=crs,
        nodata=nodata,
        origin_x=origin_x,
        origin_y=origin_y,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        height=height,
        width=width,
        mean_elevation=mean_elevation,
    )


def world_to_pixel(sampler: DSMSampler, x: float, y: float) -> tuple:
    """Geographic (x, y) in the DSM's CRS -> fractional (col, row) pixel coords."""
    col = (x - sampler.origin_x) / sampler.pixel_width
    row = (sampler.origin_y - y) / sampler.pixel_height
    return col, row


def sample_elevation_cpu(sampler: DSMSampler, x: float, y: float) -> float:
    """
    CPU fallback: given a coordinate in the DSM's CRS (e.g. UTM), return the
    DSM elevation via nearest-pixel lookup. Used by footprint.py, which only
    needs a few lookups per image. Returns sampler.mean_elevation if the
    point is outside the DSM or lands on a nodata pixel — flat-terrain
    fallback, never a hard failure.
    """
    col, row = world_to_pixel(sampler, x, y)
    col_i = int(round(col))
    row_i = int(round(row))

    if not (0 <= row_i < sampler.height and 0 <= col_i < sampler.width):
        return sampler.mean_elevation

    data = sampler.data_gpu
    value = data[row_i, col_i]
    value = float(value.get()) if hasattr(value, "get") else float(value)

    if math.isnan(value):
        return sampler.mean_elevation
    return value