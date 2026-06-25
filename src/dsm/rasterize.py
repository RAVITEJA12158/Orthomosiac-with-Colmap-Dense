"""
src/dsm/rasterize.py

Stage 9b â€” Point Cloud Rasterization.

Converts a fused 3D point cloud (.ply, produced by COLMAP stereo_fusion
DensifyPointCloud in fusion mode) into a regular elevation grid written
out as a georeferenced GeoTIFF (dsm.tif).

Design notes
------------
- Median (not mean) is used per grid cell: a handful of fusion-survivor
  outliers are common even after COLMAP stereo fusion consistency checks,
  and median rejects a single bad point without dragging the cell
  elevation toward it. On flat agricultural fields, median and mean are
  close anyway, so this costs nothing in the common case and protects
  the rare case.
- .ply reading prefers open3d (cleanest API) but falls back to a
  lightweight manual struct-based binary/ASCII PLY parser so this file
  has no hard dependency on a heavy, multi-hundred-MB package.
- Output is intentionally "raw" (nodata=-9999 for empty cells). Gap
  filling is a separate concern, handled by interpolate.py. Keeping
  these separate means rasterize.py stays simple and testable with
  pure synthetic point clouds.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import logging
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover - exercised only when rasterio absent
    _HAS_RASTERIO = False

try:
    import open3d as o3d  # type: ignore

    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False


_log = logging.getLogger(__name__)

NODATA_VALUE = -9999.0
DEFAULT_TARGET_GSD_M = 0.10  # 10cm/pixel default, see dsm_stage_context.md


class RasterioRequiredError(RuntimeError):
    """Raised when a GeoTIFF must be written but rasterio is not installed."""


# ---------------------------------------------------------------------------
# .ply reading
# ---------------------------------------------------------------------------

def _read_ply_xyz(ply_path: str) -> np.ndarray:
    """
    Read a .ply point cloud and return an (N, 3) float32 array of XYZ points.

    Tries open3d first (handles every PLY variant correctly). Falls back to
    a manual parser that covers the common cases COLMAP stereo_fusion commonly produces:
    ASCII PLY and binary_little_endian PLY with float/double x,y,z properties
    (other properties, e.g. normals/colors, are read past but ignored).
    """
    if _HAS_OPEN3D:
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        return pts

    return _manual_read_ply_xyz(ply_path)


# Map PLY scalar type names -> (struct format char, size in bytes)
_PLY_TYPE_MAP = {
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
}


@dataclass
class _PlyProperty:
    name: str
    fmt: str
    size: int


def _manual_read_ply_xyz(ply_path: str) -> np.ndarray:
    """
    Minimal dependency-free PLY reader for the 'vertex' element's x,y,z
    properties. Supports ASCII and binary_little_endian formats, which
    covers COLMAP stereo_fusion output.
    """
    with open(ply_path, "rb") as f:
        raw = f.read()

    header_lines_raw = []
    body_offset = None
    offset = 0
    for line in raw.splitlines(keepends=True):
        line_body = line.rstrip(b"\r\n")
        offset += len(line)
        if line_body == b"end_header":
            body_offset = offset
            break
        header_lines_raw.append(line)

    if body_offset is None:
        raise ValueError(f"Not a valid PLY file (no end_header): {ply_path}")

    header_text = b"".join(header_lines_raw).decode("ascii", errors="replace")
    header_lines = header_text.splitlines()

    fmt = None
    vertex_count = 0
    properties: list[_PlyProperty] = []
    in_vertex_element = False

    for line in header_lines:
        line = line.strip()
        if line.startswith("format"):
            fmt = line.split()[1]
        elif line.startswith("element"):
            parts = line.split()
            elem_name = parts[1]
            elem_count = int(parts[2])
            in_vertex_element = elem_name == "vertex"
            if in_vertex_element:
                vertex_count = elem_count
            else:
                # Once we hit a non-vertex element, stop collecting properties.
                in_vertex_element = False
        elif line.startswith("property") and in_vertex_element:
            parts = line.split()
            # "property float x" or "property list uchar int vertex_indices"
            if parts[1] == "list":
                raise ValueError(
                    f"Unsupported PLY: list property in vertex element ({ply_path})"
                )
            type_name, prop_name = parts[1], parts[2]
            if type_name not in _PLY_TYPE_MAP:
                raise ValueError(f"Unsupported PLY property type: {type_name}")
            struct_fmt, size = _PLY_TYPE_MAP[type_name]
            properties.append(_PlyProperty(prop_name, struct_fmt, size))

    if vertex_count == 0 or not properties:
        return np.zeros((0, 3), dtype=np.float32)

    prop_names = [p.name for p in properties]
    xyz_idx = [prop_names.index(c) for c in ("x", "y", "z")]

    body = raw[body_offset:]

    if fmt == "ascii":
        points = np.zeros((vertex_count, 3), dtype=np.float32)
        text = body.decode("ascii", errors="replace").splitlines()
        for i in range(vertex_count):
            tokens = text[i].split()
            points[i, 0] = float(tokens[xyz_idx[0]])
            points[i, 1] = float(tokens[xyz_idx[1]])
            points[i, 2] = float(tokens[xyz_idx[2]])
        return points

    if fmt in ("binary_little_endian", "binary_big_endian"):
        endian = "<" if fmt == "binary_little_endian" else ">"
        row_fmt = endian + "".join(p.fmt for p in properties)
        row_size = sum(p.size for p in properties)
        struct_obj = struct.Struct(row_fmt)

        points = np.zeros((vertex_count, 3), dtype=np.float32)
        offset = 0
        for i in range(vertex_count):
            row = struct_obj.unpack_from(body, offset)
            points[i, 0] = row[xyz_idx[0]]
            points[i, 1] = row[xyz_idx[1]]
            points[i, 2] = row[xyz_idx[2]]
            offset += row_size
        return points

    raise ValueError(f"Unsupported PLY format '{fmt}' in {ply_path}")


# ---------------------------------------------------------------------------
# Gridding
# ---------------------------------------------------------------------------

def _points_to_grid(
    points: np.ndarray,
    gsd_m: float,
    nodata_value: float = NODATA_VALUE,
) -> Tuple[np.ndarray, tuple]:
    """
    Bin (N, 3) XY-Z points into a regular grid using per-cell median Z.

    Returns
    -------
    grid : np.ndarray, shape (rows, cols), float32
        Median elevation per cell, nodata_value where no points fell in
        the cell.
    geotransform : tuple
        (x_min, gsd_m, 0.0, y_max, 0.0, -gsd_m) â€” GDAL-style affine
        geotransform (origin = top-left corner, y decreasing downward).
    """
    if points.shape[0] == 0:
        raise ValueError("Cannot rasterize an empty point cloud")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    # Number of bins along each axis: the bin index of the max-coordinate
    # point is floor((max - min) / gsd), so we need that many bins + 1 to
    # include it. (Using ceil here would silently add a spurious empty
    # trailing row/column whenever the extent is an exact multiple of gsd.)
    cols = max(1, int(np.floor((x_max - x_min) / gsd_m)) + 1)
    rows = max(1, int(np.floor((y_max - y_min) / gsd_m)) + 1)

    # Column increases with X. Row increases as Y decreases (image convention:
    # row 0 = top = y_max).
    col_idx = np.clip(((x - x_min) / gsd_m).astype(np.int64), 0, cols - 1)
    row_idx = np.clip(((y_max - y) / gsd_m).astype(np.int64), 0, rows - 1)

    flat_idx = row_idx * cols + col_idx
    n_cells = rows * cols

    # Group z-values by cell using argsort, then take median within each
    # contiguous run. This avoids per-cell Python loops over a huge point
    # cloud while still computing an exact median (not an approximation).
    order = np.argsort(flat_idx, kind="stable")
    sorted_idx = flat_idx[order]
    sorted_z = z[order]

    grid_flat = np.full(n_cells, nodata_value, dtype=np.float32)

    # Boundaries between runs of identical cell index.
    boundaries = np.flatnonzero(np.diff(sorted_idx)) + 1
    run_starts = np.concatenate(([0], boundaries))
    run_ends = np.concatenate((boundaries, [len(sorted_idx)]))

    for start, end in zip(run_starts, run_ends):
        cell = sorted_idx[start]
        grid_flat[cell] = np.median(sorted_z[start:end])

    grid = grid_flat.reshape(rows, cols)
    geotransform = (x_min, gsd_m, 0.0, y_max, 0.0, -gsd_m)
    return grid, geotransform


# ---------------------------------------------------------------------------
# GeoTIFF writing
# ---------------------------------------------------------------------------

def _write_geotiff(
    grid: np.ndarray,
    geotransform: tuple,
    crs: str,
    output_path: str,
    nodata_value: float = NODATA_VALUE,
) -> None:
    """Write a float32 GeoTIFF via rasterio with the given CRS/geotransform."""
    if not _HAS_RASTERIO:
        raise RasterioRequiredError(
            "rasterio is required to write GeoTIFFs. Install it with "
            "`pip install rasterio` (see CLAUDE.md install requirements)."
        )

    x_min, px_w, _, y_max, _, px_h = geotransform
    transform = from_origin(x_min, y_max, px_w, -px_h)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype=grid.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata_value,
    ) as dst:
        dst.write(grid, 1)




def _utm_epsg_from_reconstruction(reconstruction) -> str:
    """
    Derive the UTM EPSG code from the median GPS position of all registered
    images in a pycolmap.Reconstruction.

    Returns a string like "EPSG:32644" (UTM zone 44N).

    Falls back to None if no GPS priors are available on the images.
    """
    try:
        from pyproj import CRS
        from pyproj.aoi import AreaOfInterest
        from pyproj.database import query_utm_crs_info
    except ImportError:
        _log.warning(
            "_utm_epsg_from_reconstruction: pyproj not installed â€” cannot "
            "auto-detect UTM CRS. Pass crs= explicitly."
        )
        return None

    lats, lons = [], []
    try:
        images = reconstruction.images.values()
    except AttributeError:
        return None

    for img in images:
        try:
            # pycolmap.Image may expose GPS via .tvec or via .cam_from_world
            # depending on version; the most reliable source is the pose_prior
            # attached during GPS alignment (src/sfm/pose_priors.py).
            pp = getattr(img, "pose_prior", None)
            if pp is not None:
                pos = pp.position
                if abs(float(pos[0])) > 360 or abs(float(pos[1])) > 360:
                    try:
                        import pycolmap
                        import numpy as np
                        transform = pycolmap.GPSTransform()
                        ellipsoid = transform.ecef_to_ellipsoid(np.array([pos], dtype="float64"))
                        lats.append(float(ellipsoid[0, 0]))
                        lons.append(float(ellipsoid[0, 1]))
                    except Exception:
                        continue
                else:
                    lats.append(float(pos[0]))
                    lons.append(float(pos[1]))
        except Exception:
            continue

    if not lats:
        return None

    lat_med = float(sorted(lats)[len(lats) // 2])
    lon_med = float(sorted(lons)[len(lons) // 2])

    try:
        results = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=lon_med,
                south_lat_degree=lat_med,
                east_lon_degree=lon_med,
                north_lat_degree=lat_med,
            ),
        )
        if results:
            return f"EPSG:{results[0].code}"
    except Exception as exc:
        _log.warning("_utm_epsg_from_reconstruction: pyproj UTM query failed: %s", exc)

    return None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rasterize_pointcloud(
    ply_path: str,
    output_path: str,
    reconstruction=None,
    target_gsd_m: float = DEFAULT_TARGET_GSD_M,
    crs: Optional[str] = None,
) -> str:
    """
    Convert a fused point cloud (.ply) into a georeferenced DSM GeoTIFF.

    Parameters
    ----------
    ply_path : path to fused point cloud from fusion.py (run_fusion()).
    output_path : where to write the (gap-having, raw) DSM GeoTIFF.
    reconstruction : pycolmap.Reconstruction, reserved for future use if
        CRS needs to be derived from the SfM georeferencing rather than
        passed explicitly. Currently unused but kept in the signature to
        match the Stage 9 -> Stage 10 contract and avoid a breaking change
        later (e.g. deriving UTM zone automatically from GPS priors).
    target_gsd_m : output pixel size in meters/pixel. Must match the
        downstream orthomosaic target GSD, or orthorectification will
        show staircase artifacts (see dsm_stage_context.md).
    crs : output CRS string (e.g. "EPSG:4326" or a UTM zone EPSG code).

        IMPORTANT: the .ply XY coordinates must already be in this CRS's
        units.  COLMAP outputs points in the dense workspace coordinate frame (typically metric/ECEF or UTM after GPS alignment, NOT WGS84 degrees).
        If you leave crs at the default "EPSG:4326" when the data is metric,
        the DSM will have the correct elevation grid but an incorrect
        geographic tag.  dsm_sampler.py will then attempt an ill-formed
        reprojection.

        Pass the actual UTM EPSG (e.g. "EPSG:32644") that matches the SfM
        coordinate frame.  If unsure, check the output of
        src.sfm.alignment.align_to_gps() â€” it logs the UTM zone used.

    Returns
    -------
    output_path
    """
    # If the caller left crs at the default "EPSG:4326" and we have a
    # reconstruction, try to auto-detect the correct UTM zone from GPS priors.
    # If auto-detection succeeds, use it.  If it fails (no GPS priors, no
    # pyproj) we cannot proceed safely â€” the DSM would carry the wrong
    # geotag and silently break dsm_sampler.py's reprojection downstream,
    # so we raise instead of just warning.
    if crs is None or crs == "EPSG:4326":
        crs = "EPSG:4326"
        if reconstruction is not None:
            detected = _utm_epsg_from_reconstruction(reconstruction)
            if detected is not None:
                _log.info(
                    "rasterize_pointcloud: auto-detected CRS %s from reconstruction GPS priors "
                    "(overrides default EPSG:4326).",
                    detected,
                )
                crs = detected
            else:
                _log.warning(
                    "rasterize_pointcloud: crs was left at the default 'EPSG:4326' but the "
                    "fused .ply is almost certainly in a metric frame (UTM/ECEF), not WGS84 "
                    "degrees. Auto-detection from reconstruction GPS priors failed (no priors "
                    "or pyproj unavailable). Falling back to EPSG:4326, which may be incorrect."
                )

    points = _read_ply_xyz(ply_path)
    grid, geotransform = _points_to_grid(points, target_gsd_m)
    _write_geotiff(grid, geotransform, crs, output_path)
    return output_path


