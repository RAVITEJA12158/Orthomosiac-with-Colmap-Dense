"""
src/ortho/backward_project.py

Core Stage 10 logic: GPU backward projection using CuPy (falls back to NumPy
on CPU-only machines via _xp.py).

For each output pixel at geographic coordinate (X, Y):
  1. Look up DSM[X, Y] → elevation Z
  2. Form world point P_world = [X, Y, Z]
  3. Transform to camera: P_cam = R @ P_world + t
  4. Project to pixel:   u = fx*(P_cam[0]/P_cam[2]) + cx
                         v = fy*(P_cam[1]/P_cam[2]) + cy
  5. Bilinear sample raw image at (u, v)
  6. Write to output

All steps run on GPU (CuPy arrays). Only image upload and result download
touch the PCIe bus. The DSM stays resident on GPU across all images.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._xp import xp, to_numpy
from .camera_model import CameraPose
from .dsm_sampler import DSMSampler
from .footprint import BoundingBox


def _build_output_grid(
    footprint: BoundingBox,
) -> Tuple["xp.ndarray", "xp.ndarray"]:
    """
    Build (H_out, W_out) grids of geographic X and Y coordinates for each
    output pixel centre, on GPU.

    Pixel centre at output row r, col c:
        X = footprint.min_x + (c + 0.5) * gsd
        Y = footprint.max_y - (r + 0.5) * gsd   (Y decreases downward)
    """
    H_out = footprint.height_px
    W_out = footprint.width_px
    gsd = footprint.gsd_m

    cols = xp.arange(W_out, dtype=xp.float64)
    rows = xp.arange(H_out, dtype=xp.float64)

    X_grid = footprint.min_x + (cols + 0.5) * gsd          # (W_out,)
    Y_grid = footprint.max_y - (rows + 0.5) * gsd          # (H_out,)

    X_grid, Y_grid = xp.meshgrid(X_grid, Y_grid)           # both (H_out, W_out)
    return X_grid, Y_grid


def _lookup_dsm_elevations(
    dsm: DSMSampler,
    X_grid: "xp.ndarray",   # (H_out, W_out)
    Y_grid: "xp.ndarray",   # (H_out, W_out)
) -> "xp.ndarray":
    """
    For each (X, Y) output pixel centre, look up the DSM elevation via
    nearest-pixel indexing on GPU.

    Returns Z_grid (H_out, W_out) float32, NaN where outside DSM or nodata.
    """
    col_f = (X_grid - dsm.origin_x) / dsm.pixel_width        # fractional col
    row_f = (dsm.origin_y - Y_grid) / dsm.pixel_height       # fractional row

    col_i = xp.clip(xp.round(col_f).astype(xp.int32), 0, dsm.width  - 1)
    row_i = xp.clip(xp.round(row_f).astype(xp.int32), 0, dsm.height - 1)

    in_bounds = (
        (col_f >= 0) & (col_f < dsm.width) &
        (row_f >= 0) & (row_f < dsm.height)
    )

    Z_grid = dsm.data_gpu[row_i, col_i].astype(xp.float64)
    # Mark out-of-bounds pixels as NaN so they become nodata in the output
    Z_grid = xp.where(in_bounds, Z_grid, xp.nan)
    return Z_grid


def _backward_project_kernel(
    X_grid: "xp.ndarray",    # (H_out, W_out) geographic X
    Y_grid: "xp.ndarray",    # (H_out, W_out) geographic Y
    Z_grid: "xp.ndarray",    # (H_out, W_out) DSM elevation (NaN = nodata)
    R: "xp.ndarray",         # (3, 3) world-to-camera rotation
    t: "xp.ndarray",         # (3,)   world-to-camera translation
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    W_in: int,
    H_in: int,
) -> Tuple["xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    """
    Vectorised backward projection (no Python loops — runs on GPU as array ops).

    For each output pixel (r, c):
        P_world = [X, Y, Z]
        P_cam   = R @ P_world + t     (broadcast matmul via einsum)
        u = fx * Px/Pz + cx
        v = fy * Py/Pz + cy

    Returns:
        u_coords : (H_out, W_out) float32 — column coords in input image
        v_coords : (H_out, W_out) float32 — row    coords in input image
        valid    : (H_out, W_out) bool    — True = inside input image & DSM valid
    """
    H_out, W_out = X_grid.shape

    # Stack world points: (H_out, W_out, 3)
    P_world = xp.stack([X_grid, Y_grid, Z_grid], axis=-1)  # (..., 3)

    # Camera-space: P_cam = R @ P_world + t
    # einsum: 'ij,...j->...i'  maps (3,3) x (...,3) -> (...,3)
    P_cam = xp.einsum("ij,...j->...i", R, P_world) + t     # (H_out, W_out, 3)

    Px = P_cam[..., 0]
    Py = P_cam[..., 1]
    Pz = P_cam[..., 2]

    # Project to image plane
    # Guard against Pz <= 0 (point behind camera) — set to NaN so it's invalid
    Pz_safe = xp.where(Pz > 1e-6, Pz, xp.nan)

    u_coords = (fx * (Px / Pz_safe) + cx).astype(xp.float32)
    v_coords = (fy * (Py / Pz_safe) + cy).astype(xp.float32)

    # Validity: inside input image bounds AND Pz > 0 AND DSM not NaN
    dsm_valid = ~xp.isnan(Z_grid)
    depth_ok  = Pz > 1e-6
    in_image  = (
        (u_coords >= 0) & (u_coords < W_in - 1) &
        (v_coords >= 0) & (v_coords < H_in - 1)
    )
    valid = dsm_valid & depth_ok & in_image

    return u_coords, v_coords, valid


def _bilinear_sample(
    image_gpu: "xp.ndarray",   # (H_in, W_in, C) — C=3 for RGB, C=1 for MS
    u_coords: "xp.ndarray",    # (H_out, W_out) float32
    v_coords: "xp.ndarray",    # (H_out, W_out) float32
    valid: "xp.ndarray",       # (H_out, W_out) bool
) -> "xp.ndarray":
    """
    Bilinear interpolation entirely via CuPy array indexing — no custom CUDA kernel.

    For each valid (u, v):
        u0, u1 = floor(u), floor(u)+1
        v0, v1 = floor(v), floor(v)+1
        wu = u - u0   (fractional col weight)
        wv = v - v0   (fractional row weight)
        result = (1-wu)*(1-wv)*img[v0,u0] + wu*(1-wv)*img[v0,u1]
               +  (1-wu)*wv  *img[v1,u0] + wu*wv    *img[v1,u1]

    Returns (H_out, W_out, C) in the same dtype as image_gpu.
    """
    H_in, W_in = image_gpu.shape[:2]
    C = image_gpu.shape[2] if image_gpu.ndim == 3 else 1

    u0 = xp.floor(u_coords).astype(xp.int32)
    v0 = xp.floor(v_coords).astype(xp.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    # Clamp to valid image bounds for safe indexing
    u0c = xp.clip(u0, 0, W_in - 1)
    u1c = xp.clip(u1, 0, W_in - 1)
    v0c = xp.clip(v0, 0, H_in - 1)
    v1c = xp.clip(v1, 0, H_in - 1)

    wu = (u_coords - u0.astype(xp.float32))[..., None]   # (..., 1) for broadcast
    wv = (v_coords - v0.astype(xp.float32))[..., None]

    if image_gpu.ndim == 2:
        image_gpu = image_gpu[..., None]

    # Fetch the four neighbours
    img_dtype = image_gpu.dtype
    img_f = image_gpu.astype(xp.float32)

    p00 = img_f[v0c, u0c]   # (H_out, W_out, C)
    p01 = img_f[v0c, u1c]
    p10 = img_f[v1c, u0c]
    p11 = img_f[v1c, u1c]

    result = (
        (1.0 - wu) * (1.0 - wv) * p00 +
        wu          * (1.0 - wv) * p01 +
        (1.0 - wu) * wv          * p10 +
        wu          * wv          * p11
    )

    # Zero out invalid pixels (will be set to nodata by tile_writer)
    result = xp.where(valid[..., None], result, 0.0)

    if img_dtype == xp.uint8:
        result = xp.clip(result, 0, 255).astype(xp.uint8)
    else:
        result = result.astype(img_dtype)

    if C == 1:
        result = result[..., 0]   # back to (H_out, W_out) for single-band

    return result


def backward_project_image(
    raw_image: np.ndarray,      # (H_in, W_in, 3) uint8 RGB  or  (H_in, W_in) float32 MS
    pose: CameraPose,
    dsm: DSMSampler,
    footprint: BoundingBox,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run backward projection for one image on GPU.

    Args:
        raw_image : RGB uint8 (H, W, 3)  or  single-band float32 (H, W)
        pose      : camera pose from extract_camera_poses()
        dsm       : DSMSampler with data_gpu already on GPU
        footprint : BoundingBox from compute_image_footprint()

    Returns:
        ortho_out  : np.ndarray  (H_out, W_out, 3) uint8  or  (H_out, W_out) float32
        ortho_mask : np.ndarray  (H_out, W_out) bool — True = valid pixel
    """
    intr = pose.intrinsics

    # Use actual image dimensions for boundary checks and bilinear sampling.
    # Multispectral bands (e.g. Sequoia 1280×960) often differ from the RGB
    # camera intrinsics (e.g. 4000×3000) stored in the reconstruction.  When
    # they differ we scale fx/fy/cx/cy proportionally so the projection math
    # stays correct.  For RGB images the scale factors are exactly 1.0.
    H_in = raw_image.shape[0]
    W_in = raw_image.shape[1]

    scale_x = W_in / max(intr.width, 1)
    scale_y = H_in / max(intr.height, 1)
    fx = intr.fx * scale_x
    fy = intr.fy * scale_y
    cx = intr.cx * scale_x
    cy = intr.cy * scale_y

    # Upload image to GPU
    image_gpu = xp.asarray(raw_image)

    # Upload pose to GPU
    R_gpu = xp.asarray(pose.R.astype(np.float64))
    t_gpu = xp.asarray(pose.t.astype(np.float64))

    # Build output coordinate grids
    X_grid, Y_grid = _build_output_grid(footprint)

    # Lookup DSM elevations at each output pixel
    Z_grid = _lookup_dsm_elevations(dsm, X_grid, Y_grid)

    # Backward project: world → camera → image pixel
    u_coords, v_coords, valid = _backward_project_kernel(
        X_grid, Y_grid, Z_grid,
        R_gpu, t_gpu,
        fx, fy, cx, cy,
        W_in, H_in,
    )

    # Bilinear sample the raw image
    ortho_gpu = _bilinear_sample(image_gpu, u_coords, v_coords, valid)

    # Download to CPU
    ortho_out  = to_numpy(ortho_gpu)
    ortho_mask = to_numpy(valid)

    return ortho_out, ortho_mask
