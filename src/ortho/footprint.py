"""
src/ortho/footprint.py

Computes the ground footprint of a single camera image — the bounding
rectangle on the DSM grid that this image covers at the target GSD.

Uses a flat-terrain assumption: ray-plane intersection at Z = mean DSM
elevation. This is exact enough for flat agricultural fields where terrain
relief within a single image footprint is small (< 5 m typically) relative
to the flying altitude (> 80 m).

Runs entirely on CPU — called once per image, only 4 corners to project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rasterio.transform import Affine

from .camera_model import CameraPose, CameraIntrinsics
from .dsm_sampler import DSMSampler


@dataclass
class BoundingBox:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    gsd_m: float

    @property
    def width_px(self) -> int:
        return max(1, int(math.ceil((self.max_x - self.min_x) / self.gsd_m)))

    @property
    def height_px(self) -> int:
        return max(1, int(math.ceil((self.max_y - self.min_y) / self.gsd_m)))

    def to_affine(self) -> Affine:
        """Rasterio Affine transform: pixel (col, row) → CRS coordinate (x, y)."""
        return Affine(self.gsd_m, 0.0, self.min_x, 0.0, -self.gsd_m, self.max_y)


def _unproject_pixel(
    u: float,
    v: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Unproject a pixel (u, v) to a unit ray direction in camera space,
    ignoring distortion (distortion is small for nadir drone cameras,
    and footprint computation only needs corner-accuracy to ~1 m, not sub-pixel).

    Ray direction: [(u - cx)/fx, (v - cy)/fy, 1.0], then normalised.
    """
    x = (u - intrinsics.cx) / intrinsics.fx
    y = (v - intrinsics.cy) / intrinsics.fy
    ray = np.array([x, y, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def _ray_plane_intersect(
    origin: np.ndarray,       # camera center in world coords
    direction: np.ndarray,    # ray direction in world coords (unit)
    z_plane: float,           # Z = mean DSM elevation (flat-terrain assumption)
) -> Optional[np.ndarray]:
    """
    Intersect a ray  P = origin + t * direction  with the horizontal plane Z = z_plane.

    Returns the 3D world point [X, Y, Z] or None if the ray is parallel to the plane
    (camera is perfectly horizontal — degenerate for a nadir camera).
    """
    # direction[2] is the Z component of the world-space ray
    if abs(direction[2]) < 1e-8:
        return None  # ray is horizontal — won't hit the ground plane
    t = (z_plane - origin[2]) / direction[2]
    if t < 0:
        return None  # intersection is behind the camera
    return origin + t * direction


def compute_image_footprint(
    pose: CameraPose,
    dsm: DSMSampler,
    target_gsd_m: float,
    margin: float = 0.05,
) -> Optional[BoundingBox]:
    """
    Project the four image corners through the camera model onto the mean
    DSM ground plane and return the bounding box of the resulting footprint.

    Args:
        pose          : camera pose from extract_camera_poses()
        dsm           : loaded DSMSampler (only mean_elevation used here)
        target_gsd_m  : desired output GSD in metres/pixel
        margin        : fractional margin added around footprint (default 5%)

    Returns:
        BoundingBox or None if projection fails (degenerate pose).
    """
    intr = pose.intrinsics
    W, H = intr.width, intr.height

    # Four corners of the image in pixel space
    corners_uv = [
        (0.0,   0.0),
        (W - 1, 0.0),
        (0.0,   H - 1),
        (W - 1, H - 1),
    ]

    z_ground = dsm.mean_elevation  # flat-terrain plane
    R_world_from_cam = pose.R.T   # world-from-camera rotation = R^T

    ground_points = []
    for u, v in corners_uv:
        # 1. Ray in camera space
        ray_cam = _unproject_pixel(u, v, intr)

        # 2. Ray in world space: rotate by R^T
        ray_world = R_world_from_cam @ ray_cam

        # 3. Intersect with Z = z_ground plane
        pt = _ray_plane_intersect(pose.camera_center, ray_world, z_ground)
        if pt is None:
            return None  # degenerate pose
        ground_points.append(pt)

    xs = [p[0] for p in ground_points]
    ys = [p[1] for p in ground_points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Add a margin so edge pixels aren't clipped
    dx = (max_x - min_x) * margin
    dy = (max_y - min_y) * margin
    min_x -= dx;  max_x += dx
    min_y -= dy;  max_y += dy

    # Sanity: footprint must overlap DSM bounds
    dsm_min_x = dsm.origin_x
    dsm_max_x = dsm.origin_x + dsm.width  * dsm.pixel_width
    dsm_min_y = dsm.origin_y - dsm.height * dsm.pixel_height
    dsm_max_y = dsm.origin_y

    if max_x < dsm_min_x or min_x > dsm_max_x:
        return None
    if max_y < dsm_min_y or min_y > dsm_max_y:
        return None

    # Clip to DSM extent so the output tile doesn't extend into empty space
    min_x = max(min_x, dsm_min_x)
    max_x = min(max_x, dsm_max_x)
    min_y = max(min_y, dsm_min_y)
    max_y = min(max_y, dsm_max_y)

    return BoundingBox(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        gsd_m=target_gsd_m,
    )
