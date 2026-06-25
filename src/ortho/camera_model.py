"""
src/ortho/camera_model.py

Extracts camera intrinsics and pose from the pycolmap.Reconstruction (built
in Stage 6/7) into plain dataclasses of NumPy arrays — the boundary where
Stage 10 stops depending on pycolmap's object model and switches to arrays
that can be uploaded to the GPU.

COLMAP convention used throughout this module and backward_project.py:
    P_cam = R @ P_world + t          (cam_from_world)
    camera_center = -R.T @ t         (camera position in world coordinates)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# COLMAP OPENCV camera model parameter order: fx, fy, cx, cy, k1, k2, p1, p2
_OPENCV_PARAM_ORDER = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


@dataclass
class CameraPose:
    image_id: int
    image_name: str
    R: np.ndarray               # (3, 3) world-to-camera rotation
    t: np.ndarray                # (3,)   world-to-camera translation
    camera_center: np.ndarray    # (3,)   camera position in world coords
    intrinsics: CameraIntrinsics


def _rotation_matrix_from_image(image) -> np.ndarray:
    """
    Get the 3x3 world-to-camera rotation matrix from a pycolmap.Image,
    supporting both the modern (cam_from_world.rotation.matrix()) and
    legacy (qvec / Rotation3d-less) pycolmap APIs.
    """
    if hasattr(image, "cam_from_world"):
        cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        rotation = cfw.rotation
        if hasattr(rotation, "matrix"):
            return np.asarray(rotation.matrix(), dtype=np.float64)
        # Some pycolmap builds expose .quat (x, y, z, w) instead.
        return _quat_to_matrix(np.asarray(rotation.quat, dtype=np.float64))

    # Legacy pycolmap: image.qvec is (w, x, y, z).
    if hasattr(image, "qvec"):
        qw, qx, qy, qz = np.asarray(image.qvec, dtype=np.float64)
        return _quat_to_matrix(np.array([qx, qy, qz, qw]))

    raise AttributeError(
        f"Don't know how to extract rotation from pycolmap.Image of type "
        f"{type(image)} — no cam_from_world or qvec attribute found."
    )


def _translation_from_image(image) -> np.ndarray:
    if hasattr(image, "cam_from_world"):
        cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        return np.asarray(cfw.translation, dtype=np.float64)
    if hasattr(image, "tvec"):
        return np.asarray(image.tvec, dtype=np.float64)
    raise AttributeError(
        f"Don't know how to extract translation from pycolmap.Image of type "
        f"{type(image)} — no cam_from_world or tvec attribute found."
    )


def _quat_to_matrix(xyzw: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = xyzw
    n = x * x + y * y + z * z + w * w
    if n < np.finfo(np.float64).eps:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def _intrinsics_from_camera(camera) -> CameraIntrinsics:
    """
    Build CameraIntrinsics from a pycolmap.Camera. Supports the OPENCV
    model (fx, fy, cx, cy, k1, k2, p1, p2), which is what the Sequoia
    multispectral camera is calibrated with (per dsm_stage_context.md /
    project conventions). Falls back to a pinhole-only read (no distortion)
    for SIMPLE_PINHOLE/PINHOLE models, which have fewer params.
    """
    params = np.asarray(camera.params, dtype=np.float64)
    model_name = getattr(camera, "model_name", None) or str(getattr(camera, "model", ""))

    width = int(camera.width)
    height = int(camera.height)

    if "OPENCV" in model_name and len(params) >= 8:
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        return CameraIntrinsics(fx, fy, cx, cy, width, height, k1, k2, p1, p2)

    if "PINHOLE" in model_name and len(params) == 4:
        fx, fy, cx, cy = params
        return CameraIntrinsics(fx, fy, cx, cy, width, height)

    if "SIMPLE_PINHOLE" in model_name and len(params) == 3:
        f, cx, cy = params
        return CameraIntrinsics(f, f, cx, cy, width, height)

    # Last resort: assume the first 4 params are fx, fy, cx, cy and warn by
    # ignoring distortion rather than crashing on an unrecognised model.
    if len(params) >= 4:
        fx, fy, cx, cy = params[:4]
        k1 = params[4] if len(params) > 4 else 0.0
        k2 = params[5] if len(params) > 5 else 0.0
        p1 = params[6] if len(params) > 6 else 0.0
        p2 = params[7] if len(params) > 7 else 0.0
        return CameraIntrinsics(fx, fy, cx, cy, width, height, k1, k2, p1, p2)

    raise ValueError(f"Unrecognised camera model '{model_name}' with params {params}")


def extract_camera_poses(reconstruction) -> List[CameraPose]:
    """
    Parse a pycolmap.Reconstruction and return one CameraPose per
    registered image, sorted by image_id.
    """
    poses: List[CameraPose] = []

    for image_id, image in reconstruction.images.items():
        if hasattr(image, "registered") and not image.registered:
            continue

        camera_id = image.camera_id
        camera = reconstruction.cameras[camera_id]
        intrinsics = _intrinsics_from_camera(camera)

        R = _rotation_matrix_from_image(image)
        t = _translation_from_image(image)
        camera_center = -R.T @ t

        poses.append(
            CameraPose(
                image_id=int(image_id),
                image_name=str(image.name),
                R=R,
                t=t,
                camera_center=camera_center,
                intrinsics=intrinsics,
            )
        )

    poses.sort(key=lambda p: p.image_id)
    return poses


def intrinsics_to_array(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Pack intrinsics into a flat float64 array: [fx, fy, cx, cy, k1, k2, p1, p2]."""
    return np.array(
        [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
         intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2],
        dtype=np.float64,
    )


def intrinsics_to_cupy(intrinsics: CameraIntrinsics):
    """Pack intrinsics into a flat CuPy (or NumPy fallback) float64 array."""
    from ._xp import xp

    return xp.asarray(intrinsics_to_array(intrinsics))