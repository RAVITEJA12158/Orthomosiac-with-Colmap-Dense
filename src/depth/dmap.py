"""
dmap.py
=======
Read and write the pipeline .dmap depth-map compatibility format.

This module is the **explicit boundary** between Stage 8 (depth
estimation) and Stage 9 (DSM fusion). Stage 9 reads .dmap files and
does not care who produced them.  This abstraction means a future swap
from one dense stereo backend to another only requires converting backend output into .dmap
format â€” zero changes downstream.

.dmap Binary Format
-------------------
All values little-endian.

Header (fixed-size section):
    uint32      content_type
                    1 = depth only
                    3 = depth + normals + confidence
    uint32      reserved           (always 0)
    uint32      width
    uint32      height
    float32     depth_min
    float32     depth_max
    char[64]    image_name         (null-terminated, zero-padded to 64 bytes)
    float32[4]  intrinsics         (fx, fy, cx, cy)
    float32[9]  rotation           (3Ã—3 row-major R matrix)
    float32[3]  translation        (tx, ty, tz)

Data (immediately after header):
    float32[H Ã— W]      depth values in metres   (0.0 = invalid / no data)
    float32[H Ã— W Ã— 3] normal vectors            (only if content_type == 3)
    float32[H Ã— W]      confidence scores         (only if content_type == 3)

Reference: OpenMVS source â€” libs/MVS/DepthMap.h, struct DepthData.

Usage
-----
    from src.depth.dmap import DMap, read_dmap, write_dmap

    # Read back a file produced by OpenMVS
    dmap = read_dmap("depthmaps/00123.dmap")
    print(dmap.valid_pixel_ratio())     # fraction of pixels with depth > 0
    print(dmap.depth_stats())           # min/max/mean/std of valid depths

    # Write a DMap (e.g. for testing, or future ACMMP conversion)
    write_dmap(dmap, "/tmp/roundtrip.dmap")
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants â€” .dmap content_type values
# ---------------------------------------------------------------------------

CONTENT_DEPTH_ONLY: int = 1
"""content_type value: file contains depth array only."""

CONTENT_DEPTH_NORMALS_CONFIDENCE: int = 3
"""content_type value: file contains depth + normals + confidence arrays."""

# Header layout for struct.pack / struct.unpack
# < = little-endian
# I I I I f f  = uint32Ã—4, float32Ã—2
# 64s           = 64-byte char array (image_name)
# 4f            = 4 floats (intrinsics: fx, fy, cx, cy)
# 9f            = 9 floats (rotation matrix, row-major)
# 3f            = 3 floats (translation)
_HEADER_FMT = "<IIIIff64s4f9f3f"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # bytes


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DMap:
    """
    Represents a single per-image depth map in the pipeline compatibility format.

    Attributes
    ----------
    image_name : str
        Original RGB image filename (e.g. "000.jpg").
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    depth_min : float
        Minimum valid depth stored in this map (metres).
    depth_max : float
        Maximum valid depth stored in this map (metres).
    depth : np.ndarray
        float32 array of shape (H, W). Value 0.0 means invalid / no
        depth estimate for that pixel.
    normal : np.ndarray or None
        float32 array of shape (H, W, 3), or None if not present.
    confidence : np.ndarray or None
        float32 array of shape (H, W), or None if not present.
    K : np.ndarray
        (3, 3) float64 camera intrinsics matrix::

            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

    R : np.ndarray
        (3, 3) float64 world-to-camera rotation matrix.
    t : np.ndarray
        (3,) float64 world-to-camera translation vector.
    """

    image_name: str
    width: int
    height: int
    depth_min: float
    depth_max: float
    depth: np.ndarray                    # float32, shape (H, W)
    normal: Optional[np.ndarray]         # float32, shape (H, W, 3) or None
    confidence: Optional[np.ndarray]     # float32, shape (H, W) or None
    K: np.ndarray                        # float64, shape (3, 3)
    R: np.ndarray                        # float64, shape (3, 3)
    t: np.ndarray                        # float64, shape (3,)

    # ------------------------------------------------------------------
    # Quality helpers
    # ------------------------------------------------------------------

    def valid_pixel_ratio(self) -> float:
        """
        Fraction of pixels that have a valid depth estimate.

        A pixel is *invalid* when its depth value is exactly 0.0
        (.dmap convention for "no data").

        Returns
        -------
        float
            Value in [0.0, 1.0]. 1.0 means every pixel has depth data.
        """
        total = self.depth.size
        if total == 0:
            return 0.0
        valid = int(np.count_nonzero(self.depth))
        return valid / total

    def depth_stats(self) -> dict:
        """
        Descriptive statistics for *valid* depth pixels.

        Returns
        -------
        dict with keys:
            min, max, mean, std  â€” all in metres (float).
            valid_count          â€” number of pixels with depth > 0 (int).
            total_pixels         â€” total pixels in the depth map (int).

        Example::

            {'min': 91.2, 'max': 148.3, 'mean': 119.7,
             'std': 8.4, 'valid_count': 5834221, 'total_pixels': 6000000}
        """
        valid = self.depth[self.depth > 0.0]

        if valid.size == 0:
            return {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "valid_count": 0,
                "total_pixels": int(self.depth.size),
            }

        return {
            "min": float(valid.min()),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
            "std": float(valid.std()),
            "valid_count": int(valid.size),
            "total_pixels": int(self.depth.size),
        }


# ---------------------------------------------------------------------------
# Public I/O functions
# ---------------------------------------------------------------------------

def write_dmap(dmap: DMap, output_path: str) -> None:
    """
    Serialise a DMap to an pipeline-compatible .dmap binary file.

    Parameters
    ----------
    dmap : DMap
        The depth map to write.
    output_path : str
        Destination file path (created or overwritten).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    has_extras = (dmap.normal is not None) and (dmap.confidence is not None)
    content_type = CONTENT_DEPTH_NORMALS_CONFIDENCE if has_extras else CONTENT_DEPTH_ONLY

    # Encode image name: null-terminate, pad/truncate to 64 bytes
    name_bytes = dmap.image_name.encode("utf-8")[:63] + b"\x00"
    name_bytes = name_bytes.ljust(64, b"\x00")

    # Flatten K into (fx, fy, cx, cy)
    K = dmap.K.astype(np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    # Flatten R (3Ã—3 row-major) and t
    R_flat = dmap.R.astype(np.float64).ravel().tolist()    # 9 floats
    t_flat = dmap.t.astype(np.float64).ravel().tolist()    # 3 floats

    header = struct.pack(
        _HEADER_FMT,
        content_type,       # uint32
        0,                  # reserved uint32
        dmap.width,         # uint32
        dmap.height,        # uint32
        dmap.depth_min,     # float32
        dmap.depth_max,     # float32
        name_bytes,         # 64s
        fx, fy, cx, cy,     # 4f
        *R_flat,            # 9f
        *t_flat,            # 3f
    )

    with open(output_path, "wb") as f:
        f.write(header)
        # Depth array â€” always present
        f.write(dmap.depth.astype(np.float32).tobytes())
        if has_extras:
            f.write(dmap.normal.astype(np.float32).tobytes())
            f.write(dmap.confidence.astype(np.float32).tobytes())


def read_dmap(dmap_path: str) -> DMap:
    """
    Deserialise an pipeline .dmap binary file into a DMap dataclass.

    Parameters
    ----------
    dmap_path : str
        Path to the .dmap file.

    Returns
    -------
    DMap

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is too small to contain a valid header, or if
        content_type is unrecognised.
    """
    path = Path(dmap_path)
    if not path.exists():
        raise FileNotFoundError(f".dmap file not found: {dmap_path}")

    raw = path.read_bytes()
    if len(raw) < _HEADER_SIZE:
        raise ValueError(
            f"File too small to contain a valid .dmap header "
            f"(got {len(raw)} bytes, need {_HEADER_SIZE}): {dmap_path}"
        )

    # Unpack header
    (
        content_type,
        _reserved,
        width,
        height,
        depth_min,
        depth_max,
        name_bytes,
        fx, fy, cx, cy,
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22,
        tx, ty, tz,
    ) = struct.unpack_from(_HEADER_FMT, raw, offset=0)

    if content_type not in (CONTENT_DEPTH_ONLY, CONTENT_DEPTH_NORMALS_CONFIDENCE):
        raise ValueError(
            f"Unrecognised .dmap content_type {content_type} in {dmap_path}. "
            f"Expected {CONTENT_DEPTH_ONLY} or {CONTENT_DEPTH_NORMALS_CONFIDENCE}."
        )

    # Decode image name
    image_name = name_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")

    # Reconstruct K, R, t as (3,3) / (3,) float64
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    R = np.array([[r00, r01, r02],
                  [r10, r11, r12],
                  [r20, r21, r22]], dtype=np.float64)

    t = np.array([tx, ty, tz], dtype=np.float64)

    # Read pixel arrays
    n_pixels = width * height
    offset = _HEADER_SIZE

    depth_bytes = n_pixels * 4  # float32 = 4 bytes
    depth = np.frombuffer(raw[offset: offset + depth_bytes], dtype=np.float32).reshape(height, width).copy()
    offset += depth_bytes

    normal: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None

    if content_type == CONTENT_DEPTH_NORMALS_CONFIDENCE:
        normal_bytes = n_pixels * 3 * 4
        normal = np.frombuffer(
            raw[offset: offset + normal_bytes], dtype=np.float32
        ).reshape(height, width, 3).copy()
        offset += normal_bytes

        conf_bytes = n_pixels * 4
        confidence = np.frombuffer(
            raw[offset: offset + conf_bytes], dtype=np.float32
        ).reshape(height, width).copy()

    return DMap(
        image_name=image_name,
        width=width,
        height=height,
        depth_min=float(depth_min),
        depth_max=float(depth_max),
        depth=depth,
        normal=normal,
        confidence=confidence,
        K=K,
        R=R,
        t=t,
    )


def dmap_from_openmvs_output(dmap_path: str) -> DMap:
    """
    Alias for :func:`read_dmap` with semantic clarity.

    Backward-compatible alias for older callers; reads a .dmap file for validation or
    future format conversion. This name makes the call-site intention
    explicit.

    Parameters
    ----------
    dmap_path : str
        Path to the .dmap file.

    Returns
    -------
    DMap
    """
    return read_dmap(dmap_path)


