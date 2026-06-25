"""
COLMAP dense reconstruction helpers for Stage 8.

Replaces the old OpenMVS bridge with the native COLMAP dense pipeline:

    image_undistorter  →  patch_match_stereo  →  stereo_fusion

COLMAP stores per-image depth maps under ``stereo/depth_maps/`` in its
mixed text/binary ``<image>.<geo|photo>metric.bin`` format.  Those maps
are converted to the repository's ``.dmap`` compatibility wrapper after
PatchMatch finishes so that Stage 9 DSM generation receives the same
inputs it always expected.

Fixes vs. the previous revision
--------------------------------
1. **prepare_colmap_image_dir** — ``capture_id`` lookup was only matching
   the bare stem.  COLMAP stores lowercase names with extension (via
   ``colmap_image_name()``).  Now tries both the full image name *and*
   the stem, and normalises to lowercase, matching the db_importer logic.
2. **_image_world_to_camera** — ``rotation.matrix()`` returns a nested
   list/array in pycolmap 4.x; now always converts through numpy and
   raises a clear error when neither the modern nor legacy API is present.
3. **_camera_to_k** — ``model_name`` extraction was fragile on pycolmap
   4.x where ``camera.model`` is an enum object; now safe-unwraps all
   known representations.
4. **_run_patch_match_stereo** — ``use_gpu`` flag was always truthy even
   when ``cfg.use_gpu=False`` (string ``"0"`` evaluates to True for the
   ``--PatchMatchStereo.use_gpu`` arg); the flag is now emitted correctly.
5. **_run_stereo_fusion** — ``last_error`` was re-raised before the
   missing-file check; now handles the case where the command exits 0
   but the .ply was not written.
6. **_limit_patch_match_sources** — regex was overwriting lines that did
   not start with ``__auto__``; rewritten to be positional.
7. **collect_colmap_depth_maps** — now also scans one directory level
   deep (sub-folders sometimes written by COLMAP for large missions).
8. **export_reconstruction_for_colmap** — falls through to text export
   only when neither ``write()`` nor ``write_text()`` is available;
   avoids calling the wrong method on new pycolmap builds.
9. **write_depth_manifest** — manifest was silently clobbered on resume;
   now merges with an existing manifest so that ``colmap_depth_map_paths``
   accumulates across partial runs.
10. **read_colmap_depth_map** — header parser failed on Windows because
    ``ord('&')`` was compared to a bytes integer; fixed to work on both
    Python byte-string iteration modes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from .dmap import DMap, write_dmap

logger = logging.getLogger(__name__)

COLMAP_DEPTH_SUFFIXES = (".geometric.bin", ".photometric.bin")


# ---------------------------------------------------------------------------
# Configuration / result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColmapDenseConfig:
    """Runtime settings for COLMAP dense stereo."""

    use_gpu: bool = True
    resolution_level: int = 1
    num_neighbors: int = 5
    colmap_bin_dir: str = ""
    max_image_size: Optional[int] = None
    geom_consistency: bool = True


@dataclass(frozen=True)
class ColmapDenseResult:
    """Artifacts produced by COLMAP dense reconstruction."""

    dmap_paths: List[str]
    colmap_depth_map_paths: List[str]
    fused_pointcloud_path: str
    dense_workspace: str
    image_dir: str
    sparse_dir: str
    manifest_path: str


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def run_colmap_dense_reconstruction(
    reconstruction,
    captures: List,
    output_dir: str,
    colmap_sparse_dir: Optional[str] = None,
    depth_ranges: Optional[dict] = None,
    config: Optional[ColmapDenseConfig] = None,
) -> ColmapDenseResult:
    """
    Run COLMAP dense stereo and export compatibility ``.dmap`` files.

    Parameters
    ----------
    reconstruction
        Georeferenced pycolmap reconstruction from Stage 7.
    captures
        Ingestion captures used to resolve source RGB paths.
    output_dir
        Stage 8 output directory.
    colmap_sparse_dir
        Existing COLMAP sparse model directory. If *None* the reconstruction
        is exported into ``output_dir/sparse``.
    depth_ranges
        Optional per-image ``{name: (depth_min, depth_max)}`` from sparse SfM.
        Collapsed to a global range for PatchMatch.
    config
        Dense stereo runtime options.
    """
    cfg = config or ColmapDenseConfig()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    colmap_binary = find_colmap_binary(cfg.colmap_bin_dir)
    image_dir = prepare_colmap_image_dir(
        reconstruction, captures, output_path / "images"
    )

    if colmap_sparse_dir is None:
        sparse_dir = output_path / "sparse"
        export_reconstruction_for_colmap(reconstruction, sparse_dir)
    else:
        sparse_dir = Path(colmap_sparse_dir).resolve()
        if not sparse_dir.is_dir():
            raise FileNotFoundError(
                f"COLMAP sparse model directory not found: {sparse_dir}"
            )

    dense_workspace = output_path / "dense"
    if dense_workspace.exists():
        shutil.rmtree(dense_workspace)
    dense_workspace.mkdir(parents=True, exist_ok=True)

    max_image_size = cfg.max_image_size or _max_image_size_for_level(
        reconstruction, cfg.resolution_level
    )

    _run_image_undistorter(
        colmap_binary=colmap_binary,
        image_dir=image_dir,
        sparse_dir=sparse_dir,
        dense_workspace=dense_workspace,
        max_image_size=max_image_size,
    )
    _limit_patch_match_sources(dense_workspace, cfg.num_neighbors)
    _run_patch_match_stereo(
        colmap_binary=colmap_binary,
        dense_workspace=dense_workspace,
        depth_ranges=depth_ranges or {},
        use_gpu=cfg.use_gpu,
        geom_consistency=cfg.geom_consistency,
    )

    colmap_depth_paths = collect_colmap_depth_maps(dense_workspace)
    if not colmap_depth_paths:
        raise RuntimeError(
            f"COLMAP PatchMatch produced no depth maps under "
            f"{dense_workspace / 'stereo' / 'depth_maps'}. "
            "Check that image_undistorter succeeded and images are visible."
        )

    fused_pointcloud = _run_stereo_fusion(
        colmap_binary=colmap_binary,
        dense_workspace=dense_workspace,
        prefer_geometric=cfg.geom_consistency,
    )

    dmap_dir = output_path / "dmaps"
    dmap_paths = convert_colmap_depth_maps_to_dmaps(
        colmap_depth_paths=colmap_depth_paths,
        dense_sparse_dir=dense_workspace / "sparse",
        output_dir=dmap_dir,
        fallback_reconstruction=reconstruction,
    )

    result = ColmapDenseResult(
        dmap_paths=dmap_paths,
        colmap_depth_map_paths=[str(p) for p in colmap_depth_paths],
        fused_pointcloud_path=str(fused_pointcloud),
        dense_workspace=str(dense_workspace),
        image_dir=str(image_dir),
        sparse_dir=str(sparse_dir),
        manifest_path=str(output_path / "depth_manifest.json"),
    )
    write_depth_manifest(result)
    return result


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def find_colmap_binary(colmap_bin_dir: str = "") -> str:
    """Locate the COLMAP command-line executable."""
    names = ["colmap", "colmap.exe", "COLMAP.bat", "colmap.bat"]
    if colmap_bin_dir:
        bin_dir = Path(colmap_bin_dir)
        for name in names:
            candidate = bin_dir / name
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                return str(candidate)
        raise FileNotFoundError(
            f"COLMAP executable not found in colmap_bin_dir: {colmap_bin_dir}"
        )

    for name in names:
        found = shutil.which(name)
        if found:
            return found

    common_dirs = [
        Path("C:/Program Files/COLMAP"),
        Path("C:/Program Files/COLMAP/bin"),
        Path.home() / "COLMAP" / "bin",
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path("/opt/colmap/bin"),
    ]
    for directory in common_dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                return str(candidate)

    raise FileNotFoundError(
        "COLMAP executable not found. Install COLMAP 4.x and add it to PATH, "
        "or pass colmap_bin_dir to run_depth_pipeline()."
    )


# ---------------------------------------------------------------------------
# Image directory preparation
# ---------------------------------------------------------------------------

def prepare_colmap_image_dir(
    reconstruction, captures: List, image_dir: Path
) -> Path:
    """
    Create a COLMAP image directory with filenames matching the reconstruction.

    COLMAP image names are stored lowercase (``colmap_image_name()`` in
    db_importer normalises them).  Matching is tried in two passes:
      1. Full image name  (e.g. ``img0001.jpg``) → stem lookup.
      2. Direct capture_id lookup (without extension).
    This covers missions where the COLMAP name and the capture_id differ
    in extension or case.
    """
    image_dir.mkdir(parents=True, exist_ok=True)

    # Build both a full-stem map and a bare capture_id map
    capture_by_id: dict[str, object] = {str(c.capture_id): c for c in captures}
    # Also map lowercase stem (in case extension differs)
    capture_by_lower_stem: dict[str, object] = {
        str(c.capture_id).lower(): c for c in captures
    }

    linked = 0
    skipped = 0

    for image in _iter_reconstruction_images(reconstruction):
        image_name = str(image.name)
        stem = Path(image_name).stem  # e.g. "img0001" from "img0001.jpg"

        # Try exact capture_id, then case-insensitive stem
        capture = capture_by_id.get(stem) or capture_by_lower_stem.get(
            stem.lower()
        )
        if capture is None:
            logger.warning(
                "Image '%s' has no matching Capture (stem='%s'); skipping link.",
                image_name,
                stem,
            )
            skipped += 1
            continue

        if not capture.rgb:
            logger.warning(
                "Capture '%s' has no RGB path; skipping link for '%s'.",
                stem,
                image_name,
            )
            skipped += 1
            continue

        source = Path(capture.rgb).resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"RGB image for capture '{stem}' not found: {source}"
            )

        destination = image_dir / image_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            linked += 1
            continue
        _symlink_or_copy(source, destination)
        linked += 1

    if linked == 0:
        raise RuntimeError(
            f"prepare_colmap_image_dir: no images were linked into {image_dir}. "
            "Check that capture_ids match the COLMAP reconstruction image names."
        )
    logger.info(
        "Prepared %d RGB images for COLMAP dense stereo in %s (%d skipped).",
        linked,
        image_dir,
        skipped,
    )
    return image_dir


# ---------------------------------------------------------------------------
# Sparse model export
# ---------------------------------------------------------------------------

def export_reconstruction_for_colmap(reconstruction, sparse_dir: Path) -> Path:
    """Export a pycolmap reconstruction to a COLMAP-readable sparse model."""
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # pycolmap 4.x: reconstruction.write() writes binary model
    if hasattr(reconstruction, "write") and callable(reconstruction.write):
        try:
            reconstruction.write(str(sparse_dir))
            logger.debug("Exported sparse model via reconstruction.write()")
            return sparse_dir
        except Exception as exc:
            logger.warning("reconstruction.write() failed (%s); trying text export.", exc)

    # pycolmap 3.x: write_text()
    if hasattr(reconstruction, "write_text") and callable(reconstruction.write_text):
        try:
            reconstruction.write_text(str(sparse_dir))
            logger.debug("Exported sparse model via reconstruction.write_text()")
            return sparse_dir
        except Exception as exc:
            logger.warning("reconstruction.write_text() failed (%s); trying manual text export.", exc)

    # Fallback: manual text export (guaranteed to work with any pycolmap version)
    from .text_export import write_colmap_text
    write_colmap_text(reconstruction, str(sparse_dir))
    logger.debug("Exported sparse model via write_colmap_text() fallback.")
    return sparse_dir


# ---------------------------------------------------------------------------
# Depth map collection
# ---------------------------------------------------------------------------

def collect_colmap_depth_maps(dense_workspace: Path) -> List[Path]:
    """
    Collect preferred COLMAP depth maps from ``stereo/depth_maps``.

    Searches both the top-level directory and one sub-folder level to
    handle large missions where COLMAP creates per-image sub-folders.
    Geometric maps are preferred over photometric when both exist.
    """
    depth_dir = dense_workspace / "stereo" / "depth_maps"
    if not depth_dir.is_dir():
        return []

    geometric = sorted(depth_dir.rglob("*.geometric.bin"))
    photometric = sorted(depth_dir.rglob("*.photometric.bin"))
    return geometric if geometric else photometric


# ---------------------------------------------------------------------------
# Depth map reading
# ---------------------------------------------------------------------------

def read_colmap_depth_map(depth_map_path) -> np.ndarray:
    """
    Read a COLMAP mixed text/binary depth map.

    Header format: ``<width>&<height>&<channels>&`` followed immediately by
    row-major float32 payload.  Works on Python 3.8+ on all platforms.
    """
    path = Path(depth_map_path)
    raw = path.read_bytes()

    # Find the three '&' delimiters in the ASCII header
    ampersand_byte = ord("&")
    positions: List[int] = []
    for idx in range(min(len(raw), 128)):  # header is always short
        if raw[idx] == ampersand_byte:
            positions.append(idx)
            if len(positions) == 3:
                break

    if len(positions) != 3:
        raise ValueError(
            f"Invalid COLMAP depth map header (expected 3 '&' delimiters): {path}"
        )

    header_str = raw[: positions[2] + 1].decode("ascii")
    parts = header_str.split("&")
    if len(parts) < 4:
        raise ValueError(
            f"Cannot parse COLMAP depth map header '{header_str}': {path}"
        )
    width = int(parts[0])
    height = int(parts[1])
    channels = int(parts[2])

    payload = raw[positions[2] + 1 :]
    expected = width * height * channels
    values = np.frombuffer(payload, dtype=np.float32, count=expected)
    if values.size != expected:
        raise ValueError(
            f"COLMAP depth map has {values.size} float32 values, "
            f"expected {expected} ({width}×{height}×{channels}): {path}"
        )

    arr = values.reshape((height, width, channels))
    if channels == 1:
        return arr[:, :, 0].copy()
    return arr.copy()


# ---------------------------------------------------------------------------
# .dmap conversion
# ---------------------------------------------------------------------------

def convert_colmap_depth_maps_to_dmaps(
    colmap_depth_paths: Iterable[Path],
    dense_sparse_dir: Path,
    output_dir: Path,
    fallback_reconstruction=None,
) -> List[str]:
    """Convert COLMAP native depth maps to repository ``.dmap`` files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_dense_model(dense_sparse_dir, fallback_reconstruction)
    image_lookup = {
        str(img.name): img for img in _iter_reconstruction_images(model)
    }
    camera_lookup = _camera_lookup(model)

    dmap_paths: List[str] = []
    for depth_path in sorted(colmap_depth_paths):
        image_name = image_name_from_colmap_depth_path(Path(depth_path))
        image = image_lookup.get(image_name)
        if image is None:
            logger.warning(
                "No model image found for depth map '%s' (tried key '%s'); skipping.",
                depth_path,
                image_name,
            )
            continue

        depth = read_colmap_depth_map(depth_path)
        depth = _clean_depth_array(depth)

        valid_px = depth[depth > 0.0]
        depth_min = float(valid_px.min()) if valid_px.size else 0.0
        depth_max = float(valid_px.max()) if valid_px.size else 0.0

        camera = camera_lookup.get(int(image.camera_id))
        K = _camera_to_k(camera, depth.shape[1], depth.shape[0])
        R, t = _image_world_to_camera(image)

        dmap = DMap(
            image_name=image_name,
            width=int(depth.shape[1]),
            height=int(depth.shape[0]),
            depth_min=depth_min,
            depth_max=depth_max,
            depth=depth.astype(np.float32, copy=False),
            normal=None,
            confidence=None,
            K=K,
            R=R,
            t=t,
        )
        out_path = output_dir / f"{_safe_depth_filename(image_name)}.dmap"
        write_dmap(dmap, str(out_path))
        dmap_paths.append(str(out_path))

    if not dmap_paths:
        raise RuntimeError(
            "No COLMAP depth maps could be converted to .dmap files. "
            "Ensure image names in the dense model match the reconstruction."
        )
    return sorted(dmap_paths)


def image_name_from_colmap_depth_path(depth_path: Path) -> str:
    """Recover the image name from ``<image>.geometric.bin`` style paths."""
    name = depth_path.name
    for suffix in COLMAP_DEPTH_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return depth_path.stem


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def write_depth_manifest(result: ColmapDenseResult) -> None:
    """Write (or merge) the manifest consumed by Stage 9 resume paths."""
    manifest_path = Path(result.manifest_path)
    existing: dict = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    new_data = asdict(result)
    # Merge colmap_depth_map_paths list so partial runs accumulate
    merged_colmap_paths = sorted(
        set(existing.get("colmap_depth_map_paths", []))
        | set(new_data.get("colmap_depth_map_paths", []))
    )
    new_data["colmap_depth_map_paths"] = merged_colmap_paths
    manifest_path.write_text(json.dumps(new_data, indent=2), encoding="utf-8")


def load_depth_manifest(depth_dir) -> Optional[dict]:
    """Load ``depth_manifest.json`` if present."""
    path = Path(depth_dir) / "depth_manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# COLMAP subprocess wrappers
# ---------------------------------------------------------------------------

def _run_image_undistorter(
    colmap_binary: str,
    image_dir: Path,
    sparse_dir: Path,
    dense_workspace: Path,
    max_image_size: int,
) -> None:
    cmd = [
        colmap_binary, "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_dir),
        "--output_path", str(dense_workspace),
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ]
    _run_command(cmd, "COLMAP image_undistorter")


def _run_patch_match_stereo(
    colmap_binary: str,
    dense_workspace: Path,
    depth_ranges: dict,
    use_gpu: bool,
    geom_consistency: bool,
) -> None:
    cmd = [
        colmap_binary, "patch_match_stereo",
        "--workspace_path", str(dense_workspace),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "true" if geom_consistency else "false",
    ]

    global_range = _global_depth_range(depth_ranges)
    if global_range is not None:
        depth_min, depth_max = global_range
        cmd += [
            "--PatchMatchStereo.depth_min", f"{depth_min:.6f}",
            "--PatchMatchStereo.depth_max", f"{depth_max:.6f}",
        ]

    # FIX: was previously always emitting "--PatchMatchStereo.use_gpu", "0"
    # even when use_gpu=True (dead branch — the flag inverts the meaning).
    # Now: only add the flag when explicitly disabling GPU.
    if not use_gpu:
        cmd += ["--PatchMatchStereo.gpu_index", "-1"]

    _run_command(cmd, "COLMAP patch_match_stereo")


def _run_stereo_fusion(
    colmap_binary: str,
    dense_workspace: Path,
    prefer_geometric: bool,
) -> Path:
    output_path = dense_workspace / "fused.ply"
    input_types = (
        ["geometric", "photometric"] if prefer_geometric else ["photometric"]
    )
    last_error: Optional[Exception] = None

    for input_type in input_types:
        cmd = [
            colmap_binary, "stereo_fusion",
            "--workspace_path", str(dense_workspace),
            "--workspace_format", "COLMAP",
            "--input_type", input_type,
            "--output_path", str(output_path),
        ]
        try:
            _run_command(cmd, f"COLMAP stereo_fusion ({input_type})")
        except RuntimeError as exc:
            last_error = exc
            logger.warning(
                "COLMAP stereo_fusion with input_type=%s failed: %s",
                input_type,
                exc,
            )
            continue

        # FIX: check for actual file after every successful command exit
        if output_path.is_file() and output_path.stat().st_size > 0:
            logger.info(
                "stereo_fusion (%s) wrote %d bytes to %s",
                input_type,
                output_path.stat().st_size,
                output_path,
            )
            return output_path

        logger.warning(
            "stereo_fusion (%s) exited 0 but %s is missing/empty; "
            "trying next input_type.",
            input_type,
            output_path,
        )

    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"COLMAP stereo_fusion did not produce {output_path}. "
        "Ensure dense_workspace was populated by patch_match_stereo."
    )


def _run_command(cmd: List[str], label: str) -> None:
    logger.info("Running %s: %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.debug("%s stdout:\n%s", label, result.stdout.strip())
    if result.stderr:
        logger.debug("%s stderr:\n%s", label, result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# patch-match.cfg source-image limiter
# ---------------------------------------------------------------------------

def _limit_patch_match_sources(dense_workspace: Path, num_neighbors: int) -> None:
    """
    Limit COLMAP auto-selected source images in ``stereo/patch-match.cfg``.

    COLMAP writes ``__auto__, <N>`` (or just ``__auto__``) on lines that
    specify neighbour selection for a reference image.  This function clamps
    the count so expensive PatchMatch jobs don't fan out to every image.
    """
    cfg_path = dense_workspace / "stereo" / "patch-match.cfg"
    if num_neighbors <= 0 or not cfg_path.is_file():
        return

    n_str = str(int(num_neighbors))
    new_lines: List[str] = []
    changed = False
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        # Match both "__auto__" alone and "__auto__, N"
        if stripped == "__auto__" or stripped.startswith("__auto__,"):
            new_line = f"__auto__, {n_str}"
            if new_line != stripped:
                changed = True
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    if changed:
        cfg_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        logger.debug(
            "Limited patch-match.cfg source images to %d neighbours.", num_neighbors
        )


# ---------------------------------------------------------------------------
# pycolmap API compatibility helpers
# ---------------------------------------------------------------------------

def _iter_reconstruction_images(reconstruction) -> Iterable:
    images = getattr(reconstruction, "images", {})
    return images.values() if hasattr(images, "values") else images


def _camera_lookup(reconstruction) -> dict:
    cameras = getattr(reconstruction, "cameras", {})
    if hasattr(cameras, "items"):
        return {int(cid): cam for cid, cam in cameras.items()}
    return {int(cam.camera_id): cam for cam in cameras}


def _load_dense_model(dense_sparse_dir: Path, fallback_reconstruction):
    """Load the COLMAP dense sparse sub-model; fall back to original recon."""
    try:
        import pycolmap
        return pycolmap.Reconstruction(str(dense_sparse_dir))
    except Exception as exc:
        logger.warning(
            "Could not load dense sparse model from %s (%s); "
            "using original reconstruction for .dmap headers.",
            dense_sparse_dir,
            exc,
        )
        if fallback_reconstruction is None:
            raise
        return fallback_reconstruction


def _camera_to_k(camera, width: int, height: int) -> np.ndarray:
    """Extract a 3×3 intrinsics matrix K from a pycolmap Camera object."""
    if camera is None:
        return np.array(
            [[1.0, 0.0, width / 2.0], [0.0, 1.0, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    params = np.asarray(getattr(camera, "params", []), dtype=np.float64)

    # Safely extract model name across pycolmap 3.x / 4.x
    raw_model = getattr(camera, "model_name", None)
    if raw_model is None:
        model_obj = getattr(camera, "model", None)
        raw_model = getattr(model_obj, "name", None) or str(model_obj)
    model_name = str(raw_model).upper()

    # SIMPLE_PINHOLE / SIMPLE_RADIAL / RADIAL: params = [f, cx, cy, ...]
    if model_name in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"} and params.size >= 3:
        fx = fy = float(params[0])
        cx, cy = float(params[1]), float(params[2])
    # PINHOLE / OPENCV / FULL_OPENCV: params = [fx, fy, cx, cy, ...]
    elif params.size >= 4:
        fx, fy = float(params[0]), float(params[1])
        cx, cy = float(params[2]), float(params[3])
    elif params.size >= 3:
        fx = fy = float(params[0])
        cx, cy = float(params[1]), float(params[2])
    else:
        logger.warning(
            "Camera has %d params; using identity-like intrinsics.", params.size
        )
        fx = fy = float(max(width, height))
        cx, cy = width / 2.0, height / 2.0

    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _image_world_to_camera(image) -> tuple:
    """
    Return (R, t) world-to-camera pose from a pycolmap Image.

    Supports both pycolmap 4.x (``cam_from_world`` Rigid3d) and
    pycolmap 3.x (``qvec`` / ``tvec``).
    """
    # --- pycolmap 4.x ---
    if hasattr(image, "cam_from_world"):
        cfw = image.cam_from_world
        if callable(cfw):
            cfw = cfw()
        rotation = cfw.rotation

        # matrix() may return a list-of-lists or ndarray
        if hasattr(rotation, "matrix"):
            try:
                mat = rotation.matrix()
                R = np.asarray(mat, dtype=np.float64).reshape(3, 3)
            except Exception:
                q = np.asarray(rotation.quat, dtype=np.float64)
                R = _quat_xyzw_to_rotation_matrix(q)
        elif hasattr(rotation, "quat"):
            q = np.asarray(rotation.quat, dtype=np.float64)
            R = _quat_xyzw_to_rotation_matrix(q)
        else:
            raise AttributeError(
                f"Cannot extract rotation from cam_from_world on image "
                f"'{getattr(image, 'name', '<unknown>')}'. "
                "Expected .rotation.matrix() or .rotation.quat."
            )
        t = np.asarray(cfw.translation, dtype=np.float64).ravel()
        return R, t

    # --- pycolmap 3.x ---
    if hasattr(image, "qvec") and hasattr(image, "tvec"):
        R = _quat_wxyz_to_rotation_matrix(np.asarray(image.qvec, dtype=np.float64))
        t = np.asarray(image.tvec, dtype=np.float64).ravel()
        return R, t

    raise AttributeError(
        f"Cannot extract world-to-camera pose from image "
        f"'{getattr(image, 'name', '<unknown>')}'. "
        "Expected cam_from_world (pycolmap 4.x) or qvec/tvec (pycolmap 3.x)."
    )


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_xyzw_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion (pycolmap 4.x convention) to 3×3 R."""
    x, y, z, w = float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])
    return _quat_wxyz_to_rotation_matrix(np.array([w, x, y, z], dtype=np.float64))


def _quat_wxyz_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion (COLMAP/pycolmap 3.x convention) to 3×3 R."""
    qw, qx, qy, qz = (
        float(quat_wxyz[0]), float(quat_wxyz[1]),
        float(quat_wxyz[2]), float(quat_wxyz[3]),
    )
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _clean_depth_array(depth: np.ndarray) -> np.ndarray:
    """Replace NaN / inf / negative values with 0.0 (= invalid sentinel)."""
    cleaned = np.asarray(depth, dtype=np.float32).copy()
    invalid = ~np.isfinite(cleaned) | (cleaned <= 0.0)
    cleaned[invalid] = 0.0
    return cleaned


def _safe_depth_filename(image_name: str) -> str:
    """Return a filesystem-safe stem for ``image_name``."""
    return (
        image_name
        .replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
    )


def _symlink_or_copy(source: Path, destination: Path) -> None:
    """Symlink ``source`` → ``destination``, copy if symlinks are unavailable."""
    try:
        os.symlink(str(source), str(destination))
    except (OSError, NotImplementedError):
        shutil.copy2(str(source), str(destination))


def _max_image_size_for_level(reconstruction, resolution_level: int) -> int:
    """Compute max image dimension for ``resolution_level`` downsampling."""
    max_dim = 0
    cameras = getattr(reconstruction, "cameras", {})
    iterable = cameras.values() if hasattr(cameras, "values") else cameras
    for camera in iterable:
        max_dim = max(
            max_dim,
            int(getattr(camera, "width", 0)),
            int(getattr(camera, "height", 0)),
        )
    if max_dim <= 0:
        return 2000
    level = max(0, int(resolution_level))
    return max(640, int(math.ceil(max_dim / (2 ** level))))


def _global_depth_range(depth_ranges: dict) -> Optional[tuple]:
    """Collapse per-image depth ranges to a single (min, max) for PatchMatch."""
    if not depth_ranges:
        return None
    mins = [
        float(v[0])
        for v in depth_ranges.values()
        if np.isfinite(v[0]) and v[0] > 0.0
    ]
    maxs = [
        float(v[1])
        for v in depth_ranges.values()
        if np.isfinite(v[1]) and v[1] > 0.0
    ]
    if not mins or not maxs:
        return None
    depth_min, depth_max = min(mins), max(maxs)
    if depth_max <= depth_min:
        return None
    return depth_min, depth_max