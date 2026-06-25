"""
Stage 3 — Feature Extraction
Tool : ALIKED (aliked-n16-rot) via the lightglue package
Output: one .h5 file per image written to <output_dir>/features/<capture_id>.h5

Each .h5 file contains:
    keypoints   float32 (N, 2)   pixel coordinates  [x, y]
    descriptors float32 (N, 256) ALIKED descriptors
    image_size  int32   (2,)     [width, height]

Images are processed one at a time — zero OOM risk at 900 images on 16 GB VRAM.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch

from ..ingestion.capture import Capture

# ── ALIKED model name used throughout the pipeline ──────────────────────────
ALIKED_MODEL = "aliked-n16rot"

# Max keypoints per image.  8192 is a good default for nadir drone imagery.
# More keypoints → slower LightGlue matching; fewer → weaker SfM.
DEFAULT_MAX_KEYPOINTS = 8192


# ── internal helpers ─────────────────────────────────────────────────────────

def _check_lightglue():
    try:
        from lightglue import ALIKED
        return ALIKED
    except ImportError as exc:
        raise ImportError(
            "lightglue not installed.\n"
            "Run:  pip install lightglue\n"
            "Docs: https://github.com/cvg/LightGlue"
        ) from exc


def _build_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        dev = torch.device("cuda")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[extractor] GPU: {torch.cuda.get_device_name(0)}  VRAM: {vram_gb:.1f} GB")
    else:
        if use_gpu and not torch.cuda.is_available():
            print("[extractor] Warning: CUDA not available — falling back to CPU.")
        dev = torch.device("cpu")
    return dev


def _load_image_tensor(
    image_path: str,
    device: torch.device,
    resize: Optional[int],
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Load an image as a (1, C, H, W) float32 tensor in [0, 1].
    Returns (tensor, (width, height)) of the ORIGINAL image before any resize.

    lightglue's ALIKED extractor accepts either:
        - a plain (C, H, W) tensor  — we pass this
        - a dict with 'image' key   — not needed here

    Resize is applied when the image is very large to save GPU memory.
    For 12 MP drone images a resize cap of 1600 px keeps VRAM under 2 GB.
    None = no resize (use when image is already reasonably small).
    """
    from lightglue.utils import load_image

    # load_image returns a (3, H, W) float32 tensor in [0, 1]
    img: torch.Tensor = load_image(image_path, resize=resize).to(device)
    # original width/height before any resize (for COLMAP camera model)
    import PIL.Image
    with PIL.Image.open(image_path) as pil:
        orig_w, orig_h = pil.size
    return img, (orig_w, orig_h)


def _save_features(
    h5_path: Path,
    keypoints: np.ndarray,    # (N, 2) float32
    descriptors: np.ndarray,  # (N, 256) float32
    image_size: tuple[int, int],  # (width, height)  — ORIGINAL resolution
) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("keypoints",   data=keypoints,   dtype="float32")
        f.create_dataset("descriptors", data=descriptors, dtype="float32")
        f.create_dataset("image_size",  data=np.array(list(image_size), dtype="int32"))


# ── public API ───────────────────────────────────────────────────────────────

def extract_features(
    captures: List[Capture],
    output_dir: str,
    use_gpu: bool = True,
    max_keypoints: int = DEFAULT_MAX_KEYPOINTS,
    resize: Optional[int] = 1600,
) -> Path:
    """
    Run ALIKED feature extraction on all RGB images in `captures`.

    Images are processed ONE AT A TIME.  GPU memory is released after each
    image — no OOM risk regardless of mission size.

    Args:
        captures       : Capture list from ingestion.load_mission()
        output_dir     : root output folder (features/ sub-dir created inside)
        use_gpu        : use CUDA when available
        max_keypoints  : max keypoints per image (default 8192)
        resize         : cap longest image dimension to this many pixels before
                         extraction.  None = no resize.  1600 is safe for 16 GB VRAM.

    Returns:
        Path to features/ directory containing one .h5 per capture.
    """
    ALIKED = _check_lightglue()
    device = _build_device(use_gpu)

    features_dir = Path(output_dir) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extractor] Model: {ALIKED_MODEL}  |  max_keypoints: {max_keypoints}")
    print(f"[extractor] Extracting features from {len(captures)} RGB images...")
    print(f"[extractor] Output: {features_dir}")

    # Build extractor once — reused for every image
    extractor = (
        ALIKED(model=ALIKED_MODEL, max_num_keypoints=max_keypoints)
        .eval()
        .to(device)
    )

    t0 = time.perf_counter()
    skipped = 0

    for idx, cap in enumerate(captures):
        h5_path = features_dir / f"{cap.capture_id}.h5"

        if h5_path.exists():
            # Already extracted in a previous (interrupted) run — skip.
            skipped += 1
            continue

        if not cap.rgb:
            print(f"[extractor] Warning: capture {cap.capture_id} has no RGB — skipping.")
            skipped += 1
            continue

        img_tensor, (orig_w, orig_h) = _load_image_tensor(cap.rgb, device, resize)

        with torch.no_grad():
            # extractor returns dict with keys:
            #   'keypoints'   : (1, N, 2) float  — pixel coords in resized space
            #   'descriptors' : (1, N, D) float
            #   'keypoint_scores' : (1, N) float  (confidence per kp)
            result = extractor.extract(img_tensor)

        kpts = result["keypoints"][0].cpu().numpy()        # (N, 2)
        desc = result["descriptors"][0].cpu().numpy()      # (N, 256)

        # Scale keypoints back to original image resolution if resize was applied
        if resize is not None:
            _, h_resized, w_resized = img_tensor.shape
            scale_x = orig_w / w_resized
            scale_y = orig_h / h_resized
            kpts[:, 0] *= scale_x
            kpts[:, 1] *= scale_y

        _save_features(h5_path, kpts, desc, (orig_w, orig_h))

        if (idx + 1) % 10 == 0 or (idx + 1) == len(captures):
            elapsed = time.perf_counter() - t0
            done    = idx + 1
            pct     = 100.0 * done / len(captures)
            speed   = elapsed / done          # s/image
            eta     = speed * (len(captures) - done)
            print(f"[extractor] {done}/{len(captures)} ({pct:.0f}%)  "
                  f"elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s  "
                  f"speed: {speed:.2f}s/img  last: {cap.capture_id}  kpts: {len(kpts)}")

    elapsed = time.perf_counter() - t0
    extracted = len(captures) - skipped
    print(f"[extractor] Done. Extracted: {extracted}  Skipped: {skipped}  "
          f"Time: {elapsed:.1f}s  ({elapsed / max(extracted, 1):.2f}s/image)")

    return features_dir


def features_exist(output_dir: str, captures: List[Capture]) -> bool:
    """Return True if every capture already has a .h5 feature file."""
    features_dir = Path(output_dir) / "features"
    return all((features_dir / f"{cap.capture_id}.h5").exists() for cap in captures)