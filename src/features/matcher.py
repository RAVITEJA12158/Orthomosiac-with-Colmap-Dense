"""
Stage 4 — Feature Matching
Tool : LightGlue (transformer matcher, ETH Zurich)
Input : .h5 feature files from extractor.py
        GPS-filtered pairs from neighbors.py
Output: <output_dir>/matches.h5

matches.h5 layout:
    /<capture_id_a>/<capture_id_b>/matches0   int32  (M, 2)
        matches0[:, 0] = keypoint index in image A
        matches0[:, 1] = keypoint index in image B

Only pairs that pass LightGlue's internal confidence filter are written.
Pairs with zero matches are silently skipped.

LightGlue already performs geometric filtering internally — no separate RANSAC
is needed at this stage.  COLMAP RANSAC runs in Stage 5 (db_importer) when the
matches are imported into the COLMAP database.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch

from ..ingestion.capture import Capture
from .neighbors import build_neighbor_pairs


# ── internal helpers ─────────────────────────────────────────────────────────

def _check_lightglue():
    try:
        from lightglue import LightGlue
        return LightGlue
    except ImportError as exc:
        raise ImportError(
            "lightglue not installed.\n"
            "Run:  pip install lightglue"
        ) from exc


def _build_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if use_gpu and not torch.cuda.is_available():
        print("[matcher] Warning: CUDA not available — falling back to CPU.")
    return torch.device("cpu")


def _load_features(h5_path: Path, device: torch.device) -> Optional[dict]:
    """
    Load ALIKED features from .h5 file.
    Returns None if file is missing (capture was skipped at extraction).

    Returns dict with:
        keypoints   : (1, N, 2) float32 tensor
        descriptors : (1, N, 256) float32 tensor
        image_size  : (width, height) int tuple
    """
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        kpts = torch.from_numpy(f["keypoints"][:]).float().unsqueeze(0).to(device)   # (1, N, 2)
        desc = torch.from_numpy(f["descriptors"][:]).float().unsqueeze(0).to(device) # (1, N, D)
        w, h = f["image_size"][:]
    return {"keypoints": kpts, "descriptors": desc, "image_size": (int(w), int(h))}


def _normalise_keypoints(
    kpts: torch.Tensor,   # (1, N, 2)  pixel coords
    image_size: Tuple[int, int],  # (width, height)
) -> torch.Tensor:
    """
    LightGlue expects keypoints normalised to [-1, 1] by image dimensions.
    kpts[:, :, 0] = x  (width  axis)
    kpts[:, :, 1] = y  (height axis)
    """
    w, h = image_size
    scale = torch.tensor([w, h], dtype=kpts.dtype, device=kpts.device)
    return (kpts / (scale / 2.0)) - 1.0


def _append_matches(
    matches_h5: h5py.File,
    id_a: str,
    id_b: str,
    matches: np.ndarray,  # (M, 2)
) -> None:
    """Write match indices into matches.h5 under group /<id_a>/<id_b>/."""
    grp = matches_h5.require_group(id_a)
    if id_b in grp:
        del grp[id_b]   # overwrite if re-running
    sub = grp.create_group(id_b)
    sub.create_dataset("matches0", data=matches, dtype="int32")


# ── public API ───────────────────────────────────────────────────────────────

def match_features(
    captures: List[Capture],
    output_dir: str,
    n_neighbors: int = 20,
    use_gpu: bool = True,
) -> Path:
    """
    Match ALIKED features between GPS-filtered image pairs using LightGlue.

    Reads .h5 feature files written by extract_features().
    Writes matches to <output_dir>/matches.h5.

    Args:
        captures    : Capture list from ingestion (same order as extraction)
        output_dir  : same root dir passed to extract_features()
        n_neighbors : GPS neighbors per image. Set automatically from overlap
                      estimation in run_pipeline.py; 20 is the safe default.
        use_gpu     : use CUDA when available

    Returns:
        Path to matches.h5
    """
    LightGlue = _check_lightglue()
    device = _build_device(use_gpu)

    features_dir = Path(output_dir) / "features"
    matches_path = Path(output_dir) / "matches.h5"

    if not features_dir.exists():
        raise FileNotFoundError(
            f"Features directory not found: {features_dir}\n"
            "Run extract_features() first."
        )

    # GPS-filtered pairs — the key reduction: 810k → ~7200
    pairs = build_neighbor_pairs(captures, n_neighbors=n_neighbors)
    print(f"[matcher] Matching {len(pairs)} GPS-filtered pairs (GPU: {use_gpu})...")

    # Build matcher once — reused for all pairs
    matcher = (
        LightGlue(features="aliked")
        .eval()
        .to(device)
    )
    # Cache loaded features — each capture appears in multiple pairs.
    # Cap at _FEATURE_CACHE_LIMIT entries to avoid unbounded VRAM growth:
    # at ~8 MB/capture, 64 entries ≈ 512 MB GPU memory.
    _FEATURE_CACHE_LIMIT = 64
    feature_cache: dict[str, Optional[dict]] = {}  # Python 3.7+ insertion-ordered

    def get_features(capture_id: str) -> Optional[dict]:
        if capture_id not in feature_cache:
            if len(feature_cache) >= _FEATURE_CACHE_LIMIT:
                # Evict the oldest entry (first key in insertion order).
                # del drops the dict; Python ref-counting frees GPU tensors immediately.
                oldest = next(iter(feature_cache))
                del feature_cache[oldest]
            h5_path = features_dir / f"{capture_id}.h5"
            feature_cache[capture_id] = _load_features(h5_path, device)
        return feature_cache[capture_id]

    t0 = time.perf_counter()
    total_matches = 0
    matched_pairs = 0
    skipped_pairs = 0

    with h5py.File(matches_path, "w") as matches_h5:
        for idx, (id_a, id_b) in enumerate(pairs):
            feats_a = get_features(id_a)
            feats_b = get_features(id_b)

            if feats_a is None or feats_b is None:
                skipped_pairs += 1
                continue

            # Normalise to [-1, 1] as required by LightGlue
            kpts_a = _normalise_keypoints(feats_a["keypoints"], feats_a["image_size"])
            kpts_b = _normalise_keypoints(feats_b["keypoints"], feats_b["image_size"])

            input_dict = {
                "image0": {
                    "keypoints":   kpts_a,
                    "descriptors": feats_a["descriptors"],
                    "image_size":  torch.tensor(
                        [feats_a["image_size"]], dtype=torch.long, device=device
                    ),
                },
                "image1": {
                    "keypoints":   kpts_b,
                    "descriptors": feats_b["descriptors"],
                    "image_size":  torch.tensor(
                        [feats_b["image_size"]], dtype=torch.long, device=device
                    ),
                },
            }

            with torch.no_grad():
                result = matcher(input_dict)

            # matches0: (1, N) — for each kp in image0, index of matched kp in image1
            #           -1 = unmatched
            matches0 = result["matches0"][0].cpu().numpy()  # (N,)

            valid = matches0 >= 0
            if valid.sum() == 0:
                skipped_pairs += 1
                continue

            # Build (M, 2) index array  [ kp_idx_in_A,  kp_idx_in_B ]
            kp_indices_a = np.where(valid)[0].astype("int32")
            kp_indices_b = matches0[valid].astype("int32")
            match_array = np.stack([kp_indices_a, kp_indices_b], axis=1)  # (M, 2)

            _append_matches(matches_h5, id_a, id_b, match_array)
            total_matches += len(match_array)
            matched_pairs += 1

            if (idx + 1) % 100 == 0 or (idx + 1) == len(pairs):
                elapsed = time.perf_counter() - t0
                done    = idx + 1
                pct     = 100.0 * done / len(pairs)
                speed   = elapsed / done
                eta     = speed * (len(pairs) - done)
                avg_m   = total_matches / max(matched_pairs, 1)
                print(f"[matcher] {done}/{len(pairs)} ({pct:.0f}%)  "
                      f"elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s  "
                      f"matched pairs: {matched_pairs}  avg matches/pair: {avg_m:.0f}")

    elapsed   = time.perf_counter() - t0
    avg_m     = total_matches / max(matched_pairs, 1)
    match_pct = 100.0 * matched_pairs / max(len(pairs), 1)
    print(f"[matcher] Done — {matched_pairs}/{len(pairs)} pairs matched ({match_pct:.0f}%)  "
          f"skipped: {skipped_pairs}  total matches: {total_matches}  "
          f"avg: {avg_m:.0f}/pair  time: {elapsed:.1f}s")

    return matches_path