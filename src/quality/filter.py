"""
Image Quality Filter
====================
Conservative filter — only removes CLEARLY bad images.

Philosophy:
  Photogrammetry loves overlap.
  Removing too many images breaks overlap chains.
  Only reject what is OBVIOUSLY unusable.

Three rejection reasons ONLY:
  1. Corrupt / unreadable file
  2. Severely blurry (motion blur, out of focus)
  3. Missing GPS metadata

Everything else passes.
No exposure filtering.
No contrast filtering.
No aggressive thresholds.

Applied to RGB image only.
If RGB passes → entire capture (all 5 bands) passes.
If RGB fails  → entire capture rejected.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..ingestion.capture import Capture


# ── Thresholds ────────────────────────────────────────────────────────────────
# Set deliberately LOW — only catch clearly bad images.
# A slightly blurry image is better than a gap in coverage.
# Tune these DOWN if too many images are rejected on your data.

BLUR_THRESHOLD = 30
# Laplacian variance below this = severely blurry, no recoverable features
# Typical sharp aerial image: 200-2000+
# Severely blurry: < 30
# Slightly soft:   30-100  → we KEEP these (overlap matters more)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    capture_id  : str
    passed      : bool
    reason      : Optional[str]   # why rejected, None if passed
    blur_score  : Optional[float] # Laplacian variance
    has_gps     : bool

    def __repr__(self):
        if self.passed:
            blur_str = f"{self.blur_score:.1f}" if self.blur_score is not None else "None"
            return f"QualityResult({self.capture_id} PASS blur={blur_str})"
        return f"QualityResult({self.capture_id} FAIL reason='{self.reason}')"


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_corrupt(rgb_path: str) -> tuple[bool, Optional[str], Optional[np.ndarray]]:
    """
    Check 1 — File integrity.
    Tries to read the image. Rejects if unreadable or empty.
    This is the safest rejection — a corrupt file is 100% useless.

    Returns the loaded BGR image array (on success) alongside the
    pass/fail result so _check_blur() can reuse it instead of re-reading
    the same file from disk a second time.
    """
    try:
        img = cv2.imread(rgb_path)
        if img is None:
            return False, "file unreadable by OpenCV", None
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return False, f"empty image dimensions ({w}x{h})", None
        return True, None, img
    except Exception as e:
        return False, f"read error: {e}", None


def _check_blur(img: np.ndarray) -> tuple[bool, Optional[str], float]:
    """
    Check 2 — Severe blur detection via Laplacian variance.
    Only rejects images that are SO blurry no keypoints are recoverable.
    Slightly soft images pass — overlap is more important.

    Takes the already-loaded image from _check_corrupt() rather than a
    path — avoids a second cv2.imread() of the same file.

    Laplacian measures second derivative (edge sharpness).
    High variance = sharp edges present = image usable.
    Very low variance = no edges = completely blurred.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if score < BLUR_THRESHOLD:
        return False, f"severely blurry (blur_score={score:.1f} < {BLUR_THRESHOLD})", score

    return True, None, score


def _check_gps(capture: Capture) -> tuple[bool, Optional[str]]:
    """
    Check 3 — GPS metadata presence.
    Without GPS the image cannot participate in:
      - neighbor filtering
      - GPS distance rejection
      - georeferencing
    It becomes an orphan that slows SfM and contributes nothing.

    Uses the GPS already parsed by ingestion (grouper.py calls
    exif_reader.read_gps() once per RGB image while grouping captures) via
    capture.latitude/longitude instead of re-reading and re-parsing the
    EXIF tags from disk again here. Re-parsing was a third redundant
    file read per image with no new information — the result is always
    identical to what's already cached on the Capture.
    """
    if capture.latitude is None or capture.longitude is None:
        return False, "missing GPS metadata in EXIF"
    return True, None


# ── Per-capture check ─────────────────────────────────────────────────────────

def check_capture(capture: Capture) -> QualityResult:
    """
    Run all quality checks on one capture's RGB image.
    Returns QualityResult with pass/fail and reason.

    Checks run in order — stops at first failure (fast path).
    Order: corrupt → blur → GPS
    (corrupt must be first — other checks need readable image)

    The RGB file is read from disk exactly once (in _check_corrupt) and
    reused for the blur check; GPS comes from the Capture's cached
    latitude/longitude rather than a second EXIF read.
    """
    # Guard: captures with no RGB path at all cannot be checked.
    # This happens when ingestion found only multispectral files for a capture ID.
    if not capture.rgb:
        return QualityResult(
            capture_id=capture.capture_id,
            passed=False,
            reason="no RGB image path — capture has no RGB file",
            blur_score=None,
            has_gps=False,
        )

    rgb_path = capture.rgb

    # Check 1 — corrupt
    ok, reason, img = _check_corrupt(rgb_path)
    if not ok:
        return QualityResult(
            capture_id = capture.capture_id,
            passed     = False,
            reason     = reason,
            blur_score = None,
            has_gps    = False,
        )

    # Check 2 — blur (reuses `img` loaded above — no second disk read)
    ok, reason, blur_score = _check_blur(img)
    if not ok:
        return QualityResult(
            capture_id = capture.capture_id,
            passed     = False,
            reason     = reason,
            blur_score = blur_score,
            has_gps    = capture.latitude is not None and capture.longitude is not None,
        )

    # Check 3 — GPS (reuses capture.latitude/longitude — no EXIF re-read)
    ok, reason = _check_gps(capture)
    if not ok:
        return QualityResult(
            capture_id = capture.capture_id,
            passed     = False,
            reason     = reason,
            blur_score = blur_score,
            has_gps    = False,
        )

    # All passed
    return QualityResult(
        capture_id = capture.capture_id,
        passed     = True,
        reason     = None,
        blur_score = blur_score,
        has_gps    = True,
    )


# ── Main filter function ──────────────────────────────────────────────────────

def filter_quality(
    captures        : list[Capture],
    strict          : bool = False,
    min_pass_ratio  : float = 0.80,
) -> list[Capture]:
    """
    Filter captures by image quality.
    Conservative — only removes clearly bad images.

    Args:
        captures       : list of Capture objects from ingestion
        strict         : if True, raise error on any rejection
        min_pass_ratio : if fewer than this fraction pass, raise error
                         default 0.80 = warn if >20% rejected
                         safety net against wrong thresholds

    Returns:
        list of captures that passed all checks

    Raises:
        ValueError if strict=True and any image rejected
        ValueError if pass rate < min_pass_ratio
    """
    if not captures:
        raise ValueError("[quality] No captures to filter.")

    passed   : list[Capture]       = []
    rejected : list[QualityResult] = []

    print(f"[quality] Checking {len(captures)} captures (blur threshold={BLUR_THRESHOLD})...")

    for cap in captures:
        result = check_capture(cap)
        if result.passed:
            passed.append(cap)
        else:
            rejected.append(result)
            print(f"[quality] REJECT capture {cap.capture_id}: {result.reason}")

    # Summary
    total       = len(captures)
    n_passed    = len(passed)
    n_rejected  = len(rejected)
    pass_ratio  = n_passed / total

    print(f"[quality] {n_passed}/{total} captures passed "
          f"({n_rejected} rejected, {pass_ratio*100:.1f}% pass rate)")

    # Safety net — if too many rejected something is wrong with thresholds
    if pass_ratio < min_pass_ratio:
        raise ValueError(
            f"[quality] Only {pass_ratio*100:.1f}% of captures passed "
            f"(minimum {min_pass_ratio*100:.1f}%). "
            f"Thresholds may be too aggressive. "
            f"Check BLUR_THRESHOLD={BLUR_THRESHOLD}."
        )

    if strict and n_rejected > 0:
        reasons = "\n".join(f"  {r.capture_id}: {r.reason}" for r in rejected)
        raise ValueError(f"[quality] {n_rejected} captures rejected in strict mode:\n{reasons}")

    return passed


def print_quality_report(captures: list[Capture]) -> None:
    """
    Run checks and print full report without filtering.
    Useful for tuning thresholds on real data.
    """
    print(f"\n{'='*60}")
    print(f"QUALITY REPORT — {len(captures)} captures")
    print(f"{'='*60}")

    blur_scores = []
    for cap in captures:
        result = check_capture(cap)
        status = "PASS" if result.passed else f"FAIL ({result.reason})"
        blur   = f"{result.blur_score:.1f}" if result.blur_score else "N/A"
        print(f"  {cap.capture_id}  blur={blur:>8}  {status}")
        if result.blur_score:
            blur_scores.append(result.blur_score)

    if blur_scores:
        print("\nBlur score stats:")
        print(f"  min  : {min(blur_scores):.1f}")
        print(f"  max  : {max(blur_scores):.1f}")
        print(f"  mean : {np.mean(blur_scores):.1f}")
        print(f"  p5   : {np.percentile(blur_scores, 5):.1f}")
        print(f"  p25  : {np.percentile(blur_scores, 25):.1f}")
        print(f"\nCurrent BLUR_THRESHOLD = {BLUR_THRESHOLD}")
        print(f"Would reject captures with score < {BLUR_THRESHOLD}")
        n_would_reject = sum(1 for s in blur_scores if s < BLUR_THRESHOLD)
        print(f"Would reject: {n_would_reject}/{len(blur_scores)} captures")
    print(f"{'='*60}\n")