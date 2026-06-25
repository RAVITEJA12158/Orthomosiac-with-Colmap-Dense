import os
import re
from typing import Optional
from .capture import Capture
from .exif_reader import read_gps, detect_rtk, read_focal_length_px


# ── Pattern 1: existing IMG_* convention ─────────────────────────────────────
# Matches: IMG_<optional_prefix>_<capture>_<band>.<ext>
# Supports formats like:
# - IMG_0001_000_RGB.jpg (test format)
# - IMG_260315_083045_0000_RGB.JPG (user format with date and time)
IMG_PATTERN = re.compile(
    r"^IMG_(?:\d+_)*(\d+)_(RGB|GRE|NIR|RED|REG)\.(jpg|jpeg|tiff|tif)$",
    re.IGNORECASE
)

# Keep the old name as an alias so any external code that imported it still works.
FILENAME_PATTERN = IMG_PATTERN

IMG_BAND_MAP = {
    "RGB": "rgb",
    "GRE": "green",
    "NIR": "nir",
    "RED": "red",
    "REG": "reg",
}

# ── Pattern 2: DJI Zenmuse / Mavic Multispectral convention ──────────────────
# rgb/   → DJI_<timestamp14>_<frame4>_D.(JPG|JPEG)
# multi/ → DJI_<timestamp14>_<frame4>_MS_(G|NIR|R|RE).(TIF|TIFF)
#
# capture_id = "<timestamp>_<frame>"  e.g. "20240405154706_0001"
# This keeps IDs unique across flights even if frame counters reset.
DJI_PATTERN = re.compile(
    r"^DJI_(\d{14})_(\d{4})_(D|MS_G|MS_NIR|MS_R|MS_RE)\.(jpg|jpeg|tif|tiff)$",
    re.IGNORECASE
)

DJI_BAND_MAP = {
    "D":      "rgb",
    "MS_G":   "green",
    "MS_NIR": "nir",
    "MS_R":   "red",
    "MS_RE":  "reg",
}

# Combined alias used by the rest of the module
BAND_MAP = {**IMG_BAND_MAP, **DJI_BAND_MAP}


def _parse_filename(filename: str) -> Optional[tuple[str, str]]:
    """
    Parse filename and return (capture_id, band) or None if no match.

    Tries the IMG_* pattern first, then the DJI pattern.
    The returned band is always the internal key used in BAND_MAP
    (e.g. "RGB", "GRE", "D", "MS_G", …).
    """
    # ── IMG_* ─────────────────────────────────────────────────────────────────
    m = IMG_PATTERN.match(filename)
    if m:
        return m.group(1), m.group(2).upper()

    # ── DJI ───────────────────────────────────────────────────────────────────
    m = DJI_PATTERN.match(filename)
    if m:
        # Use only the frame number (group 2) as capture_id, NOT the timestamp.
        # DJI RGB and multispectral sensors fire ~1 s apart, so the timestamp
        # embedded in the filename differs between the _D.JPG and the _MS_*.TIF
        # files of the same physical capture.  The 4-digit frame counter is the
        # only field that is identical across all 5 bands of a single burst.
        capture_id = m.group(2)       # e.g. "0001", "0125"
        band = m.group(3).upper()     # "D", "MS_G", "MS_NIR", …
        return capture_id, band

    return None


def group_captures(mission_dir: str) -> dict[str, Capture]:
    """
    Scan rgb/ and multi/ folders inside mission_dir.
    Group files by capture ID.
    Returns dict of capture_id -> Capture.
    """
    rgb_dir   = os.path.join(mission_dir, "rgb")
    multi_dir = os.path.join(mission_dir, "multi")

    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"rgb/ folder not found in {mission_dir}")
    if not os.path.isdir(multi_dir):
        raise FileNotFoundError(f"multi/ folder not found in {mission_dir}")

    captures: dict[str, Capture] = {}

    # Scan rgb/
    for filename in sorted(os.listdir(rgb_dir)):
        result = _parse_filename(filename)
        if result is None:
            continue
        capture_id, band = result
        # Both IMG "RGB" and DJI "D" are the downward RGB image.
        if BAND_MAP.get(band) != "rgb":
            print(f"[grouper] Warning: unexpected band {band} in rgb/ folder: {filename}")
            continue

        filepath = os.path.join(rgb_dir, filename)

        if capture_id not in captures:
            captures[capture_id] = Capture(capture_id=capture_id)

        captures[capture_id].rgb = filepath

        # Read GPS and RTK status from the RGB image (the only band guaranteed
        # to carry full EXIF/XMP on all drone models).
        lat, lon, alt = read_gps(filepath)
        captures[capture_id].latitude  = lat
        captures[capture_id].longitude = lon
        captures[capture_id].altitude  = alt
        captures[capture_id].has_rtk   = detect_rtk(filepath)

        focal_px, ref_width = read_focal_length_px(filepath)
        captures[capture_id].focal_length_px       = focal_px
        captures[capture_id].focal_length_ref_width = ref_width

    # Scan multi/
    for filename in sorted(os.listdir(multi_dir)):
        result = _parse_filename(filename)
        if result is None:
            continue
        capture_id, band = result
        # Reject any file whose band resolves to "rgb" (belongs in rgb/ folder).
        if BAND_MAP.get(band) == "rgb":
            print(f"[grouper] Warning: RGB file found in multi/ folder: {filename}")
            continue

        filepath = os.path.join(multi_dir, filename)

        if capture_id not in captures:
            captures[capture_id] = Capture(capture_id=capture_id)
            # Warn: this capture has no RGB image yet. GPS is only read from
            # RGB images, so latitude/longitude will remain None for this
            # capture, which will cause it to be rejected at the quality
            # filter or keyframe selection stage.
            print(
                f"[grouper] Warning: capture '{capture_id}' found in multi/ "
                f"but not in rgb/. GPS will be None (no RGB to read EXIF from). "
                f"Ensure RGB file follows the naming convention IMG_<frame>_{capture_id}_RGB.<ext>."
            )

        field = BAND_MAP[band]
        setattr(captures[capture_id], field, filepath)

    return captures