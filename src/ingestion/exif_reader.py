"""
EXIF / XMP reader utilities for the ingestion stage.

GPS reading  : read_gps()    — standard EXIF, works for all cameras
RTK detection: detect_rtk()  — three layered signals, DJI-first, EXIF fallback

RTK detection signals (checked in priority order)
──────────────────────────────────────────────────
1. DJI XMP  drone-dji:RtkFlag == 50
       50 = RTK Fixed (centimeter-level)   ← gold standard
       34 = RTK Float (decimeter-level)     ← NOT treated as RTK here
       16 = Single-point GNSS              ← NOT RTK
   Source: DJI SDK / Agisoft / Pix4D docs

2. DJI XMP  drone-dji:RtkStdLon + RtkStdLat both < 0.05 m
   Standard deviations written by DJI RTK drones alongside RtkFlag.
   A horizontal std < 5 cm is physically only achievable with RTK Fixed,
   so this acts as an independent cross-check (and catches any firmware
   variants that use a different RtkFlag encoding).

3. Standard EXIF  GPS GPSDifferential == 1
   Defined in EXIF spec as "differential correction applied".
   Non-DJI RTK hardware (e.g. senseFly, Wingtra, Trimble UAS) commonly
   writes this field instead of proprietary XMP tags.

All three signals must fail to return False. A single True is sufficient.
Any exception is swallowed and returns False (conservative — falls back
to the user's --rtk flag or treats mission as non-RTK).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

# DJI RtkFlag value that unambiguously means "RTK Fixed" (centimeter-level).
# Float (34) is intentionally excluded — it is decimeter-level, which is not
# reliable enough to justify the stronger GLOMAP path / tighter priors.
_DJI_RTK_FIXED_FLAG = 50

# Maximum horizontal standard deviation (metres) that we consider RTK quality.
# DJI RTK Fixed typically gives < 0.01 m; 0.05 m (5 cm) is a generous threshold
# that still cleanly separates RTK from standard GPS (which is 1–5 m).
_RTK_STD_THRESHOLD_M = 0.05

# XMP namespace for DJI proprietary metadata
_NS_DJI = "drone-dji"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _convert_dms_to_decimal(dms_values, ref: str) -> Optional[float]:
    """Convert degrees/minutes/seconds to decimal degrees."""
    try:
        d = float(dms_values[0].num) / float(dms_values[0].den)
        m = float(dms_values[1].num) / float(dms_values[1].den)
        s = float(dms_values[2].num) / float(dms_values[2].den)
    except ZeroDivisionError:
        return None
    decimal = d + (m / 60.0) + (s / 3600.0)
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def _extract_xmp_block(image_path: str) -> Optional[str]:
    """
    Extract the raw XMP XML string embedded in a JPEG or TIFF file.

    Reads only the first 128 KB of the file — XMP is always written near
    the start of the file by camera firmware, so this avoids loading large
    TIFF/RAW files entirely into memory.

    Returns the XMP string, or None if no XMP packet was found.
    """
    try:
        with open(image_path, "rb") as f:
            data = f.read(131072)  # 128 KB is always enough for XMP header

        # XMP packet markers
        start = data.find(b"<?xpacket begin")
        if start == -1:
            start = data.find(b"<x:xmpmeta")
        if start == -1:
            return None

        end = data.find(b"<?xpacket end", start)
        if end != -1:
            end = data.find(b"?>", end)
            if end != -1:
                end += 2
        else:
            end = data.find(b"</x:xmpmeta>", start)
            if end != -1:
                end += len(b"</x:xmpmeta>")

        if end == -1:
            return None

        return data[start:end].decode("utf-8", errors="ignore")
    except Exception:
        return None


def _parse_dji_rtk_xmp(xmp_str: str) -> Optional[bool]:
    """
    Parse DJI XMP metadata for RTK signals.

    Returns:
        True  — at least one DJI RTK signal confirms RTK Fixed quality
        False — DJI XMP found but no RTK Fixed signal
        None  — no DJI XMP namespace found (not a DJI image or no XMP)
    """
    try:
        root = ET.fromstring(xmp_str)
    except ET.ParseError:
        return None

    # Collect all text content under any element in the drone-dji namespace.
    # ElementTree uses Clark notation: {namespace_uri}local_name.
    # DJI's XMP namespace URI varies slightly across firmware; we match
    # by the local name prefix rather than the full URI to be robust.
    dji_values: dict[str, str] = {}

    for elem in root.iter():
        tag = elem.tag  # e.g. "{http://www.dji.com/drone-dji/1.0/}RtkFlag"
        if _NS_DJI not in tag:
            continue
        # Strip Clark notation: {uri}LocalName → LocalName
        local = tag.rsplit("}", 1)[-1]
        if elem.text:
            dji_values[local] = elem.text.strip()

    # Also scan attributes on any element (DJI sometimes writes as attrs).
    for elem in root.iter():
        for attr, val in elem.attrib.items():
            if _NS_DJI not in attr:
                continue
            local = attr.rsplit("}", 1)[-1]
            dji_values.setdefault(local, val.strip())

    if not dji_values:
        return None  # no DJI namespace found

    # ── Signal 1: RtkFlag == 50 (RTK Fixed) ──────────────────────────────────
    rtk_flag_raw = dji_values.get("RtkFlag")
    if rtk_flag_raw is not None:
        try:
            if int(rtk_flag_raw) == _DJI_RTK_FIXED_FLAG:
                return True
        except ValueError:
            pass

    # ── Signal 2: RtkStdLon + RtkStdLat both below threshold ─────────────────
    std_lon_raw = dji_values.get("RtkStdLon")
    std_lat_raw = dji_values.get("RtkStdLat")
    if std_lon_raw is not None and std_lat_raw is not None:
        try:
            std_lon = float(std_lon_raw)
            std_lat = float(std_lat_raw)
            if std_lon < _RTK_STD_THRESHOLD_M and std_lat < _RTK_STD_THRESHOLD_M:
                return True
        except ValueError:
            pass

    return False  # DJI XMP found but no RTK Fixed signal


# ── Public API ───────────────────────────────────────────────────────────────

def read_gps(image_path: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Read GPS coordinates from image EXIF.
    Returns (latitude, longitude, altitude) or (None, None, None) if not found.
    """
    try:
        import exifread  # lazy import: only required at runtime, not at module load
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        lat = None
        lon = None
        alt = None

        if "GPS GPSLatitude" in tags and "GPS GPSLatitudeRef" in tags:
            lat = _convert_dms_to_decimal(
                tags["GPS GPSLatitude"].values,
                str(tags["GPS GPSLatitudeRef"])
            )

        if "GPS GPSLongitude" in tags and "GPS GPSLongitudeRef" in tags:
            lon = _convert_dms_to_decimal(
                tags["GPS GPSLongitude"].values,
                str(tags["GPS GPSLongitudeRef"])
            )

        if "GPS GPSAltitude" in tags:
            a = tags["GPS GPSAltitude"].values[0]
            alt = float(a.num) / float(a.den)

        return lat, lon, alt

    except Exception as e:
        print(f"[exif_reader] Warning: could not read GPS from {image_path}: {e}")
        return None, None, None


def read_focal_length_px(image_path: str) -> tuple[Optional[float], Optional[int]]:
    """
    Compute the camera's focal length in pixels directly from EXIF, instead
    of relying on a generic heuristic.

    Formula (standard EXIF focal-plane convention):
        px_per_mm = FocalPlaneXResolution / mm_per_unit(FocalPlaneResolutionUnit)
        focal_px  = FocalLength_mm * px_per_mm

    This is independent of any later resizing the pipeline does to the
    image (e.g. extractor.py's `resize=1600`), because EXIF focal-plane
    tags describe the sensor as captured. The returned `ref_width_px`
    (EXIF ExifImageWidth/ImageWidth) records what width that focal length
    corresponds to, so callers can rescale it correctly if the image was
    later resized: focal_px_at_w = focal_px * (w / ref_width_px).

    Returns:
        (focal_length_px, ref_width_px) — both None if the required EXIF
        tags are missing, malformed, or produce an implausible value.
        A None result means "do not trust this as a calibrated prior" —
        callers must fall back to a heuristic and leave
        has_prior_focal_length=False on the COLMAP camera.
    """
    try:
        import exifread
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        focal_tag = tags.get("EXIF FocalLength")
        plane_res_tag = tags.get("EXIF FocalPlaneXResolution")
        if focal_tag is None or plane_res_tag is None:
            return None, None

        focal_mm = float(focal_tag.values[0].num) / float(focal_tag.values[0].den)
        plane_res = float(plane_res_tag.values[0].num) / float(plane_res_tag.values[0].den)
        if focal_mm <= 0 or plane_res <= 0:
            return None, None

        # EXIF FocalPlaneResolutionUnit: 2 = inches (default per spec), 3 = cm.
        unit_tag = tags.get("EXIF FocalPlaneResolutionUnit")
        unit_code = 2
        if unit_tag is not None:
            try:
                unit_code = int(str(unit_tag))
            except ValueError:
                pass
        mm_per_unit = 25.4 if unit_code == 2 else 10.0

        px_per_mm = plane_res / mm_per_unit
        focal_px = focal_mm * px_per_mm

        # Sanity guard: reject implausible values rather than silently
        # poisoning the reconstruction with a "trusted" bad prior.
        if not (100.0 < focal_px < 50000.0):
            return None, None

        ref_width = None
        width_tag = tags.get("EXIF ExifImageWidth") or tags.get("Image ImageWidth")
        if width_tag is not None:
            try:
                ref_width = int(str(width_tag))
            except ValueError:
                ref_width = None

        return focal_px, ref_width

    except Exception as e:
        print(f"[exif_reader] Warning: could not read focal length from {image_path}: {e}")
        return None, None


def detect_rtk(image_path: str) -> bool:
    """
    Detect whether an image was captured with RTK-quality GPS.

    Three signals are checked in priority order; a single True is sufficient:

    1. DJI XMP drone-dji:RtkFlag == 50  (RTK Fixed — centimeter-level)
    2. DJI XMP RtkStdLon + RtkStdLat < 0.05 m  (< 5 cm horizontal std)
    3. EXIF GPS GPSDifferential == 1  (differential correction applied)

    Returns False conservatively on any error so that a bad image does not
    incorrectly promote the whole mission to RTK mode.
    """
    try:
        # ── Signals 1 & 2: DJI XMP ───────────────────────────────────────────
        xmp = _extract_xmp_block(image_path)
        if xmp is not None:
            result = _parse_dji_rtk_xmp(xmp)
            if result is True:
                return True
            if result is False:
                # DJI image but no RTK Fixed confirmed — skip EXIF fallback
                # (DJI always writes RtkFlag when RTK hardware is present, so
                # a False here means the drone genuinely wasn't in RTK Fixed mode)
                return False

        # ── Signal 3: Standard EXIF GPSDifferential ──────────────────────────
        # Reached only for non-DJI images (result is None above).
        import exifread
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        diff_tag = tags.get("GPS GPSDifferential")
        if diff_tag is not None:
            try:
                if int(str(diff_tag)) == 1:
                    return True
            except ValueError:
                pass

        return False

    except Exception:
        return False