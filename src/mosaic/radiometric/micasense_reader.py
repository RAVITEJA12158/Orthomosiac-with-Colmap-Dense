"""
src/mosaic/radiometric/micasense_reader.py

Reads MicaSense RedEdge XMP radiometric metadata into the same
camera-agnostic RadiometricParams used for Sequoia. Field names differ
from Sequoia; the downstream calibrate_image() math is identical.
"""

from __future__ import annotations

import logging

from ._exif_utils import get_float, get_str, parse_coeffs
from .params import RadiometricParams

logger = logging.getLogger(__name__)

_BASE_ISO = 100.0


def read_micasense_params(exif_dict: dict, band_name: str) -> RadiometricParams:
    """
    Parse MicaSense RedEdge XMP fields:
        XMP:BlackLevel              -> black_level
        XMP:ExposureTime            -> exposure_time
        XMP:ISOSpeed                -> gain = iso / 100
        XMP:Irradiance              -> irradiance
        XMP:VignettingPolynomial    -> vignetting_coeffs
    """
    black_level = get_float(exif_dict, ["XMP:BlackLevel", "EXIF:BlackLevel"], default=0.0)
    exposure_time = get_float(
        exif_dict, ["XMP:ExposureTime", "EXIF:ExposureTime"], default=1.0
    )
    iso = get_float(exif_dict, ["XMP:ISOSpeed", "XMP:IsoSpeed", "EXIF:ISOSpeedRatings"], default=_BASE_ISO)
    gain = iso / _BASE_ISO if iso > 0 else 1.0
    irradiance = get_float(exif_dict, ["XMP:Irradiance"], default=1.0)
    vignetting_raw = exif_dict.get("XMP:VignettingPolynomial")
    vignetting_coeffs = parse_coeffs(vignetting_raw)

    camera_make = get_str(exif_dict, ["EXIF:Make", "XMP:Make"], default="MicaSense")
    camera_model = get_str(exif_dict, ["EXIF:Model", "XMP:Model"], default="RedEdge")

    if exposure_time <= 0:
        logger.warning(
            "read_micasense_params: non-positive exposure_time for band %s, defaulting to 1.0",
            band_name,
        )
        exposure_time = 1.0
    if irradiance <= 0:
        logger.warning(
            "read_micasense_params: non-positive irradiance for band %s, defaulting to 1.0 "
            "(sunshine sensor correction disabled for this image)",
            band_name,
        )
        irradiance = 1.0

    return RadiometricParams(
        black_level=black_level,
        exposure_time=exposure_time,
        gain=gain,
        irradiance=irradiance,
        vignetting_coeffs=vignetting_coeffs,
        band_name=band_name,
        camera_make=camera_make,
        camera_model=camera_model,
    )
