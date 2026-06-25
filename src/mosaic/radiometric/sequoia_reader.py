"""
src/mosaic/radiometric/sequoia_reader.py

Reads Parrot Sequoia-specific XMP radiometric metadata (via pyexiftool)
into a camera-agnostic RadiometricParams.

Sequoia captures each band as a separate TIFF file with its own embedded
XMP block, so exposure/gain/irradiance can genuinely differ band-to-band
and capture-to-capture (the sunshine sensor logs irradiance per shot).
"""

from __future__ import annotations

import logging

from ._exif_utils import get_float, get_str, parse_coeffs
from .params import RadiometricParams

logger = logging.getLogger(__name__)

_BASE_ISO = 100.0


def read_sequoia_params(exif_dict: dict, band_name: str) -> RadiometricParams:
    """
    Parse Parrot Sequoia XMP fields:
        XMP:BlackLevel              -> black_level
        XMP:ExposureTime / EXIF:ExposureTime -> exposure_time
        XMP:IsoSpeed                -> gain = iso / 100
        XMP:Irradiance              -> irradiance (sunshine sensor, W/m^2)
        XMP:VignettingPolynomial    -> vignetting_coeffs
    """
    black_level = get_float(exif_dict, ["XMP:BlackLevel", "EXIF:BlackLevel"], default=0.0)
    exposure_time = get_float(
        exif_dict, ["XMP:ExposureTime", "EXIF:ExposureTime"], default=1.0
    )
    iso = get_float(exif_dict, ["XMP:IsoSpeed", "XMP:ISOSpeed", "EXIF:ISOSpeedRatings"], default=_BASE_ISO)
    gain = iso / _BASE_ISO if iso > 0 else 1.0
    irradiance = get_float(exif_dict, ["XMP:Irradiance"], default=1.0)
    vignetting_raw = exif_dict.get("XMP:VignettingPolynomial") or exif_dict.get(
        "XMP:VignettingPolynomial2"
    )
    vignetting_coeffs = parse_coeffs(vignetting_raw)

    camera_make = get_str(exif_dict, ["EXIF:Make", "XMP:Make"], default="Parrot")
    camera_model = get_str(exif_dict, ["EXIF:Model", "XMP:Model"], default="Sequoia")

    if exposure_time <= 0:
        logger.warning(
            "read_sequoia_params: non-positive exposure_time for band %s, defaulting to 1.0",
            band_name,
        )
        exposure_time = 1.0
    if irradiance <= 0:
        logger.warning(
            "read_sequoia_params: non-positive irradiance for band %s, defaulting to 1.0 "
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
