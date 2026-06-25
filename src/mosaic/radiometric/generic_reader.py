"""
src/mosaic/radiometric/generic_reader.py

Fallback reader for cameras that aren't Sequoia or MicaSense. Reads
standard EXIF only — no sunshine sensor, no vignetting polynomial. The
result is a degraded but safe RadiometricParams: calibrate_image() will
still run without crashing, it just won't have absolute-reflectance
accuracy.
"""

from __future__ import annotations

import logging

import numpy as np

from ._exif_utils import get_float, get_str
from .params import RadiometricParams

logger = logging.getLogger(__name__)

_BASE_ISO = 100.0


def read_generic_params(exif_dict: dict, band_name: str) -> RadiometricParams:
    """
    Fallback for unrecognised cameras.
        EXIF:ExposureTime      -> exposure_time
        EXIF:ISOSpeedRatings   -> approximate gain
        irradiance             = 1.0   (no sunshine sensor data)
        vignetting_coeffs      = [0, 0, 0]  (identity — no correction)
        black_level             = 0.0
    """
    logger.warning(
        "read_generic_params: unrecognised camera for band %s — radiometric "
        "calibration is DEGRADED (no sunshine-sensor correction, no vignetting "
        "polynomial). NDVI from this image will not be directly comparable to "
        "images from a calibrated camera.",
        band_name,
    )

    exposure_time = get_float(exif_dict, ["EXIF:ExposureTime"], default=1.0)
    iso = get_float(exif_dict, ["EXIF:ISOSpeedRatings", "EXIF:ISOSpeed"], default=_BASE_ISO)
    gain = iso / _BASE_ISO if iso > 0 else 1.0

    camera_make = get_str(exif_dict, ["EXIF:Make"], default="unknown")
    camera_model = get_str(exif_dict, ["EXIF:Model"], default="unknown")

    return RadiometricParams(
        black_level=0.0,
        exposure_time=exposure_time if exposure_time > 0 else 1.0,
        gain=gain,
        irradiance=1.0,
        vignetting_coeffs=np.zeros(3, dtype=np.float32),
        band_name=band_name,
        camera_make=camera_make,
        camera_model=camera_model,
    )
