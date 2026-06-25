"""
src/mosaic/radiometric/params.py

Camera-agnostic container for the per-image values needed to convert raw
sensor DN into absolute reflectance. The calibration math in
radiometric/__init__.py never changes — only how these fields are read
from EXIF/XMP differs per camera (see sequoia_reader.py /
micasense_reader.py / generic_reader.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RadiometricParams:
    """Per-image radiometric parameters, camera-agnostic."""

    black_level: float  # sensor black level, subtract before anything else
    exposure_time: float  # seconds
    gain: float  # linear gain (ISO / base_ISO, or a direct sensor gain factor)
    irradiance: float  # W/m^2, from the sunshine sensor, per band
    vignetting_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    band_name: str = ""  # "GRE" | "RED" | "REG" | "NIR"
    camera_make: str = ""
    camera_model: str = ""
