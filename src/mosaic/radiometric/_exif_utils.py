"""
src/mosaic/radiometric/_exif_utils.py

Small shared helpers for pulling values out of a pyexiftool output dict.
Not part of the public module spec — internal to the reader
implementations, which otherwise differ only in which exact key names they
try.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


def get_first(exif_dict: dict, keys: List[str], default: Any = None) -> Any:
    """Return the value of the first key in `keys` present in exif_dict."""
    for key in keys:
        if key in exif_dict and exif_dict[key] not in (None, ""):
            return exif_dict[key]
    return default


def get_float(exif_dict: dict, keys: List[str], default: float = 0.0) -> float:
    value = get_first(exif_dict, keys, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_str(exif_dict: dict, keys: List[str], default: str = "") -> str:
    value = get_first(exif_dict, keys, default)
    return str(value) if value is not None else default


def parse_coeffs(value: Optional[Any]) -> np.ndarray:
    """Parse a vignetting-polynomial-style value (string or list) into a
    1D float32 array. Returns zeros (identity) on any parse failure."""
    if value is None:
        return np.zeros(3, dtype=np.float32)
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            return np.array([float(v) for v in value], dtype=np.float32)
        text = str(value).strip()
        parts = text.replace(";", ",").split(",") if "," in text else text.split()
        coeffs = np.array([float(p) for p in parts if p.strip() != ""], dtype=np.float32)
        return coeffs if coeffs.size > 0 else np.zeros(3, dtype=np.float32)
    except (ValueError, TypeError):
        return np.zeros(3, dtype=np.float32)
