from dataclasses import dataclass
from typing import Optional


@dataclass
class Capture:
    capture_id: str
    rgb: Optional[str] = None
    green: Optional[str] = None
    nir: Optional[str] = None
    red: Optional[str] = None
    reg: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    has_rtk: bool = False
    # Focal length derived from EXIF (FocalLength + FocalPlaneXResolution),
    # in pixels, measured at `focal_length_ref_width` image width.
    # None if EXIF did not contain usable focal-plane tags — callers must
    # fall back to a heuristic and must NOT mark the COLMAP camera as
    # having a prior focal length in that case.
    focal_length_px: Optional[float] = None
    focal_length_ref_width: Optional[int] = None

    def is_complete(self) -> bool:
        return all([
            self.rgb,
            self.green,
            self.nir,
            self.red,
            self.reg,
        ])

    def missing_bands(self) -> list[str]:
        missing = []
        if not self.rgb:   missing.append("RGB")
        if not self.green: missing.append("GRE")
        if not self.nir:   missing.append("NIR")
        if not self.red:   missing.append("RED")
        if not self.reg:   missing.append("REG")
        return missing

    def __repr__(self):
        status = "complete" if self.is_complete() else f"missing {self.missing_bands()}"
        return f"Capture(id={self.capture_id}, gps=({self.latitude}, {self.longitude}), {status})"