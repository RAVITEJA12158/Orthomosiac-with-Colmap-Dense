import re
from pathlib import Path
from typing import List

from ..ingestion.capture import Capture
from ..ingestion.exif_reader import read_gps


RGB_PATTERN = re.compile(
    r"^IMG_\d+_(\d+)_RGB\.(jpg|jpeg|tiff|tif)$",
    re.IGNORECASE,
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff"}


def _capture_id_from_filename(path: Path) -> str:
    match = RGB_PATTERN.match(path.name)
    if match:
        return match.group(1)
    return path.stem


def load_rgb_captures(rgb_dir: str) -> List[Capture]:
    """
    Build Capture objects from a folder containing RGB images only.
    """
    image_dir = Path(rgb_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"RGB folder not found: {rgb_dir}")

    captures = []
    seen_ids = set()

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        capture_id = _capture_id_from_filename(image_path)
        if capture_id in seen_ids:
            raise ValueError(f"Duplicate capture ID '{capture_id}' from {image_path.name}")

        lat, lon, alt = read_gps(str(image_path))
        captures.append(
            Capture(
                capture_id=capture_id,
                rgb=str(image_path),
                latitude=lat,
                longitude=lon,
                altitude=alt,
            )
        )
        seen_ids.add(capture_id)

    if not captures:
        raise ValueError(f"No RGB images found in {rgb_dir}")

    return captures


def gps_summary(captures: List[Capture]) -> tuple[int, int]:
    with_gps = sum(c.latitude is not None and c.longitude is not None for c in captures)
    return with_gps, len(captures) - with_gps
