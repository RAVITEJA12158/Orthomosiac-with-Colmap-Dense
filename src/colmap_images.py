"""Shared COLMAP image-name helpers."""

from __future__ import annotations

from pathlib import Path

from .ingestion.capture import Capture


def colmap_image_name(capture: Capture) -> str:
    """
    Return the canonical image name stored in the COLMAP database.

    The importer normalizes these names to lowercase. Every mapper/filter/
    lookup path must use the same normalization; otherwise COLMAP loads zero
    images when an image-name allowlist is provided.
    """
    ext = Path(capture.rgb).suffix if capture.rgb else ".jpg"
    return f"{capture.capture_id}{ext}".lower()
