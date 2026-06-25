import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion import ValidationError, load_mission

WORK_DIR = Path(__file__).parent / "_work" / "ingestion"


def make_dummy_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff")  # minimal fake image bytes


def test_complete_mission():
    mission_dir = WORK_DIR / "complete"
    captures = ["000", "001", "002"]

    for cid in captures:
        make_dummy_file(mission_dir / "rgb" / f"IMG_0001_{cid}_RGB.jpg")
        make_dummy_file(mission_dir / "multi" / f"IMG_0002_{cid}_GRE.tiff")
        make_dummy_file(mission_dir / "multi" / f"IMG_0003_{cid}_NIR.tiff")
        make_dummy_file(mission_dir / "multi" / f"IMG_0004_{cid}_RED.tiff")
        make_dummy_file(mission_dir / "multi" / f"IMG_0005_{cid}_REG.tiff")

    result = load_mission(str(mission_dir))

    assert len(result) == 3
    assert result[0].capture_id == "000"
    assert result[0].is_complete()


def test_incomplete_strict():
    mission_dir = WORK_DIR / "incomplete_strict"
    cid = "000"
    make_dummy_file(mission_dir / "rgb" / f"IMG_0001_{cid}_RGB.jpg")
    make_dummy_file(mission_dir / "multi" / f"IMG_0002_{cid}_GRE.tiff")

    with pytest.raises(ValidationError):
        load_mission(str(mission_dir), strict=True)


def test_incomplete_lenient():
    mission_dir = WORK_DIR / "incomplete_lenient"
    make_dummy_file(mission_dir / "rgb" / "IMG_0001_000_RGB.jpg")
    make_dummy_file(mission_dir / "multi" / "IMG_0002_000_GRE.tiff")
    make_dummy_file(mission_dir / "multi" / "IMG_0003_000_NIR.tiff")
    make_dummy_file(mission_dir / "multi" / "IMG_0004_000_RED.tiff")
    make_dummy_file(mission_dir / "multi" / "IMG_0005_000_REG.tiff")

    make_dummy_file(mission_dir / "rgb" / "IMG_0001_001_RGB.jpg")
    make_dummy_file(mission_dir / "multi" / "IMG_0002_001_GRE.tiff")
    make_dummy_file(mission_dir / "multi" / "IMG_0004_001_RED.tiff")
    make_dummy_file(mission_dir / "multi" / "IMG_0005_001_REG.tiff")

    result = load_mission(str(mission_dir), strict=False)

    assert len(result) == 1
    assert result[0].capture_id == "000"


def test_unknown_files_ignored():
    mission_dir = WORK_DIR / "unknown_files"
    cid = "000"
    make_dummy_file(mission_dir / "rgb" / f"IMG_0001_{cid}_RGB.jpg")
    make_dummy_file(mission_dir / "rgb" / "thumbnail.jpg")
    make_dummy_file(mission_dir / "multi" / f"IMG_0002_{cid}_GRE.tiff")
    make_dummy_file(mission_dir / "multi" / f"IMG_0003_{cid}_NIR.tiff")
    make_dummy_file(mission_dir / "multi" / f"IMG_0004_{cid}_RED.tiff")
    make_dummy_file(mission_dir / "multi" / f"IMG_0005_{cid}_REG.tiff")

    result = load_mission(str(mission_dir))

    assert len(result) == 1
