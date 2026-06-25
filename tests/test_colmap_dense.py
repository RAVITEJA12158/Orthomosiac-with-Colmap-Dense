import json
from pathlib import Path

import numpy as np
import pytest

from src.depth.colmap_dense import image_name_from_colmap_depth_path, read_colmap_depth_map
from src.dsm.fusion import resolve_fused_pointcloud


def _write_colmap_depth(path, array):
    height, width = array.shape
    header = f"{width}&{height}&1&".encode("ascii")
    path.write_bytes(header + array.astype(np.float32).tobytes())


def test_read_colmap_depth_map(tmp_path):
    expected = np.array([[1.0, 2.5, 0.0], [3.25, 4.5, 5.75]], dtype=np.float32)
    path = tmp_path / "image.jpg.geometric.bin"
    _write_colmap_depth(path, expected)

    actual = read_colmap_depth_map(path)

    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_image_name_from_colmap_depth_path():
    assert image_name_from_colmap_depth_path(Path("IMG_0001.JPG.geometric.bin")) == "IMG_0001.JPG"
    assert image_name_from_colmap_depth_path(Path("IMG_0002.JPG.photometric.bin")) == "IMG_0002.JPG"


def test_resolve_fused_pointcloud_from_manifest(tmp_path):
    depth_root = tmp_path / "depth"
    dmap_dir = depth_root / "dmaps"
    dmap_dir.mkdir(parents=True)
    dmap_path = dmap_dir / "IMG_0001.JPG.dmap"
    dmap_path.write_bytes(b"not-empty")

    fused = depth_root / "dense" / "fused.ply"
    fused.parent.mkdir()
    fused.write_text("ply\n", encoding="utf-8")
    (depth_root / "depth_manifest.json").write_text(
        json.dumps({"fused_pointcloud_path": str(fused)}),
        encoding="utf-8",
    )

    assert resolve_fused_pointcloud([str(dmap_path)]) == str(fused.resolve())


def test_resolve_fused_pointcloud_missing(tmp_path):
    dmap = tmp_path / "depth" / "dmaps" / "missing.dmap"
    dmap.parent.mkdir(parents=True)
    dmap.write_bytes(b"not-empty")

    with pytest.raises(FileNotFoundError, match="COLMAP fused point cloud not found"):
        resolve_fused_pointcloud([str(dmap)])
