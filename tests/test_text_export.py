import tempfile
from pathlib import Path
import numpy as np
import pytest
from src.depth.text_export import write_colmap_text

class MockCamera:
    def __init__(self, model_name, width, height, params):
        self.model_name = model_name
        self.width = width
        self.height = height
        self.params = params

class MockPoint2D:
    def __init__(self, x, y, point3D_id):
        self.xy = (x, y)
        self.point3D_id = point3D_id

class MockTrackElement:
    def __init__(self, image_id, point2D_idx):
        self.image_id = image_id
        self.point2D_idx = point2D_idx

class MockPoint3D:
    def __init__(self, xyz, color, error, track_elements):
        self.xyz = xyz
        self.color = color
        self.error = error
        self.track = track_elements

# pycolmap 4.x mock objects
class MockRotation3d:
    def __init__(self, quat_xyzw):
        self.quat = quat_xyzw

class MockRigid3d:
    def __init__(self, quat_xyzw, translation):
        self.rotation = MockRotation3d(quat_xyzw)
        self.translation = translation

class MockImageV4:
    def __init__(self, name, camera_id, quat_xyzw, translation, points2D, callable_cfw=False):
        self.name = name
        self.camera_id = camera_id
        self.points2D = points2D
        cfw_obj = MockRigid3d(quat_xyzw, translation)
        if callable_cfw:
            self.cam_from_world = lambda: cfw_obj
        else:
            self.cam_from_world = cfw_obj

# pycolmap 3.x mock image
class MockImageV3:
    def __init__(self, name, camera_id, qvec, tvec, points2D):
        self.name = name
        self.camera_id = camera_id
        self.qvec = qvec
        self.tvec = tvec
        self.points2D = points2D

class MockReconstruction:
    def __init__(self, cameras, images, points3D):
        self.cameras = cameras
        self.images = images
        self.points3D = points3D

def test_write_colmap_text_v3(tmp_path):
    cameras = {1: MockCamera("PINHOLE", 640, 480, [500.0, 500.0, 320.0, 240.0])}
    images = {
        1: MockImageV3("img1.jpg", 1, [0.707, 0.0, 0.707, 0.0], [1.0, 2.0, 3.0], [MockPoint2D(100.0, 200.0, 5)])
    }
    points3D = {
        5: MockPoint3D([10.0, 20.0, 30.0], [255, 128, 64], 1.5, [MockTrackElement(1, 0)])
    }
    
    recon = MockReconstruction(cameras, images, points3D)
    output_dir = tmp_path / "colmap_text_v3"
    write_colmap_text(recon, str(output_dir))
    
    # Verify cameras.txt
    cam_file = output_dir / "cameras.txt"
    assert cam_file.exists()
    cam_lines = cam_file.read_text().splitlines()
    assert cam_lines[-1] == "1 PINHOLE 640 480 500.0 500.0 320.0 240.0"
    
    # Verify images.txt
    img_file = output_dir / "images.txt"
    assert img_file.exists()
    img_lines = img_file.read_text().splitlines()
    # Find non-comment lines
    data_lines = [l for l in img_lines if not l.startswith("#")]
    assert len(data_lines) == 2
    assert data_lines[0] == "1 0.707 0.0 0.707 0.0 1.0 2.0 3.0 1 img1.jpg"
    assert data_lines[1] == "100.0 200.0 5"
    
    # Verify points3D.txt
    pts_file = output_dir / "points3D.txt"
    assert pts_file.exists()
    pts_lines = pts_file.read_text().splitlines()
    pts_data = [l for l in pts_lines if not l.startswith("#")]
    assert len(pts_data) == 1
    assert pts_data[0] == "5 10.0 20.0 30.0 255 128 64 1.5 1 0"

def test_write_colmap_text_v4_direct_and_callable(tmp_path):
    cameras = {1: MockCamera("PINHOLE", 640, 480, [500.0, 500.0, 320.0, 240.0])}
    images = {
        # Image 1: cam_from_world is direct attribute
        # xyzw = [0.0, 0.707, 0.0, 0.707] -> qw, qx, qy, qz = [0.707, 0.0, 0.707, 0.0]
        1: MockImageV4("img1.jpg", 1, [0.0, 0.707, 0.0, 0.707], [1.0, 2.0, 3.0], [MockPoint2D(100.0, 200.0, 5)], callable_cfw=False),
        # Image 2: cam_from_world is a callable method (bound method)
        2: MockImageV4("img2.jpg", 1, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [MockPoint2D(150.0, 250.0, -1)], callable_cfw=True)
    }
    points3D = {
        5: MockPoint3D([10.0, 20.0, 30.0], [255, 128, 64], 1.5, [MockTrackElement(1, 0)])
    }
    
    recon = MockReconstruction(cameras, images, points3D)
    output_dir = tmp_path / "colmap_text_v4"
    write_colmap_text(recon, str(output_dir))
    
    # Verify images.txt
    img_file = output_dir / "images.txt"
    assert img_file.exists()
    img_lines = img_file.read_text().splitlines()
    data_lines = [l for l in img_lines if not l.startswith("#")]
    assert len(data_lines) == 4
    
    # Verify first image (q_w, q_x, q_y, q_z = 0.707, 0.0, 0.707, 0.0)
    assert data_lines[0] == "1 0.707 0.0 0.707 0.0 1.0 2.0 3.0 1 img1.jpg"
    assert data_lines[1] == "100.0 200.0 5"
    
    # Verify second image (q_w, q_x, q_y, q_z = 1.0, 0.0, 0.0, 0.0)
    assert data_lines[2] == "2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 img2.jpg"
    assert data_lines[3] == "150.0 250.0 -1"
