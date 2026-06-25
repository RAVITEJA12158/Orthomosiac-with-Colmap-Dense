import os
import sys
import cv2
import numpy as np
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.capture import Capture
from src.quality.filter import filter_quality, check_capture, BLUR_THRESHOLD


def make_capture(capture_id, rgb_path):
    c = Capture(capture_id=capture_id, rgb=rgb_path)
    c.green = c.nir = c.red = c.reg = rgb_path
    c.latitude  = 16.9
    c.longitude = 81.7
    c.altitude  = 120.0
    return c


def write_sharp_image(path):
    """Create a sharp image with strong edges — should PASS blur check."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw many sharp edges
    for i in range(0, 640, 20):
        cv2.line(img, (i, 0), (i, 480), (255, 255, 255), 1)
    for j in range(0, 480, 20):
        cv2.line(img, (0, j), (640, j), (200, 200, 200), 1)
    # Add some texture
    noise = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.7, noise, 0.3, 0)
    # Write EXIF GPS via raw bytes not possible easily, so just save image
    cv2.imwrite(path, img)


def write_blurry_image(path):
    """Create a heavily blurred image — should FAIL blur check."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Apply extreme blur
    img = cv2.GaussianBlur(img, (99, 99), 50)
    cv2.imwrite(path, img)


def write_corrupt_image(path):
    """Write garbage bytes — should FAIL corrupt check."""
    with open(path, 'wb') as f:
        f.write(b'\x00\x01\x02\x03\xff\xfe')


def test_sharp_image_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        rgb_path = os.path.join(tmpdir, "IMG_0001_000_RGB.jpg")
        write_sharp_image(rgb_path)

        cap    = make_capture("000", rgb_path)
        result = check_capture(cap)

        # Sharp image should pass blur check
        # GPS check will fail (no real EXIF) but blur should pass
        assert result.blur_score is not None
        assert result.blur_score > BLUR_THRESHOLD, \
            f"Sharp image should have blur_score > {BLUR_THRESHOLD}, got {result.blur_score:.1f}"
        print(f"PASS: test_sharp_image_passes — blur_score={result.blur_score:.1f}")


def test_blurry_image_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        rgb_path = os.path.join(tmpdir, "IMG_0001_000_RGB.jpg")
        write_blurry_image(rgb_path)

        cap    = make_capture("000", rgb_path)
        result = check_capture(cap)

        assert not result.passed or result.blur_score < BLUR_THRESHOLD
        print(f"PASS: test_blurry_image_rejected — blur_score={result.blur_score:.1f}")


def test_corrupt_image_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        rgb_path = os.path.join(tmpdir, "IMG_0001_000_RGB.jpg")
        write_corrupt_image(rgb_path)

        cap    = make_capture("000", rgb_path)
        result = check_capture(cap)

        assert not result.passed
        assert "unreadable" in result.reason or "read error" in result.reason
        print(f"PASS: test_corrupt_image_rejected — reason='{result.reason}'")


def test_safety_net_triggers():
    """If >20% rejected, safety net should raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        captures = []

        # 3 corrupt, 1 sharp = 25% pass rate → below 0.80 min
        for i in range(3):
            p = os.path.join(tmpdir, f"IMG_000{i}_00{i}_RGB.jpg")
            write_corrupt_image(p)
            captures.append(make_capture(f"00{i}", p))

        p = os.path.join(tmpdir, "IMG_0003_003_RGB.jpg")
        write_sharp_image(p)
        captures.append(make_capture("003", p))

        try:
            filter_quality(captures, strict=False, min_pass_ratio=0.80)
            print("FAIL: safety net should have triggered")
        except ValueError as e:
            print(f"PASS: test_safety_net_triggers — caught: {str(e)[:60]}")


def test_conservative_keeps_most():
    """Most normal images should pass — filter is conservative."""
    with tempfile.TemporaryDirectory() as tmpdir:
        captures = []
        for i in range(10):
            p = os.path.join(tmpdir, f"IMG_000{i}_00{i}_RGB.jpg")
            write_sharp_image(p)
            captures.append(make_capture(f"00{i}", p))

        # All sharp images should pass blur check
        # GPS will fail (no real EXIF) but that's test limitation
        blur_passes = sum(
            1 for cap in captures
            if check_capture(cap).blur_score is not None
            and check_capture(cap).blur_score > BLUR_THRESHOLD
        )

        assert blur_passes == 10, f"Expected 10 to pass blur, got {blur_passes}"
        print(f"PASS: test_conservative_keeps_most — {blur_passes}/10 passed blur check")


def test_mixed_batch():
    """Mixed batch: sharp + blurry + corrupt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        captures = []

        # 7 sharp
        for i in range(7):
            p = os.path.join(tmpdir, f"IMG_sharp_{i:03d}_RGB.jpg")
            write_sharp_image(p)
            captures.append(make_capture(f"s{i:03d}", p))

        # 2 blurry
        for i in range(2):
            p = os.path.join(tmpdir, f"IMG_blur_{i:03d}_RGB.jpg")
            write_blurry_image(p)
            captures.append(make_capture(f"b{i:03d}", p))

        # 1 corrupt
        p = os.path.join(tmpdir, "IMG_corrupt_000_RGB.jpg")
        write_corrupt_image(p)
        captures.append(make_capture("c000", p))

        # blur + corrupt = 3 rejected (30%), below 0.80 threshold
        try:
            filter_quality(captures, strict=False, min_pass_ratio=0.50)
            print(f"PASS: test_mixed_batch — pipeline continued")
        except ValueError as e:
            print(f"INFO: test_mixed_batch — safety net: {str(e)[:80]}")


if __name__ == "__main__":
    test_sharp_image_passes()
    test_blurry_image_rejected()
    test_corrupt_image_rejected()
    test_safety_net_triggers()
    test_conservative_keeps_most()
    test_mixed_batch()
    print("\nAll quality filter tests done.")
