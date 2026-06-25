import sys
print(f"Python: {sys.version}")

# Core
import numpy; print(f"NumPy: {numpy.__version__}")
import cv2; print(f"OpenCV: {cv2.__version__}")
import scipy; print(f"SciPy: {scipy.__version__}")
import h5py; print(f"h5py: {h5py.__version__}")

# PyTorch + CUDA
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# LightGlue / ALIKED
try:
    from lightglue import ALIKED, LightGlue
    print("LightGlue: OK")
except ImportError as e:
    print(f"LightGlue: MISSING - {e}")

# pycolmap
try:
    import pycolmap
    print(f"pycolmap: {pycolmap.__version__}")
except ImportError as e:
    print(f"pycolmap: MISSING - {e}")

# poselib
try:
    import poselib
    print("poselib: OK")
except ImportError as e:
    print(f"poselib: MISSING - {e}")

# rasterio + GDAL
try:
    import rasterio
    print(f"rasterio: {rasterio.__version__}")
    from osgeo import gdal
    print(f"GDAL: {gdal.__version__}")
except ImportError as e:
    print(f"rasterio/GDAL: MISSING - {e}")

# CuPy (optional)
try:
    import cupy
    print(f"CuPy: {cupy.__version__} (GPU orthorectification: ENABLED)")
except ImportError:
    print("CuPy: not installed (GPU orthorectification will use CPU NumPy fallback)")

# COLMAP binary
import shutil
colmap = shutil.which("colmap") or shutil.which("colmap.exe") or shutil.which("COLMAP.bat")
print(f"COLMAP binary: {colmap or 'NOT FOUND - install COLMAP 4.x and add to PATH'}")

print("\nAll critical checks complete.")
