# Agri Orthomosaic Engine — Complete Setup & Usage Guide

> **Who is this for?** Anyone who wants to install and run the engine from scratch.
> No prior knowledge of photogrammetry is assumed. Just follow the steps in order.

---

## Table of Contents

1. [What You Need (Hardware)](#1-what-you-need-hardware)
2. [What You Need (Software Pre-requisites)](#2-what-you-need-software-pre-requisites)
3. [Install Python](#3-install-python)
4. [Clone the Project](#4-clone-the-project)
5. [Create a Virtual Environment](#5-create-a-virtual-environment)
6. [Install Python Packages](#6-install-python-packages)
7. [Install COLMAP (CLI binary)](#7-install-colmap-cli-binary)
8. [Install OpenMVS (depth maps)](#8-install-openmvs-depth-maps)
9. [Install CUDA (for GPU acceleration)](#9-install-cuda-for-gpu-acceleration)
10. [Verify Everything Works](#10-verify-everything-works)
11. [Prepare Your Mission Folder](#11-prepare-your-mission-folder)
12. [How to Run the Pipeline](#12-how-to-run-the-pipeline)
13. [Understanding the Outputs](#13-understanding-the-outputs)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. What You Need (Hardware)

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GPU with 8 GB VRAM | NVIDIA GPU with 16 GB VRAM |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 50 GB free | 200 GB SSD |
| **CPU** | 8 cores | 16+ cores |
| **OS** | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |

> ⚠️ **The GPU must be NVIDIA.** AMD GPUs are not supported — CUDA is required.
> The engine will run on CPU only, but depth estimation will take 10–20× longer.

---

## 2. What You Need (Software Pre-requisites)

Before installing anything, make sure you have these already:

### Windows
- **Git for Windows**: https://git-scm.com/download/win
- **Visual Studio Build Tools 2022**: https://visualstudio.microsoft.com/downloads/
  *(required for some packages that compile C extensions)*

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake wget curl libgl1
```

---

## 3. Install Python

The engine requires **Python 3.10 or 3.11**. Python 3.12+ is not recommended (pycolmap wheel availability).

### Windows
1. Go to https://www.python.org/downloads/
2. Download Python **3.11.x** (the latest 3.11 release)
3. Run the installer — **tick "Add Python to PATH"** before clicking Install
4. Open PowerShell and verify:
   ```powershell
   python --version
   # Should print: Python 3.11.x
   ```

### Linux
```bash
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
# Make python3.11 the default python3
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
python3 --version   # should print Python 3.11.x
```

---

## 4. Clone the Project

```bash
# Windows (PowerShell) or Linux (Terminal)
git clone https://github.com/your-org/orthomosaic_engine.git
cd orthomosaic_engine
```

---

## 5. Create a Virtual Environment

A virtual environment keeps all the engine's packages separate from your system Python.
**Always activate it before running any commands.**

### Windows (PowerShell)
```powershell
# Create the environment (only once)
python -m venv .venv

# Activate it (every time you open a new terminal)
.\.venv\Scripts\Activate.ps1

# You should see (.venv) in your prompt
```

> If PowerShell says "execution of scripts is disabled", run this once as Administrator:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Linux
```bash
# Create (only once)
python3.11 -m venv .venv

# Activate (every new terminal)
source .venv/bin/activate

# Prompt will show (.venv)
```

---

## 6. Install Python Packages

All packages install with a single command. Make sure your virtual environment is activated first.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### What gets installed and why

| Package | Version | Stage | What it does |
| :--- | :--- | :--- | :--- |
| `ExifRead` | ≥3.0 | Stage 1, 2 | Reads GPS coordinates from JPEG EXIF metadata |
| `pyexiftool` | ≥0.5.6 | Stage 12 | Reads Sequoia/MicaSense XMP radiometric fields |
| `numpy` | ≥1.26 | All stages | Core array mathematics |
| `torch` | ≥2.2 | Stage 3, 4 | GPU tensor operations for ALIKED + LightGlue |
| `torchvision` | ≥0.17 | Stage 3 | Image loading utilities used by LightGlue |
| `lightglue` | ≥0.1 | Stage 3, 4 | ALIKED feature extractor + LightGlue matcher |
| `kornia` | ≥0.7.3 | Stage 3, 4 | Image geometry utilities (LightGlue dependency) |
| `h5py` | ≥3.10 | Stage 3, 4 | HDF5 file storage for features and matches |
| `pycolmap` | ≥4.0.4 | Stage 5, 6, 7 | Python bindings to COLMAP (SfM) |
| `poselib` | latest | Stage 5 | Fast LO-RANSAC geometric verification |
| `rasterio` | ≥1.3 | Stage 9–12 | Read/write georeferenced GeoTIFF files |
| `GDAL` | ≥3.8 | Stage 9 | DSM gap filling (fillnodata) + reprojection |
| `pyproj` | ≥3.6 | Stage 9, 10 | CRS transformations (WGS84 ↔ UTM) |
| `scipy` | ≥1.12 | Stage 9, 12 | DSM gap interpolation + MS blend weights |
| `cupy-cuda12x` | ≥13.0 | Stage 10 | GPU backward projection (requires CUDA 12.x) |
| `opencv-python` | ≥4.9 | Stage 1, 10, 11, 12 | Image processing, mosaicking, ArUco detection |
| `Pillow` | auto | Stage 3 | Image reading (LightGlue dependency) |
| `open3d` | optional | Stage 9 | Faster PLY point cloud reading (fallback exists) |
| `pytest` | ≥9.0 | Testing | Run the test suite |

### Install PyTorch with CUDA (important — do this separately)

The `requirements.txt` installs the CPU version of PyTorch by default. For GPU acceleration:

**Find your CUDA version first:**
```bash
# Windows PowerShell or Linux
nvidia-smi
# Look for "CUDA Version: 12.x" in the top-right corner
```

**Then install the matching PyTorch:**
```bash
# For CUDA 12.1 (most common with recent NVIDIA drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8 (older GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

PyTorch official install selector: https://pytorch.org/get-started/locally/

### Install CuPy (GPU orthorectification)

CuPy must match your CUDA version exactly:

```bash
# CUDA 13.x 
pip install cupy-cuda13x

# CUDA 12.x (any 12.x version)
pip install cupy-cuda12x

# CUDA 11.x
pip install cupy-cuda11x

# No GPU / just testing
# CuPy is optional — the engine falls back to NumPy automatically
```

CuPy documentation: https://docs.cupy.dev/en/stable/install.html

### Install GDAL (tricky on Windows)

GDAL has C++ binaries and is the trickiest package to install.

**Windows — use the pre-built wheel:**
```powershell
# Go to: https://github.com/cgohlke/geospatial-wheels/releases
# Download GDAL-3.x.x-cpXXX-cpXXX-win_amd64.whl matching your Python version
# Then install it:
pip install GDAL-3.x.x-cpXXX-cpXXX-win_amd64.whl

# After GDAL, install rasterio:
pip install rasterio
```

**Linux:**
```bash
sudo apt-get install -y gdal-bin libgdal-dev python3-gdal
pip install GDAL==$(gdal-config --version) rasterio pyproj
```

> **Alternative (both OS) — use conda:**
> ```bash
> conda install -c conda-forge gdal rasterio pyproj
> ```

### Install LightGlue

LightGlue is from ETH Zurich (not on PyPI by default — install from GitHub):
```bash
pip install lightglue
# OR directly from source (latest):
pip install git+https://github.com/cvg/LightGlue.git
```

LightGlue repo: https://github.com/cvg/LightGlue

### Install PoseLib
```bash
pip install poselib
```
PoseLib repo: https://github.com/PoseLib/PoseLib

---

## 7. Install COLMAP (CLI binary)

COLMAP is a separate program (not a Python package). The engine calls it as a subprocess.

### Windows
1. Go to https://github.com/colmap/colmap/releases
2. Download the file named: `COLMAP-x.x.x-windows-cuda.zip` (the CUDA version)
3. Extract to `C:\COLMAP`
4. Add to PATH:
   - Press `Win + S` → search "Environment Variables"
   - Click "Edit the system environment variables"
   - Click "Environment Variables" button
   - Under "System Variables", find `Path`, click Edit
   - Click "New" → type `C:\COLMAP`
   - Click OK on all windows
5. Open a **new** PowerShell and verify:
   ```powershell
   colmap help
   # Should print COLMAP usage information
   ```

### Linux
```bash
# Option A: conda (easiest, CUDA-enabled)
conda install -c conda-forge colmap

# Option B: apt (may not have CUDA)
sudo apt-get install -y colmap

# Option C: build from source (most control)
# https://colmap.github.io/install.html

# Verify
colmap help
```

---

## 8. Install OpenMVS (depth maps)
| Package | Version | Stage | What it does |
| :--- | :--- | :--- | :--- |
| `ExifRead` | ≥3.0 | Stage 1, 2 | Reads GPS coordinates from JPEG EXIF metadata |
| `pyexiftool` | ≥0.5.6 | Stage 12 | Reads Sequoia/MicaSense XMP radiometric fields |
| `numpy` | ≥1.26 | All stages | Core array mathematics |
| `torch` | ≥2.2 | Stage 3, 4 | GPU tensor operations for ALIKED + LightGlue |
| `torchvision` | ≥0.17 | Stage 3 | Image loading utilities used by LightGlue |
| `lightglue` | ≥0.1 | Stage 3, 4 | ALIKED feature extractor + LightGlue matcher |
| `kornia` | ≥0.7.3 | Stage 3, 4 | Image geometry utilities (LightGlue dependency) |
| `h5py` | ≥3.10 | Stage 3, 4 | HDF5 file storage for features and matches |
| `pycolmap` | ≥4.0.4 | Stage 5, 6, 7 | Python bindings to COLMAP (SfM) |
| `poselib` | latest | Stage 5 | Fast LO-RANSAC geometric verification |

OpenMVS generates the depth maps (Stage 8) and DSM (Stage 9). It is also a separate program.

### Windows
```powershell
# Option A: conda (recommended — easiest)
conda install -c conda-forge openmvs

# Verify
DensifyPointCloud --help
```

If conda is not available, download a pre-built binary:
1. Go to https://github.com/cdcseacave/openMVS/releases
2. Download the Windows `.zip` release
3. Extract to `C:\OpenMVS`
4. Add `C:\OpenMVS\bin` to your PATH (same steps as COLMAP above)

### Linux
```bash
# Option A: conda (easiest)
conda install -c conda-forge openmvs

# Option B: apt (Ubuntu 22.04+)
sudo apt-get install -y libopenmvs-dev openmvs

# Option C: build from source
# https://github.com/cdcseacave/openMVS/wiki/Building

# Verify
DensifyPointCloud --help
```
| Package | Version | Stage | What it does |
| :--- | :--- | :--- | :--- |
| `ExifRead` | ≥3.0 | Stage 1, 2 | Reads GPS coordinates from JPEG EXIF metadata |
| `pyexiftool` | ≥0.5.6 | Stage 12 | Reads Sequoia/MicaSense XMP radiometric fields |
| `numpy` | ≥1.26 | All stages | Core array mathematics |
| `torch` | ≥2.2 | Stage 3, 4 | GPU tensor operations for ALIKED + LightGlue |
| `torchvision` | ≥0.17 | Stage 3 | Image loading utilities used by LightGlue |
| `lightglue` | ≥0.1 | Stage 3, 4 | ALIKED feature extractor + LightGlue matcher |
| `kornia` | ≥0.7.3 | Stage 3, 4 | Image geometry utilities (LightGlue dependency) |
| `h5py` | ≥3.10 | Stage 3, 4 | HDF5 file storage for features and matches |
| `pycolmap` | ≥4.0.4 | Stage 5, 6, 7 | Python bindings to COLMAP (SfM) |
| `poselib` | latest | Stage 5 | Fast LO-RANSAC geometric verification |

OpenMVS repo: https://github.com/cdcseacave/openMVS

---

## 9. Install CUDA (for GPU acceleration)

If you don't have CUDA installed yet:

### Windows
1. Go to https://developer.nvidia.com/cuda-downloads
2. Select: Windows → x86_64 → your Windows version → exe (local)
3. Download and run the installer
4. Restart your computer
5. Verify:
   ```powershell
   nvidia-smi
   # Should show your GPU and CUDA version
   ```

### Linux
```bash
# Ubuntu — follow NVIDIA's official guide:
# https://developer.nvidia.com/cuda-downloads
# Select: Linux → x86_64 → Ubuntu → your version → deb (local)

# Quick verify after install:
nvidia-smi
| Package | Version | Stage | What it does |
| :--- | :--- | :--- | :--- |
| `ExifRead` | ≥3.0 | Stage 1, 2 | Reads GPS coordinates from JPEG EXIF metadata |
| `pyexiftool` | ≥0.5.6 | Stage 12 | Reads Sequoia/MicaSense XMP radiometric fields |
| `numpy` | ≥1.26 | All stages | Core array mathematics |
| `torch` | ≥2.2 | Stage 3, 4 | GPU tensor operations for ALIKED + LightGlue |
| `torchvision` | ≥0.17 | Stage 3 | Image loading utilities used by LightGlue |
| `lightglue` | ≥0.1 | Stage 3, 4 | ALIKED feature extractor + LightGlue matcher |
| `kornia` | ≥0.7.3 | Stage 3, 4 | Image geometry utilities (LightGlue dependency) |
| `h5py` | ≥3.10 | Stage 3, 4 | HDF5 file storage for features and matches |
| `pycolmap` | ≥4.0.4 | Stage 5, 6, 7 | Python bindings to COLMAP (SfM) |
| `poselib` | latest | Stage 5 | Fast LO-RANSAC geometric verification |
nvcc --version
```

CUDA download page: https://developer.nvidia.com/cuda-downloads

> 💡 **CUDA 12.1 or 12.4** is recommended. It has the widest support across PyTorch, CuPy, and COLMAP.

---

## 10. Verify Everything Works

Run this quick check to make sure all critical components are installed:

```python
# Save as check_install.py and run: python check_install.py

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
    print(f"LightGlue: MISSING — {e}")

# pycolmap
try:
    import pycolmap
    print(f"pycolmap: {pycolmap.__version__}")
except ImportError as e:
    print(f"pycolmap: MISSING — {e}")

# poselib
try:
    import poselib
    print("poselib: OK")
except ImportError as e:
    print(f"poselib: MISSING — {e}")

# rasterio + GDAL
try:
    import rasterio
    print(f"rasterio: {rasterio.__version__}")
    from osgeo import gdal
    print(f"GDAL: {gdal.__version__}")
except ImportError as e:
    print(f"rasterio/GDAL: MISSING — {e}")

# CuPy (optional)
try:
    import cupy
    print(f"CuPy: {cupy.__version__} (GPU orthorectification: ENABLED)")
except ImportError:
    print("CuPy: not installed (GPU orthorectification will use CPU NumPy fallback)")

# COLMAP binary
import shutil
colmap = shutil.which("colmap")
print(f"COLMAP binary: {colmap or 'NOT FOUND — add to PATH'}")

# OpenMVS binary
openmvs = shutil.which("DensifyPointCloud")
print(f"OpenMVS binary: {openmvs or 'NOT FOUND — add to PATH'}")

print("\nAll critical checks complete.")
```

Run it:
```bash
python check_install.py
```

**Expected output (with GPU):**
```
Python: 3.11.x
NumPy: 1.26.x
OpenCV: 4.9.x
SciPy: 1.12.x
h5py: 3.x.x
PyTorch: 2.2.x+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX ...
VRAM: 16.0 GB
LightGlue: OK
pycolmap: 4.x.x
poselib: OK
rasterio: 1.3.x
GDAL: 3.8.x
CuPy: 13.x.x (GPU orthorectification: ENABLED)
COLMAP binary: /usr/local/bin/colmap  (or C:\COLMAP\colmap.exe on Windows)
OpenMVS binary: /usr/local/bin/DensifyPointCloud
```

---

## 11. Prepare Your Mission Folder

### Required folder structure

```
your_mission/
├── rgb/
│   ├── IMG_0001_000_RGB.jpg
│   ├── IMG_0002_001_RGB.jpg
│   ├── IMG_0003_002_RGB.jpg
│   └── ...
└── multi/
    ├── IMG_0001_000_GRE.tiff
    ├── IMG_0001_000_RED.tiff
    ├── IMG_0001_000_REG.tiff
    ├── IMG_0001_000_NIR.tiff
    ├── IMG_0002_001_GRE.tiff
    ├── IMG_0002_001_RED.tiff
    ├── IMG_0002_001_REG.tiff
    ├── IMG_0002_001_NIR.tiff
    └── ...
```

### Filename format explained

```
IMG_{frame}_{capture_id}_{band}.{ext}
```

| Part | Example | Meaning |
| :--- | :--- | :--- |
| `frame` | `0001` | Random number — the engine ignores this |
| `capture_id` | `000` | **Grouping key** — all 5 files with the same capture_id belong to one moment in time |
| `band` | `RGB`, `GRE`, `RED`, `REG`, `NIR` | Which sensor band |
| `ext` | `.jpg` for RGB, `.tiff` for multispectral | File format |

**One capture group = 5 files with the same capture_id:**
- `IMG_XXXX_000_RGB.jpg`
- `IMG_XXXX_000_GRE.tiff`
- `IMG_XXXX_000_RED.tiff`
- `IMG_XXXX_000_REG.tiff`
- `IMG_XXXX_000_NIR.tiff`

### EXIF requirements on RGB images

Each RGB `.jpg` **must** have GPS data embedded in its EXIF:
- GPS Latitude
- GPS Longitude
- GPS Altitude

This is automatically set by the Parrot Sequoia + drone GPS.
If GPS is missing from any image, the quality filter will reject it.

### Calibration panel images

The Parrot Sequoia kit includes a **grey calibration panel** with 4 ArUco QR markers.
You must photograph this panel **at the start and/or end of every mission** using the multispectral camera.

The panel images go into the same `multi/` folder — the engine automatically detects them.

**What the panel looks like:**
- A flat grey square
- 4 black-and-white QR-like patterns at the corners (ArUco markers)
- Photographed on the ground before takeoff / after landing

If you do not have panel images, radiometric calibration will be skipped and NDVI values will be **relative, not absolute**.

---

## 12. How to Run the Pipeline

### Create a run script

Create a file called `run_pipeline.py` in the project root:

```python
"""
run_pipeline.py  —  Full Agri Orthomosaic Engine pipeline
Usage: python run_pipeline.py --mission /path/to/your_mission --output /path/to/outputs
"""

import argparse
from pathlib import Path

from src.ingestion import load_mission
from src.quality.filter import filter_captures
from src.features.extractor import extract_features
from src.features.matcher import match_features
from src.features.db_importer import import_to_colmap_db
from src.features.geometric_verification import verify_matches_poselib
from src.sfm import run_sfm
from src.depth import run_depth_pipeline
from src.dsm import run_dsm_pipeline
from src.ortho import run_ortho_pipeline
from src.mosaic import run_rgb_mosaic, run_ms_mosaic


def main():
    parser = argparse.ArgumentParser(description="Agri Orthomosaic Engine")
    parser.add_argument("--mission", required=True,
                        help="Path to mission folder (contains rgb/ and multi/)")
    parser.add_argument("--output",  required=True,
                        help="Output directory for all results")
    parser.add_argument("--gsd",     type=float, default=0.05,
                        help="Target ground sampling distance in metres/pixel (default 0.05 = 5cm)")
    parser.add_argument("--rtk",     action="store_true",
                        help="Set this flag if your drone has RTK GPS")
    parser.add_argument("--no-gpu",  action="store_true",
                        help="Disable GPU (run on CPU only — much slower)")
    args = parser.parse_args()

    mission_dir = args.mission
    output_dir  = Path(args.output)
    use_gpu     = not args.no_gpu

    # ── Stage 1+2: Ingest + quality filter ───────────────────────────────────
    print("\n=== Stage 1+2: Ingestion & Quality Filter ===")
    captures = load_mission(mission_dir)
    captures = filter_captures(captures)
    print(f"  {len(captures)} valid captures ready")

    # ── Stage 3: Feature extraction ───────────────────────────────────────────
    print("\n=== Stage 3: Feature Extraction (ALIKED) ===")
    extract_features(captures, output_dir=str(output_dir), use_gpu=use_gpu)

    # ── Stage 4: Feature matching ─────────────────────────────────────────────
    print("\n=== Stage 4: Feature Matching (LightGlue) ===")
    match_features(captures, output_dir=str(output_dir), n_neighbors=8, use_gpu=use_gpu)

    # ── Stage 5: Geometric verification ──────────────────────────────────────
    print("\n=== Stage 5: Geometric Verification (PoseLib) ===")
    db_path = str(output_dir / "database.db")
    import_to_colmap_db(captures, output_dir=str(output_dir), db_path=db_path)
    verify_matches_poselib(db_path)

    # ── Stage 6+7: SfM + Georeferencing ──────────────────────────────────────
    print("\n=== Stage 6+7: SfM Mapping + Georeferencing (COLMAP) ===")
    reconstruction = run_sfm(
        database_path = db_path,
        image_dir     = str(Path(mission_dir) / "rgb"),
        output_dir    = str(output_dir / "sparse"),
        captures      = captures,
        has_rtk       = args.rtk,
    )
    if reconstruction is None:
        print("ERROR: SfM failed. Check your images have GPS and enough overlap.")
        return

    # ── Stage 8+9: Depth maps + DSM ──────────────────────────────────────────
    print("\n=== Stage 8: Depth Maps (OpenMVS) ===")
    dmap_paths, mvs_scene = run_depth_pipeline(
        reconstruction = reconstruction,
        captures       = captures,
        output_dir     = str(output_dir / "depth"),
        use_gpu        = use_gpu,
    )

    print("\n=== Stage 9: DSM Generation (OpenMVS fusion) ===")
    dsm_path = run_dsm_pipeline(
        dmap_paths     = dmap_paths,
        mvs_scene_path = mvs_scene,
        reconstruction = reconstruction,
        output_dir     = str(output_dir / "dsm"),
        target_gsd_m   = args.gsd,
    )
    print(f"  DSM written to: {dsm_path}")

    # ── Stage 10: Orthorectification ─────────────────────────────────────────
    print("\n=== Stage 10: Orthorectification (CuPy) ===")
    ortho_result = run_ortho_pipeline(
        reconstruction        = reconstruction,
        captures              = captures,
        dsm_path              = dsm_path,
        output_dir            = str(output_dir / "ortho"),
        target_gsd_m          = args.gsd,
        process_multispectral = True,
    )
    print(f"  {len(ortho_result.rgb_tile_paths)} RGB tiles written")

    # ── Stage 11: RGB Mosaicking ──────────────────────────────────────────────
    print("\n=== Stage 11: RGB Mosaicking (OpenCV) ===")
    rgb_mosaic_path, seamlines = run_rgb_mosaic(
        tile_paths        = ortho_result.rgb_tile_paths,
        output_path       = str(output_dir / "rgb_orthomosaic.tif"),
        seamlines_save_dir= str(output_dir / "seamlines"),
        target_gsd_m      = args.gsd,
    )
    print(f"  RGB mosaic: {rgb_mosaic_path}")

    # ── Stage 12: Multispectral Mosaicking ───────────────────────────────────
    print("\n=== Stage 12: Multispectral Mosaicking (NumPy) ===")
    ms_mosaic_path = run_ms_mosaic(
        multi_tile_paths = ortho_result.multi_tile_paths,
        captures         = captures,
        seamline_set     = seamlines,
        output_path      = str(output_dir / "multispectral_orthomosaic.tif"),
        target_gsd_m     = args.gsd,
    )
    print(f"  MS mosaic: {ms_mosaic_path}")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n=== Pipeline Complete ===")
    print(f"  RGB orthomosaic:            {output_dir}/rgb_orthomosaic.tif")
    print(f"  Multispectral orthomosaic:  {output_dir}/multispectral_orthomosaic.tif")
    print(f"  DSM:                        {output_dir}/dsm/dsm.tif")


if __name__ == "__main__":
    main()
```

### Run it

```bash
# Activate virtual environment first
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux:
source .venv/bin/activate

# Basic run (normal GPS drone, 5 cm/pixel output)
python run_pipeline.py \
    --mission  /path/to/your_mission \
    --output   /path/to/outputs

# With custom GSD (10 cm/pixel — faster processing)
python run_pipeline.py \
    --mission  /path/to/your_mission \
    --output   /path/to/outputs \
    --gsd      0.10

# RTK GPS drone (enables faster GLOMAP solver)
python run_pipeline.py \
    --mission  /path/to/your_mission \
    --output   /path/to/outputs \
    --rtk

# CPU only (no GPU — very slow, only for testing)
python run_pipeline.py \
    --mission  /path/to/your_mission \
    --output   /path/to/outputs \
    --no-gpu
```

### Expected runtime (800–900 images, 16 GB VRAM)

| Stage | Time |
| :--- | :--- |
| Ingestion + Quality Filter | ~30 sec |
| Feature Extraction (ALIKED) | ~2 min |
| Feature Matching (LightGlue) | ~1 min |
| Geometric Verification | ~5 sec |
| SfM Mapping (COLMAP) | ~25–35 min |
| Depth Maps (OpenMVS) | ~8–15 min |
| DSM Generation | ~2–4 min |
| Orthorectification (CuPy GPU) | ~1–3 min |
| RGB Mosaicking | ~4–6 min |
| Multispectral Mosaicking | ~3–4 min |
| **Total** | **~49–73 min** |

---

## 13. Understanding the Outputs

After the pipeline finishes, your output folder contains:

```
outputs/
├── rgb_orthomosaic.tif           ← Main RGB map — open in QGIS or ArcGIS
├── multispectral_orthomosaic.tif ← 4-band MS map (Green, Red, RedEdge, NIR)
├── dsm/
│   └── dsm.tif                   ← Digital Surface Model (elevation)
├── ortho/
│   ├── rgb/                      ← Individual rectified RGB tiles (one per image)
│   └── multi/                    ← Individual rectified MS tiles per band
├── sparse/                       ← COLMAP 3D sparse model
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── depth/                        ← Per-image depth maps (.dmap files)
├── seamlines/
│   └── seamlines.npz             ← Saved seamlines for MS alignment
└── database.db                   ← COLMAP feature database
```

### How to open the results

**QGIS (free, recommended):**
1. Download QGIS: https://qgis.org/en/site/forusers/download.html
2. Open QGIS → Layer → Add Layer → Add Raster Layer
3. Select `rgb_orthomosaic.tif` — it will automatically georeference on the map

**Google Earth Pro (free):**
1. File → Import → `rgb_orthomosaic.tif`
2. GDAL converts it to KML overlay automatically

**Python / Rasterio:**
```python
import rasterio
import matplotlib.pyplot as plt
import numpy as np

with rasterio.open("outputs/rgb_orthomosaic.tif") as src:
    rgb = src.read([1, 2, 3])   # bands 1=R, 2=G, 3=B
    rgb = np.moveaxis(rgb, 0, -1)   # → (H, W, 3)

plt.figure(figsize=(15, 10))
plt.imshow(rgb)
plt.title("RGB Orthomosaic")
plt.axis("off")
plt.show()
```

**Read multispectral bands:**
```python
with rasterio.open("outputs/multispectral_orthomosaic.tif") as src:
    green   = src.read(1)   # Green band (550nm)
    red     = src.read(2)   # Red band (660nm)
    rededge = src.read(3)   # RedEdge band (735nm)
    nir     = src.read(4)   # NIR band (790nm)

# Compute NDVI
ndvi = (nir - red) / (nir + red + 1e-8)
print(f"NDVI range: {ndvi.min():.3f} to {ndvi.max():.3f}")
# Healthy crops: 0.4–0.9  |  Bare soil: 0.1–0.2  |  Water: negative
```

---

## 14. Troubleshooting
    print("\n=== Stage 5: Geometric Verification (PoseLib) ===")
    db_path = str(output_dir / "database.db")
    import_to_colmap_db(captures, output_dir=str(output_dir), db_path=db_path)
    verify_matches_poselib(db_path)

    # ── Stage 6+7: SfM + Georeferencing ──────────────────────────────────────
    print("\n=== Stage 6+7: SfM Mapping + Georeferencing (COLMAP) ===")
    reconstruction = run_sfm(
        database_path = db_path,
        image_dir     = str(Path(mission_dir) / "rgb"),

### "CUDA not available" — PyTorch can't find GPU
```bash
# Check CUDA install
nvidia-smi            # should show your GPU
nvcc --version        # should show CUDA version

# Reinstall PyTorch with the right CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### "colmap: command not found"
- Windows: Make sure you added the COLMAP folder to PATH and opened a **new** terminal
- Linux: `conda install -c conda-forge colmap` and make sure conda is activated

### "DensifyPointCloud: command not found"
- Same as COLMAP — check OpenMVS is on PATH
- Or pass the path explicitly: `run_depth_pipeline(..., openmvs_bin_dir="C:\\OpenMVS\\bin")`

### "GDAL import error" on Windows
```powershell
# Download the correct .whl from:
# https://github.com/cgohlke/geospatial-wheels/releases
# Choose the .whl matching: GDAL-3.x.x-cp311-cp311-win_amd64.whl (for Python 3.11)
pip install GDAL-3.x.x-cp311-cp311-win_amd64.whl
```

### "SfM failed — no reconstruction" 
Checklist:
- [ ] Do all RGB images have GPS in EXIF? (check with `exiftool image.jpg | grep GPS`)
- [ ] Is overlap ≥ 60%? (SfM needs enough shared features between images)
- [ ] Did feature extraction produce `.h5` files? (check `outputs/features/` folder)
- [ ] Did matching produce `matches.h5`? (check `outputs/matches.h5` exists and is >1MB)

### "No .dmap files produced" (OpenMVS fails)
- Make sure OpenMVS was compiled with CUDA support: `DensifyPointCloud --help` should mention CUDA
- Try with `resolution_level=2` (quarter resolution) to reduce VRAM usage for a test
- Check the OpenMVS log output for specific errors

### CuPy version mismatch
```bash
# Check your CUDA version:
nvidia-smi | grep "CUDA Version"

# Install matching CuPy:
pip install cupy-cuda12x    # for CUDA 12.x
pip install cupy-cuda11x    # for CUDA 11.x
```

### Out of memory (OOM) during depth maps
- Reduce resolution: set `resolution_level=2` in `run_depth_pipeline()`
- Default is `resolution_level=1` (half resolution)

### Images load as all-black in RGB mosaic
- Make sure the orthotile `.tif` files in `outputs/ortho/rgb/` are non-empty (>0 bytes)
- Check that the DSM covers the same area as the images (same UTM zone)

---

## Quick Reference Card

```bash
# 1. Activate environment
source .venv/bin/activate          # Linux
.\.venv\Scripts\Activate.ps1      # Windows

# 2. Verify install
python check_install.py

# 3. Run pipeline
python run_pipeline.py --mission /data/mission_2024_06_15 --output /data/outputs

# 4. Open result in QGIS
# File → Add Raster Layer → /data/outputs/rgb_orthomosaic.tif
```

---

## Dependency Summary Table

| Dependency | Type | Install |
| :--- | :--- | :--- |
| Python 3.11 | Language | https://www.python.org |
| NVIDIA CUDA 12.x | System | https://developer.nvidia.com/cuda-downloads |
| COLMAP | Binary (CLI) | https://github.com/colmap/colmap/releases |
| OpenMVS | Binary (CLI) | `conda install -c conda-forge openmvs` |
| numpy | pip | `pip install numpy` |
| opencv-python | pip | `pip install opencv-python` |
| torch + torchvision | pip | `pip install torch torchvision --index-url ...cu121` |
| lightglue | pip | `pip install lightglue` |
| kornia | pip | `pip install kornia` |
| h5py | pip | `pip install h5py` |
| pycolmap | pip | `pip install pycolmap` |
| poselib | pip | `pip install poselib` |
| rasterio | pip | `pip install rasterio` |
| GDAL | pip (special) | Pre-built wheel — see Section 6 |
| pyproj | pip | `pip install pyproj` |
| scipy | pip | `pip install scipy` |
| cupy-cuda12x | pip | `pip install cupy-cuda12x` |
| ExifRead | pip | `pip install ExifRead` |
| pyexiftool | pip | `pip install pyexiftool` |
