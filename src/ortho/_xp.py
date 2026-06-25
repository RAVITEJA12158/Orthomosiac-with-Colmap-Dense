"""
src/ortho/_xp.py

Tiny CuPy/NumPy compatibility shim.

CuPy's array API is intentionally NumPy-compatible, so the entire Stage 10
pipeline is written against a single `xp` module reference. On a machine
with a CUDA GPU and `cupy-cuda12x` installed, `xp` is `cupy` and every
array op in backward_project.py / dsm_sampler.py actually runs on the GPU.
If CuPy isn't importable (no GPU, or running tests on a CPU-only dev box),
`xp` silently falls back to NumPy — same code path, just slower.

`GPU_AVAILABLE` lets call sites log which mode they're in, since silently
running 900 images on CPU instead of GPU is a 50-100x slowdown worth
knowing about.
"""

try:
    import cupy as xp  # type: ignore

    GPU_AVAILABLE = True
except ImportError:

    GPU_AVAILABLE = False


def to_numpy(array):
    """Download a GPU array to host memory; no-op if already NumPy."""
    if GPU_AVAILABLE and hasattr(array, "get"):
        return array.get()
    return array
