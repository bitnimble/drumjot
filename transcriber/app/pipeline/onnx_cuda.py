"""Make onnxruntime's CUDA provider find its runtime libs in a torch-free process.

The transcriber ships torch-free, so nothing pulls torch's bundled CUDA runtime
(`libcublasLt.so.12` / `cublasLt64_12.dll`, cuDNN, ...) into the process.
onnxruntime-gpu then can't load its CUDA provider (`libcublasLt.so.12: cannot open
shared object file`) and silently falls back to the CPU EP -- fatal for the fp16
GRU models (onset heads / ADTOF), which the CPU EP can't run, and ~7x slower for
the rest. The desktop broker spawns the sidecar with no `LD_LIBRARY_PATH` (and a
bundled app has no reason to set it), so we make the libs findable ourselves.

The libs ARE present in the venv, just not on the loader path:
  - Linux: torch's `nvidia-*` wheels under `site-packages/nvidia/*/lib/*.so`. We
    preload them with `RTLD_GLOBAL` so the provider's `dlopen` resolves against
    the loaded copies.
  - Windows: torch bundles the CUDA DLLs in `site-packages/torch/lib` (and/or the
    `nvidia-*` wheels' `bin` dirs). We add those to the DLL search path via
    `os.add_dll_directory` (what torch itself does on import).
  - macOS: no CUDA (uses the CoreML EP), and CPU-only installs have no `nvidia/`
    dir -- both are a no-op here.

Idempotent + best-effort. Call `preload_cuda_libs()` before creating any
onnxruntime session with a GPU EP (done at `app/sidecar.py::main`).
"""
from __future__ import annotations

import ctypes
import glob
import logging
import os
import sysconfig

log = logging.getLogger(__name__)

_done = False
# Keeps the os.add_dll_directory handles alive (closing them drops the dir).
_dll_dirs: list = []


def preload_cuda_libs() -> None:
    """Make the venv's CUDA runtime libs findable by onnxruntime-gpu. Idempotent;
    no-op on a box with no CUDA libs (CPU-only, or macOS/CoreML)."""
    global _done
    if _done:
        return
    _done = True

    purelib = sysconfig.get_paths()["purelib"]
    if os.name == "nt":
        _add_windows_dll_dirs(purelib)
    else:
        _preload_unix(os.path.join(purelib, "nvidia"))


def _add_windows_dll_dirs(purelib: str) -> None:
    # torch/lib carries the CUDA DLLs on Windows (torch bundles them); the
    # nvidia-*-cu12 wheels, if present, put theirs under nvidia/<pkg>/bin. Add
    # both so onnxruntime's LoadLibrary of its CUDA provider resolves them.
    dirs = [os.path.join(purelib, "torch", "lib")]
    dirs += glob.glob(os.path.join(purelib, "nvidia", "*", "bin"))
    added = 0
    for d in dirs:
        if os.path.isdir(d):
            try:
                _dll_dirs.append(os.add_dll_directory(d))  # type: ignore[attr-defined]  # win-only
                added += 1
            except OSError:
                pass
    log.info("preload_cuda_libs: added %d CUDA DLL dir(s)", added)


def _try_load(path: str) -> bool:
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def _preload_unix(base: str) -> None:
    if not os.path.isdir(base):
        return  # no torch CUDA wheels (CPU-only box, or macOS) -> nothing to do
    libs = glob.glob(os.path.join(base, "*", "lib", "*.so*"))
    if not libs:
        return
    # A few passes so a lib whose dependency is loaded by a later entry still gets
    # in; the dynamic loader dedups, so re-attempting a loaded lib is cheap.
    remaining = list(libs)
    for _ in range(4):
        still = [lib for lib in remaining if not _try_load(lib)]
        if len(still) == len(remaining):
            break
        remaining = still
    log.info("preload_cuda_libs: loaded %d/%d nvidia libs", len(libs) - len(remaining), len(libs))
