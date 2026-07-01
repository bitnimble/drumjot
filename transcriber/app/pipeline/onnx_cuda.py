"""Make onnxruntime-gpu find the CUDA runtime libs in a torch-free process.

The transcriber ships torch-free, so nothing pulls torch's bundled CUDA runtime
(`libcublasLt.so.12`, `libcudnn.so.9`, ...) into the process. onnxruntime-gpu then
can't `dlopen` its CUDA provider (`libcublasLt.so.12: cannot open shared object
file`) and silently falls back to the CPU EP -- fatal for the fp16 GRU models
(onset heads / ADTOF), which the CPU EP can't run, and ~7x slower for the rest.

The libs ARE present (torch's `nvidia-*` wheels under the venv's
`site-packages/nvidia/*/lib`), just not on the dynamic-loader path: the desktop
Tauri sidecar broker spawns us without `LD_LIBRARY_PATH`, and a bundled app has
no reason to have it set either. Rather than depend on the launcher, we preload
the libs into the process with `RTLD_GLOBAL` so the provider's `dlopen` resolves
against the already-loaded copies. Idempotent + best-effort: a CPU-only box with
no `nvidia/` dir is a no-op, and libs ORT doesn't need (nccl/nvshmem) are allowed
to fail.

Call `preload_cuda_libs()` before creating any onnxruntime session with a GPU EP.
"""
from __future__ import annotations

import ctypes
import glob
import logging
import os
import sysconfig

log = logging.getLogger(__name__)

_done = False


def _try_load(path: str) -> bool:
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def preload_cuda_libs() -> None:
    """Preload the venv's bundled NVIDIA CUDA libs (RTLD_GLOBAL). Idempotent."""
    global _done
    if _done:
        return
    _done = True

    base = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
    libs = glob.glob(os.path.join(base, "*", "lib", "*.so*"))
    if not libs:
        return  # CPU-only install (no torch CUDA wheels) -> nothing to do

    # A few passes so a lib whose dependency is loaded by a later entry still
    # gets in; the dynamic loader dedups, so re-attempting a loaded lib is cheap.
    remaining = list(libs)
    for _ in range(4):
        still = [lib for lib in remaining if not _try_load(lib)]
        if len(still) == len(remaining):
            break
        remaining = still
    log.info("preload_cuda_libs: loaded %d/%d nvidia libs", len(libs) - len(remaining), len(libs))
