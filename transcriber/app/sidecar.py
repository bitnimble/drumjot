"""`python -m app.sidecar` -- the stdio backend the Tauri broker spawns.

stdout carries the control protocol; everything else (logging, library chatter)
goes to stderr so it can't corrupt the frame stream.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

from app.comms.runners import build_registry
from app.comms.stdio_adapter import StdioAdapter
from app.pipeline.onnx_cuda import preload_cuda_libs

# Age past which an orphaned runner scratch dir is safe to reap. Far longer than
# any real job, so an in-flight concurrent sidecar's dir (mtime ~now) is never
# touched.
_SCRATCH_MAX_AGE_SEC = 2 * 60 * 60


def _sweep_stale_scratch() -> None:
    """Reap runner scratch dirs (`drumjot_*` under the temp dir) orphaned by a
    previously cancelled job: a cancel SIGKILLs the sidecar before its own
    work-dir cleanup runs. Age-gated and best-effort. (The frontend's staged
    inputs live under `drumjot/`, which `drumjot_` doesn't match.)"""
    cutoff = time.time() - _SCRATCH_MAX_AGE_SEC
    try:
        entries = list(Path(tempfile.gettempdir()).glob("drumjot_*"))
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    # The broker spawns us without LD_LIBRARY_PATH, so onnxruntime-gpu can't find
    # the CUDA libs on its own; preload them so GPU inference works (no-op on a
    # CPU-only box). Must run before any ORT session is created.
    preload_cuda_libs()
    _sweep_stale_scratch()
    # The control protocol owns the real stdout. Hand the adapter that stream,
    # then repoint sys.stdout at stderr so a stray print() in any dependency
    # (e.g. adtof_pytorch's weight-load message) can't inject a non-JSON line
    # into the frame stream. The adapter keeps its own reference, so its writes
    # still go to the real stdout.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    adapter = StdioAdapter(build_registry(), stdin=sys.stdin, stdout=protocol_out)
    asyncio.run(adapter.run())


if __name__ == "__main__":
    main()
