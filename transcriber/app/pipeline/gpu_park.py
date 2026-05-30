"""GPU-residency control for warm model singletons.

The pipeline worker holds several model singletons across the process
lifetime: the two drum separators (eagerly loaded at startup), the
vocals separator (lazy, /lyrics/align only), Beat Transformer + ADTOF
(lazy, /transcribe only), and the lyrics aligner's CTC checkpoints
(lazy). On a 6 GB consumer GPU all of these resident simultaneously
exceeds the budget once CTC forced alignment tries to allocate its
emissions tensor.

This module exposes a "park to CPU" primitive: move an nn.Module's
parameters/buffers to host RAM and `torch.cuda.empty_cache()` so the
freed VRAM is actually returned to the allocator. Unpark is the
inverse - a CUDA <-> host memcpy, ~hundreds of ms, vs reloading from
disk + state_dict which is multi-second.

Coordinator functions swap "warm but parked" sets at endpoint entry:

  - /transcribe entry   -> park lyrics models, unpark drum models
  - /lyrics/align entry -> park drum models, unpark vocals + CTC

Process-wide serialization across the two endpoints is the caller's
responsibility (see main.py::_gpu_lock); parking a model while another
request is mid-stream through it would device-mismatch the inputs.

Each helper is idempotent (a model already on the target device is a
no-op) and a no-op when the underlying model hasn't been loaded yet
(lazy-cached models start absent).
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _empty_cache() -> None:
    """Return freshly-freed VRAM to the PyTorch allocator so the next
    allocation can actually use it. Called once per coordinator after a
    batch of parks rather than once per park to amortise the syscall."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _mem_allocated_mb() -> float:
    """Allocated VRAM in MB; used purely for the before/after log
    line. Returns 0.0 when CUDA isn't available."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def park_module(module: Any, label: str) -> None:
    """Move `module`'s parameters/buffers to CPU. Idempotent: a module
    already on CPU (or with no parameters at all) is a no-op. Any
    exception is swallowed and logged - parking is a memory-pressure
    optimisation, not a correctness primitive, so a torch oddity must
    never crash the request."""
    if module is None:
        return
    try:
        first_param = next(module.parameters(), None)
        if first_param is None or first_param.device.type == "cpu":
            return
        module.to("cpu")
        log.info("gpu_park: parked %s to CPU", label)
    except Exception as exc:
        log.warning("gpu_park: failed to park %s: %s", label, exc)


def unpark_module(module: Any, label: str) -> None:
    """Move `module` back to CUDA. Idempotent and a no-op when CUDA
    isn't available or the module is already there."""
    if module is None or not _cuda_available():
        return
    try:
        first_param = next(module.parameters(), None)
        if first_param is None or first_param.device.type == "cuda":
            return
        module.to("cuda")
        log.info("gpu_park: unparked %s to CUDA", label)
    except Exception as exc:
        log.warning("gpu_park: failed to unpark %s: %s", label, exc)


def park_for_lyrics(separator: Any, aligner: Any) -> None:
    """Free VRAM held by drum-pipeline models so /lyrics/align can
    safely load its CTC aligner. Drum separators (_stems_all,
    _stems_per) and the lazy lru_caches for Beat Transformer and ADTOF
    get parked to CPU; the lyrics side (vocals separator + any
    previously-loaded CTC aligners) gets unparked so the request runs
    on a clean GPU.

    Called at the top of /lyrics/align under the process-wide GPU
    lock, so no in-flight transcribe stage can be holding a CUDA
    tensor whose source module is about to move host-side."""
    before = _mem_allocated_mb()
    separator.park_drum_models()
    # Lazy caches; no-op if their lru_cache hasn't been hit yet
    from app.pipeline import adtof_onsets, beat_transformer

    beat_transformer.park_model()
    adtof_onsets.park_model()
    # Lyrics side: vocals separator and CTC aligner(s) need to be on
    # GPU for the upcoming separate() / generate_emissions() calls.
    # Both are no-ops if never loaded.
    separator.unpark_vocals()
    aligner.unpark()
    _empty_cache()
    after = _mem_allocated_mb()
    log.info(
        "gpu_park: park_for_lyrics: VRAM %.0f MB -> %.0f MB",
        before, after,
    )


def park_for_transcribe(separator: Any, aligner: Any) -> None:
    """Free VRAM held by lyrics-side models so /transcribe and
    /transcribe/resume can run on a freshly cleared GPU. Drum
    separators come back to CUDA; BT and ADTOF lazy caches are
    unparked too (no-op if they've never loaded).

    Called at the top of /transcribe / /transcribe/resume under the
    process-wide GPU lock."""
    before = _mem_allocated_mb()
    separator.park_vocals()
    aligner.park()
    separator.unpark_drum_models()
    from app.pipeline import adtof_onsets, beat_transformer

    beat_transformer.unpark_model()
    adtof_onsets.unpark_model()
    _empty_cache()
    after = _mem_allocated_mb()
    log.info(
        "gpu_park: park_for_transcribe: VRAM %.0f MB -> %.0f MB",
        before, after,
    )


def park_vocals_after_extraction(separator: Any) -> None:
    """Park the vocals separator after /lyrics/align has extracted
    the vocals stem and BEFORE the CTC aligner loads. Frees ~1.5 GB
    so the wav2vec2 CTC aligner can allocate without OOM. Idempotent
    (no-op when _vocals was a cache hit on disk and never loaded
    into GPU this request)."""
    before = _mem_allocated_mb()
    separator.park_vocals()
    _empty_cache()
    after = _mem_allocated_mb()
    log.info(
        "gpu_park: park_vocals_after_extraction: VRAM %.0f MB -> %.0f MB",
        before, after,
    )
