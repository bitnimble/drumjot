"""GPU precision/throughput knobs: TF32 + bf16 autocast, guarded for old GPUs.

The frozen MERT encoder is a 24-layer transformer, so it dominates wall time
and is almost pure matmul, exactly the workload tensor cores accelerate. On an
Ampere+ card (e.g. RTX 3080) bf16 autocast is a large, low-risk speedup
(bf16 has FP32's exponent range, so no loss scaling and no GradScaler). On a
pre-Ampere card (e.g. GTX 1660 Super, compute capability < 8.0, no native bf16
tensor cores) these helpers degrade to FP32 no-ops, so the same code path is
correct on both machines.

Native bf16 is gated on compute capability >= 8.0 (Ampere), NOT on
`torch.cuda.is_bf16_supported()`: recent torch reports that True on Turing too
because it counts *emulated* bf16, and emulated bf16 on a 1660 is far slower
than plain FP32 (a MERT forward that should take seconds takes minutes). The
capability check is also torch-version-independent.

torch is imported lazily so the rest of the package stays importable on a host
without a working CUDA torch.
"""
from __future__ import annotations

import contextlib


def tee_stdio(log_path) -> None:
    """Mirror stdout+stderr to `log_path` (append) IN ADDITION to the console, so
    long runs self-log without a manual `… | tee` / `>> file 2>&1` redirect.

    Captures everything the process emits, prints, library warnings, tqdm bars
    (stderr) and tracebacks. Call once at the very top of a script's `main()`
    (before the encoder/heavy imports run) so nothing is missed. Creates parent
    dirs; line-buffered so `tail -f` is live. No-op if `log_path` is falsy.

    NOTE: this replaces a manual shell redirect, don't do both, or every line
    lands in the file twice (the script writes it AND the shell captures the
    tee'd stdout)."""
    import sys
    from pathlib import Path

    if not log_path:
        return
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # noqa SIM115: the handle is the process-lifetime stdio sink, it MUST stay
    # open (no `with`); line-buffered so a tailing reader sees each line live.
    fh = open(p, "a", buffering=1)  # noqa: SIM115

    class _Tee:
        def __init__(self, stream):
            self._stream = stream

        def write(self, s):
            self._stream.write(s)
            fh.write(s)
            return len(s)

        def flush(self):
            self._stream.flush()
            fh.flush()

        def __getattr__(self, name):  # delegate isatty()/fileno()/encoding/…
            return getattr(self._stream, name)

    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)


def configure_backends() -> None:
    """Enable TF32 matmul/cuDNN where supported. Idempotent; no-op off CUDA.

    TF32 gives ~2x on float32 matmuls on Ampere+ for free (the float32 code
    path is unchanged); it is silently ignored on hardware without it."""
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def amp_dtype():
    """bf16 if the active CUDA device has NATIVE bf16, else None (stay FP32).

    Gated on compute capability >= 8.0 (Ampere+) rather than
    `is_bf16_supported()`, which also returns True for *emulated* bf16 on
    pre-Ampere cards (e.g. the 1660 Super) where emulation is slower than FP32."""
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return None


def autocast():
    """CUDA bf16 autocast context, or a null context on FP32-only hardware.

    Use to wrap encoder/head forwards. bf16 needs no GradScaler, so the
    training loop wraps only the forward+loss and leaves backward as-is."""
    import torch

    dtype = amp_dtype()
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)
