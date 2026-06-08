"""GPU precision/throughput knobs: TF32 + bf16 autocast, guarded for old GPUs.

The frozen MERT encoder is a 24-layer transformer, so it dominates wall time
and is almost pure matmul, exactly the workload tensor cores accelerate. On an
Ampere+ card (e.g. RTX 3080) bf16 autocast is a large, low-risk speedup
(bf16 has FP32's exponent range, so no loss scaling and no GradScaler). On a
tensor-core-less card (e.g. GTX 1660 Super, no bf16 support) these helpers
degrade to FP32 no-ops, so the same code path is correct on both machines.

torch is imported lazily so the rest of the package stays importable on a host
without a working CUDA torch.
"""
from __future__ import annotations

import contextlib


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
    """bf16 if the active CUDA device supports it, else None (stay FP32)."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
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
