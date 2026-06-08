import contextlib

import drumjot_training.runtime as runtime


def test_configure_backends_runs():
    # idempotent, must not raise on any device (no-op without a tensor-core GPU)
    runtime.configure_backends()
    runtime.configure_backends()


def test_amp_dtype_is_bf16_or_none():
    import torch

    assert runtime.amp_dtype() in (None, torch.bfloat16)


def test_autocast_returns_context_manager():
    with runtime.autocast():
        pass


def test_autocast_is_null_when_no_bf16(monkeypatch):
    # FP32-only hardware (e.g. GTX 1660) must get a true no-op, not a cuda
    # autocast that would error off an Ampere card.
    monkeypatch.setattr(runtime, "amp_dtype", lambda: None)
    assert isinstance(runtime.autocast(), contextlib.nullcontext)


def test_head_forward_under_bf16_autocast_yields_finite_fp32():
    # The integration contract the wiring relies on: a forward under bf16
    # autocast must convert back to finite float32 before .numpy() (a bare
    # bf16 tensor .numpy() raises). Uses CPU autocast so it runs without an
    # Ampere GPU.
    import numpy as np
    import torch

    from drumjot_training.model import MultiLaneHeads

    model = MultiLaneHeads(in_dim=16, hidden=8, num_layers=1).eval()
    x = torch.randn(1, 20, 16)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = torch.sigmoid(model(x))
    arr = out[0].float().cpu().numpy()
    assert arr.dtype == np.float32
    assert np.isfinite(arr).all()
