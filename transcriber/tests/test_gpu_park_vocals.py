"""Regression tests for parking the MDX/ONNX vocals separator.

The vocals model runs through ONNX Runtime, whose `model_instance.
model_run` is a plain lambda (no `.parameters()`) and whose CUDA arena
torch can't move host-side. The old `park_vocals` fed that lambda to
`gpu_park.park_module`, which raised `'function' object has no attribute
'parameters'` and freed nothing, so the CTC aligner OOMed. `park_vocals`
now releases the ORT session for that path instead; `park_module` /
`unpark_module` also no-op cleanly on non-`nn.Module` objects.

These tests use lightweight stand-ins (no torch model, no GPU, no
audio-separator load) so they run on CPU in CI.
"""
from __future__ import annotations

from app.pipeline import gpu_park
from app.pipeline.separate import Separator


class _FakeInner:
    """Stand-in for a torch-backed inner module: exposes `.parameters()`
    (empty, so `park_module` treats it as already-on-CPU) and `.to()`."""

    def parameters(self):
        return iter(())

    def to(self, device):  # noqa: D401 - torch nn.Module surface
        return self


class _FakeModelInstance:
    def __init__(self, model_run: object) -> None:
        self.model_run = model_run


class _FakeSeparatorWrapper:
    """Stand-in for `audio_separator.separator.Separator`."""

    def __init__(self, model_run: object) -> None:
        self.model_instance = _FakeModelInstance(model_run)


def _ort_run(spek):
    """An ONNX Runtime-style inference lambda equivalent (a function with
    no `.parameters()`), matching `MDXSeparator.model_run`."""
    return spek


def _sep_with_vocals(model_run: object) -> Separator:
    sep = Separator.__new__(Separator)
    sep._vocals = _FakeSeparatorWrapper(model_run)  # type: ignore[attr-defined]
    return sep


def test_park_module_noop_on_non_module() -> None:
    """A bare function (the ORT lambda case) must be a clean no-op, not a
    caught `'function' object has no attribute 'parameters'` warning."""
    gpu_park.park_module(_ort_run, "vocals")
    gpu_park.unpark_module(_ort_run, "vocals")


def test_park_vocals_releases_ort_session() -> None:
    """ORT path: the inner is a lambda, so park_vocals drops the whole
    separator (freeing its CUDA arena) rather than torch-parking it."""
    sep = _sep_with_vocals(_ort_run)
    sep.park_vocals()
    assert sep._vocals is None


def test_park_vocals_keeps_torch_backed_model_warm() -> None:
    """onnx2torch path: the inner exposes `.parameters()`, so park_vocals
    parks it to CPU and keeps the separator loaded (cheaper than a
    reload)."""
    sep = _sep_with_vocals(_FakeInner())
    sep.park_vocals()
    assert sep._vocals is not None


def test_release_vocals_is_idempotent() -> None:
    sep = Separator.__new__(Separator)
    sep._vocals = None  # type: ignore[attr-defined]
    sep._release_vocals()
    assert sep._vocals is None
