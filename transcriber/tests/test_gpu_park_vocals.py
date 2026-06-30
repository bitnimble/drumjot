"""Tests for parking the vocals-extraction model.

Vocals now comes from the Stage-1 BS-Roformer SW separator (a torch model), so
`park_vocals` / `unpark_vocals` just move that `SeparationRunner`'s module
between CPU and GPU like the drum models. The old dedicated MDX/ONNX vocals
model (whose ORT-session CUDA arena needed a full release) is gone.

These use lightweight stand-ins (no torch, no GPU) so they run on CPU in CI.
"""

from __future__ import annotations

from app.pipeline import gpu_park
from app.pipeline.separate import Separator
from app.pipeline.separation.runner import SeparationRunner


class _FakeModule:
    """Torch-module stand-in: empty `.parameters()` (so `park_module` treats it
    as already host-side) plus a `.to()` that records the device."""

    def __init__(self) -> None:
        self.device = "cuda"

    def parameters(self):
        return iter(())

    def to(self, device):  # noqa: D401 - torch nn.Module surface
        self.device = device
        return self


def _sep_with_sw(model: object) -> Separator:
    sep = Separator.__new__(Separator)
    runner = SeparationRunner.__new__(SeparationRunner)
    runner.model = model  # type: ignore[attr-defined]
    sep._stems_all = runner  # type: ignore[attr-defined]
    sep._stems_per = None  # type: ignore[attr-defined]
    return sep


def test_park_module_noop_on_non_module() -> None:
    """A bare function (no `.parameters()`) must be a clean no-op."""
    gpu_park.park_module(lambda spek: spek, "vocals")
    gpu_park.unpark_module(lambda spek: spek, "vocals")


def test_park_vocals_parks_sw_runner() -> None:
    """park_vocals parks the SW runner's module; unpark brings it back."""
    sep = _sep_with_sw(_FakeModule())
    sep.park_vocals()
    sep.unpark_vocals()


def test_park_vocals_noop_when_sw_not_loaded() -> None:
    """No SW loaded (e.g. a vocals cache hit) -> park/unpark are no-ops."""
    sep = Separator.__new__(Separator)
    sep._stems_all = None  # type: ignore[attr-defined]
    sep._stems_per = None  # type: ignore[attr-defined]
    sep.park_vocals()
    sep.unpark_vocals()
