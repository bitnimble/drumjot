"""End-to-end check that a SHIPPED ONNX model actually loads + produces sane
output through its real loader -- the smallest/fastest model path (Beat This!, a
self-contained ~40 MB fp16 transformer that runs on the CPU EP; the GRU models
would need a GPU EP for fp16).

This validates the download -> load -> run path the desktop app relies on: the
loader prefers the provisioned `beat_this.fp16.onnx` (`provision.shipped_onnx`).
The test is GATED on that file being present, so it stays skipped until the fp16
set is uploaded to `bitnimble/drumjot-onnx` and provisioned into `models_dir`.

To enable it once the upload is done, provision the model (or point at a dir that
already has it):

    DRUMJOT_MODELS_DIR=/path/with/beat_this.fp16.onnx bun ... / pytest
    # or: python -m app.pipeline.provision transcription   (downloads into models_dir)
"""

import numpy as np
import pytest

from app.config import settings
from app.pipeline.provision import shipped_onnx

pytestmark = pytest.mark.skipif(
    shipped_onnx("beat_this") is None,
    reason=(
        "beat_this.fp16.onnx not provisioned into settings.models_dir "
        "(upload the fp16 set to drumjot-onnx, then `python -m app.pipeline.provision transcription`)"
    ),
)


def _click_track(path, dur=20.0, sr=22050, bpm=120.0):
    """A metronomic, lightly-accented click track Beat This! can lock onto."""
    import soundfile as sf

    n = int(sr * dur)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    y = 0.05 * rng.standard_normal(n) + 0.2 * np.sin(2 * np.pi * 220 * t)
    period = 60.0 / bpm
    for k in range(int(dur / period)):
        i = int(k * period * sr)
        env = np.exp(-np.arange(min(3000, n - i)) / 300.0)
        amp = 1.0 if k % 4 == 0 else 0.6  # accent the downbeats
        y[i : i + len(env)] += amp * env * rng.standard_normal(len(env))
    sf.write(str(path), (y / np.abs(y).max() * 0.9).astype(np.float32), sr)


def test_shipped_beat_onnx_tracks_a_click_track(tmp_path):
    """The provisioned Beat This! ONNX model tracks a 120 BPM click track: it
    returns beats at ~0.5 s spacing (proving the model ran and produced usable
    output, not just that the file loads)."""
    from app.pipeline.beat_onnx import OnnxBeatThis, load_beat_session

    clip = tmp_path / "click.wav"
    _click_track(clip)

    model = load_beat_session(settings.models_dir, providers=["CPUExecutionProvider"])
    assert isinstance(model, OnnxBeatThis)  # used the shipped onnx, not the torch path

    beats, downbeats = model(str(clip))
    assert len(beats) > 10, f"expected a beat grid, got {len(beats)} beats"
    ibi = float(np.median(np.diff(np.asarray(beats, dtype=float))))
    assert 0.4 < ibi < 0.6, f"median inter-beat interval {ibi:.3f}s not ~0.5s (120 BPM)"
    assert len(downbeats) >= 1, "no downbeats detected"
