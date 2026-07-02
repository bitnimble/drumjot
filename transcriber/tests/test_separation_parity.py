"""torch <-> ONNX full-model parity guard for the SEPARATION models.

Complements the DSP-level `test_np_stft_parity.py` (which pins the numpy STFT/
iSTFT to torch): this runs the WHOLE separation of both shipped models through
BOTH paths on the same input and asserts the stems agree, so a future re-export
or an onnxruntime bump can't silently shift the separated stems.

  * torch reference: `separation/loader.load_model` + `runner.SeparationRunner`
    (fp32, no bf16 -- the plain runner path, not the bf16-patched app path).
  * ONNX path:       `separation/np_inference.NumpySeparator` over the shipped
    fp16 `.onnx` body (numpy STFT/chunking + onnxruntime).

The shipped `.onnx` is fp16 and fp16 separation only runs on a GPU EP (ORT's
CPU EP can't run the fp16 GRU/attention), so per-stem correlation is the metric
that survives the fp32<->fp16 magnitude drift; the review measured mask corr
~1.0, so >= 0.9998 is a real guard, not a flake.

GATED (mirrors `test_onnx_model_e2e.py`): skipped unless BOTH the onnx bodies
and the torch ckpts+yamls are on disk, and unless a CUDA GPU EP is available for
the fp16 onnx. Path resolution (env-overridable):

    DRUMJOT_SEP_CKPT_DIR   ckpts + yamls   (default MODELS_DIR, else the dev NAS)
    DRUMJOT_SEP_ONNX_DIR   fp16 .onnx set  (default the dev NAS onnx-export dir)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.pipeline.provision import yaml_for_ckpt

# (torch ckpt filename, expected kind) for both shipped separators.
_MODELS = [
    (settings.demucs_model, "bs_roformer"),  # model_bs_roformer_sw.ckpt
    (settings.drum_pieces_model, "mdx23c"),  # drumsep_5stems_mdx23c_jarredou.ckpt
]

_DEFAULT_CKPT_DIR = "/codebox-workspace/drumjot/models-cache"
_DEFAULT_ONNX_DIR = "/codebox-workspace/drumjot/onnx-export"


def _ckpt_dir() -> Path:
    return Path(os.environ.get("DRUMJOT_SEP_CKPT_DIR") or os.environ.get("MODELS_DIR") or _DEFAULT_CKPT_DIR)


def _onnx_dir() -> Path:
    return Path(os.environ.get("DRUMJOT_SEP_ONNX_DIR") or _DEFAULT_ONNX_DIR)


def _paths(ckpt_filename: str) -> tuple[Path, Path, Path]:
    """(ckpt, yaml, fp16 onnx) for a separation model filename."""
    ckpt_dir, onnx_dir = _ckpt_dir(), _onnx_dir()
    ckpt = ckpt_dir / ckpt_filename
    yaml = ckpt_dir / yaml_for_ckpt(ckpt_filename)
    onnx = onnx_dir / f"{Path(ckpt_filename).stem}.fp16.onnx"
    return ckpt, yaml, onnx


def _all_present() -> bool:
    for ckpt_filename, _ in _MODELS:
        for p in _paths(ckpt_filename):
            if not (p.exists() and p.stat().st_size > 0):
                return False
    return True


def _cuda_ep_available() -> bool:
    try:
        import onnxruntime as ort
    except Exception:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


pytestmark = [
    pytest.mark.skipif(
        not _all_present(),
        reason=(
            f"separation weights not on disk (ckpts/yamls under {_ckpt_dir()}, "
            f"fp16 onnx under {_onnx_dir()}); set DRUMJOT_SEP_CKPT_DIR / DRUMJOT_SEP_ONNX_DIR"
        ),
    ),
    pytest.mark.skipif(
        not _cuda_ep_available(),
        reason="no CUDA ExecutionProvider; the shipped fp16 separation onnx is GPU-only",
    ),
]


def _stereo_noise(seconds: float = 3.5, sr: int = 44100, seed: int = 0) -> np.ndarray:
    """A short seeded stereo waveform, shape (samples, channels) -- the shape both
    `separate()` entry points expect for a 2D ndarray (they transpose to channels-
    first internally). Tones + noise + a broadband click train: the transients give
    the percussion separators (MDX23C kick/snare/toms) real energy to route, so the
    correlation check covers more than one stem, not just whichever stem the steady
    tones land in. Decorrelated L/R so the stereo path is non-trivial."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    tone = 0.15 * np.sin(2 * np.pi * 220.0 * t) + 0.1 * np.sin(2 * np.pi * 1760.0 * t)
    clicks = np.zeros(n, dtype=np.float64)
    for k in range(int(seconds * 4)):  # 4 clicks/sec -> broadband transients
        i = int(k * sr / 4)
        if i < n:
            clicks[i:i + 64] = np.hanning(128)[:64] * rng.choice([-1.0, 1.0])
    left = tone + 0.5 * clicks + 0.15 * rng.standard_normal(n)
    right = tone + 0.5 * clicks + 0.15 * rng.standard_normal(n)
    mix = np.stack([left, right], axis=1).astype(np.float32)  # (samples, channels)
    return (mix / np.abs(mix).max() * 0.7).astype(np.float32)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float((a * b).sum() / denom)


@pytest.mark.parametrize("ckpt_filename, expected_kind", _MODELS, ids=[m[0] for m in _MODELS])
def test_onnx_separation_matches_torch(ckpt_filename, expected_kind):
    pytest.importorskip("torch")

    from app.pipeline.separation.loader import load_model
    from app.pipeline.separation.np_inference import NumpySeparator
    from app.pipeline.separation.runner import SeparationRunner

    ckpt, yaml, onnx = _paths(ckpt_filename)
    mix = _stereo_noise()

    # torch reference on CPU: it's the fp32 ground truth (numerically
    # device-independent, no bf16/TF32), and it leaves the whole GPU for the fp16
    # onnx session, which MUST run on the CUDA EP.
    loaded = load_model(ckpt, yaml, device="cpu")
    assert loaded.kind == expected_kind
    torch_stems = SeparationRunner(loaded, device="cpu").separate(mix)

    onnx_sep = NumpySeparator(onnx, yaml)
    assert onnx_sep.kind == expected_kind
    assert "CUDAExecutionProvider" in onnx_sep.session.get_providers()
    onnx_stems = onnx_sep.separate(mix)

    assert set(torch_stems) == set(onnx_stems), (sorted(torch_stems), sorted(onnx_stems))

    # All stems are fractions of the same mix; the loudest one sets the reference
    # scale for the (magnitude) error bound. RMS gates the correlation check: a stem
    # that's near-silent on this synthetic input (e.g. guitar/cymbals -- there's no
    # real such content in tones+noise) has an ill-conditioned corr (residual-vs-
    # residual noise), so its corr is uninformative; the tiny abs-error bound still
    # guards it against corruption. Real energy-carrying stems (drums, bass) get the
    # tight corr guard that would catch a shifted/re-exported/onnxruntime-drifted model.
    def _rms(a: np.ndarray) -> float:
        return float(np.sqrt((a.astype(np.float64) ** 2).mean()))

    peak_all = max(float(np.abs(s).max()) for s in torch_stems.values()) or 1.0
    rms_all = max(_rms(s) for s in torch_stems.values()) or 1.0
    for name in sorted(torch_stems):
        ta, oa = torch_stems[name], onnx_stems[name]
        assert ta.shape == oa.shape, f"{name}: {ta.shape} vs {oa.shape}"
        corr = _correlation(ta, oa)
        max_abs = float(np.abs(ta.astype(np.float64) - oa.astype(np.float64)).max())
        rms_frac = _rms(ta) / rms_all
        print(f"{ckpt_filename} :: {name}: corr={corr:.6f} max_abs_diff={max_abs:.4e} rms_frac={rms_frac:.3f}")
        assert max_abs < 1e-2 * peak_all, (
            f"{ckpt_filename} stem {name}: max_abs_diff {max_abs:.4e} > 1% of peak {peak_all:.4e}")
        if rms_frac >= 0.05:  # only assert corr on stems with real energy
            assert corr >= 0.9998, f"{ckpt_filename} stem {name}: corr {corr:.6f} < 0.9998"
