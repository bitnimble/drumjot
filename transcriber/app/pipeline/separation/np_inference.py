"""Torch-free numpy inference path for the ONNX separation bodies.

Runs the full separation in numpy + onnxruntime: numpy STFT (`np_stft`) for the
spectrogram prep/post, numpy chunking / overlap-add (mirroring
`runner.SeparationRunner` exactly), and an onnxruntime session for the model
body. No `import torch`, so a deployment can run inference without PyTorch
(torch is needed only for the one-time `.onnx` export and the opt-out torch
fallback path).

Output matches the torch ONNX runner to fp32 rounding (see the parity test).

The spectrogram packing mirrors the vendored model classes:
  - MDX23C (`TFC_TDF_net.STFT`): channels-as-complex `b (c*2) dim_f t`.
  - BS-Roformer (`BSRoformer._stft_prep`/`_apply_mask`/`_istft_post`):
    frequency-leading real-view `b (f s) t c`, plus the complex mask multiply.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import librosa
import numpy as np
import yaml
from scipy import signal

from . import np_stft

NORMALIZATION_THRESHOLD = 0.9
AMPLIFICATION_THRESHOLD = 0.0
SAMPLE_RATE = 44100
MDXC_OVERLAP = 8

ProgressCallback = Callable[[int, int], None]


def _normalize(wave: np.ndarray, max_peak: float = 1.0, min_peak: float | None = None) -> np.ndarray:
    maxv = np.abs(wave).max()
    if maxv > max_peak:
        wave = wave * (max_peak / maxv)
    elif min_peak is not None and min_peak > 0 and maxv < min_peak:
        wave = wave * (min_peak / maxv)
    return wave


def _prepare_mix(audio: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        mix = audio.T if audio.ndim == 2 else audio
    else:
        mix, _ = librosa.load(str(audio), mono=False, sr=SAMPLE_RATE)
    if mix.ndim == 1:
        mix = np.asfortranarray([mix, mix])
    return mix.astype(np.float32)


# ---- MDX23C spectrogram packing (mirrors TFC_TDF_net.STFT) ----------------


def mdx_pack(audio: np.ndarray, n_fft: int, hop: int, dim_f: int, window: np.ndarray) -> np.ndarray:
    """`(b, c, t)` audio -> `(b, c*2, dim_f, T)` real spectrogram."""
    b, c, t = audio.shape
    spec = np_stft.stft(audio.reshape(b * c, t), n_fft, hop, window)  # (b*c, F, T) complex
    real = np.stack([spec.real, spec.imag], axis=1)  # (b*c, 2, F, T)
    real = real.reshape(b, c, 2, real.shape[-2], real.shape[-1]).reshape(
        b, c * 2, real.shape[-2], real.shape[-1]
    )
    return real[..., :dim_f, :].astype(np.float32)


def mdx_unpack(spec: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    """`(b, n, c*2, dim_f, T)` masked spectrogram -> `(b, n, c, samples)` audio."""
    *batch, c, f, t = spec.shape
    n_bins = n_fft // 2 + 1
    pad = np.zeros((*batch, c, n_bins - f, t), dtype=np.float32)
    x = np.concatenate([spec, pad], axis=-2).reshape(*batch, c // 2, 2, n_bins, t).reshape(
        -1, 2, n_bins, t
    )
    cplx = (x[:, 0] + 1j * x[:, 1]).astype(np.complex64)
    audio = np_stft.istft(cplx, n_fft, hop, window)
    return audio.reshape(*batch, 2, -1).astype(np.float32)


# ---- BS-Roformer spectrogram packing (mirrors BSRoformer prep/post) -------


def bs_pack(audio: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    """`(b, s, t)` audio -> `(b, (f s), T, 2)` real-view stft_repr."""
    b, s, t = audio.shape
    spec = np_stft.stft(audio.reshape(b * s, t), n_fft, hop, window)  # (b*s, F, T) complex
    f_, tt = spec.shape[-2], spec.shape[-1]
    real = np.stack([spec.real, spec.imag], axis=-1).reshape(b, s, f_, tt, 2)  # (b, s, f, t, c)
    real = real.transpose(0, 2, 1, 3, 4).reshape(b, f_ * s, tt, 2)  # b (f s) t c
    return real.astype(np.float32)


def bs_apply_mask(stft_repr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """real-view `(b, (f s), T, 2)` * `(b, n, (f s), T, 2)` -> `(b, n, (f s), T)` complex."""
    sr = (stft_repr[..., 0] + 1j * stft_repr[..., 1])[:, None]  # (b, 1, (f s), T)
    mk = mask[..., 0] + 1j * mask[..., 1]  # (b, n, (f s), T)
    return (sr * mk).astype(np.complex64)


def bs_unpack(
    masked: np.ndarray, n_fft: int, hop: int, window: np.ndarray, audio_channels: int, num_stems: int
) -> np.ndarray:
    """`(b, n, (f s), T)` complex -> `(b, n, s, samples)` audio."""
    b, n, fs, t = masked.shape
    s = audio_channels
    f_ = fs // s
    m = masked.reshape(b, n, f_, s, t).transpose(0, 1, 3, 2, 4).reshape(b * n * s, f_, t)
    audio = np_stft.istft(m, n_fft, hop, window)
    audio = audio.reshape(b, n, s, -1).astype(np.float32)
    return audio[:, 0] if num_stems == 1 else audio


# ---- the separator --------------------------------------------------------


def _ort_session(onnx_path, providers):
    import onnxruntime as ort

    if providers is None:
        providers = ort.get_available_providers()
    try:
        return ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


class NumpySeparator:
    """Torch-free numpy + onnxruntime separator for one model's `.onnx` body."""

    def __init__(self, onnx_path, yaml_path, kind: str | None = None, providers=None) -> None:
        with open(yaml_path, encoding="utf-8") as fh:
            cfg = yaml.load(fh, Loader=yaml.FullLoader)
        # Mirrors loader._detect_kind: roformer configs carry freqs_per_bands.
        self.kind = kind or ("bs_roformer" if "freqs_per_bands" in cfg.get("model", {}) else "mdx23c")
        self.cfg = cfg
        self.instruments = list(cfg["training"]["instruments"])
        self.target = cfg["training"].get("target_instrument")
        self.session = _ort_session(onnx_path, providers)
        self._in = self.session.get_inputs()[0].name

    def _run(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self._in: x})[0]

    def separate(self, audio, *, progress_callback: ProgressCallback | None = None) -> dict[str, np.ndarray]:
        mix = _normalize(
            _prepare_mix(audio), max_peak=NORMALIZATION_THRESHOLD, min_peak=AMPLIFICATION_THRESHOLD
        )
        sources = (
            self._demix_roformer(mix, progress_callback)
            if self.kind == "bs_roformer"
            else self._demix_mdx23c(mix, progress_callback)
        )
        return {
            name: _normalize(w, max_peak=NORMALIZATION_THRESHOLD, min_peak=AMPLIFICATION_THRESHOLD)
            for name, w in sources.items()
        }

    def _demix_mdx23c(self, mix, progress_callback):
        cfg = self.cfg
        n_fft = cfg["audio"]["n_fft"]
        hop = cfg["audio"]["hop_length"]
        dim_f = cfg["audio"]["dim_f"]
        segment = cfg["inference"]["dim_t"]
        num_stems = len(self.instruments)
        window = np_stft.hann_window(n_fft)

        chunk_size = hop * (segment - 1)
        hop_size = chunk_size // MDXC_OVERLAP
        mix_shape = mix.shape[1]
        pad_size = hop_size - (mix_shape - chunk_size) % hop_size
        mix_p = np.concatenate(
            [
                np.zeros((2, chunk_size - hop_size), np.float32),
                mix,
                np.zeros((2, pad_size + chunk_size - hop_size), np.float32),
            ],
            axis=1,
        )
        n_chunks = (mix_p.shape[1] - chunk_size) // hop_size + 1
        accumulated = np.zeros((num_stems, *mix_p.shape), np.float32)
        for c in range(n_chunks):
            chunk = mix_p[:, c * hop_size : c * hop_size + chunk_size][None]  # (1, 2, chunk)
            spec = mdx_pack(chunk, n_fft, hop, dim_f, window)
            out = self._run(spec)  # (1, n, c*2, dim_f, T)
            audio = mdx_unpack(out, n_fft, hop, window)[0]  # (n, 2, chunk)
            accumulated[..., c * hop_size : c * hop_size + chunk_size] += audio
            if progress_callback is not None:
                progress_callback(c + 1, n_chunks)
        inferenced = accumulated[..., chunk_size - hop_size : -(pad_size + chunk_size - hop_size)] / MDXC_OVERLAP
        return dict(zip(self.instruments, inferenced, strict=True))

    def _demix_roformer(self, mix, progress_callback):
        cfg = self.cfg
        n_fft = cfg["model"]["stft_n_fft"]
        hop = cfg["model"]["stft_hop_length"]
        segment = cfg["inference"]["dim_t"]
        audio_channels = 2 if cfg["model"].get("stereo") else 1
        num_stems = 1 if self.target else len(self.instruments)
        window = np_stft.hann_window(n_fft)

        chunk_size = hop * (segment - 1)
        desired_step = int(MDXC_OVERLAP * cfg["audio"]["sample_rate"])
        step = chunk_size if desired_step <= 0 else min(desired_step, chunk_size)
        ham = signal.windows.hamming(chunk_size).astype(np.float32)

        orig_len = mix.shape[1]
        if orig_len < chunk_size:
            mix = np.concatenate([mix, np.zeros((mix.shape[0], chunk_size - orig_len), np.float32)], axis=1)
        starts = list(range(0, mix.shape[1], step))
        req = (len(self.instruments), *mix.shape)
        result = np.zeros(req, np.float32)
        counter = np.zeros(req, np.float32)
        for done, i in enumerate(starts):
            part = mix[:, i : i + chunk_size]
            length = part.shape[-1]
            at_tail = i + chunk_size > mix.shape[1]
            if at_tail:
                part = mix[:, -chunk_size:]
                length = chunk_size
            stft_repr = bs_pack(part[None], n_fft, hop, window)  # (1, (f s), T, 2)
            mask = self._run(stft_repr)  # (1, n, (f s), T, 2)
            masked = bs_apply_mask(stft_repr, mask)
            x = bs_unpack(masked, n_fft, hop, window, audio_channels, num_stems)[0]  # (n, s, chunk)
            start = result.shape[-1] - chunk_size if at_tail else i
            safe = min(length, x.shape[-1], ham.shape[0])
            if safe > 0:
                result[..., start : start + safe] += x[..., :safe] * ham[:safe]
                counter[..., start : start + safe] += ham[:safe]
            if progress_callback is not None:
                progress_callback(done + 1, len(starts))
        inferenced = (result / np.clip(counter, 1e-10, None))[..., :orig_len]
        if num_stems > 1:
            return dict(zip(self.instruments, inferenced, strict=True))
        return {self.target: inferenced[0]}
