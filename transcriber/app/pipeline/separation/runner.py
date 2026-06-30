"""Chunked overlap-add separation runner, reimplemented from
audio-separator's `architectures/mdxc_separator.py` so it runs the vendored
models without importing `audio-separator`.

The chunking + overlap-add + peak-normalization all happen *outside* the
model, exactly as audio-separator does it. Two model families, two schemes:

  * **BS-Roformer** (`mdxc_separator.py:272-343`, the `is_roformer` branch):
    an explicit `for i in range(0, n, step)` loop with a Hamming-window
    overlap-add (`signal.windows.hamming(chunk_size)`) and a per-sample
    `counter` accumulator; final result is `result / counter.clamp(min=1e-10)`.
    `overlap` is in *seconds* -> `step = min(overlap*sr, chunk_size)`. The tail
    chunk re-anchors to `mix[:, -chunk_size:]` and is overlap-added at the end.

  * **MDX23C** (`mdxc_separator.py:345-402`, the else branch): pad, then
    `mix.unfold(1, chunk_size, hop_size)` into fixed chunks, accumulate model
    outputs in place, slice off the pad, divide by `overlap`. `overlap` is a
    *divider* -> `hop_size = chunk_size // overlap`.

Peak normalization before demix and per-stem after demix mirrors
`spec_utils.normalize` (audio-separator default `normalization_threshold=0.9`,
`amplification_threshold=0.0`), and input prep mirrors
`common_separator.py::prepare_mix` (`librosa.load(mono=False, sr=44100)`,
mono->stereo broadcast).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy import signal

from .loader import LoadedModel

# audio-separator Separator() defaults (separator.py:116-120).
NORMALIZATION_THRESHOLD = 0.9
AMPLIFICATION_THRESHOLD = 0.0
SAMPLE_RATE = 44100

# mdxc_params defaults (separator.py:128).
MDXC_OVERLAP = 8
MDXC_OVERRIDE_SEGMENT_SIZE = False

ProgressCallback = Callable[[int, int], None]


def _normalize(wave: np.ndarray, max_peak: float = 1.0, min_peak: float | None = None) -> np.ndarray:
    """spec_utils.normalize (spec_utils.py:99-115), in place on a copy-safe array."""
    maxv = np.abs(wave).max()
    if maxv > max_peak:
        wave = wave * (max_peak / maxv)
    elif min_peak is not None and min_peak > 0 and maxv < min_peak:
        wave = wave * (min_peak / maxv)
    return wave


def _prepare_mix(audio: str | Path | np.ndarray) -> np.ndarray:
    """common_separator.py::prepare_mix (lines 217-282): load to
    (channels, samples) @ 44.1k, broadcast mono to stereo."""
    if isinstance(audio, np.ndarray):
        mix = audio.T if audio.ndim == 2 else audio
    else:
        mix, _ = librosa.load(str(audio), mono=False, sr=SAMPLE_RATE)
    if mix.ndim == 1:
        mix = np.asfortranarray([mix, mix])
    return mix


class SeparationRunner:
    """Runs a `LoadedModel` over an audio file/array and returns in-memory
    stems. No disk writes; the caller persists what it wants."""

    def __init__(self, loaded: LoadedModel, *, device: str | torch.device = "cpu") -> None:
        self.loaded = loaded
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = loaded.model

    def separate(
        self,
        audio: str | Path | np.ndarray,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, np.ndarray]:
        """Separate `audio` into `{stem_name: np.ndarray}` of shape
        (channels, samples), matching what audio-separator returns pre-write
        (before its transpose for the soundfile write)."""
        mix = _prepare_mix(audio)
        mix = _normalize(mix, max_peak=NORMALIZATION_THRESHOLD, min_peak=AMPLIFICATION_THRESHOLD)

        if self.loaded.kind == "bs_roformer":
            sources = self._demix_roformer(mix, progress_callback)
        else:
            sources = self._demix_mdx23c(mix, progress_callback)

        out: dict[str, np.ndarray] = {}
        for name, wave in sources.items():
            out[name] = _normalize(
                wave, max_peak=NORMALIZATION_THRESHOLD, min_peak=AMPLIFICATION_THRESHOLD
            )
        return out

    def _run_model(self, x):
        """Run the model body on a chunk. Routes through onnxruntime when an ONNX
        session is attached (loader.attach_onnx_session), else torch. Both return
        the same (b, n, s, t) recon tensor, so call sites are agnostic."""
        session = self.loaded.onnx_session
        if session is not None:
            return self.model.forward_onnx(x, session)
        return self.model(x)

    # ---- BS-Roformer demix (mdxc_separator.py:272-343) -----------------

    def _demix_roformer(
        self, mix: np.ndarray, progress_callback: ProgressCallback | None
    ) -> dict[str, np.ndarray]:
        cfg = self.loaded.config
        instruments = self.loaded.instruments
        target = self.loaded.target_instrument

        mix_t = torch.tensor(mix, dtype=torch.float32)

        # Segment size: model default unless override (we never override).
        mdx_segment_size = cfg.inference.dim_t

        num_stems = 1 if target else len(instruments)

        # chunk_size from stft_hop_length (mdxc_separator.py:289-301).
        stft_hop_len = getattr(cfg.model, "stft_hop_length", None)
        if stft_hop_len is None:
            stft_hop_len = cfg.audio.hop_length
        chunk_size = int(stft_hop_len) * (int(mdx_segment_size) - 1)

        # overlap is in seconds; clamp the resulting step to chunk_size.
        desired_step = int(MDXC_OVERLAP * cfg.audio.sample_rate)
        step = chunk_size if desired_step <= 0 else min(desired_step, chunk_size)

        window = torch.tensor(signal.windows.hamming(chunk_size), dtype=torch.float32)

        # audio-separator's roformer loop produces a negative start index for a
        # mix shorter than one chunk (`result.shape[-1] - chunk_size < 0`), so
        # zero-pad up to chunk_size and trim the output back. A long mix is
        # unchanged (orig_len == padded len).
        orig_len = mix_t.shape[1]
        if orig_len < chunk_size:
            mix_t = torch.cat([mix_t, torch.zeros(mix_t.shape[0], chunk_size - orig_len)], dim=1)

        starts = list(range(0, mix_t.shape[1], step))
        total = len(starts)

        with torch.no_grad():
            req_shape = (len(instruments),) + tuple(mix_t.shape)
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            for done, i in enumerate(starts):
                part = mix_t[:, i : i + chunk_size]
                length = part.shape[-1]
                if i + chunk_size > mix_t.shape[1]:
                    part = mix_t[:, -chunk_size:]
                    length = chunk_size
                part = part.to(self.device)
                x = self._run_model(part.unsqueeze(0))[0]
                x = x.cpu()
                if i + chunk_size > mix_t.shape[1]:
                    start_idx = result.shape[-1] - chunk_size
                    self._overlap_add(result, x, window, start_idx, length)
                    safe_len = min(length, x.shape[-1], window.shape[0])
                    if safe_len > 0:
                        counter[..., start_idx : start_idx + safe_len] += window[:safe_len]
                else:
                    self._overlap_add(result, x, window, i, length)
                    safe_len = min(length, x.shape[-1], window.shape[0])
                    if safe_len > 0:
                        counter[..., i : i + safe_len] += window[:safe_len]
                if progress_callback is not None:
                    progress_callback(done + 1, total)

        inferenced = result / counter.clamp(min=1e-10)
        inferenced = inferenced[..., :orig_len]

        if num_stems > 1:
            return {
                key: value
                for key, value in zip(instruments, inferenced.cpu().detach().numpy(), strict=True)
            }
        # Single-target roformer (not our case, kept for parity).
        name = target
        return {name: inferenced[0].cpu().detach().numpy()}

    @staticmethod
    def _overlap_add(result, x, weights, start, length):
        """mdxc_separator.py::overlap_add (lines 246-255)."""
        safe_len = min(length, x.shape[-1], weights.shape[0])
        if safe_len > 0:
            result[..., start : start + safe_len] += x[..., :safe_len] * weights[:safe_len]
        return result

    # ---- MDX23C demix (mdxc_separator.py:345-402) ----------------------

    def _demix_mdx23c(
        self, mix: np.ndarray, progress_callback: ProgressCallback | None
    ) -> dict[str, np.ndarray]:
        cfg = self.loaded.config
        instruments = self.loaded.instruments

        mix_t = torch.tensor(mix, dtype=torch.float32)
        num_stems = self.model.num_target_instruments

        mdx_segment_size = cfg.inference.dim_t
        chunk_size = cfg.audio.hop_length * (mdx_segment_size - 1)
        hop_size = chunk_size // MDXC_OVERLAP

        mix_shape = mix_t.shape[1]
        pad_size = hop_size - (mix_shape - chunk_size) % hop_size

        mix_t = torch.cat(
            [
                torch.zeros(2, chunk_size - hop_size),
                mix_t,
                torch.zeros(2, pad_size + chunk_size - hop_size),
            ],
            1,
        )

        chunks = mix_t.unfold(1, chunk_size, hop_size).transpose(0, 1)

        # batch_size default is 1 (mdxc_params); one chunk per batch.
        batches = [chunks[i : i + 1] for i in range(0, len(chunks), 1)]
        total = len(batches)

        accumulated = (
            torch.zeros(num_stems, *mix_t.shape) if num_stems > 1 else torch.zeros_like(mix_t)
        )

        with torch.no_grad():
            count = 0
            for done, batch in enumerate(batches):
                single_batch_result = self._run_model(batch.to(self.device))
                for individual_output in single_batch_result:
                    individual_output_cpu = individual_output.cpu()
                    accumulated[..., count * hop_size : count * hop_size + chunk_size] += (
                        individual_output_cpu
                    )
                    count += 1
                if progress_callback is not None:
                    progress_callback(done + 1, total)

        inferenced = (
            accumulated[..., chunk_size - hop_size : -(pad_size + chunk_size - hop_size)]
            / MDXC_OVERLAP
        )
        del accumulated

        if num_stems > 1:
            return {
                key: value
                for key, value in zip(instruments, inferenced.cpu().detach().numpy(), strict=True)
            }
        return {instruments[0]: inferenced.cpu().detach().numpy()}
