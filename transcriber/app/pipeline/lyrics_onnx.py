"""Torch-free ONNX inference for the CTC forced-alignment model.

The `/lyrics` aligner runs a wav2vec2-family CTC model over the audio to get
per-frame emissions, then a C++ Viterbi (`ctc_forced_aligner.forced_align`, numpy
in/out) aligns the lyric tokens. Only the model + `log_softmax` were torch; this
module exports the model to ONNX and reproduces `generate_emissions` /
`get_alignments` in numpy, reusing the package's numpy `forced_align`,
`merge_repeats`, `get_spans`, and `postprocess_results`.

Torch is needed only for the one-time export (cached); inference is torch-free.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

SR = 16000


def export_ctc_model(model_path: str, out_path: str | Path, *, opset: int = 17) -> Path:
    """Export a HF `AutoModelForCTC` (waveform -> logits). Returns the path."""
    import torch
    from transformers import AutoModelForCTC

    model = AutoModelForCTC.from_pretrained(model_path).eval()

    class Body(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_values):  # (batch, samples)
            return self.m(input_values).logits  # (batch, frames, vocab)

    body = Body(model).eval()
    dummy = torch.zeros(1, SR)
    out_path = Path(out_path)
    with torch.no_grad():
        torch.onnx.export(
            body, (dummy,), str(out_path),
            input_names=["input_values"], output_names=["logits"],
            dynamic_axes={"input_values": {0: "batch", 1: "samples"},
                          "logits": {0: "batch", 1: "frames"}},
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
    return out_path


def load_audio_np(audio_file: str | Path) -> np.ndarray:
    """ffmpeg -> mono 16 kHz float32 in [-1, 1] (numpy port of the package's load_audio)."""
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", str(audio_file),
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SR), "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def _time_to_frame(t: float) -> int:
    return int(t * (1000 / 20))  # 20 ms stride -> 50 fps


def generate_emissions_np(session, audio: np.ndarray, *, window_length=30, context_length=2,
                          batch_size=4):
    """Numpy port of `ctc_forced_aligner.generate_emissions`; returns `(emissions, stride)`.
    `emissions` is `(T, vocab+1)` log-probs (star token appended)."""
    from scipy.special import logsumexp

    window = int(window_length * SR)
    n = audio.shape[0]
    if n < window:
        extension = context = 0
        chunks = audio[None].astype(np.float32)
    else:
        context = int(context_length * SR)
        extension = math.ceil(n / window) * window - n
        padded = np.pad(audio, (context, context + extension)).astype(np.float32)
        chunk_len = window + 2 * context
        n_chunks = (len(padded) - chunk_len) // window + 1
        chunks = np.stack([padded[i * window : i * window + chunk_len] for i in range(n_chunks)])

    name = session.get_inputs()[0].name
    outs = [
        session.run(None, {name: chunks[i : i + batch_size]})[0]
        for i in range(0, chunks.shape[0], max(batch_size, 1))
    ]
    emissions = np.concatenate(outs, axis=0)  # (n_chunks, frames, vocab)
    if context > 0:
        emissions = emissions[:, _time_to_frame(context_length) : -_time_to_frame(context_length) + 1]
    emissions = emissions.reshape(-1, emissions.shape[-1])  # flatten(0, 1)
    if _time_to_frame(extension / SR) > 0:
        emissions = emissions[: -_time_to_frame(extension / SR)]
    emissions = emissions - logsumexp(emissions, axis=-1, keepdims=True)  # log_softmax
    emissions = np.concatenate([emissions, np.zeros((emissions.shape[0], 1), emissions.dtype)], axis=1)
    stride = float(n * 1000 / emissions.shape[0] / SR)
    return emissions.astype(np.float32), math.ceil(stride)


def get_alignments_np(emissions: np.ndarray, tokens: list, tokenizer):
    """Numpy port of `ctc_forced_aligner.get_alignments` (reuses the C++ Viterbi)."""
    from ctc_forced_aligner.alignment_utils import forced_align, merge_repeats

    assert len(tokens) > 0, "Empty transcript"
    dictionary = {k.lower(): v for k, v in tokenizer.get_vocab().items()}
    dictionary["<star>"] = len(dictionary)
    token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    blank_id = dictionary.get("<blank>", tokenizer.pad_token_id)
    targets = np.asarray([token_indices], dtype=np.int64)
    path, scores = forced_align(emissions[None].astype(np.float32), targets, blank=blank_id)
    idx_to_token = {v: k for k, v in dictionary.items()}
    segments = merge_repeats(path.squeeze().tolist(), idx_to_token)
    return segments, scores, idx_to_token[blank_id]


def _ort_session(onnx_path, providers):
    import onnxruntime as ort

    if providers is None:
        providers = ort.get_available_providers()
    try:
        return ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


class OnnxCtcAligner:
    """Torch-free CTC aligner: an ORT wav2vec2 session + the HF tokenizer."""

    def __init__(self, onnx_path, tokenizer, providers=None) -> None:
        self.session = _ort_session(onnx_path, providers)
        self.tokenizer = tokenizer

    def generate_emissions(self, audio, *, batch_size=4):
        return generate_emissions_np(self.session, audio, batch_size=batch_size)

    def get_alignments(self, emissions, tokens):
        return get_alignments_np(emissions, tokens, self.tokenizer)


def _sanitize(model_path: str) -> str:
    return model_path.replace("/", "__")


def load_onnx_aligner(model_path: str, models_dir, *, providers=None) -> OnnxCtcAligner:
    """Build the torch-free aligner for `model_path`, exporting the `.onnx` once
    (cached in `models_dir`). The tokenizer loads via HF (torch-free)."""
    from transformers import AutoTokenizer

    onnx_path = Path(models_dir) / f"ctc_align__{_sanitize(model_path)}.onnx"
    if not onnx_path.exists():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        export_ctc_model(model_path, onnx_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, word_delimiter_token=None)
    return OnnxCtcAligner(onnx_path, tokenizer, providers=providers)
