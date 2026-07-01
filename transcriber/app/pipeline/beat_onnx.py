"""Torch-free ONNX inference for the Beat This! beat/downbeat tracker.

`OnnxBeatThis.__call__(audio_path) -> (beats, downbeats)` reproduces
`beat_this.inference.File2Beats` (dbn=False) with no torch: the transformer runs
on onnxruntime, and the log-mel frontend, chunk/aggregate, and minimal
peak-pick postprocessing are numpy (reusing `separation.np_stft` for the STFT).

The mel frontend matches torchaudio's LogMelSpect to ~1e-5: STFT magnitude *
1/sqrt(n_fft) (torch.stft normalized=True), a slaney mel filterbank (librosa,
norm=None), then log1p(1000 * mel). The model `.onnx` is exported once (cached;
that step needs torch), after which inference is torch-free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR, N_FFT, HOP, N_MELS, FMIN, FMAX = 22050, 1024, 441, 128, 30, 11000
FPS = 50
CHUNK, BORDER = 1500, 6
_STFT_NORM = 1.0 / np.sqrt(N_FFT)


def export_beatthis(out_path: str | Path, *, checkpoint: str = "final0", opset: int = 17,
                    fp16: bool = False) -> Path:
    """Export the BeatThis transformer (spectrogram -> beat/downbeat logits)."""
    import torch
    from beat_this.inference import load_model
    from torch import nn

    model = load_model(checkpoint, "cpu")

    class Body(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):  # x: (1, T, 128)
            o = self.m(x)
            return o["beat"], o["downbeat"]  # each (1, T)

    body = Body(model).eval()
    dummy = torch.zeros(1, CHUNK, N_MELS)
    out_path = Path(out_path)
    with torch.no_grad():
        torch.onnx.export(
            body, (dummy,), str(out_path),
            input_names=["spect"], output_names=["beat", "downbeat"],
            dynamic_axes={"spect": {1: "frames"}, "beat": {1: "frames"}, "downbeat": {1: "frames"}},
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
    if fp16:
        from app.pipeline.onnx_fp16 import to_fp16

        to_fp16(out_path)
    return out_path


def _mel_fb() -> np.ndarray:
    import librosa

    return librosa.filters.mel(
        sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, htk=False, norm=None
    ).astype(np.float32)


def _logmel(wave: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """(samples,) waveform @ 22050 -> (T, 128) log-mel, matching torchaudio LogMelSpect."""
    from app.pipeline.separation import np_stft

    window = np_stft.hann_window(N_FFT)
    spec = np_stft.stft(wave[None].astype(np.float32), N_FFT, HOP, window)[0]  # (F, T) complex
    mag = np.abs(spec) * _STFT_NORM
    return np.log1p(1000.0 * (fb @ mag)).T.astype(np.float32)


def _zeropad(x: np.ndarray, left: int, right: int) -> np.ndarray:
    return np.pad(x, ((left, right), (0, 0))) if (left or right) else x


def _split(spect: np.ndarray):
    """Mirror beat_this.inference.split_piece (chunk 1500, border 6, avoid_short_end)."""
    t = spect.shape[0]
    starts = np.arange(-BORDER, t - BORDER, CHUNK - 2 * BORDER)
    if t > CHUNK - 2 * BORDER:
        starts[-1] = t - (CHUNK - BORDER)
    chunks = [
        _zeropad(spect[max(s, 0) : min(s + CHUNK, t)], max(0, -s), max(0, min(BORDER, s + CHUNK - t)))
        for s in starts
    ]
    return chunks, starts


def _aggregate(preds, starts, t: int):
    """Mirror aggregate_prediction (border cut + keep_first overwrite)."""
    if BORDER > 0:
        preds = [(b[BORDER:-BORDER], d[BORDER:-BORDER]) for b, d in preds]
    beat = np.full(t, -1000.0, np.float32)
    down = np.full(t, -1000.0, np.float32)
    for s, (b, d) in reversed(list(zip(starts, preds, strict=True))):  # keep_first
        beat[s + BORDER : s + CHUNK - BORDER] = b
        down[s + BORDER : s + CHUNK - BORDER] = d
    return beat, down


def _deduplicate(peaks: np.ndarray, width: int = 1) -> np.ndarray:
    """Port of beat_this.model.postprocessor.deduplicate_peaks (running-mean merge)."""
    result: list[float] = []
    it = (int(p) for p in peaks)
    try:
        p = next(it)
    except StopIteration:
        return np.array(result)
    c = 1
    for p2 in it:
        if p2 - p <= width:
            c += 1
            p += (p2 - p) / c
        else:
            result.append(p)
            p = p2
            c = 1
    result.append(p)
    return np.array(result)


def _peaks(logits: np.ndarray) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d

    mx = maximum_filter1d(logits, size=7, mode="constant", cval=-1e9)  # ~= F.max_pool1d(7,1,3)
    frames = np.nonzero((logits == mx) & (logits > 0))[0]
    return _deduplicate(frames, 1) / FPS


def _postp(beat_logits: np.ndarray, downbeat_logits: np.ndarray):
    """Port of Postprocessor.postp_minimal for one piece."""
    beat_time = _peaks(beat_logits)
    downbeat_time = _peaks(downbeat_logits)
    if len(beat_time) > 0:
        for i, d in enumerate(downbeat_time):
            downbeat_time[i] = beat_time[np.argmin(np.abs(beat_time - d))]
    return beat_time, np.unique(downbeat_time)


class OnnxBeatThis:
    """Torch-free Beat This!; call with an audio path -> `(beats, downbeats)` (seconds)."""

    def __init__(self, onnx_path, providers=None) -> None:
        import onnxruntime as ort

        if providers is None:
            providers = ort.get_available_providers()
        try:
            self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        except Exception:
            self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self._in = self.sess.get_inputs()[0].name
        self.fb = _mel_fb()

    def __call__(self, audio_path):
        import librosa

        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        spect = _logmel(np.asarray(y, np.float32), self.fb)
        chunks, starts = _split(spect)
        preds = [
            tuple(v[0] for v in self.sess.run(None, {self._in: ch[None].astype(np.float32)}))
            for ch in chunks
        ]
        beat_logits, downbeat_logits = _aggregate(preds, starts, spect.shape[0])
        return _postp(beat_logits, downbeat_logits)


def load_beat_session(models_dir, *, providers=None) -> OnnxBeatThis:
    """Build the torch-free Beat This!, exporting the `.onnx` once (cached)."""
    onnx_path = Path(models_dir) / "beat_this.onnx"
    if not onnx_path.exists():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        export_beatthis(onnx_path)
    return OnnxBeatThis(onnx_path, providers=providers)
