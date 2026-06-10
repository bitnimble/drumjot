"""Frozen music-SSL encoder features (MERT) + on-disk cache.

The encoder is frozen; we extract one intermediate hidden layer (design
spec §4 / N2N use MERT layer ~10) at ~75 Hz, 1024-dim, and cache per-clip
features as .npy so epochs are cheap. Encoder-agnostic by intent: MERT now,
MusicFM later behind the same `encode` interface.

Sandbox-verified facts (m-a-p/MERT-v1-330M): sampling_rate 24000, 25 hidden
states, layer-10 output (frames, 1024) at ~75 fps. The model's `nnAudio`
warning is harmless, it only affects an auxiliary CQT feature we don't use.

torch / transformers / librosa are imported lazily so the pure helpers
(cache keys) and the rest of the package stay importable on a host without
a working CUDA torch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from drumjot_training import runtime

MERT_NAME = "m-a-p/MERT-v1-330M"
MERT_SR = 24000
MERT_FPS = 75.0
MERT_DIM = 1024

# High-band spectral pathway: MERT's 24 kHz input caps it at a 12 kHz Nyquist,
# discarding the hat/cymbal sizzle band (8-16 kHz+) BEFORE the encoder sees
# anything -- a structural blind spot for open/closed-hat and ride/crash
# discrimination that no training data can recover. We append a small log-mel
# block computed from the ORIGINAL-rate audio (resampled to 44.1 kHz), band
# 6-20 kHz, frame-aligned to MERT's 75 fps (44100/75 = 588-sample hop, exact).
HB_SR = 44100
HB_BANDS = 16
HB_FMIN = 6000.0
HB_FMAX = 20000.0
HB_NFFT = 2048
FEAT_VARIANT = "hb16"  # cache-key token; bump when the feature recipe changes
FEAT_DIM = MERT_DIM + HB_BANDS  # model input width (1040)


def cache_key(
    audio_path: str | Path,
    encoder: str,
    layer: int,
    window: float | None = None,
    variant: str = FEAT_VARIANT,
) -> str:
    """Stable cache key for a clip's features under encoder+layer (+window cap).

    `variant` names the feature recipe (e.g. the appended high-band block);
    changing the recipe invalidates old caches by changing the key. Pass "" for
    raw MERT-only features (the layer-sweep probe)."""
    raw = f"{Path(audio_path).expanduser().absolute()}|{encoder}|{layer}|{window}|{variant}"
    return hashlib.sha1(raw.encode()).hexdigest()


def highband_from_wave(y44: np.ndarray, n_frames: int) -> np.ndarray:
    """(n_frames, HB_BANDS) log-mel energies of the 6-20 kHz band at 75 fps.

    `y44` must be mono at `HB_SR`. dB-scaled to [0, 1] ([-80, 0] dB clipped),
    deterministic (no learned/per-clip normalisation, so absolute level
    survives -- bleed suppression needs it). Padded/trimmed to `n_frames` to
    line up with the MERT frames."""
    import librosa

    hop = int(round(HB_SR / MERT_FPS))  # 588 (exact at 44100/75)
    if y44.size < HB_NFFT:
        y44 = np.pad(y44, (0, HB_NFFT - y44.size))
    S = librosa.feature.melspectrogram(
        y=y44, sr=HB_SR, n_fft=HB_NFFT, hop_length=hop,
        n_mels=HB_BANDS, fmin=HB_FMIN, fmax=HB_FMAX, power=2.0,
    )
    db = librosa.power_to_db(S, ref=1.0, top_db=None)
    feat = (np.clip(db, -80.0, 0.0) / 80.0 + 1.0).astype(np.float32).T  # (T', bands) in [0,1]
    if feat.shape[0] < n_frames:
        feat = np.pad(feat, ((0, n_frames - feat.shape[0]), (0, 0)))
    return feat[:n_frames]


def highband_features(
    audio_path: str | Path, n_frames: int, max_seconds: float | None = None
) -> np.ndarray:
    """`highband_from_wave` for a file: loads at HB_SR (resampling if needed;
    sources at <=24 kHz simply yield near-zero bands, degrading gracefully)."""
    import librosa

    y44, _ = librosa.load(str(audio_path), sr=HB_SR, mono=True)
    if max_seconds is not None:
        y44 = y44[: int(max_seconds * HB_SR)]
    return highband_from_wave(y44, n_frames)


def load_audio(path: str | Path, sr: int = MERT_SR) -> np.ndarray:
    """Mono float32 audio at `sr` (lazy librosa import)."""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32, copy=False)


class MertEncoder:
    """Frozen MERT feature extractor. `encode(waveform, sr) -> (frames, dim)`."""

    def __init__(self, name: str = MERT_NAME, layer: int = 10, device: str | None = None):
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.name = name
        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._fe = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(name, trust_remote_code=True).to(self.device)
        self._model.eval()
        self.sr = int(self._fe.sampling_rate)

    def encode(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Return frozen layer-`layer` features (frames, dim) for `waveform`."""
        import torch

        if sr != self.sr:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sr)
        inputs = self._fe(waveform, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad(), runtime.autocast():
            out = self._model(**inputs, output_hidden_states=True)
        # .float() before numpy: autocast may return bf16, and the on-disk
        # cache stays FP32 so features are identical across machines.
        feat = out.hidden_states[self.layer][0]  # (frames, dim)
        return feat.float().cpu().numpy()

    def encode_layers(self, waveform: np.ndarray, sr: int, layers: list[int]) -> dict[int, np.ndarray]:
        """One forward pass -> {layer: (frames, dim)} for several hidden layers.
        Used by the layer-sweep probe (scripts/layer_sweep.py) so sweeping N
        layers costs one encode, not N."""
        import torch

        if sr != self.sr:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sr)
        inputs = self._fe(waveform, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad(), runtime.autocast():
            out = self._model(**inputs, output_hidden_states=True)
        return {int(li): out.hidden_states[li][0].float().cpu().numpy() for li in layers}


def embed_clip(
    audio_path: str | Path,
    encoder: MertEncoder,
    cache_dir: str | Path | None = None,
    max_seconds: float | None = None,
    cache_dtype: str = "float16",
) -> np.ndarray:
    """Features for one clip, reading/writing `cache_dir` when given.

    `max_seconds` caps the audio before encoding (bounds MERT's sequence
    length on long clips); it's part of the cache key so capped and full
    features don't collide.

    `cache_dtype` is the on-disk feature precision: **float16 by default**,
    which halves the cache size + per-epoch read bandwidth (so the cache fits
    in the OS page cache) at no real cost, MERT features sit well within fp16
    range and training autocasts to bf16 anyway. Pass `"float32"` for a
    full-precision cache. The dtype is not part of the cache key, so an
    existing cache is reused as-is (loaded at whatever precision it was
    written); delete `_cache_mert` to re-encode at a new precision.
    """
    cache_file: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = cache_key(audio_path, encoder.name, encoder.layer, max_seconds)
        cache_file = cache_dir / f"{key}.npy"
        if cache_file.exists():
            return np.load(cache_file)
    y = load_audio(audio_path, sr=encoder.sr)
    if max_seconds is not None:
        y = y[: int(max_seconds * encoder.sr)]
    feat = encoder.encode(y, encoder.sr)
    # append the high-band block (the 6-20 kHz sizzle MERT's 24 kHz input
    # discards), frame-aligned; cached together as one (T, FEAT_DIM) array
    hb = highband_features(audio_path, feat.shape[0], max_seconds)
    feat = np.concatenate([feat, hb], axis=1).astype(cache_dtype, copy=False)
    if cache_file is not None:
        np.save(cache_file, feat)
    return feat
