"""Frozen music-SSL encoder features (MERT) + on-disk cache.

The encoder is frozen; we extract one intermediate hidden layer (design
spec §4 / N2N use MERT layer ~10) at ~75 Hz, 1024-dim, and cache per-clip
features as .npy so epochs are cheap. Encoder-agnostic: MERT (via transformers)
and MuQ (via the `muq` package) are both wired behind the same `encode` /
`encode_layers` interface (pick with `make_encoder`); MuQ runs at 25 fps so the
appended high-band/cym blocks align to `encoder.fps`. MusicFM is a future drop-in.

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

# MuQ (Tencent, wav2vec2-conformer + Mel-RVQ SSL): same 24 kHz input and 1024-dim
# hidden as MERT, but 25 fps (40 ms hop) vs MERT's 75 fps. Loaded via the `muq`
# pip package, not transformers. Drop-in behind the same encode/encode_layers
# interface; the only knock-on is that the high-band/cym blocks must be aligned to
# the encoder's fps (passed through below), not hardcoded to 75. 24 conformer
# layers -> 25 hidden states (embeddings + 24), indexed like MERT.
MUQ_NAME = "OpenMuQ/MuQ-large-msd-iter"
MUQ_SR = 24000
MUQ_FPS = 25.0
MUQ_DIM = 1024

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
FEAT_DIM = MERT_DIM + HB_BANDS  # model input width (1040) -- default (hb on, cym off)

# Sub-6 kHz ride/crash/hi-hat timbre pathway: a frame-wise (75 fps) port of the
# deterministic cymbal_split.py features the transcriber measures per onset. The
# high band tells cymbals apart from non-cymbals; THIS band tells cymbals apart
# from each OTHER (ride ping vs crash wash), which the high band can't.
CYM_BANDS = 5
CYM_VARIANT = "cym5"


def feat_variant(high_band: bool = True, cym: bool = False) -> str:
    """Cache-key token for a feature recipe, e.g. "hb16", "cym5", "hb16+cym5",
    or "" (raw MERT). Order is fixed so keys are stable."""
    parts = []
    if high_band:
        parts.append(FEAT_VARIANT)
    if cym:
        parts.append(CYM_VARIANT)
    return "+".join(parts)


def feat_dim(high_band: bool = True, cym: bool = False) -> int:
    """Model input width for a feature recipe: MERT + optional appended blocks."""
    return MERT_DIM + (HB_BANDS if high_band else 0) + (CYM_BANDS if cym else 0)


def cache_key(
    audio_path: str | Path,
    encoder: str,
    layer: int,
    window: float | None = None,
    variant: str = FEAT_VARIANT,
    start: float = 0.0,
) -> str:
    """Stable cache key for a clip's features under encoder+layer (+window cap).

    `variant` names the feature recipe (e.g. the appended high-band block);
    changing the recipe invalidates old caches by changing the key. Pass "" for
    raw MERT-only features (the layer-sweep probe).

    `start` is the window offset (seconds) for multi-window clips; it's appended
    to the key ONLY when non-zero, so the default (whole-clip-from-0) keys are
    byte-identical to pre-windowing caches and existing features are reused."""
    raw = f"{Path(audio_path).expanduser().absolute()}|{encoder}|{layer}|{window}|{variant}"
    if start:
        raw += f"|s{start:g}"
    return hashlib.sha1(raw.encode()).hexdigest()


def highband_from_wave(y44: np.ndarray, n_frames: int, fps: float = MERT_FPS) -> np.ndarray:
    """(n_frames, HB_BANDS) log-mel energies of the 6-20 kHz band at `fps`.

    `y44` must be mono at `HB_SR`. dB-scaled to [0, 1] ([-80, 0] dB clipped),
    deterministic (no learned/per-clip normalisation, so absolute level
    survives -- bleed suppression needs it). Padded/trimmed to `n_frames` to line
    up with the encoder frames. `fps` is the ENCODER's frame rate (75 for MERT,
    25 for MuQ) so the hop matches; default keeps the MERT alignment."""
    import librosa

    hop = int(round(HB_SR / fps))  # 588 @ 75 fps, 1764 @ 25 fps (exact at 44100/fps)
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
    audio_path: str | Path, n_frames: int, max_seconds: float | None = None,
    start_seconds: float = 0.0, fps: float = MERT_FPS,
) -> np.ndarray:
    """`highband_from_wave` for a file: loads at HB_SR (resampling if needed;
    sources at <=24 kHz simply yield near-zero bands, degrading gracefully).
    `start_seconds`/`max_seconds` select the [start, start+max] window. `fps` is
    the encoder frame rate the block must align to (MERT 75 / MuQ 25)."""
    import librosa

    y44, _ = librosa.load(str(audio_path), sr=HB_SR, mono=True)
    a = int(start_seconds * HB_SR)
    b = a + int(max_seconds * HB_SR) if max_seconds is not None else None
    y44 = y44[a:b]
    return highband_from_wave(y44, n_frames, fps)


def cym_features(
    audio_path: str | Path, n_frames: int, max_seconds: float | None = None,
    start_seconds: float = 0.0, fps: float = MERT_FPS,
) -> np.ndarray:
    """(n_frames, CYM_BANDS) sub-6 kHz ride/crash/hi-hat timbre channels at 75 fps,
    frame-aligned to MERT. A frame-wise port of the deterministic per-onset
    features in the transcriber's `cymbal_split.py` (the ear-confirmed ride/crash
    discriminators), so the model gets the cue MERT abstracts away and the high
    band can't carry (it separates cymbals from non-cymbals, not from each other):

      low_mid : fundamental band (250-800 Hz) over wash band (1.5-5 kHz), dB --
                a ride's pitched ping fills the low band (high), a crash is mid
                wash (low). cymbal_split's ear-confirmed winner.
      crest   : low-band (200-1500 Hz) spectral crest, dB -- tall narrow partial
                (ride) vs flat noise (crash).
      flat    : spectral flatness 250 Hz-14 kHz (band-restricted exactly as the
                split does; full-range collapses on the dead >14 kHz bins).
      + low_mid smoothed at ~250 ms and ~1 s: the split measured low_mid on a
        POST-attack window (the attack transient blurs the ratio); the smoothed
        copies hand the GRU that post-attack tone directly.

    Deterministic [0,1] scaling (no per-clip norm), so absolute level survives.
    Computed at HB_SR (44.1 kHz), hop 588 -> exact 75 fps."""
    import librosa

    y, _ = librosa.load(str(audio_path), sr=HB_SR, mono=True)
    a = int(start_seconds * HB_SR)
    b = a + int(max_seconds * HB_SR) if max_seconds is not None else None
    y = y[a:b]
    hop = int(round(HB_SR / fps))  # 588 @ 75 fps, 1764 @ 25 fps (exact at 44100/fps)
    if y.size < HB_NFFT:
        y = np.pad(y, (0, HB_NFFT - y.size))
    S = np.abs(librosa.stft(y, n_fft=HB_NFFT, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=HB_SR, n_fft=HB_NFFT)

    def band(lo: float, hi: float) -> np.ndarray:
        return S[(freqs >= lo) & (freqs <= hi), :]

    fund = band(250.0, 800.0).sum(axis=0)
    wash = band(1500.0, 5000.0).sum(axis=0)
    low_mid_db = 10.0 * np.log10(np.maximum(fund, 1e-20) / np.maximum(wash, 1e-20))
    low_mid = np.clip((low_mid_db + 40.0) / 80.0, 0.0, 1.0)  # [-40,+40] dB -> [0,1]
    tb = band(200.0, 1500.0)
    crest_db = 10.0 * np.log10(
        np.maximum(tb.max(axis=0), 1e-20) / np.maximum(tb.mean(axis=0), 1e-20)
    )
    crest = np.clip(crest_db / 30.0, 0.0, 1.0)  # 0..30 dB -> [0,1]
    fb = band(250.0, 14000.0) + 1e-10
    flat = np.exp(np.mean(np.log(fb), axis=0)) / np.mean(fb, axis=0)  # already [0,1]

    def smooth(x: np.ndarray, win: int) -> np.ndarray:
        return np.convolve(x, np.ones(win) / win, mode="same")

    w_250ms, w_1s = max(1, int(round(0.25 * fps))), max(1, int(round(fps)))  # fps-scaled
    feat = np.stack(  # ~250 ms / 1 s post-attack smoothing windows (19/75 @ 75 fps)
        [low_mid, crest, flat, smooth(low_mid, w_250ms), smooth(low_mid, w_1s)], axis=1
    ).astype(np.float32)
    if feat.shape[0] < n_frames:
        feat = np.pad(feat, ((0, n_frames - feat.shape[0]), (0, 0)))
    return feat[:n_frames]


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
        self.fps = MERT_FPS
        self.dim = MERT_DIM
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._fe = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(name, trust_remote_code=True).to(self.device)
        except Exception as e:  # noqa: BLE001
            # The package forces HF offline (see __init__), so this means the
            # model isn't in the local cache -- never a network hiccup. Point at
            # the one-time fetch step instead of leaking a cryptic HF OSError.
            raise RuntimeError(
                f"MERT model {name!r} is not available offline (HF_HUB_OFFLINE is on by "
                f"default). Fetch it ONCE with:\n"
                f"    python training/scripts/fetch_models.py\n"
                f"then re-run. (underlying error: {type(e).__name__}: {e})"
            ) from e
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

    def n_hidden_states(self) -> int:
        """How many hidden-state tensors `output_hidden_states` returns (embedding
        + one per layer); valid `layer` indices are 0..n-1. Lets a layer sweep
        clamp to the encoder's real depth instead of crashing on an out-of-range
        index. MERT-v1-330M returns 25."""
        import torch

        y = np.zeros(int(0.2 * self.sr), dtype=np.float32)
        inputs = self._fe(y, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad(), runtime.autocast():
            out = self._model(**inputs, output_hidden_states=True)
        return len(out.hidden_states)


class MuQEncoder:
    """Frozen MuQ feature extractor (Tencent wav2vec2-conformer, 24 kHz, 25 fps,
    1024-dim). Same `encode` / `encode_layers` / `.name` / `.layer` / `.sr` /
    `.fps` / `.dim` interface as `MertEncoder`, so `embed_clip` and the sweeps
    treat the two interchangeably. Loaded via the `muq` package (not transformers)
    and run in FP32 -- MuQ documents that bf16 can NaN, so no autocast here."""

    def __init__(self, name: str = MUQ_NAME, layer: int = 10, device: str | None = None):
        import torch

        try:
            from muq import MuQ
        except ImportError as e:
            raise RuntimeError(
                "The MuQ encoder needs the `muq` package, which isn't installed.\n"
                "    (cd transcriber && uv pip install muq)\n"
                "then pre-fetch the weights once with training/scripts/fetch_models.py."
            ) from e

        self.name = name
        self.layer = layer
        self.fps = MUQ_FPS
        self.dim = MUQ_DIM
        self.sr = MUQ_SR
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._model = MuQ.from_pretrained(name).to(self.device).eval()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"MuQ model {name!r} is not available offline (HF_HUB_OFFLINE is on by "
                f"default). Fetch it ONCE with:\n"
                f"    python training/scripts/fetch_models.py\n"
                f"then re-run. (underlying error: {type(e).__name__}: {e})"
            ) from e

    def _hidden(self, waveform: np.ndarray, sr: int):
        import torch

        if sr != self.sr:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sr)
        wavs = torch.as_tensor(waveform, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():  # FP32, no autocast (MuQ warns bf16 can NaN)
            return self._model(wavs, output_hidden_states=True).hidden_states

    def encode(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Frozen layer-`layer` features (frames, dim) for `waveform`."""
        return self._hidden(waveform, sr)[self.layer][0].float().cpu().numpy()

    def encode_layers(self, waveform: np.ndarray, sr: int, layers: list[int]) -> dict[int, np.ndarray]:
        """One forward pass -> {layer: (frames, dim)} for several hidden layers."""
        hs = self._hidden(waveform, sr)
        return {int(li): hs[li][0].float().cpu().numpy() for li in layers}

    def n_hidden_states(self) -> int:
        """How many hidden-state tensors the model returns (valid layers 0..n-1).
        MuQ-large-msd-iter exposes **13** (embedding + 12 layers), NOT the 24 the
        bundled w2v2 config implies, so a MERT-style 0..24 sweep must be clamped."""
        return len(self._hidden(np.zeros(int(0.2 * self.sr), dtype=np.float32), self.sr))


def _encoder_class(name: str):
    """Pick the encoder implementation for a model `name` (no instantiation, so
    it's importable/testable without the heavy deps). MuQ ids route to MuQEncoder,
    everything else (MERT, MusicFM later) to MertEncoder's transformers loader."""
    return MuQEncoder if "muq" in name.lower() else MertEncoder


def make_encoder(name: str = MERT_NAME, layer: int = 10, device: str | None = None):
    """Construct the right frozen encoder for `name`. Both encoders share the
    `encode`/`encode_layers`/`.fps`/`.dim`/`.sr` interface, so callers (embed_clip,
    materialize, the sweeps) need no encoder-specific branches."""
    return _encoder_class(name)(name=name, layer=layer, device=device)


def embed_clip(
    audio_path: str | Path,
    encoder: MertEncoder | MuQEncoder,
    cache_dir: str | Path | None = None,
    max_seconds: float | None = None,
    cache_dtype: str = "float16",
    high_band: bool = True,
    cym: bool = False,
    start_seconds: float = 0.0,
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

    `high_band` (default True) appends the 6-20 kHz block; `cym` (default False)
    appends the sub-6 kHz ride/crash block. The model width is
    `feat_dim(high_band, cym)` and the cache key carries `feat_variant(...)` so
    every recipe (raw / hb / cym / hb+cym) lands in its own cache, never colliding.
    """
    variant = feat_variant(high_band, cym)  # distinct cache per recipe
    cache_file: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = cache_key(audio_path, encoder.name, encoder.layer, max_seconds, variant, start_seconds)
        cache_file = cache_dir / f"{key}.npy"
        if cache_file.exists():
            return np.load(cache_file)
    y = load_audio(audio_path, sr=encoder.sr)
    a = int(start_seconds * encoder.sr)
    b = a + int(max_seconds * encoder.sr) if max_seconds is not None else None
    y = y[a:b]  # the [start, start+max] window (one MERT forward, always <= max)
    feat = encoder.encode(y, encoder.sr)
    fps = getattr(encoder, "fps", MERT_FPS)  # align the spectral blocks to the encoder
    blocks = [feat]
    if high_band:
        # the 6-20 kHz sizzle MERT's 24 kHz input discards (cymbal vs non-cymbal)
        blocks.append(highband_features(audio_path, feat.shape[0], max_seconds, start_seconds, fps))
    if cym:
        # sub-6 kHz ride/crash timbre (cymbal vs cymbal); frame-aligned
        blocks.append(cym_features(audio_path, feat.shape[0], max_seconds, start_seconds, fps))
    feat = np.concatenate(blocks, axis=1) if len(blocks) > 1 else feat
    feat = feat.astype(cache_dtype, copy=False)
    if cache_file is not None:
        np.save(cache_file, feat)
    return feat
