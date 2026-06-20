"""Label-free signal features for the param predictor.

Two groups (design spec §features):

- **Activation-curve** features (per lane): shape statistics of the model's
  per-frame onset-likelihood curve - where the noise floor sits, how separated
  the real peaks are, how rhythmic the curve is, and the deterministic knee
  threshold. These are exactly the quantities a threshold/prominence depends on.
- **Audio** features (per stem, shared by that stem's lanes): coarse timbre /
  dynamics descriptors - spectral shape, crest factor, loudness spread, and
  high-band energy (cymbal-relevant).

`feature_dict` / `feature_vector` join one lane's activation features with its
stem's audio features into the predictor's input row, in a fixed name order.
Pure numpy/scipy/librosa, no torch.
"""
from __future__ import annotations

import numpy as np

from drumjot_training.parampred import baseline

ACT_FEATURES: tuple[str, ...] = (
    "act_noise_floor", "act_p50", "act_p75", "act_p90", "act_max",
    "act_cand_per_s", "act_top_median_ratio", "act_beat_autocorr", "act_knee",
)
AUDIO_FEATURES: tuple[str, ...] = (
    "aud_centroid", "aud_rolloff", "aud_flatness", "aud_bandwidth",
    "aud_crest", "aud_rms_p50", "aud_rms_p90", "aud_highband",
)
FEATURE_NAMES: tuple[str, ...] = ACT_FEATURES + AUDIO_FEATURES

# High-band energy ratio window (cymbal-relevant brightness), matches the
# model's auxiliary high-band block (drumjot_training.embeddings).
_HB_LO_HZ, _HB_HI_HZ = 6000.0, 20000.0


def activation_features(
    activation: np.ndarray, fps: float, min_distance_s: float, *, beat_period_s: float | None = None
) -> dict[str, float]:
    """Shape statistics of one lane's per-frame activation curve."""
    a = np.asarray(activation, dtype=np.float64)
    n = a.size
    heights = baseline.candidate_peak_heights(a, fps, min_distance_s)
    if heights.size:
        p50, p75, p90, pmax = (float(np.percentile(heights, q)) for q in (50, 75, 90, 100))
        median = max(p50, 1e-6)
        top_median = pmax / median
    else:
        p50 = p75 = p90 = pmax = 0.0
        top_median = 0.0
    duration = n / fps if n else 1.0
    return {
        "act_noise_floor": float(np.percentile(a, 10)) if n else 0.0,
        "act_p50": p50,
        "act_p75": p75,
        "act_p90": p90,
        "act_max": pmax,
        "act_cand_per_s": heights.size / duration,
        "act_top_median_ratio": top_median,
        "act_beat_autocorr": _beat_autocorr(a, fps, beat_period_s),
        "act_knee": baseline.knee_threshold(a, fps, min_distance_s),
    }


def _beat_autocorr(activation: np.ndarray, fps: float, beat_period_s: float | None) -> float:
    """Normalized autocorrelation of the curve at the beat-period lag in [-1, 1];
    a proxy for how periodic (rhythmically regular) the activation is. 0 when no
    beat period is supplied or the lag is out of range."""
    if beat_period_s is None or beat_period_s <= 0 or activation.size < 4:
        return 0.0
    lag = round(beat_period_s * fps)
    if lag < 1 or lag >= activation.size:
        return 0.0
    x = activation - activation.mean()
    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def audio_features(waveform: np.ndarray, sr: int) -> dict[str, float]:
    """Coarse timbre / dynamics descriptors of a stem waveform. Spectral-shape
    features are normalized by the Nyquist frequency so they land in ~[0, 1]."""
    import librosa

    y = np.asarray(waveform, dtype=np.float32)
    nyq = sr / 2.0
    if y.size == 0 or float(np.max(np.abs(y))) < 1e-9:
        return dict.fromkeys(AUDIO_FEATURES, 0.0)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    rms = librosa.feature.rms(S=S)[0]
    peak = float(np.max(np.abs(y)))
    rms_overall = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) or 1e-9
    band = (freqs >= _HB_LO_HZ) & (freqs <= _HB_HI_HZ)
    total_e = float(np.sum(S ** 2)) or 1e-9
    highband = float(np.sum(S[band, :] ** 2) / total_e)
    return {
        "aud_centroid": centroid / nyq,
        "aud_rolloff": rolloff / nyq,
        "aud_flatness": flatness,
        "aud_bandwidth": bandwidth / nyq,
        "aud_crest": peak / rms_overall,
        "aud_rms_p50": float(np.percentile(rms, 50)),
        "aud_rms_p90": float(np.percentile(rms, 90)),
        "aud_highband": highband,
    }


def feature_dict(
    activation: np.ndarray,
    fps: float,
    min_distance_s: float,
    waveform: np.ndarray,
    sr: int,
    *,
    beat_period_s: float | None = None,
) -> dict[str, float]:
    """One predictor input row: this lane's activation features joined with its
    stem's audio features, in `FEATURE_NAMES` order."""
    act = activation_features(activation, fps, min_distance_s, beat_period_s=beat_period_s)
    aud = audio_features(waveform, sr)
    merged = {**act, **aud}
    return {k: merged[k] for k in FEATURE_NAMES}


def feature_vector(
    activation: np.ndarray,
    fps: float,
    min_distance_s: float,
    waveform: np.ndarray,
    sr: int,
    *,
    beat_period_s: float | None = None,
) -> np.ndarray:
    """`feature_dict` as a dense vector in `FEATURE_NAMES` order."""
    d = feature_dict(activation, fps, min_distance_s, waveform, sr, beat_period_s=beat_period_s)
    return np.array([d[k] for k in FEATURE_NAMES], dtype=np.float64)
