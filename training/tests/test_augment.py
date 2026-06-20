import numpy as np
import pytest

from drumjot_training.parampred import augment


def _clicks(sr, positions_s, n_s=2.0):
    """Sharp unit impulses at known times on a quiet noise bed - onset times are
    exactly the impulse samples, so any time shift is directly measurable."""
    y = (1e-4 * np.random.default_rng(0).standard_normal(int(sr * n_s))).astype(np.float32)
    for t in positions_s:
        y[int(t * sr)] += 1.0
    return y


def _peak_positions(y, sr, positions_s, search_ms=30.0):
    """For each expected click, the local-argmax position (s) within +/-search."""
    half = int(search_ms / 1000.0 * sr)
    out = []
    for t in positions_s:
        c = int(t * sr)
        lo, hi = max(0, c - half), min(len(y), c + half)
        out.append((lo + int(np.argmax(np.abs(y[lo:hi])))) / sr)
    return out


def test_gain_scales_without_moving_onsets():
    sr = 44100
    y = _clicks(sr, [0.5, 1.2])
    out = augment.apply_gain(y, 6.0)
    assert len(out) == len(y)
    assert np.max(np.abs(out)) > np.max(np.abs(y))
    np.testing.assert_allclose(_peak_positions(out, sr, [0.5, 1.2]), [0.5, 1.2], atol=2e-3)


def test_eq_tilt_changes_brightness_same_length():
    sr = 44100
    rng = np.random.default_rng(1)
    y = rng.standard_normal(sr).astype(np.float32) * 0.1
    bright = augment.apply_eq_tilt(y, sr, 12.0)
    dark = augment.apply_eq_tilt(y, sr, -12.0)
    assert len(bright) == len(y)

    def centroid(x):
        S = np.abs(np.fft.rfft(x))
        f = np.fft.rfftfreq(len(x), 1 / sr)
        return float((f * S).sum() / (S.sum() + 1e-9))

    assert centroid(bright) > centroid(dark)


def test_reverb_preserves_length_and_onset_position():
    sr = 44100
    y = _clicks(sr, [0.5, 1.2])
    out = augment.apply_reverb(y, sr, decay_s=0.3, wet=0.4)
    assert len(out) == len(y)  # tail truncated to original length
    np.testing.assert_allclose(_peak_positions(out, sr, [0.5, 1.2]), [0.5, 1.2], atol=3e-3)


def test_compression_reduces_crest_factor():
    sr = 44100
    # a loud sustained burst over a quieter bed: the burst exceeds threshold, so
    # the compressor attenuates it and the make-up gain lifts the bed -> crest drops.
    rng = np.random.default_rng(4)
    y = (0.05 * rng.standard_normal(sr * 2)).astype(np.float32)
    y[int(0.5 * sr):int(0.6 * sr)] += np.sin(2 * np.pi * 220 * np.arange(int(0.1 * sr)) / sr).astype(np.float32)

    def crest(x):
        return np.max(np.abs(x)) / (np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)

    out = augment.apply_compression(y, sr, threshold_db=-30, ratio=6.0)
    assert crest(out) < crest(y)
    assert len(out) == len(y)


def test_compression_stays_finite_with_loud_bursts_over_silence():
    # a loud burst next to true silence: FFT-conv roundoff makes the silent-region
    # envelope dip below -eps, and log10 of a negative used to yield NaN. The
    # envelope must be clamped non-negative. (Real drum stems look exactly like
    # this: sharp hits separated by gaps.)
    sr = 44100
    y = np.zeros(sr, dtype=np.float32)
    y[:1500] = 5.0 * np.random.default_rng(0).standard_normal(1500).astype(np.float32)
    out = augment.apply_compression(y, sr, threshold_db=-20, ratio=4.0)
    assert np.isfinite(out).all()


def test_random_chain_output_is_always_finite():
    sr = 44100
    y = (2.5 * np.random.default_rng(1).standard_normal(sr)).astype(np.float32)
    for s in range(20):
        out, _ = augment.random_chain(y, sr, np.random.default_rng(s), use_codec=False)
        assert np.isfinite(out).all(), f"seed {s} produced non-finite audio"


def test_noise_lowers_snr_without_moving_onsets():
    sr = 44100
    y = _clicks(sr, [0.5, 1.2])
    out = augment.apply_noise(y, snr_db=20.0, rng=np.random.default_rng(2))
    assert len(out) == len(y)
    assert np.std(out - y) > 0
    np.testing.assert_allclose(_peak_positions(out, sr, [0.5, 1.2]), [0.5, 1.2], atol=2e-3)


def test_codec_roundtrip_is_delay_compensated():
    if not augment.has_ffmpeg():
        pytest.skip("ffmpeg not available")
    sr = 44100
    y = _clicks(sr, [0.5, 1.2])
    out = augment.apply_codec(y, sr, bitrate_kbps=128)
    assert len(out) == len(y)
    # without delay compensation an MP3 round-trip shifts onsets by ~half a frame
    np.testing.assert_allclose(_peak_positions(out, sr, [0.5, 1.2]), [0.5, 1.2], atol=5e-3)


def test_random_chain_preserves_length_and_returns_description():
    sr = 44100
    y = _clicks(sr, [0.5, 1.2])
    out, desc = augment.random_chain(y, sr, np.random.default_rng(3), use_codec=False)
    assert len(out) == len(y)
    assert isinstance(desc, str) and desc
