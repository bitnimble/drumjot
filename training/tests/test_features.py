import numpy as np

import drumjot_training.targets as targets
from drumjot_training.parampred import features


def _curve(onset_times, n_frames, fps, *, scale=1.0, sigma=1.0):
    return targets.onsets_to_target(onset_times, n_frames=n_frames, fps=fps, sigma_frames=sigma) * scale


def _sine(freq, sr, secs=1.0):
    t = np.arange(int(sr * secs)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_activation_features_are_complete_and_finite():
    fps = 100.0
    curve = _curve([0.5, 1.5, 2.5], 400, fps)
    d = features.activation_features(curve, fps, min_distance_s=0.02)
    assert set(d) == set(features.ACT_FEATURES)
    assert all(np.isfinite(v) for v in d.values())


def test_top_median_ratio_separates_clean_from_noisy():
    fps = 100.0
    n = 600
    clean = np.maximum(_curve([0.5, 1.5, 2.5], n, fps),
                       _curve([1.0, 2.0, 3.0], n, fps, scale=0.05))
    rng = np.random.default_rng(1)
    noisy = np.clip(0.4 * rng.random(n), 0, 1)
    rc = features.activation_features(clean, fps, 0.02)["act_top_median_ratio"]
    rn = features.activation_features(noisy, fps, 0.02)["act_top_median_ratio"]
    assert rc > rn


def test_beat_autocorr_higher_for_periodic_curve():
    fps = 100.0
    n = 800
    period = 0.5
    periodic = _curve([period * i for i in range(1, 15)], n, fps)
    rng = np.random.default_rng(2)
    random = np.clip(_curve(list(rng.uniform(0, 7.5, 14)), n, fps), 0, 1)
    fp = features.activation_features(periodic, fps, 0.02, beat_period_s=period)["act_beat_autocorr"]
    fr = features.activation_features(random, fps, 0.02, beat_period_s=period)["act_beat_autocorr"]
    assert fp > fr


def test_audio_features_complete_and_brightness_ordered():
    sr = 44100
    bright = features.audio_features(_sine(9000, sr), sr)
    dark = features.audio_features(_sine(200, sr), sr)
    assert set(bright) == set(features.AUDIO_FEATURES)
    assert all(np.isfinite(v) for v in bright.values())
    assert bright["aud_centroid"] > dark["aud_centroid"]


def test_feature_vector_matches_named_order():
    fps = 100.0
    sr = 44100
    curve = _curve([0.5, 1.5, 2.5], 400, fps)
    wave = _sine(1000, sr)
    vec = features.feature_vector(curve, fps, 0.02, wave, sr)
    d = features.feature_dict(curve, fps, 0.02, wave, sr)
    assert vec.shape == (len(features.FEATURE_NAMES),)
    assert list(d) == list(features.FEATURE_NAMES)
    np.testing.assert_allclose(vec, [d[k] for k in features.FEATURE_NAMES])
