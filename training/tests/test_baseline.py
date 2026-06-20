import numpy as np

import drumjot_training.targets as targets
from drumjot_training.parampred import baseline


def _curve(onset_times, n_frames, fps, *, scale=1.0, sigma=1.0):
    return targets.onsets_to_target(onset_times, n_frames=n_frames, fps=fps, sigma_frames=sigma) * scale


def test_otsu_splits_a_bimodal_set():
    lo = np.full(50, 0.1)
    hi = np.full(50, 0.9)
    thr = baseline.otsu_threshold(np.concatenate([lo, hi]))
    assert 0.1 < thr < 0.9


def test_knee_threshold_sits_between_noise_and_real_peaks():
    fps = 100.0
    n = 600
    # real peaks at 1.0, low decay-wobble bumps at 0.2
    curve = np.maximum(_curve([0.5, 1.5, 2.5, 3.5], n, fps),
                       _curve([1.0, 2.0, 3.0, 4.0, 4.5, 5.0], n, fps, scale=0.2))
    thr = baseline.knee_threshold(curve, fps, min_distance_s=0.02)
    assert 0.2 < thr < 1.0


def test_knee_threshold_floors_when_too_few_candidates():
    fps = 100.0
    curve = _curve([1.0], 300, fps)  # a single peak: not enough to fit a knee
    thr = baseline.knee_threshold(curve, fps, min_distance_s=0.02, floor=0.15)
    assert thr == 0.15


def test_higher_noise_floor_yields_higher_threshold():
    fps = 100.0
    n = 600
    reals = [0.5, 1.5, 2.5, 3.5]
    noise_lo = [1.0, 2.0, 3.0, 4.0, 4.5, 5.0]
    quiet = np.maximum(_curve(reals, n, fps), _curve(noise_lo, n, fps, scale=0.15))
    loud = np.maximum(_curve(reals, n, fps), _curve(noise_lo, n, fps, scale=0.45))
    assert baseline.knee_threshold(loud, fps, 0.02) > baseline.knee_threshold(quiet, fps, 0.02)


def test_deterministic_params_fills_every_param_from_seed():
    fps = 100.0
    curve = np.maximum(_curve([0.5, 1.5, 2.5], 400, fps),
                       _curve([1.0, 2.0, 3.0], 400, fps, scale=0.2))
    seed = {"threshold": 0.5, "prominence": 0.10, "min_distance_s": 0.05,
            "decay_reset_frac": 0.6, "decay_reset_floor": 0.05}
    p = baseline.deterministic_params(curve, fps, seed)
    assert set(p) == set(baseline.PARAM_NAMES)
    assert 0.2 < p["threshold"] < 1.0           # derived from the knee
    assert p["min_distance_s"] == 0.05          # non-derived params fall back to seed
    assert p["decay_reset_frac"] == 0.6
    assert p["prominence"] > 0.0
