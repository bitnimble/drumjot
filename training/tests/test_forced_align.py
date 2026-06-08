import numpy as np

import drumjot_training.forced_align as fa


def _env_with_peak(n, peak_idx, height=1.0):
    env = np.zeros(n, dtype=np.float64)
    env[peak_idx] = height
    return env


def test_supported_onset_snaps_to_the_peak():
    env = _env_with_peak(200, 50)  # peak at 0.50 s @ 100 fps
    out = fa.align_lane([0.48], env, env_fps=100.0, window_s=0.05, support_floor=0.5)
    (t, supported), = out
    assert supported is True
    assert abs(t - 0.50) < 1e-6


def test_unsupported_onset_is_flagged_and_kept_in_place():
    env = np.zeros(200, dtype=np.float64)  # no transient anywhere
    out = fa.align_lane([0.48], env, env_fps=100.0, window_s=0.05, support_floor=0.5)
    (t, supported), = out
    assert supported is False
    assert t == 0.48  # not snapped to a phantom max


def test_peak_below_floor_is_unsupported():
    env = _env_with_peak(200, 50, height=0.2)  # real-ish but weak
    out = fa.align_lane([0.49], env, env_fps=100.0, window_s=0.05, support_floor=0.5)
    (_t, supported), = out
    assert supported is False


def test_onset_out_of_range_is_kept_and_unsupported():
    env = _env_with_peak(50, 10)
    out = fa.align_lane([10.0], env, env_fps=100.0, window_s=0.05, support_floor=0.5)
    (t, supported), = out
    assert supported is False
    assert t == 10.0


def test_align_chart_runs_every_lane():
    env = _env_with_peak(200, 50)
    chart = {"k": [0.48], "s": [], "t": [], "h": [0.50], "cy": []}
    out = fa.align_chart(chart, env, env_fps=100.0, window_s=0.05, support_floor=0.5)
    assert set(out) == {"k", "s", "t", "h", "cy"}
    assert out["k"][0][1] is True
    assert out["s"] == []
