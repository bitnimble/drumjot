import numpy as np

from drumjot_training import postfilter


def test_filter_lane_drops_unsupported_and_snaps_supported():
    env = np.zeros(100)
    env[10] = 5.0  # a single transient at frame 10 (0.10s @ 100fps)
    # one onset near the transient (kept + snapped), one in silence (dropped)
    out = postfilter.filter_lane([0.108, 0.50], env, 100.0, window_s=0.05, support_floor=1.0)
    assert len(out) == 1
    assert abs(out[0] - 0.10) < 1e-6  # snapped onto the envelope peak


def test_filter_lane_keeps_all_when_floor_zero():
    env = np.ones(100)
    out = postfilter.filter_lane([0.10, 0.20, 0.30], env, 100.0, window_s=0.03, support_floor=0.0)
    assert len(out) == 3


def test_support_floor_from_env_is_percentile():
    env = np.arange(101, dtype=float)  # 0..100
    assert postfilter.support_floor_from_env(env, 50) == 50.0
    assert postfilter.support_floor_from_env(env, 90) == 90.0
