import numpy as np

from drumjot_training import clean


def _env_with_peaks(frames, n=1000):
    env = np.full(n, 0.01)
    for f in frames:
        env[f] = 1.0
    return env


def test_drops_lane_whose_onsets_miss_transients():
    fps = 100.0
    env = _env_with_peaks([100, 200, 300])  # transients at 1.0, 2.0, 3.0 s
    onsets = {
        "hc": [1.0, 2.0, 3.0],   # on the transients -> supported
        "rd": [1.5, 2.5, 3.5],   # in the gaps -> unsupported
    }
    filt, sup = clean.filter_lanes_by_support(onsets, env, fps, support_floor=0.5, min_support=0.95)
    assert filt["hc"] == [1.0, 2.0, 3.0]   # kept
    assert filt["rd"] == []                # dropped
    assert sup["hc"] == 1.0 and sup["rd"] < 0.95


def test_partial_support_below_threshold_drops_the_lane():
    fps = 100.0
    env = _env_with_peaks([100, 200])      # only 2 of the 3 ride onsets land on a transient
    onsets = {"rd": [1.0, 2.0, 3.5]}       # support = 2/3 = 0.67
    filt95, sup = clean.filter_lanes_by_support(onsets, env, fps, support_floor=0.5, min_support=0.95)
    assert filt95["rd"] == []              # 0.67 < 0.95 -> dropped
    assert abs(sup["rd"] - 2 / 3) < 1e-9
    # a lenient threshold keeps it
    filt60, _ = clean.filter_lanes_by_support(onsets, env, fps, support_floor=0.5, min_support=0.60)
    assert filt60["rd"] == [1.0, 2.0, 3.5]


def test_empty_lane_passes_through_and_is_not_reported():
    filt, sup = clean.filter_lanes_by_support(
        {"hc": [], "rd": [1.0]}, _env_with_peaks([100]), 100.0, support_floor=0.5, min_support=0.95)
    assert filt["hc"] == []
    assert "hc" not in sup and "rd" in sup
