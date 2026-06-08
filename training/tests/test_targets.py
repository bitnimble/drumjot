import numpy as np

import drumjot_training.targets as targets


def test_single_onset_peaks_at_its_frame():
    arr = targets.onsets_to_target([0.5], n_frames=200, fps=100.0, sigma_frames=1.0)
    assert arr.shape == (200,)
    assert int(np.argmax(arr)) == 50
    assert arr[50] == 1.0  # the true peak sits exactly on the true frame


def test_empty_onsets_give_all_zeros():
    arr = targets.onsets_to_target([], n_frames=50, fps=100.0, sigma_frames=1.0)
    assert arr.shape == (50,)
    assert not np.any(arr)


def test_values_never_exceed_one_for_close_onsets():
    # two onsets one frame apart: overlapping Gaussians combine by max, not sum
    arr = targets.onsets_to_target([0.50, 0.51], n_frames=200, fps=100.0, sigma_frames=2.0)
    assert arr.max() <= 1.0 + 1e-6


def test_onset_past_the_end_is_ignored():
    arr = targets.onsets_to_target([10.0], n_frames=50, fps=100.0, sigma_frames=1.0)
    assert not np.any(arr)


def test_falloff_is_symmetric_around_center():
    arr = targets.onsets_to_target([1.0], n_frames=300, fps=100.0, sigma_frames=2.0)
    c = 100
    assert arr[c] == 1.0
    assert arr[c - 1] == arr[c + 1]
    assert arr[c - 1] < arr[c]


def test_pos_weights_higher_for_sparser_lanes():
    t = np.zeros((2, 100), dtype=np.float32)
    t[0, :50] = 1.0  # dense lane: 50 positive frames
    t[1, :5] = 1.0  # sparse lane: 5 positive frames
    w = targets.pos_weights_from_targets([t])
    assert w[1] > w[0]
    assert abs(w[0] - 1.0) < 1e-6  # 50 neg / 50 pos = 1.0


def test_pos_weights_clamped_to_cap():
    t = np.zeros((1, 1000), dtype=np.float32)
    t[0, 0] = 1.0  # 1 pos / 999 neg -> 999, clamped
    w = targets.pos_weights_from_targets([t], cap=50.0)
    assert w[0] == 50.0


def test_pos_weights_empty_lane_is_one():
    t = np.zeros((1, 100), dtype=np.float32)  # no positives
    w = targets.pos_weights_from_targets([t])
    assert w[0] == 1.0
