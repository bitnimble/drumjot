import numpy as np

from drumjot_dsp import peakpick


def test_pick_peaks_height_and_min_distance():
    a = np.zeros(100, dtype=np.float32)
    a[10] = 0.9
    a[13] = 0.6  # 3 frames after (inside 5-frame min-distance) and lower -> dropped
    a[50] = 0.2  # below threshold 0.5
    a[80] = 0.8
    frames = peakpick.pick_peaks(a, fps=100.0, threshold=0.5, min_distance_s=0.05)
    assert frames.tolist() == [10, 80]


def test_pick_peaks_prominence_rejects_plateau_wobble():
    a = np.full(60, 0.8, dtype=np.float32)
    a[30] = 0.82  # tiny ripple on the plateau
    a[10] = 1.0  # a real, prominent peak
    no_prom = peakpick.pick_peaks(a, fps=100.0, threshold=0.5, min_distance_s=0.01)
    with_prom = peakpick.pick_peaks(a, fps=100.0, threshold=0.5, min_distance_s=0.01, prominence=0.1)
    assert 30 in no_prom.tolist()
    assert with_prom.tolist() == [10]


def test_decay_reset_collapses_sustain_but_keeps_separated_hits():
    sustain = np.full(40, 0.9, dtype=np.float32)
    sustain[10] = 1.0
    sustain[20] = 0.95
    sustain[30] = 0.95
    kept = peakpick.decay_reset_filter(sustain, np.array([10, 20, 30]), reset_frac=0.6, reset_floor=0.05)
    assert kept.tolist() == [10]  # never decayed -> one event

    sep = np.zeros(40, dtype=np.float32)
    sep[10] = 1.0
    sep[30] = 1.0
    kept2 = peakpick.decay_reset_filter(sep, np.array([10, 30]), reset_frac=0.6, reset_floor=0.05)
    assert kept2.tolist() == [10, 30]


def test_resolve_threshold_fixed_vs_adaptive():
    a = np.concatenate([np.zeros(95), np.full(5, 1.0)]).astype(np.float32)
    assert peakpick.resolve_threshold(a, fixed=0.3) == 0.3
    assert peakpick.resolve_threshold(a, fixed=0.3, adaptive=True, k=0.5, pct=95.0, floor=0.1) >= 0.1
    quiet = np.full(100, 0.01, dtype=np.float32)
    assert peakpick.resolve_threshold(quiet, fixed=0.3, adaptive=True, k=0.5, floor=0.2) == 0.2
