"""Convergence detection for the training early-stop (train._lane_converged):
flat trend AND low jitter, both required, over the last `window` epochs."""
import numpy as np

from drumjot_training.train import _lane_converged

W, SLOPE, JIT = 8, 0.002, 0.015


def test_too_few_points_not_converged():
    # fewer than `window` epochs -> can't judge yet
    assert not _lane_converged([0.5, 0.5, 0.5], W, SLOPE, JIT)


def test_flat_and_settled_converged():
    # ~constant 0.60 with tiny noise: slope ~0, jitter small -> converged
    rng = np.random.default_rng(0)
    curve = list(0.60 + rng.normal(0, 0.002, 12))
    assert _lane_converged(curve, W, SLOPE, JIT)


def test_still_climbing_not_converged():
    # steady +0.01/epoch climb: slope exceeds threshold (even though jitter ~0)
    assert not _lane_converged([0.40 + 0.01 * i for i in range(12)], W, SLOPE, JIT)


def test_flat_but_jittery_not_converged():
    # zero-trend but bouncing +/-0.05: slope ~0 yet jitter too high -> keep training
    curve = [0.55, 0.65] * 4
    assert not _lane_converged(curve, W, SLOPE, JIT)


def test_only_recent_window_matters():
    # early chaos, then a settled tail >= window -> converged on the recent window
    curve = [0.1, 0.9, 0.2, 0.8] + [0.600, 0.601, 0.599, 0.600, 0.601, 0.599, 0.600, 0.601]
    assert _lane_converged(curve, W, SLOPE, JIT)
