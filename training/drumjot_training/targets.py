"""Per-frame onset targets from onset times.

Each onset becomes a Gaussian bump (peak 1.0) on a per-frame activation
curve, the standard ADT target-smoothing trick: it eases optimization and
absorbs a few frames of label jitter while keeping the true peak on the
true frame. Overlapping bumps combine by element-wise max (not sum) so the
target stays in [0, 1]. See HIHAT.md §6 (target encoding) and the design
spec §4.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def onsets_to_target(
    onset_times_sec: Sequence[float],
    n_frames: int,
    fps: float,
    sigma_frames: float,
) -> np.ndarray:
    """Render `onset_times_sec` as a (n_frames,) float32 activation curve.

    A Gaussian of width `sigma_frames` is centered on each onset's frame
    (`round(t * fps)`); curves combine by element-wise max, so the result
    is bounded in [0, 1] with the exact peak (1.0) on the onset frame.
    Onsets whose support falls entirely outside [0, n_frames) contribute
    nothing (and are silently skipped).
    """
    target = np.zeros(n_frames, dtype=np.float32)
    if n_frames <= 0 or sigma_frames <= 0.0:
        return target
    half = max(1, math.ceil(4.0 * sigma_frames))  # render +/- 4 sigma, the rest is ~0
    two_sigma_sq = 2.0 * sigma_frames * sigma_frames
    for t in onset_times_sec:
        center = int(round(float(t) * fps))
        lo = max(0, center - half)
        hi = min(n_frames, center + half + 1)
        if lo >= hi:
            continue
        idx = np.arange(lo, hi)
        bump = np.exp(-((idx - center) ** 2) / two_sigma_sq).astype(np.float32)
        np.maximum(target[lo:hi], bump, out=target[lo:hi])
    return target


def pos_weights_from_targets(
    targets: Iterable[np.ndarray],
    threshold: float = 0.5,
    cap: float = 50.0,
) -> np.ndarray:
    """Per-lane BCE `pos_weight` = neg/pos frame ratio, clamped to [1, cap].

    `targets` is an iterable of (n_lanes, T) target arrays; a frame counts as
    positive when its target exceeds `threshold`. Sparser lanes get larger
    weights; lanes with no positives get 1.0. Used to counter the heavy
    negative imbalance per lane (design spec §4 / STAR imbalance finding).
    """
    targets = list(targets)
    if not targets:
        return np.ones(0, dtype=np.float32)
    n_lanes = targets[0].shape[0]
    pos = np.zeros(n_lanes, dtype=np.float64)
    total = 0
    for t in targets:
        pos += (t > threshold).sum(axis=1)
        total += t.shape[1]
    neg = total - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return np.clip(w, 1.0, cap).astype(np.float32)
