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


# Lanes whose identity lives in their SUSTAIN, not just the attack: the
# auxiliary frame-activity objective is supervised for these only.
SUSTAINED_LANES: tuple[str, ...] = ("ho", "rd", "cr")


def ring_spans(
    y: np.ndarray,
    sr: int,
    onset_times: Sequence[float],
    fps: float,
    decay_frac: float = 0.15,
    max_ring_s: float = 3.0,
    min_ring_s: float = 0.05,
) -> list[tuple[float, float]]:
    """Per-onset `(t, dur)` ring spans from the audio's RMS energy envelope.

    For each labeled onset, the ring lasts until the RMS envelope first decays
    below `decay_frac` x its local post-onset peak (or the next onset on the
    same list, or `max_ring_s`). Open hi-hat / cymbal identity lives in this
    tail; the spans become the auxiliary frame-activity target
    (`spans_to_activity`). Heuristic by design: computed on the clean training
    stem where the ring is the dominant energy after the hit."""
    if not len(onset_times):
        return []
    hop = max(1, int(round(sr / fps)))
    n = max(1, 1 + (len(y) - 1) // hop)
    rms = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = y[i * hop : i * hop + hop]
        rms[i] = np.sqrt(np.mean(seg * seg)) if seg.size else 0.0
    spans: list[tuple[float, float]] = []
    times = sorted(float(t) for t in onset_times)
    for j, t in enumerate(times):
        f0 = int(round(t * fps))
        if f0 >= n:
            continue
        peak = float(rms[f0 : min(n, f0 + 4)].max(initial=0.0))
        limit_t = min(t + max_ring_s, times[j + 1] if j + 1 < len(times) else np.inf)
        f_lim = min(n, int(round(limit_t * fps)))
        end = f_lim
        floor = decay_frac * peak
        for f in range(min(n, f0 + 2), f_lim):
            if rms[f] < floor:
                end = f
                break
        spans.append((t, max(min_ring_s, end / fps - t)))
    return spans


def spans_to_activity(
    spans: Sequence[tuple[float, float]], n_frames: int, fps: float
) -> np.ndarray:
    """Render `(t, dur)` spans as a binary (n_frames,) activity curve."""
    out = np.zeros(n_frames, dtype=np.float32)
    for t, dur in spans:
        lo = max(0, int(round(t * fps)))
        hi = min(n_frames, int(round((t + dur) * fps)) + 1)
        if lo < hi:
            out[lo:hi] = 1.0
    return out


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
