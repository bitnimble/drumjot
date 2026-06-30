"""Deterministic align-onset generators for the beat-tracker A/B.

`align_beats_to_onsets` consumes `(time, strength)` tuples and snaps the
grid to the *strongest* onset within +-50 ms of each beat. These builders
turn a clip's ground-truth per-lane MIDI onsets into such a list under
three regimes:

- `gt`         : the true onsets, undegraded (a "perfect aligner" ceiling).
- `synthetic`  : true onsets degraded to emulate an imperfect CPU-only
                 detector (recall loss + uncorrelated FPs + stem bleed).

Everything is seeded off a stable per-clip hash, so a re-run is
bit-identical. Both trackers in a single A/B run receive the *same* list
per clip, so the paired delta stays fair regardless of regime.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .beat_gt import LANE_GROUPS, LaneOnset

# Recall: drop this fraction of true onsets -> recall ~= 0.85.
DROP_FRAC = 0.15
# Precision, uncorrelated: add this fraction (of kept count) of
# uniform-random spurious onsets.
UNIFORM_FP_FRAC = 0.10
# Precision, correlated (stem bleed): this fraction of onsets get a ghost
# copy transferred from a spectrally-similar lane.
BLEED_FRAC = 0.12
BLEED_JITTER_SEC = 0.015          # ghost lands within +-this of the source
BLEED_VEL_SCALE = (0.30, 0.60)    # ghost velocity = source * U(this)
# A uniform-random FP is rejected if it lands within this of any real
# onset (it would otherwise be a free true positive, not a confounder).
MIN_FP_GAP_SEC = 0.030


def stable_seed(track_id: str) -> int:
    digest = hashlib.sha256(track_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _lane_to_group(lane: str) -> frozenset[str] | None:
    for members in LANE_GROUPS.values():
        if lane in members:
            return members
    return None


def _strength(velocity: float) -> float:
    return float(velocity) / 127.0


def gt_align_onsets(onsets: list[LaneOnset]) -> list[tuple[float, float]]:
    return sorted((o.time, _strength(o.velocity)) for o in onsets)


def synthesize_align_onsets(
    onsets: list[LaneOnset],
    seed: int,
    duration: float,
) -> list[tuple[float, float]]:
    """Degrade true per-lane onsets into a realistic detector's output."""
    if not onsets:
        return []
    rng = np.random.default_rng(seed)
    true_times = np.array([o.time for o in onsets], dtype=np.float64)

    # 1. Recall: drop a fraction of true onsets.
    keep_mask = rng.random(len(onsets)) >= DROP_FRAC
    kept = [o for o, k in zip(onsets, keep_mask, strict=True) if k]
    if not kept:
        kept = [onsets[0]]
    out: list[tuple[float, float]] = [(o.time, _strength(o.velocity)) for o in kept]

    kept_vels = np.array([o.velocity for o in kept], dtype=np.float64)

    # 2. Uncorrelated FPs: uniform-random times not coinciding with a real
    #    onset, velocities resampled from the kept distribution.
    n_fp = int(round(UNIFORM_FP_FRAC * len(kept)))
    added = 0
    attempts = 0
    while added < n_fp and attempts < n_fp * 20:
        attempts += 1
        t = float(rng.uniform(0.0, max(duration, true_times[-1])))
        nearest = float(np.min(np.abs(true_times - t)))
        if nearest < MIN_FP_GAP_SEC:
            continue
        vel = float(rng.choice(kept_vels))
        out.append((t, _strength(vel)))
        added += 1

    # 3. Stem bleed: correlated FPs. A fraction of onsets leak into a
    #    sibling lane in the same spectral group as a jittered, attenuated
    #    ghost. Pooled, this is a near-duplicate that can outrank the true
    #    transient as the "strongest nearby onset" and bias the offset.
    bleed_mask = rng.random(len(onsets)) < BLEED_FRAC
    for o, leaks in zip(onsets, bleed_mask, strict=True):
        if not leaks or _lane_to_group(o.lane) is None:
            continue
        jitter = float(rng.uniform(-BLEED_JITTER_SEC, BLEED_JITTER_SEC))
        scale = float(rng.uniform(*BLEED_VEL_SCALE))
        out.append((max(0.0, o.time + jitter), _strength(o.velocity * scale)))

    out.sort()
    return out
