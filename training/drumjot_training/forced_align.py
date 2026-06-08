"""Per-note forced alignment against an onset-strength envelope (spec §3.0/§3.3).

Snaps each chart onset onto the nearest local maximum of the audio's
onset-strength envelope within a +/-window, the per-note Tier-2 step the
scoring v1 deferred. The reference is the same librosa onset-strength
envelope the transcriber refines against (`adtof_onsets._refine_peak_times_audio`
/ `envelope.py`): verified-accurate AND license-clean, and non-circular
(we align to the audio signal, not to a detector's decisions).

A support gate guards the failure mode the spec calls out: a chart hit with
no real transient nearby would otherwise snap to a phantom max. Onsets whose
window peak is below `support_floor` are flagged unsupported (kept in place);
the cleaning stage discards or quarantines those.

Pure numpy here (host-testable); building the envelope from audio is a thin
lazy-librosa wrapper.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def align_lane(
    onsets_sec: Sequence[float],
    env: np.ndarray,
    env_fps: float,
    window_s: float,
    support_floor: float,
) -> list[tuple[float, bool]]:
    """Snap each onset to its window's envelope peak; flag support.

    Returns one `(time, supported)` per input onset. Supported onsets are
    moved to the peak's time; unsupported ones (no peak >= `support_floor`,
    or out of range) keep their original time and `supported=False`.
    """
    env = np.asarray(env)
    n = env.size
    half = max(1, round(window_s * env_fps))
    out: list[tuple[float, bool]] = []
    for t in onsets_sec:
        center = int(round(float(t) * env_fps))
        lo = max(0, center - half)
        hi = min(n, center + half + 1)
        if lo >= hi:
            out.append((float(t), False))
            continue
        idx = lo + int(np.argmax(env[lo:hi]))
        if float(env[idx]) >= support_floor:
            out.append((idx / float(env_fps), True))
        else:
            out.append((float(t), False))
    return out


def align_chart(
    onsets_by_lane: Mapping[str, Sequence[float]],
    env: np.ndarray,
    env_fps: float,
    window_s: float,
    support_floor: float,
) -> dict[str, list[tuple[float, bool]]]:
    """`align_lane` for every lane. (In practice each lane aligns against its
    own stem's envelope; callers pass the right `env` per lane.)"""
    return {
        lane: align_lane(onsets, env, env_fps, window_s, support_floor)
        for lane, onsets in onsets_by_lane.items()
    }


def onset_envelope(audio_path, hop_length: int = 64, max_seconds: float | None = None):
    """(env, fps) onset-strength envelope of `audio_path` (lazy librosa).

    hop=64 @ 44.1 kHz -> ~1.45 ms/frame, matching the transcriber's
    sample-honest refine resolution. `max_seconds` caps the load (the model
    only predicts within the encoded window). Not unit-tested here (IO);
    verified in the sandbox.
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=max_seconds)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    return env.astype(np.float64), float(sr) / float(hop_length)
