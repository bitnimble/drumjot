"""Onset-strength envelope of a stem, sampled per slot by the quantise stage.

The quantise passes otherwise reason only about onset *times* and the beat
grid; they never re-consult the audio. `OnsetEnvelope` carries a stem's
onset-strength envelope (the same signal `adtof_onsets._refine_peak_times_audio`
uses, but kept around) so the envelope re-snap in `quantise.py` can pull a
note onto the slot whose time-bin actually holds the transient, fixing
placements where detection locked the onset time onto the wrong (early)
envelope max inside its narrow refine window.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ~1.45 ms/frame at 44.1 kHz, matches the detection-time refine resolution
# (`adtof_onsets._refine_peak_times_audio`), so slot-bin sampling is
# sample-honest rather than blurred by a coarse hop.
_HOP = 64


@dataclass
class OnsetEnvelope:
    """A stem's onset-strength envelope plus a high-percentile reference.

    `frame_times` is ascending seconds in the same coordinate as onset
    `time` / the aligned beat grid; `env` is the per-frame onset strength;
    `ref` is a high percentile of `env` used as the "this is a real
    transient" floor by consumers.
    """

    frame_times: np.ndarray
    env: np.ndarray
    ref: float

    def peak_in(self, t_lo: float, t_hi: float) -> float:
        """Largest envelope value in the half-open time window [t_lo, t_hi).

        Returns 0.0 for an empty envelope or an inverted window. When the
        window is narrower than a frame it falls back to the nearest
        sample so a tiny slot still yields its local energy.
        """
        if self.env.size == 0 or t_hi <= t_lo:
            return 0.0
        lo = int(np.searchsorted(self.frame_times, t_lo, side="left"))
        hi = int(np.searchsorted(self.frame_times, t_hi, side="right"))
        if lo >= hi:
            idx = min(max(lo, 0), self.env.size - 1)
            return float(self.env[idx])
        return float(self.env[lo:hi].max())


def compute_onset_envelope(
    audio_path: Path, *, hop_length: int = _HOP
) -> OnsetEnvelope | None:
    """Onset-strength envelope of `audio_path`, or None if it can't be read.

    A load / analysis failure logs and returns None so the caller degrades
    cleanly (the envelope re-snap simply skips that lane) rather than
    aborting the request.
    """
    import librosa

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as exc:
        log.warning("envelope: librosa.load(%s) failed (%s)", audio_path, exc)
        return None
    if y.size == 0:
        return None
    env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length
    ).astype(np.float64)
    if env.size == 0 or not np.any(env):
        return None
    frame_times = librosa.frames_to_time(
        np.arange(env.size), sr=sr, hop_length=hop_length
    )
    ref = float(np.percentile(env, 99.0))
    return OnsetEnvelope(frame_times=frame_times, env=env, ref=ref)
