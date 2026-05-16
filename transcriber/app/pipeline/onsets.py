"""High-recall onset detection per per-instrument drum stem.

We deliberately tune `librosa.onset.onset_detect` toward over-triggering so
the LLM downstream is the one deciding which candidates are real hits.
`pipeline/beats.py` then attaches each onset to a `(bar, beat_in_bar)`
position using madmom's beat structure.
"""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from app.config import settings
from app.models import OnsetCandidate

log = logging.getLogger(__name__)


def detect_onsets(audio_path: Path, sample_rate: int = 44100) -> list[OnsetCandidate]:
    """Run a high-recall onset detector and return a list of candidates.

    Returns candidates with absolute onset time (seconds from start) and a
    relative strength (the onset detection function value at that time).
    `bar` / `beat_in_bar` are filled in by the caller using `BeatStructure`.
    """
    audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if audio.size == 0:
        return []

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
    times = librosa.times_like(onset_env, sr=sr, hop_length=512)

    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        units="time",
        backtrack=True,
        pre_max=settings.onset_pre_max,
        post_max=settings.onset_post_max,
        pre_avg=settings.onset_pre_avg,
        post_avg=settings.onset_post_avg,
        delta=settings.onset_delta,
        wait=settings.onset_wait,
    )

    candidates: list[OnsetCandidate] = []
    for t in onset_times:
        idx = int(np.argmin(np.abs(times - t)))
        strength = float(onset_env[idx])
        candidates.append(
            OnsetCandidate(
                time=float(t),
                strength=strength,
                bar=-1,
                beat_in_bar=-1.0,
            )
        )
    log.info(
        "Detected %d onset candidates in %s (median strength=%.2f)",
        len(candidates),
        audio_path.name,
        float(np.median([c.strength for c in candidates])) if candidates else 0.0,
    )
    return candidates


def attach_beat_positions(
    candidates: list[OnsetCandidate],
    structure,  # BeatStructure (not imported to avoid cycle in typing)
) -> list[OnsetCandidate]:
    """Annotate each candidate with `(bar, beat_in_bar)` using `structure`.

    Candidates whose timestamps fall outside the tracked range are kept
    with `bar=-1, beat_in_bar=-1.0` and downstream code should treat
    them as "out of song" / drop.
    """
    out: list[OnsetCandidate] = []
    for c in candidates:
        pos = structure.position(c.time)
        if pos is None:
            out.append(
                OnsetCandidate(time=c.time, strength=c.strength, bar=-1, beat_in_bar=-1.0)
            )
            continue
        bar, beat = pos
        out.append(
            OnsetCandidate(
                time=c.time, strength=c.strength, bar=int(bar), beat_in_bar=float(beat)
            )
        )
    return out
