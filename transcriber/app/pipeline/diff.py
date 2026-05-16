"""Detect typed issues between the LLM's Jot and the source drum stems.

Each issue is a structured record with a `confidence` so the critic step
can rank-order them, and a `notes` string so the generator step has a
human-readable hint of what to do. Notes reference `(bar, beat)` rather
than seconds when possible, because that's the coordinate space the LLM
emits the DSL in.

We do NOT actually re-render audio. We extract predicted onsets from the
Jot DSL via `jot_extract.py` and compare them directly against the
librosa onsets on the source stems. This is equivalent to (and cheaper
than) rendering and re-detecting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.jot_extract import ExtractedJot, PredictedOnset

log = logging.getLogger(__name__)

ONSET_TOLERANCE_SECONDS = 0.03
STRENGTH_FILTER_RATIO = 0.30  # drop source onsets below 30% of median strength

IssueType = Literal[
    "missing_onset",
    "extra_onset",
    "velocity_mismatch",
    "tempo_mismatch",
    "time_sig_mismatch",
    "structure_refactor",
]


@dataclass
class Issue:
    type: IssueType
    pitch: str
    confidence: float
    notes: str
    time: float | None = None
    expected_velocity: int | None = None
    current_velocity: int | None = None
    expected_bpm: float | None = None
    current_bpm: float | None = None


# ---------- onset diff ----------

def diff_onsets(
    pitch: str,
    predicted: list[PredictedOnset],
    actual_candidates: list[OnsetCandidate],
    structure: BeatStructure | None = None,
) -> list[Issue]:
    """Match predicted vs actual onsets greedily; emit missing/extra issues
    for unmatched onsets on either side.

    `structure` (if provided) is used to annotate issue notes with
    `(bar, beat)` references in addition to the absolute time. The LLM
    fixes its DSL more accurately when it knows which bar to look at.
    """
    if not actual_candidates and not predicted:
        return []

    strengths = [c.strength for c in actual_candidates]
    median_strength = float(np.median(strengths)) if strengths else 1.0

    # Filter analysis noise on the source side
    actual_filtered = [
        c for c in actual_candidates
        if c.strength >= STRENGTH_FILTER_RATIO * median_strength
    ]
    actual_sorted = sorted(actual_filtered, key=lambda c: c.time)
    pred_sorted = sorted(predicted, key=lambda p: p.time)

    used_actual: set[int] = set()
    matched: list[tuple[int, int]] = []

    for pi, p in enumerate(pred_sorted):
        best_ai = -1
        best_dist = ONSET_TOLERANCE_SECONDS
        for ai, c in enumerate(actual_sorted):
            if ai in used_actual:
                continue
            d = abs(p.time - c.time)
            if d < best_dist:
                best_dist = d
                best_ai = ai
        if best_ai >= 0:
            matched.append((pi, best_ai))
            used_actual.add(best_ai)

    matched_pred_idx = {pi for pi, _ in matched}
    issues: list[Issue] = []

    for pi, p in enumerate(pred_sorted):
        if pi in matched_pred_idx:
            continue
        location = _location_string(structure, p.time)
        issues.append(Issue(
            type="extra_onset",
            pitch=pitch,
            time=p.time,
            confidence=0.85,
            notes=(
                f"Jot has a {pitch} hit at {location} ({p.time:.3f}s) with "
                f"no nearby onset in the source stem."
            ),
        ))

    for ai, c in enumerate(actual_sorted):
        if ai in used_actual:
            continue
        # In source but missing from Jot
        strength_ratio = c.strength / max(median_strength, 1e-6)
        confidence = float(min(0.95, 0.60 + 0.35 * min(strength_ratio, 1.5)))
        location = (
            f"bar {c.bar}, beat {c.beat_in_bar:.3f}"
            if c.bar >= 0
            else f"{c.time:.3f}s"
        )
        issues.append(Issue(
            type="missing_onset",
            pitch=pitch,
            time=c.time,
            confidence=confidence,
            notes=(
                f"Source has a {pitch} onset at {location} "
                f"(strength={c.strength:.2f}, {strength_ratio:.0%} of median) "
                f"but Jot omits it."
            ),
        ))

    return issues


def _location_string(structure: BeatStructure | None, t: float) -> str:
    """Return a `(bar, beat)` reference for time `t` if possible."""
    if structure is None:
        return f"{t:.3f}s"
    pos = structure.position(t)
    if pos is None:
        return f"{t:.3f}s"
    bar, beat = pos
    return f"bar {bar}, beat {beat:.3f}"


# ---------- velocity diff ----------

def diff_velocities(
    pitch: str,
    predicted: list[PredictedOnset],
    audio: np.ndarray,
    sr: int,
    structure: BeatStructure | None = None,
) -> list[Issue]:
    """Compare predicted velocity vs RMS at each matched onset.

    Uses a 50ms window around the predicted time. The 'median predicted
    velocity at median RMS' is the calibration anchor; everything else is
    scaled relative to that. This works for any audio level because we're
    comparing the predicted dynamic *contour*, not absolute amplitude.
    """
    if not predicted or audio.size == 0:
        return []

    rms_at_each = np.array([_rms_at(audio, sr, p.time) for p in predicted])
    velocity_each = np.array([p.velocity for p in predicted], dtype=float)
    if not np.any(rms_at_each > 1e-6):
        return []

    median_rms = float(np.median(rms_at_each))
    median_vel = float(np.median(velocity_each))
    if median_rms < 1e-6 or median_vel <= 0:
        return []

    issues: list[Issue] = []
    for p, rms in zip(predicted, rms_at_each):
        if rms < 1e-6:
            continue
        actual_velocity = int(round(median_vel * rms / median_rms))
        actual_velocity = max(1, min(127, actual_velocity))
        gap = actual_velocity - p.velocity
        if abs(gap) < 18:
            continue
        confidence = float(min(0.85, abs(gap) / 80.0 + 0.25))
        direction = "louder" if gap > 0 else "softer"
        location = _location_string(structure, p.time)
        issues.append(Issue(
            type="velocity_mismatch",
            pitch=pitch,
            time=p.time,
            current_velocity=p.velocity,
            expected_velocity=actual_velocity,
            confidence=confidence,
            notes=(
                f"{pitch} at {location}: Jot velocity {p.velocity}, audio "
                f"suggests ~{actual_velocity} (much {direction}). Consider "
                f"adding :a / :g or adjusting vol metadata."
            ),
        ))

    return issues


def _rms_at(audio: np.ndarray, sr: int, t: float, win: float = 0.05) -> float:
    a = int(max(0.0, t - win / 2) * sr)
    b = int(min(audio.size / sr, t + win / 2) * sr)
    if b <= a:
        return 0.0
    chunk = audio[a:b].astype(np.float64)
    return float(np.sqrt(np.mean(chunk * chunk) + 1e-12))


# ---------- macro: tempo / time sig ----------

def diff_tempo(extracted: ExtractedJot, actual_tempo: float) -> list[Issue]:
    issues: list[Issue] = []
    if actual_tempo <= 0:
        return issues
    delta = abs(extracted.bpm - actual_tempo)
    if delta < 2.0:
        return issues

    # Octave errors (half / double tempo) are common in beat trackers; flag
    # but downweight if it looks like a 2x / 0.5x ambiguity.
    is_octave = (
        abs(extracted.bpm - 2 * actual_tempo) < 2.0
        or abs(2 * extracted.bpm - actual_tempo) < 2.0
    )
    confidence = float(min(0.95, delta / 10.0))
    if is_octave:
        confidence *= 0.7
        notes_suffix = " (this looks like a half/double-tempo ambiguity)"
    else:
        notes_suffix = ""
    issues.append(Issue(
        type="tempo_mismatch",
        pitch="",
        current_bpm=extracted.bpm,
        expected_bpm=actual_tempo,
        confidence=confidence,
        notes=(
            f"Jot global bpm is {extracted.bpm:.1f}, beat tracker on source "
            f"audio says {actual_tempo:.1f}{notes_suffix}."
        ),
    ))
    return issues


# ---------- structure: pure refactor hint ----------

def structure_refactor_hint() -> list[Issue]:
    """The structure pass is audio-independent: just ask the LLM to factor
    out repeating bar-level patterns into [Name=(...)] definitions.
    """
    return [Issue(
        type="structure_refactor",
        pitch="",
        confidence=1.0,
        notes=(
            "Scan the Jot for bar-level repetition. Where two or more bars "
            "are identical or near-identical, define them as a named "
            "pattern (`[Name=(...)]`) and replace later occurrences with "
            "`[Name]`. Don't change the underlying notes - only the "
            "representation. If no useful factoring exists, leave the Jot "
            "as-is."
        ),
    )]
