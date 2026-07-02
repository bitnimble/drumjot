"""Per-bar feel classification + LLM-friendly summary rendering.

`detect_feel_for_bars` scores each bar's intra-beat onset fractions against the
`_FEEL_GRIDS` hypotheses (straight16/straight8/triplet/shuffle) and tags the
bar's `feel`. `candidates_with_beat_positions` / `summarize_bar_for_prompt`
render onsets + bars into the compact beat-relative shapes the prompt template
embeds. Depends only on `beats_types`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.pipeline.beats_types import BarInfo, BeatStructure, Feel

# Reference fraction sets for each feel. We score a bar against each
# hypothesis by summing the minimum distance from each onset's intra-beat
# fraction to the nearest reference fraction. Whichever set has the
# smallest total residual wins.
_FEEL_GRIDS: dict[Feel, list[float]] = {
    "straight16": [0.0, 0.25, 0.5, 0.75],
    "straight8": [0.0, 0.5],
    "triplet": [0.0, 1.0 / 3.0, 2.0 / 3.0],
    "shuffle": [0.0, 2.0 / 3.0],
}


def detect_feel_for_bars(
    structure: BeatStructure, onset_times: list[float]
) -> None:
    """Annotate each `BarInfo` in-place with a detected `feel`.

    `onset_times` is a flat list of onset times across all instruments
    in seconds. Per-instrument granularity isn't needed for feel detection
    - we just need to see where the strikes fall within each beat.
    """
    if not structure.bars or not onset_times:
        return
    # Bucket onsets per bar
    onset_arr = np.asarray(onset_times)
    for bar in structure.bars:
        in_bar = onset_arr[(onset_arr >= bar.start_time) & (onset_arr < bar.end_time)]
        if in_bar.size < 4:
            # Not enough hits to classify; mark sparse and skip.
            bar.feel = "sparse"
            continue
        # Convert onset times to intra-beat fractions in [0, 1)
        fractions: list[float] = []
        for t in in_bar:
            pos = _intra_beat_fraction(bar, float(t))
            if pos is None:
                continue
            fractions.append(pos)
        if len(fractions) < 4:
            bar.feel = "sparse"
            continue
        bar.feel = _score_feel(fractions)


def _intra_beat_fraction(bar: BarInfo, t: float) -> float | None:
    # Find enclosing beat in this bar.
    for i, beat in enumerate(bar.beats):
        next_t = (
            bar.beats[i + 1].time if i + 1 < len(bar.beats) else bar.end_time
        )
        if beat.time <= t < next_t:
            gap = max(next_t - beat.time, 1e-6)
            return (t - beat.time) / gap
    return None


def _score_feel(fractions: list[float]) -> Feel:
    """Pick the feel whose reference fractions best match the data."""
    best: tuple[Feel, float] = ("mixed", float("inf"))
    for label, grid in _FEEL_GRIDS.items():
        total = 0.0
        for f in fractions:
            # closest distance considering wrap-around at 1.0
            total += min(
                min(abs(f - g), 1.0 - abs(f - g)) for g in grid
            )
        if total < best[1]:
            best = (label, total)
    # If even the best is too noisy, call it mixed (LLM should treat as
    # a hint to fall back to a free grid).
    avg_residual = best[1] / max(len(fractions), 1)
    if avg_residual > 0.10:
        return "mixed"
    return best[0]


# ---------- LLM-friendly summary ----------

@dataclass
class CandidateOnset:
    """One onset attached to its beat-relative position."""

    pitch: str
    time: float
    strength: float
    bar: int
    beat_in_bar: float  # 1-indexed, uniform across the bar; see BeatStructure.position


def candidates_with_beat_positions(
    onsets_by_pitch: dict[str, list[tuple[float, float]]],
    structure: BeatStructure,
) -> dict[str, list[CandidateOnset]]:
    """For each (time, strength) per pitch, compute (bar, beat_in_bar).

    Onsets outside the tracked range are dropped.
    """
    out: dict[str, list[CandidateOnset]] = {}
    for pitch, items in onsets_by_pitch.items():
        kept: list[CandidateOnset] = []
        for time, strength in items:
            pos = structure.position(time)
            if pos is None:
                continue
            bar, beat = pos
            kept.append(
                CandidateOnset(
                    pitch=pitch,
                    time=float(time),
                    strength=float(strength),
                    bar=int(bar),
                    beat_in_bar=float(beat),
                )
            )
        if kept:
            out[pitch] = kept
    return out


def summarize_bar_for_prompt(bar: BarInfo) -> dict[str, object]:
    """Render `BarInfo` as a compact dict the prompt template can embed.

    Stable key order so the prompt is reproducible across runs.
    """
    return {
        "bar": bar.index,
        "time_signature": f"{bar.time_signature[0]}/{bar.time_signature[1]}",
        "tempo_bpm": round(bar.tempo_bpm, 1),
        "feel": bar.feel,
        "start_time": round(bar.start_time, 3),
    }
