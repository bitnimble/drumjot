"""Beat tracking + downbeat detection + per-bar feel analysis.

This module replaces the older `tempo.py` + `quantize.py` pair. Rather
than assuming a constant tempo + fixed grid (1/16 by default), we use
madmom's RNN+DBN downbeat tracker to extract per-beat anchors from the
audio. Everything downstream then works in **beat-relative** time -
that is, "1/3 of the way through beat 2 of bar 3" rather than
"slot 47 of a 1/16 grid".

Why it matters:

- Tempo changes are handled naturally: each beat has its own absolute
  time, so a song that accelerates or hits a click drop still maps
  onsets onto the right beats.
- Time-signature changes are detected from the downbeat-classifier
  output (gaps of 4 beats between downbeats -> 4/4; 7 beats -> 7/4 or
  7/8 depending on tempo).
- Triplet and swing feel become a property of intra-beat fractions and
  are encoded as the bar's `feel` field; the LLM can then emit
  `(...)_N` groups vs straight grids per bar.
- The grid is implicit and per-bar, not a global constant - the LLM
  decides how to position each onset in DSL space based on the bar's
  feel and the onset's beat fraction.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

# Common beat-counts the downbeat tracker should consider. Covers all
# popular-music time signatures (and a few odd-meter ones).
BEATS_PER_BAR_CANDIDATES = [2, 3, 4, 5, 6, 7]

# Feel classification labels emitted on each `BarInfo`.
Feel = Literal["straight16", "straight8", "triplet", "shuffle", "sparse", "mixed"]


@dataclass
class BeatTick:
    """One beat detected by the tracker."""

    time: float
    beat_in_bar: int  # 1-indexed; 1 = downbeat
    bar_index: int    # 0-indexed bar this beat belongs to


@dataclass
class BarInfo:
    """A single bar after downbeat classification."""

    index: int                          # 0-indexed
    start_time: float
    end_time: float
    beats: list[BeatTick]               # at least 1
    time_signature: tuple[int, int]     # (count, unit) - unit is always 4 by default
    tempo_bpm: float                    # avg over this bar's beats
    feel: Feel = "straight16"           # filled in once onsets are known


@dataclass
class BeatStructure:
    """Sequence of beats + bar groupings + global summary fields."""

    beats: list[BeatTick] = field(default_factory=list)
    bars: list[BarInfo] = field(default_factory=list)
    initial_tempo: float = 120.0
    initial_time_signature: tuple[int, int] = (4, 4)
    has_tempo_changes: bool = False
    has_time_sig_changes: bool = False

    def position(self, t: float) -> tuple[int, float] | None:
        """Map an absolute time `t` (seconds) to `(bar_index, beat_in_bar)`.

        The returned `beat_in_bar` is a float - the integer part is the
        beat index (1-indexed), the fractional part is how far into that
        beat the onset sits. E.g. `(3, 2.333)` = "bar 3, 1/3 into beat 2".

        Returns None if `t` falls outside the tracked range (e.g. silence
        before the first beat, or after the last). Callers should treat
        that as "drop this onset" - it's noise or chart padding.
        """
        if not self.beats:
            return None
        # Find the beat tick that t belongs to (the one whose interval
        # [tick.time, next_tick.time) contains t).
        first_t = self.beats[0].time
        last_t = self.beats[-1].time
        if t < first_t:
            # Onsets before the first beat get attributed to bar 0, fractional
            # *before* beat 1 of that bar. We extrapolate using the first beat
            # gap as a tempo proxy.
            if len(self.beats) < 2:
                return None
            gap = self.beats[1].time - self.beats[0].time
            if gap <= 0:
                return None
            frac = (t - first_t) / gap  # negative
            return (self.beats[0].bar_index, 1.0 + frac)
        if t > last_t:
            if len(self.beats) < 2:
                return None
            gap = self.beats[-1].time - self.beats[-2].time
            if gap <= 0:
                return None
            frac = (t - last_t) / gap
            return (
                self.beats[-1].bar_index,
                float(self.beats[-1].beat_in_bar) + frac,
            )

        # Binary search for the enclosing beat
        lo, hi = 0, len(self.beats) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.beats[mid].time <= t:
                lo = mid
            else:
                hi = mid - 1
        bt = self.beats[lo]
        next_t = self.beats[lo + 1].time if lo + 1 < len(self.beats) else None
        if next_t is None:
            return (bt.bar_index, float(bt.beat_in_bar))
        frac = (t - bt.time) / max(next_t - bt.time, 1e-6)
        return (bt.bar_index, float(bt.beat_in_bar) + frac)


# ---------- Detection ----------

def analyze_beats(audio_path: Path) -> BeatStructure:
    """Run beat + downbeat detection on the audio at `audio_path`.

    Uses madmom (RNN + DBN) for the actual ML; falls back to a
    librosa-based heuristic if madmom isn't importable, so the rest of
    the pipeline degrades gracefully rather than failing outright.
    """
    try:
        return _madmom_beats(audio_path)
    except Exception as exc:
        log.warning(
            "madmom beat tracking failed (%s); falling back to librosa "
            "(no downbeat classification - assuming 4/4 throughout)",
            exc,
        )
        return _librosa_fallback(audio_path)


def _madmom_beats(audio_path: Path) -> BeatStructure:
    # Imports kept local so a madmom install issue doesn't block the
    # rest of the service from starting up.
    from madmom.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )

    log.info("madmom: extracting downbeat activations from %s", audio_path.name)
    activations = RNNDownBeatProcessor()(str(audio_path))
    tracker = DBNDownBeatTrackingProcessor(
        beats_per_bar=BEATS_PER_BAR_CANDIDATES,
        fps=100,
    )
    raw = tracker(activations)  # Nx2 array: (time, beat_pos_in_bar)
    if raw.size == 0:
        log.warning("madmom returned no beats")
        return BeatStructure()

    return _from_madmom_raw(raw)


def _from_madmom_raw(raw: np.ndarray) -> BeatStructure:
    """Convert madmom's Nx2 output into our typed BeatStructure."""
    beats: list[BeatTick] = []
    bars: list[BarInfo] = []
    current_bar_beats: list[BeatTick] = []
    current_bar_index = -1

    for row in raw:
        time = float(row[0])
        beat_pos = int(round(float(row[1])))  # 1-indexed in bar
        if beat_pos == 1:
            # Close out the prior bar (if any)
            if current_bar_beats:
                bars.append(_finalize_bar(current_bar_index, current_bar_beats))
            current_bar_index += 1
            current_bar_beats = []
        if current_bar_index < 0:
            # Beats before the first detected downbeat - park them in bar 0.
            current_bar_index = 0
        tick = BeatTick(
            time=time, beat_in_bar=max(1, beat_pos), bar_index=current_bar_index
        )
        beats.append(tick)
        current_bar_beats.append(tick)

    if current_bar_beats:
        bars.append(_finalize_bar(current_bar_index, current_bar_beats))

    return _summarize(beats, bars)


def _finalize_bar(index: int, beats: list[BeatTick]) -> BarInfo:
    """Build a BarInfo from the beats falling inside it."""
    if not beats:
        return BarInfo(
            index=index,
            start_time=0.0,
            end_time=0.0,
            beats=[],
            time_signature=(4, 4),
            tempo_bpm=120.0,
        )
    start = beats[0].time
    # `end_time` is the start of the next bar; approximate with the last
    # beat's time + average gap. Will be overwritten by `_summarize` once
    # we know the next bar's actual start.
    if len(beats) >= 2:
        gap = beats[-1].time - beats[-2].time
    else:
        gap = 60.0 / 120.0
    end = beats[-1].time + max(gap, 0.0)
    count = len(beats)
    # `unit` defaults to 4 (quarter-note beats) - the spec form for most
    # popular meters. Compound meters (6/8, 12/8) are perceived as fewer
    # but longer beats by the tracker, so we don't naively divide.
    time_sig = (count, 4)
    # Average tempo over this bar's beats
    if len(beats) >= 2:
        gaps = np.diff([b.time for b in beats])
        tempo_bpm = float(60.0 / np.mean(gaps))
    else:
        tempo_bpm = 120.0
    return BarInfo(
        index=index,
        start_time=start,
        end_time=end,
        beats=list(beats),
        time_signature=time_sig,
        tempo_bpm=tempo_bpm,
    )


def _summarize(beats: list[BeatTick], bars: list[BarInfo]) -> BeatStructure:
    """Fill end_time of each bar from the next bar's start, and compute
    global summary fields (initial tempo, change flags)."""
    for i in range(len(bars) - 1):
        bars[i].end_time = bars[i + 1].start_time
    if bars:
        initial_tempo = bars[0].tempo_bpm
        initial_ts = bars[0].time_signature
    else:
        initial_tempo = 120.0
        initial_ts = (4, 4)

    tempo_set = {round(b.tempo_bpm, 1) for b in bars}
    sig_set = {b.time_signature for b in bars}
    return BeatStructure(
        beats=beats,
        bars=bars,
        initial_tempo=float(initial_tempo),
        initial_time_signature=initial_ts,
        has_tempo_changes=len(tempo_set) > 1,
        has_time_sig_changes=len(sig_set) > 1,
    )


def _librosa_fallback(audio_path: Path) -> BeatStructure:
    """Plain librosa beat tracking, no downbeat classification.

    Used only when madmom is unavailable. Produces a 4/4 structure with
    constant time signature; tempo follows whatever librosa returned.
    """
    import librosa

    log.info("librosa fallback: tracking beats on %s", audio_path.name)
    audio, sr = librosa.load(str(audio_path), sr=44100, mono=True)
    tempo, beat_times = librosa.beat.beat_track(y=audio, sr=sr, units="time", trim=False)
    tempo = float(np.atleast_1d(tempo)[0]) if np.isfinite(tempo) else 120.0
    if not isinstance(beat_times, np.ndarray) or beat_times.size == 0:
        return BeatStructure(initial_tempo=tempo)

    # Group every 4 beats into a 4/4 bar.
    beats: list[BeatTick] = []
    bars: list[BarInfo] = []
    cur_bar_beats: list[BeatTick] = []
    bar_index = 0
    for i, t in enumerate(beat_times):
        beat_in_bar = (i % 4) + 1
        if beat_in_bar == 1 and cur_bar_beats:
            bars.append(_finalize_bar(bar_index, cur_bar_beats))
            bar_index += 1
            cur_bar_beats = []
        tick = BeatTick(time=float(t), beat_in_bar=beat_in_bar, bar_index=bar_index)
        beats.append(tick)
        cur_bar_beats.append(tick)
    if cur_bar_beats:
        bars.append(_finalize_bar(bar_index, cur_bar_beats))

    return _summarize(beats, bars)


# ---------- Feel detection ----------

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
    beat_in_bar: float  # 1-indexed; integer = on the beat, fractional = inside


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
