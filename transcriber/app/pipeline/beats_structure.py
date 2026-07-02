"""Structure assembly + robust global summary.

Turns a raw `(time, beat_pos_in_bar)` grid into a typed `BeatStructure`
(`_raw_to_structure` + `_finalize_bar` + `_choose_time_signature`) and derives
the song-level summary fields (modal meter, robust initial tempo, change flags)
via `_summarize`. `_rebuild_bar_fields` refreshes the per-bar + summary fields
after any pass re-times the beats in place.

These are the shared substrate the meter, tempo, alignment and detection
modules all build on; it depends only on `beats_types`.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from app.pipeline.beats_types import BarInfo, BeatStructure, BeatTick


def _raw_to_structure(raw: np.ndarray) -> BeatStructure:
    """Convert an Nx2 `(time, beat_pos_in_bar)` array into a BeatStructure."""
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
    gap = beats[-1].time - beats[-2].time if len(beats) >= 2 else 60.0 / 120.0
    end = beats[-1].time + max(gap, 0.0)
    count = len(beats)
    # Average tempo over this bar's beats
    if len(beats) >= 2:
        gaps = np.diff([b.time for b in beats])
        tempo_bpm = float(60.0 / np.mean(gaps))
    else:
        tempo_bpm = 120.0
    time_sig = _choose_time_signature(count, tempo_bpm)
    return BarInfo(
        index=index,
        start_time=start,
        end_time=end,
        beats=list(beats),
        time_signature=time_sig,
        tempo_bpm=tempo_bpm,
    )


def _choose_time_signature(count: int, tempo_bpm: float) -> tuple[int, int]:
    """Pick a `(numerator, denominator)` for a bar with `count` beats.

    The tracker only reports a count of beats per bar; it doesn't tell us
    whether they're quarter notes (simple meter) or dotted quarters
    (compound meter). For 4-, 5- and 7-beat bars 'quarter notes' is
    nearly always right in popular music. For 6 beats it's genuinely
    ambiguous: a slow rock waltz is 6/4, but a fast jazz waltz or
    Irish jig is 6/8 with the same beat count.

    ASSUMPTION (refine when we have ground-truth data): treat 6-beat
    bars as 6/8 whenever the detected tempo is on the fast side
    (>= 100 BPM in the beat-tracker's "quarter-note" sense). Anything
    slower stays at 6/4. 12-beat bars get the same split for 12/8 vs
    12/4.

    Returns `(count, unit)` where `unit` is the note value of the
    denominator (4 = quarter, 8 = eighth).
    """
    if count == 6 and tempo_bpm >= 100.0:
        return (6, 8)
    if count == 12 and tempo_bpm >= 100.0:
        return (12, 8)
    return (count, 4)


def _reference_bars(bars: list[BarInfo]) -> list[BarInfo]:
    """Bars used for global summary: skip bar 0 (likely anacrusis/pickup)."""
    return bars[1:] if len(bars) >= 2 else bars


def _modal_time_signature(bars: list[BarInfo]) -> tuple[int, int]:
    """The song's dominant meter (most common per-bar time signature).

    Beat This!'s per-bar downbeat placement is jittery on some songs (a 4/4
    track can show scattered 2/4/8-beat bars even when the median is a clean
    4), so the song-level meter must be the modal bar, not any single bar; reading one bar (even bar 1, past the anacrusis) routinely mislabels the
    whole song."""
    return Counter(b.time_signature for b in bars).most_common(1)[0][0]


def _robust_initial_tempo(bars: list[BarInfo]) -> float:
    """Median tempo of the first few bars: start-representative but robust to
    a glitchy bar whose mis-sized duration yields a wild per-bar BPM."""
    head = bars[: min(8, len(bars))]
    return float(np.median([b.tempo_bpm for b in head])) if head else 120.0


def _has_sustained_meter_change(bars: list[BarInfo], modal: tuple[int, int],
                                min_run: int = 2) -> bool:
    """True iff some meter ≠ `modal` holds for ≥`min_run` consecutive bars.
    A single off-meter bar is per-bar detection noise, not a real change."""
    run = 0
    for b in bars:
        run = run + 1 if b.time_signature != modal else 0
        if run >= min_run:
            return True
    return False


def _summarize(beats: list[BeatTick], bars: list[BarInfo]) -> BeatStructure:
    """Fill end_time of each bar from the next bar's start, and compute
    global summary fields (initial tempo, meter, change flags).

    The summary is **modal/median over bars**, not read off a single bar:
    Beat This! places downbeats with enough per-bar jitter that any one bar
    (bar 0 anacrusis OR bar 1) routinely mislabels the song's meter/tempo.
    The per-bar `time_signature`/`tempo_bpm` are left as detected (they drive
    onset→bar mapping locally); only the song-level fields are robustified.
    """
    for i in range(len(bars) - 1):
        bars[i].end_time = bars[i + 1].start_time
    reference_bars = _reference_bars(bars)
    if reference_bars:
        initial_ts = _modal_time_signature(reference_bars)
        initial_tempo = _robust_initial_tempo(reference_bars)
        has_time_sig_changes = _has_sustained_meter_change(reference_bars, initial_ts)
    else:
        initial_tempo = 120.0
        initial_ts = (4, 4)
        has_time_sig_changes = False

    tempo_set = {round(b.tempo_bpm, 1) for b in reference_bars}
    return BeatStructure(
        beats=beats,
        bars=bars,
        initial_tempo=float(initial_tempo),
        initial_time_signature=initial_ts,
        has_tempo_changes=len(tempo_set) > 1,
        has_time_sig_changes=has_time_sig_changes,
    )


def _rebuild_bar_fields(structure: BeatStructure) -> None:
    """Recompute per-bar start/end/tempo + global summary after beats
    have been re-timed in place."""
    for i, bar in enumerate(structure.bars):
        if not bar.beats:
            continue
        bar.start_time = bar.beats[0].time
        if i + 1 < len(structure.bars) and structure.bars[i + 1].beats:
            bar.end_time = structure.bars[i + 1].beats[0].time
        elif len(bar.beats) >= 2:
            gap = bar.beats[-1].time - bar.beats[-2].time
            bar.end_time = bar.beats[-1].time + max(gap, 0.0)
        if len(bar.beats) >= 2:
            gaps = np.diff([b.time for b in bar.beats])
            bar.tempo_bpm = float(60.0 / np.mean(gaps))

    reference_bars = _reference_bars(structure.bars)
    if reference_bars:
        structure.initial_tempo = _robust_initial_tempo(reference_bars)
        tempo_set = {round(b.tempo_bpm, 1) for b in reference_bars}
        structure.has_tempo_changes = len(tempo_set) > 1
