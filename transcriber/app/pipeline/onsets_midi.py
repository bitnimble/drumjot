"""Render the raw onset list (one MIDI hit per detected onset) to bytes.

Auto-emitted by the onset stage alongside `onsets.json` for diagnostic
playback: drop the resulting `onsets_only.mid` into any DAW or web MIDI
player and you'll hear exactly what `librosa.onset.onset_detect`
produced, without the LLM's filtering or quantization in the way. The
canonical use case is "did the detector miss / invent hits on this
track?" — a question that's almost impossible to answer from a JSON
dump and an audio file in separate windows.

Design choices:

- Single track, channel 10 (drum channel; MIDI uses 1-indexed so the
  byte value is 9). Matches the convention used elsewhere in the
  project (`src/midi/to_midi.ts`).
- Tempo / time signature: when a `BeatStructure` is supplied, the
  bars are laid on a musical tick grid (each bar occupies its
  `numerator * 4/denominator` quarter-notes' worth of ticks) with a
  `set_tempo` + `time_signature` meta at every bar where either
  changes. The per-bar MIDI tempo is derived from the bar's actual
  audio duration (`num * 4/den * 60 / (bars[i+1].start_time -
  bars[i].start_time)`), **not** from `bar.tempo_bpm`: the beats
  pipeline smooths / pins `bar.tempo_bpm` for LLM stability and it
  isn't constrained to satisfy `num_beats * 60/bpm == bar_duration`,
  so reusing it for tick scaling would make MIDI bar boundaries
  drift against the audio (each bar contributes a small error and
  it accumulates linearly across the track). Onsets are placed at
  `bar_start_tick + local_seconds * ticks_per_second(midi_tempo)`
  with the same derived tempo, so a DAW shows correct bar lines /
  meter / tempo and absolute onset times round-trip exactly (tick
  rounding aside). The tick-0 `set_tempo` covers only the empty
  lead-in (audio time before bar 0); it starts from the song-level
  `initial_tempo` and is then nudged so the lead-in spans a whole
  number of bar-0-length tick blocks (see in-function comment). The
  nudge is what keeps bar 0's downbeat on a bar boundary in any
  reader that walks bars from tick 0 — SMF has no anacrusis field, so
  a non-rounded lead-in would otherwise misalign every subsequent bar
  in `src/midi/from_midi.ts` and most generic MIDI parsers. A DAW
  reading the full tempo map still sees per-bar `set_tempo` events at
  every bar boundary, so bar 0's audio duration stays exact. The
  lead-in itself is the MIDI counterpart of the DSL path's
  `{{ drumsT0Sec }}`. Without a structure (the raw
  `onsets_only.mid` diagnostic, where onsets may have no bar) it
  falls back to a single `set_tempo` at tick 0 from
  `initial_tempo_bpm` and a flat seconds->ticks scaling.
- Velocity is a percentile-normalized mapping of onset strength: p10
  -> 64, p90 -> 104, linearly interpolated and clamped to [1, 127].
  The narrow range / high floor keeps it as gentle dynamics rather
  than a wide loud/quiet swing (see `VEL_FLOOR` / `VEL_CEIL`).
  Per-pitch percentiles rather than global, because each stem has a
  very different strength distribution (snares typically 5-10x
  louder than hi-hats post-separation) and global percentiles would
  squash hi-hats into uniform pianissimo.
- One-tick `note_off` after each `note_on`. Drum hits decay through
  the sample envelope, not the MIDI gate; the short note_off just
  keeps the file MIDI-spec-compliant.
"""
from __future__ import annotations

import io
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import mido
import numpy as np

if TYPE_CHECKING:
    from app.pipeline.beats import BeatStructure

log = logging.getLogger(__name__)

# Standard PPQ matches `src/midi/to_midi.ts:TICKS_PER_BEAT` so a file
# round-tripped through both paths reads identically.
TICKS_PER_BEAT = 480

DRUM_CHANNEL = 9  # GM percussion (channel 10 in 1-indexed parlance)

# DSL pitch letter -> default MIDI note. Ported from
# `src/midi/gm.ts::defaultMidiNote`. We don't see modifiers here
# (the onset stage produces raw pitches), so this is the bare mapping.
PITCH_TO_MIDI: dict[str, int] = {
    "k": 36,  # Kick
    "s": 38,  # Snare
    "h": 42,  # Closed hi-hat
    # `H` is the synthetic open-hi-hat routing pitch introduced by
    # `pipeline/hihat_split.py`; renders as GM open hi-hat in the
    # onset-diagnostic MIDI even before the H->h:o recompose merge lands.
    "H": 46,  # Open hi-hat
    "c": 49,  # Crash 1
    "d": 51,  # Ride 1
    "t": 50,  # High tom
    "f": 41,  # Low floor tom
    "p": 39,  # Hand clap
    "b": 56,  # Cowbell
}

# Velocity mapping anchors. p10 of strength -> VEL_FLOOR, p90 -> VEL_CEIL.
# Linear in between, clamped at the ends. The range is deliberately
# narrow and the floor high so the strength->velocity mapping reads as
# gentle dynamics rather than a wide loud/quiet swing: the quietest hit
# still lands at a clearly-audible velocity, and the loudest isn't far
# above it.
VEL_FLOOR = 64
VEL_CEIL = 104


def onsets_to_midi_bytes(
    onsets_by_pitch: dict[str, list[Any]],
    initial_tempo_bpm: float = 120.0,
    structure: BeatStructure | None = None,
) -> bytes:
    """Render all onsets to a single-track MIDI file and return its bytes.

    `onsets_by_pitch` is the same shape produced by
    `pipeline/onsets.py`: pitch letter -> list of objects with `time`
    (seconds, absolute), `strength`, and (when `structure` is given)
    `bar` (0-indexed, < 0 = out of tracked range).

    When `structure` is supplied the bars are laid on a musical tick
    grid with a per-bar tempo + time-signature map, so the file opens
    correctly in a DAW (right meter, tempo, bar lines), and bar 0 is
    offset from tick 0 by the lead-in so absolute times round-trip.
    Onsets with no
    in-range bar are dropped (consistent with the filter pathway, which
    only keeps in-range onsets). Without a structure, a single tempo +
    flat seconds->ticks scaling is used and every onset is kept.
    """
    velocity_lookup = _build_velocity_lookup(onsets_by_pitch)

    # (abs_tick, order, message). `order` keeps meta (0) ahead of
    # note_off (1) / note_on (2) at the same tick so a bar-start tempo
    # change applies before that bar's notes.
    timeline: list[tuple[int, int, int, Any]] = []

    if structure is not None and getattr(structure, "bars", None):
        bars = list(structure.bars)
        bar_start_tick, midi_tempos, _lead_bars, lead_tempo = (
            compute_bar_tick_grid(structure, initial_tempo_bpm)
        )

        # Tick-0 set_tempo at the song-level baseline. The per-bar
        # set_tempo loop below then re-emits each bar's actual tempo at
        # its start tick, so a DAW reading the full tempo map still gets
        # exact per-bar audio durations.
        lead_micros = int(round(60_000_000 / max(lead_tempo, 1e-6)))
        timeline.append((0, 0, 0, mido.MetaMessage(
            "set_tempo", tempo=lead_micros, time=0)))
        prev_micros: int | None = lead_micros
        prev_ts: tuple[int, int] | None = None
        for i, b in enumerate(bars):
            st = bar_start_tick[i]
            # Per-bar tempo at the bar's start tick (not tick 0). The
            # lead-in already has its own set_tempo above; emitting
            # bar 0's tempo here means a DAW gets exact per-bar audio
            # durations while a "first-tempo-only" player keeps using
            # the song-level lead_tempo (the right behaviour for a
            # constant-tempo recording).
            micros = int(round(60_000_000 / max(midi_tempos[i], 1e-6)))
            if micros != prev_micros:
                timeline.append((st, 0, 0, mido.MetaMessage(
                    "set_tempo", tempo=micros, time=0)))
                prev_micros = micros
            ts = (int(b.time_signature[0]), int(b.time_signature[1]))
            if ts != prev_ts:
                # TS for bar 0 still goes at tick 0 so the DAW knows the
                # meter for the lead-in; later TS changes ride the bar
                # start tick like the set_tempo events.
                ts_tick = 0 if i == 0 else st
                timeline.append((ts_tick, 0, 0, mido.MetaMessage(
                    "time_signature",
                    numerator=ts[0],
                    denominator=_safe_denominator(ts[1]),
                    time=0,
                )))
                prev_ts = ts

        skipped = 0
        for pitch, cands in onsets_by_pitch.items():
            midi_note = PITCH_TO_MIDI.get(pitch)
            if midi_note is None:
                log.info("Skipping unmapped pitch %r in onsets MIDI", pitch)
                continue
            for c in cands:
                bar = int(getattr(c, "bar", -1))
                if bar < 0 or bar >= len(bars):
                    skipped += 1
                    continue
                b = bars[bar]
                t = float(getattr(c, "time", 0.0) or 0.0)
                local = max(0.0, t - float(b.start_time))
                tick = bar_start_tick[bar] + int(round(
                    local * TICKS_PER_BEAT * midi_tempos[bar] / 60.0
                ))
                vel = velocity_lookup(
                    pitch, float(getattr(c, "strength", 0.0) or 0.0)
                )
                timeline.append((tick, 2, midi_note, mido.Message(
                    "note_on", channel=DRUM_CHANNEL, note=midi_note,
                    velocity=vel, time=0)))
                timeline.append((tick + 1, 1, midi_note, mido.Message(
                    "note_off", channel=DRUM_CHANNEL, note=midi_note,
                    velocity=0, time=0)))
        if skipped:
            log.info(
                "onsets->MIDI: dropped %d onset(s) with no in-range bar",
                skipped,
            )
    else:
        tempo_bpm = (
            float(initial_tempo_bpm)
            if initial_tempo_bpm and initial_tempo_bpm > 0
            else 120.0
        )
        micros = int(round(60_000_000 / tempo_bpm))
        seconds_to_ticks = TICKS_PER_BEAT * tempo_bpm / 60.0
        timeline.append((0, 0, 0, mido.MetaMessage(
            "set_tempo", tempo=micros, time=0)))
        for pitch, cands in onsets_by_pitch.items():
            midi_note = PITCH_TO_MIDI.get(pitch)
            if midi_note is None:
                log.info("Skipping unmapped pitch %r in onsets MIDI", pitch)
                continue
            for c in cands:
                t = float(getattr(c, "time", 0.0) or 0.0)
                if t < 0:
                    continue
                vel = velocity_lookup(
                    pitch, float(getattr(c, "strength", 0.0) or 0.0)
                )
                tick = max(0, int(round(t * seconds_to_ticks)))
                timeline.append((tick, 2, midi_note, mido.Message(
                    "note_on", channel=DRUM_CHANNEL, note=midi_note,
                    velocity=vel, time=0)))
                timeline.append((tick + 1, 1, midi_note, mido.Message(
                    "note_off", channel=DRUM_CHANNEL, note=midi_note,
                    velocity=0, time=0)))

    timeline.sort(key=lambda e: (e[0], e[1], e[2]))

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    last_tick = 0
    for abs_tick, _order, _tie, msg in timeline:
        msg.time = max(0, abs_tick - last_tick)
        track.append(msg)
        last_tick = abs_tick

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def compute_bar_tick_grid(
    structure: BeatStructure,
    initial_tempo_bpm: float | None = None,
) -> tuple[list[int], list[float], int, float]:
    """Build the per-bar tick grid `onsets_to_midi_bytes` lays out.

    Returns `(bar_start_tick, midi_tempos, lead_bars, lead_tempo)`:

      - `bar_start_tick[i]` is the absolute MIDI tick of bar `i`'s downbeat
        (with the lead-in pre-rolled as `lead_bars` bar-0-sized empty
        blocks before bar 0).
      - `midi_tempos[i]` is the BPM derived from bar `i`'s actual audio
        duration (`num * 4/den * 60 / bar_duration`). Distinct from
        `bar.tempo_bpm`, which the beats pipeline pins/smooths.
      - `lead_bars` is how many bar-0-sized empty blocks were laid down
        before bar 0 to absorb the pre-drum lead-in.
      - `lead_tempo` is the back-solved tempo emitted as the tick-0
        `set_tempo` so the rounded `lead_bars * bar0_ticks` maps exactly
        back to `bars[0].start_time` in audio time. See the lead-in
        rounding rationale in `onsets_to_midi_bytes`.
    """
    bars = list(structure.bars)
    if not bars:
        # Falls back to a sane default; the structureless code path inside
        # `onsets_to_midi_bytes` doesn't call us, so this branch is mostly
        # defensive for direct callers (provenance builder, tests).
        fallback = float(initial_tempo_bpm) if initial_tempo_bpm and initial_tempo_bpm > 0 else 120.0
        return [], [], 0, fallback
    midi_tempos = [_bar_duration_tempo_bpm(bars, i) for i in range(len(bars))]
    lead_tempo = float(initial_tempo_bpm or 0.0)
    if lead_tempo <= 0:
        lead_tempo = midi_tempos[0]
    num0, den0 = int(bars[0].time_signature[0]), int(bars[0].time_signature[1])
    bar0_ticks = max(1, int(round(num0 * (4.0 / max(den0, 1)) * TICKS_PER_BEAT)))
    lead_in_secs = float(bars[0].start_time)
    raw_lead_ticks = max(0.0, lead_in_secs * TICKS_PER_BEAT * lead_tempo / 60.0)
    lead_bars = int(round(raw_lead_ticks / bar0_ticks))
    acc = lead_bars * bar0_ticks
    if acc > 0 and lead_in_secs > 0:
        lead_tempo = acc * 60.0 / (TICKS_PER_BEAT * lead_in_secs)
    bar_start_tick: list[int] = []
    for b in bars:
        bar_start_tick.append(acc)
        num, den = int(b.time_signature[0]), int(b.time_signature[1])
        acc += max(1, int(round(num * (4.0 / max(den, 1)) * TICKS_PER_BEAT)))
    return bar_start_tick, midi_tempos, lead_bars, lead_tempo


def _bar_duration_tempo_bpm(bars: list[Any], i: int) -> float:
    """Quarter-note BPM that makes bar `i`'s fixed MIDI tick length
    (`num * 4/den * TICKS_PER_BEAT`) map back to its actual audio
    duration.

    Duration is taken from `bars[i+1].start_time - bars[i].start_time`
    when a next bar exists (the most reliable signal — directly anchored
    to the next downbeat), else from `bar.end_time - start_time`, else
    a final fallback to `bar.tempo_bpm` so structures without `end_time`
    (e.g. duck-typed test bars) degrade gracefully rather than dividing
    by zero.
    """
    b = bars[i]
    start = float(getattr(b, "start_time", 0.0))
    dur = 0.0
    if i + 1 < len(bars):
        nxt = float(getattr(bars[i + 1], "start_time", start))
        if nxt > start:
            dur = nxt - start
    if dur <= 0.0:
        end = getattr(b, "end_time", None)
        if end is not None and float(end) > start:
            dur = float(end) - start
    if dur <= 0.0:
        return float(getattr(b, "tempo_bpm", 120.0))
    num = int(b.time_signature[0])
    den = int(b.time_signature[1])
    quarter_notes = num * 4.0 / max(den, 1)
    return quarter_notes * 60.0 / dur


def _safe_denominator(den: int) -> int:
    """MIDI time-signature denominators must be powers of two; `mido`
    derives the stored exponent via log2. Fall back to 4 for anything
    the beat tracker emits that isn't a clean power of two."""
    if den >= 1 and (den & (den - 1)) == 0:
        return den
    log.info("onsets->MIDI: non-power-of-two denominator %r; using 4", den)
    return 4


def _build_velocity_lookup(
    onsets_by_pitch: dict[str, list[Any]],
):
    """Return a `(pitch, strength) -> velocity` callable.

    Pre-computes per-pitch p10 / p90 strength so the linear mapping
    only does a comparison + interpolation per onset.
    """
    per_pitch_range: dict[str, tuple[float, float]] = {}
    for pitch, cands in onsets_by_pitch.items():
        strengths = [
            float(getattr(c, "strength", 0.0) or 0.0) for c in cands
        ]
        per_pitch_range[pitch] = _percentile_range(strengths)

    def velocity_for(pitch: str, strength: float) -> int:
        lo, hi = per_pitch_range.get(pitch, (0.0, 1.0))
        if hi <= lo:
            return (VEL_FLOOR + VEL_CEIL) // 2
        t = (strength - lo) / (hi - lo)
        v = VEL_FLOOR + t * (VEL_CEIL - VEL_FLOOR)
        return max(1, min(127, int(round(v))))

    return velocity_for


def _percentile_range(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(arr, 10.0))
    hi = float(np.percentile(arr, 90.0))
    if hi <= lo:
        # Pathologically narrow distribution (e.g. ≤ 2 hits): fall back to
        # min/max so we don't divide by zero downstream.
        return float(arr.min()), float(arr.max() or 1.0)
    return lo, hi
