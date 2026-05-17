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
- One `setTempo` at tick 0 derived from `BeatStructure.initial_tempo`,
  defaulting to 120 BPM when the structure is empty. The DBN+RNN
  beat tracker may have produced a slightly off tempo; we honour it
  so the audio time -> tick conversion stays consistent with what
  the rest of the pipeline saw.
- Velocity is a percentile-normalized mapping of onset strength: p10
  -> 32, p90 -> 110, linearly interpolated and clamped to [1, 127].
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
from typing import Any, Iterable

import mido
import numpy as np

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
    "c": 49,  # Crash 1
    "d": 51,  # Ride 1
    "t": 50,  # High tom
    "f": 41,  # Low floor tom
    "p": 39,  # Hand clap
    "b": 56,  # Cowbell
}

# Velocity mapping anchors. p10 of strength -> VEL_FLOOR, p90 -> VEL_CEIL.
# Linear in between, clamped at the ends. Compresses dynamic range slightly
# so even the quietest hit is audible.
VEL_FLOOR = 32
VEL_CEIL = 110


def onsets_to_midi_bytes(
    onsets_by_pitch: dict[str, list[Any]],
    initial_tempo_bpm: float = 120.0,
) -> bytes:
    """Render all onsets to a single-track MIDI file and return its bytes.

    `onsets_by_pitch` is the same shape produced by
    `pipeline/onsets.py`: pitch letter -> list of objects with `time`
    (seconds, absolute) and `strength` (unnormalized peak amplitude
    from `librosa.onset.onset_strength`).
    """
    tempo_bpm = (
        float(initial_tempo_bpm)
        if initial_tempo_bpm and initial_tempo_bpm > 0
        else 120.0
    )
    micros_per_beat = int(round(60_000_000 / tempo_bpm))
    seconds_to_ticks = TICKS_PER_BEAT * tempo_bpm / 60.0

    velocity_lookup = _build_velocity_lookup(onsets_by_pitch)

    # Flatten to (tick, midi_note, velocity) and sort by absolute tick so
    # we can emit deltaTime via mido cleanly.
    events: list[tuple[int, int, int]] = []
    for pitch, cands in onsets_by_pitch.items():
        midi_note = PITCH_TO_MIDI.get(pitch)
        if midi_note is None:
            log.info("Skipping unmapped pitch %r when rendering onsets MIDI", pitch)
            continue
        for c in cands:
            time = float(getattr(c, "time", 0.0) or 0.0)
            if time < 0:
                continue
            strength = float(getattr(c, "strength", 0.0) or 0.0)
            velocity = velocity_lookup(pitch, strength)
            tick = max(0, int(round(time * seconds_to_ticks)))
            events.append((tick, midi_note, velocity))

    events.sort(key=lambda e: (e[0], e[1]))

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=micros_per_beat, time=0))

    last_tick = 0
    for tick, note, vel in events:
        dt_on = max(0, tick - last_tick)
        track.append(
            mido.Message(
                "note_on",
                channel=DRUM_CHANNEL,
                note=note,
                velocity=vel,
                time=dt_on,
            )
        )
        # One-tick gate so the file is well-formed; drum samples decay
        # by their own envelope, the gate length isn't musically
        # meaningful.
        track.append(
            mido.Message(
                "note_off",
                channel=DRUM_CHANNEL,
                note=note,
                velocity=0,
                time=1,
            )
        )
        last_tick = tick + 1

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


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
