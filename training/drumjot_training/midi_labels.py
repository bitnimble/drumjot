"""E-GMD / MIDI label extraction: per-lane drum onset times.

Reads onset times (seconds) for each drum lane from a MIDI file. Iterates
`mido.MidiFile` directly (NOT `.tracks`), so message times are tempo-aware
and already in seconds, avoiding the multi-track raw-ticks trap the
scoring `midi_read.py` also calls out. A `note_on` with velocity > 0 is an
onset; a velocity-0 `note_on` (a note-off in disguise) and `note_off` are
ignored. Notes outside the five-lane kit are dropped.
"""
from __future__ import annotations

from pathlib import Path

import mido

from drumjot_training.lanes import WEIGHT_LANES, lane_for_gm_note, negative_lane_for_gm_note


def onsets_by_lane(midi: mido.MidiFile) -> dict[str, list[float]]:
    """Per-lane ascending onset times (seconds) for `midi`.

    Always returns all output lanes (empty list when absent) PLUS the catch-all
    negative lane `x`: non-kit GM percussion (clap/tambourine/cowbell + latin/aux
    perc) the kit map drops, kept for hard-negative loss weighting and ignored by
    output-lane consumers.
    """
    out: dict[str, list[float]] = {lane: [] for lane in WEIGHT_LANES}
    t = 0.0
    for msg in midi:  # tempo-aware iteration: msg.time is delta seconds
        t += msg.time
        if msg.type != "note_on" or msg.velocity <= 0:
            continue
        lane = lane_for_gm_note(msg.note) or negative_lane_for_gm_note(msg.note)
        if lane is not None:
            out[lane].append(t)
    return out


def onsets_from_path(path: str | Path) -> dict[str, list[float]]:
    """`onsets_by_lane` for a MIDI file on disk."""
    return onsets_by_lane(mido.MidiFile(str(path)))
