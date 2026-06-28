"""The `transcription.json` sidecar: Drumjot-native, higher-fidelity data
that `prediction.mid` can only approximate.

`prediction.mid` is the portable, lossy tier, any DAW or a bare re-import
into Drumjot understands it, and `src/midi/from_midi.ts` reconstructs what
it can (e.g. ramps via `detectRampRuns`). `transcription.json` is the
lossless tier: a versioned container, written into the debug bundle beside
the MIDI, that the frontend prefers when present (skipping MIDI tempo
parsing entirely). We own its schema, so it carries things MIDI represents
poorly or not at all.

Today it holds one field, `tempoMap` (a tick-based constant/ramp tempo map
built from `BeatStructure.tempo_segments`); it's intentionally a general
container so future rich data (feel, swing, sections, lane mapping, …) can
join it without a new sidecar. The `format` integer guards against silent
schema drift the same way `note_provenance.json` does.
"""
from __future__ import annotations

from typing import Any

from app.pipeline.onsets_midi import build_tempo_map

# Bump on any breaking change to the container schema. The frontend loader
# (`src/editing/provenance/debug_zip.ts`) checks it and ignores newer
# formats it doesn't understand, falling back to the MIDI tempo track.
# (The bundle filename lives in `debug_bundle.TRANSCRIPTION_FILENAME`.)
TRANSCRIPTION_FORMAT = 1


def build_transcription(structure: Any) -> dict[str, Any]:
    """Assemble the `transcription.json` payload from a finalized structure."""
    return {
        "format": TRANSCRIPTION_FORMAT,
        "tempoMap": build_tempo_map(structure),
        "barDrift": build_bar_drift(structure),
    }


def build_bar_drift(structure: Any) -> list[float]:
    """Per-bar performance drift in seconds, indexed by the (drum) bar.

    `barDrift[i]` is how far bar `i`'s *real* downbeat sits past where the
    clean uniform tempo puts it (`beats.BarInfo.drift_sec`), the deviation
    the tempo map smoothed away. The frontend keeps the uniform tempo for
    display but uses this to align the bar lines + waveform to the recording
    (it maps bar `i` to `layers[0].bars[leadBars + i]`). All-zero when the
    song needed no regularization, so it's cheap and harmless to ship.
    """
    return [
        round(float(getattr(bar, "drift_sec", 0.0)), 4)
        for bar in getattr(structure, "bars", None) or []
    ]
