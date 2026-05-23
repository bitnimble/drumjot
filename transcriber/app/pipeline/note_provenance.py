"""Build the per-note debug provenance sidecar for filter-mode transcribe.

For every detected onset (kept OR rejected by the filter LLM) we record
everything we know about its origin and fate:

  - the original detector hit (time + strength + backend),
  - the beat-tracker placement (bar + beat_in_bar) — already
    grid-aligned by `align_beats_to_onsets` before this runs,
  - the MIDI tick the kept note ended up at, or `null` for rejected
    onsets that never reach the MIDI,
  - the filter LLM's keep/reject decision.

The frontend's debug-bundle loader reads this sidecar to:

  - annotate each rendered MIDI note with per-note debug details in the
    selection label (keyed by the unique `(tick, pitch)` identifier we
    preserve through `from_midi.ts`), and
  - render rejected onsets as ghost overlays at their detected (bar,
    beat_in_bar) position so the operator can see *what* was filtered
    out and *why*, gated by the toolbar "Show filtered" checkbox.

Sidecar-only: this file is generated alongside `prediction.mid` in the
filter pathway and shipped inside the debug bundle. The pipeline does
not depend on it; an absent provenance file just means the UI shows the
score without per-note debug details.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.onsets_midi import (
    PITCH_TO_MIDI,
    TICKS_PER_BEAT,
    compute_bar_tick_grid,
)

# File format version. Bump on any schema change so the frontend can
# guard against loading older bundles with newer code (or vice versa).
FORMAT_VERSION = 1


def build_note_provenance(
    *,
    all_onsets_by_pitch: dict[str, list[OnsetCandidate]],
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    onset_backend: str,
    beat_alignment_offset_sec: float | None = None,
) -> dict[str, Any]:
    """Return the JSON-serialisable provenance payload for one filter run.

    `all_onsets_by_pitch` is the post-split candidates (what the filter LLM
    saw); `kept_by_pitch` is whatever survived. Identity is by object —
    we check `id(c)` membership in the kept set, so the caller MUST pass
    the same `OnsetCandidate` instances through both maps (the filter
    pathway already does — `filter_onsets_for_instrument` keeps the
    candidates verbatim).
    """
    bar_start_tick, midi_tempos, lead_bars, _lead_tempo = compute_bar_tick_grid(
        structure, structure.initial_tempo
    )
    kept_ids: dict[str, set[int]] = {
        pitch: {id(c) for c in cands}
        for pitch, cands in kept_by_pitch.items()
    }

    per_pitch: dict[str, list[dict[str, Any]]] = {}
    for pitch, candidates in all_onsets_by_pitch.items():
        midi_note = PITCH_TO_MIDI.get(pitch)
        kept_set = kept_ids.get(pitch, set())
        entries: list[dict[str, Any]] = []
        for c in candidates:
            bar = int(c.bar)
            in_range = bar >= 0 and bar < len(bar_start_tick)
            kept = id(c) in kept_set
            tick: int | None = None
            if kept and in_range and midi_note is not None:
                b = structure.bars[bar]
                local = max(0.0, float(c.time) - float(b.start_time))
                tick = bar_start_tick[bar] + int(round(
                    local * TICKS_PER_BEAT * midi_tempos[bar] / 60.0
                ))
            entries.append({
                "pitch": pitch,
                "midi_note": midi_note,
                # Unique (tick, pitch) key — the frontend matches each
                # rendered Note's `metadata.midi.tick` against this.
                "tick": tick,
                "detected_time_sec": float(c.time),
                "detection_backend": onset_backend,
                "strength": float(c.strength),
                "bar": bar,
                "beat_in_bar": float(c.beat_in_bar),
                # In-range hits are the only ones the filter ever sees;
                # an out-of-range onset is dropped before the LLM call,
                # so a `kept=False, out_of_range=True` entry should read
                # as "padding noise" rather than "the LLM rejected it".
                "out_of_range": not in_range,
                "kept": kept,
                "rejected_by": None if kept or not in_range else "filter_llm",
            })
        # Score order makes the JSON readable; out-of-range entries
        # (bar=-1) sort to the front.
        entries.sort(
            key=lambda e: (e["bar"], e["beat_in_bar"], e["detected_time_sec"])
        )
        per_pitch[pitch] = entries

    return {
        "format": FORMAT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "onset_backend": onset_backend,
        # Audio-time shift applied uniformly to the beat grid before
        # beat positions were computed. The detected `time` fields here
        # predate this shift; the `bar`/`beat_in_bar` values are
        # post-shift. Useful for the operator to see how much grid
        # correction was needed.
        "beat_alignment_offset_sec": beat_alignment_offset_sec,
        # Mapping bar 0 (struct.bars) -> the rendered MIDI bar index a
        # consumer that walks bars from tick 0 would see. The MIDI lays
        # down `lead_bars` empty bar-0-sized blocks before bar 0 to
        # absorb the audio lead-in (see `compute_bar_tick_grid`). The
        # frontend uses this to find a struct bar's displayed bar in
        # the rendered jot (the rendered jot's bar indices are 1-based:
        # `displayed_bar_index = lead_bars + struct_bar + 1`).
        "lead_bars": lead_bars,
        "per_pitch": per_pitch,
    }
