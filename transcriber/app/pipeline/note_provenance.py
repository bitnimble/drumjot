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
#   v1: original kept/rejected entries.
#   v2: adds `reason_code` / `reason_text` from the filter LLM on
#       rejected entries (null on kept/upstream-vetted entries).
#   v3: surfaces every per-stage time shift that affects a kept onset's
#       final position. Per-entry: `raw_model_time_sec` (pre-envelope-
#       refine ADTOF time), per-quantise-pass shifts
#       (`geometric_shift_slots` / `envelope_shift_slots` /
#       `grid_shift_slots` / `llm_shift_slots`),
#       `quantised_residual_slots` (signed sub-slot residual from the
#       geometric pass), and an explicit `off_grid` flag. File-level:
#       `beat_align_coarse_offset_sec` + `beat_align_fine_offset_sec`
#       (the previously-collapsed coarse/fine alignment split; their
#       sum equals `beat_alignment_offset_sec`).
FORMAT_VERSION = 3


def build_note_provenance(
    *,
    all_onsets_by_pitch: dict[str, list[OnsetCandidate]],
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    beat_alignment_offset_sec: float = 0.0,
    beat_align_coarse_offset_sec: float = 0.0,
    beat_align_fine_offset_sec: float = 0.0,
    rejected_by_pitch: dict[str, str] | None = None,
    reasons_by_pitch: dict[str, dict[int, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Return the JSON-serialisable provenance payload for one filter run.

    `all_onsets_by_pitch` is the post-split candidates (what the filter LLM
    saw); `kept_by_pitch` is whatever survived. Identity is by object; we check `id(c)` membership in the kept set, so the caller MUST pass
    the same `OnsetCandidate` instances through both maps (the filter
    pathway already does; `filter_onsets_for_instrument` keeps the
    candidates verbatim).

    `rejected_by_pitch` overrides the `rejected_by` label per pitch.
    Defaults to `"filter_llm"` for any pitch not in the map; the
    historical behaviour. Hi-hat lanes (`h`, `H`) pass
    `"hihat_split"` since their discards come from the unified ternary
    classifiers upstream of the filter LLM.

    `reasons_by_pitch`, when provided, supplies the filter LLM's
    `{reason, reason_text}` for each rejected onset, keyed by `id(c)` of
    the rejected candidate. Pitches with no entry (or the upstream-vetted
    `h`/`H`/`c`/`d` lanes whose rejections don't go through the filter
    LLM) get `reason_code = null` on their rejected entries.
    """
    bar_start_tick, midi_tempos, lead_bars, _lead_tempo = compute_bar_tick_grid(
        structure, structure.initial_tempo
    )
    kept_ids: dict[str, set[int]] = {
        pitch: {id(c) for c in cands}
        for pitch, cands in kept_by_pitch.items()
    }
    rejected_source = rejected_by_pitch or {}
    reasons_source = reasons_by_pitch or {}

    per_pitch: dict[str, list[dict[str, Any]]] = {}
    for pitch, candidates in all_onsets_by_pitch.items():
        midi_note = PITCH_TO_MIDI.get(pitch)
        kept_set = kept_ids.get(pitch, set())
        reject_label = rejected_source.get(pitch, "filter_llm")
        reason_map = reasons_source.get(pitch, {})
        entries: list[dict[str, Any]] = []
        for c in candidates:
            bar = int(c.bar)
            in_range = bar >= 0 and bar < len(bar_start_tick)
            kept = id(c) in kept_set
            tick: int | None = None
            # When the quantise stage has shifted a kept onset, the
            # rendered MIDI tick is derived from `quantised_time` (see
            # `onsets_to_midi_bytes`), so the provenance tick must
            # match; otherwise the frontend's `(tick, pitch)` key
            # lookup misses every shifted note.
            quantised_time = getattr(c, "quantised_time", None)
            tick_time = quantised_time if quantised_time is not None else float(c.time)
            if kept and in_range and midi_note is not None:
                b = structure.bars[bar]
                local = max(0.0, float(tick_time) - float(b.start_time))
                tick = bar_start_tick[bar] + int(round(
                    local * TICKS_PER_BEAT * midi_tempos[bar] / 60.0
                ))
            reason_info = (
                reason_map.get(id(c))
                if not kept and in_range
                else None
            )
            raw_model_time = getattr(c, "raw_model_time", None)
            entries.append({
                "pitch": pitch,
                "midi_note": midi_note,
                # Unique (tick, pitch) key; the frontend matches each
                # rendered Note's `metadata.midi.tick` against this.
                "tick": tick,
                # The raw detector hit; unchanged by quantise.
                "detected_time_sec": float(c.time),
                # ADTOF model peak time BEFORE `_refine_peak_times_audio`
                # snapped it to the audio's onset-strength envelope
                # local-max. `null` for non-ADTOF detection paths (none
                # in production today) and for older bundles. Lets the
                # debug popup surface the envelope refinement as its
                # own per-onset stage in the detected → final chain.
                "raw_model_time_sec": (
                    float(raw_model_time) if raw_model_time is not None else None
                ),
                # Post-quantise absolute time. None when the quantise
                # stage didn't run or this onset wasn't shifted; the
                # rendered MIDI tick falls back to `detected_time_sec`
                # in that case.
                "quantised_time_sec": (
                    float(quantised_time) if quantised_time is not None else None
                ),
                "quantised_shift_slots": getattr(c, "quantised_shift_slots", None),
                # Per-pass quantise contributions, so the popup can show
                # "geometric +1, env +0, grid 0, llm -1" instead of one
                # collapsed sum. `null` when the pass didn't run for
                # this onset (off-grid for any later pass; envelope pass
                # skipped because no envelope was available; grid/LLM
                # pass turned off; LLM pass cancelled/errored; or all
                # of the above on older bundles). `0` means the pass
                # ran but didn't shift (or its shift was rejected by
                # the monotonic-injective guard).
                "geometric_shift_slots": getattr(c, "geometric_shift_slots", None),
                "envelope_shift_slots": getattr(c, "envelope_shift_slots", None),
                "grid_shift_slots": getattr(c, "grid_shift_slots", None),
                "llm_shift_slots": getattr(c, "llm_shift_slots", None),
                # Sub-slot residual from the geometric pass: how far the
                # raw natural slot position sat from the nearest integer
                # slot, range (-0.5, +0.5]. Surfaces the "performer was
                # consistently ~0.3 slots late" feel that a clean snap
                # erases. `null` for off-grid onsets and for older
                # bundles.
                "quantised_residual_slots": getattr(
                    c, "quantised_residual_slots", None
                ),
                # Explicit off-grid flag from the geometric snap (= no
                # free slot within the match band). Previously inferred
                # from `quantised_time_sec === null`; surfaced as its
                # own field so the popup can distinguish "off-grid"
                # from "stage didn't run / no shift needed".
                "off_grid": bool(getattr(c, "off_grid", False)),
                # ADTOF model confidence at the peak frame, in [0, 1].
                # Surfaced as "Onset confidence" in the per-note debug
                # popup. Distinct from `amplitude` (raw audio loudness)
                # which drives velocity mapping; see
                # `OnsetCandidate.amplitude` for the split.
                "strength": float(c.strength),
                # Raw audio amplitude (|sample| in [0, 1]) in a ±20ms
                # window around the onset, on the source stem. Drives
                # the per-pitch percentile-normalised velocity mapping.
                # `null` for non-ADTOF detection paths and re-loaded
                # legacy bundles produced before this field existed;
                # consumers fall back to `strength` in that case.
                "amplitude": (
                    float(c.amplitude) if c.amplitude is not None else None
                ),
                "bar": bar,
                "beat_in_bar": float(c.beat_in_bar),
                # In-range hits are the only ones the filter ever sees;
                # an out-of-range onset is dropped before the LLM call,
                # so a `kept=False, out_of_range=True` entry should read
                # as "padding noise" rather than "the LLM rejected it".
                "out_of_range": not in_range,
                "kept": kept,
                "rejected_by": None if kept or not in_range else reject_label,
                # Filter-LLM rejection reason. `reason_code` is one of
                # the short codes in `filter_llm.REASON_CODES`
                # (`bleed`/`double_trigger`/`noise`/`custom`); `null`
                # when the rejection didn't come from the filter LLM
                # (upstream-vetted lanes, out-of-range, or kept onsets)
                # or the bundle predates this field. `reason_text` is
                # free-text detail; always populated for `custom`,
                # optional otherwise.
                "reason_code": reason_info["reason"] if reason_info else None,
                "reason_text": (
                    reason_info.get("reason_text") if reason_info else None
                ),
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
        # Audio-time shift applied uniformly to the beat grid before
        # beat positions were computed. The detected `time` fields here
        # predate this shift; the `bar`/`beat_in_bar` values are
        # post-shift. Useful for the operator to see how much grid
        # correction was needed. The coarse / fine split surfaces the
        # two alignment passes separately so the debug popup can show
        # them as distinct stages; their sum always equals
        # `beat_alignment_offset_sec` and downstream code reading just
        # the combined field keeps working.
        "beat_alignment_offset_sec": beat_alignment_offset_sec,
        "beat_align_coarse_offset_sec": beat_align_coarse_offset_sec,
        "beat_align_fine_offset_sec": beat_align_fine_offset_sec,
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
