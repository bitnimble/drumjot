"""Optional `quantise` pipeline stage: snap kept onsets to the slot grid.

Runs between `filter` and `transcribe`. Two passes:

  1. **Geometric snap** (`_geometric_snap`). Per drum lane and per bar, a
     monotonic-injective dynamic program (`geometric_snap.snap_lane`)
     assigns each onset to an integer slot, minimising total squared
     slot-distance. No two onsets in a lane+bar share a slot, detected
     order is preserved, and an onset with no free slot within the match
     band (`_MATCH_BAND`) is left *off-grid*: `quantised_time` stays None
     and `off_grid` is set, so the MIDI emitter keeps its raw `time` and
     the frontend records the sub-slot residual as the note's `offset`
     (swing / ghost-flam / push-pull feel survives instead of snapping).

  2. **LLM residual pass** (`_llm_residual_pass`). Sends every *on-grid*
     kept onset with its current slot to Haiku in one joint call. The
     model returns an integer slot shift per onset, clamped server-side
     to `_MAX_LLM_SHIFT` (= 2). The joint call is intentional: this is
     where *cross-instrument* musical reasoning earns its keep ("the
     kick is on the &, so the snare ~1 slot later belongs on the same &").
     Off-grid onsets are presented as context but never shifted; their
     position is the geometric finding, not jitter.

Placed onsets get `quantised_time` set to their exact slot time (even at
zero shift) so the emitted MIDI is grid-aligned and the frontend adds no
spurious offset; off-grid onsets keep `quantised_time = None`. The
original `time` / `beat_in_bar` fields are left untouched so per-note
provenance still reports the original detector hit.

Snapping is bounded to within a bar (the per-bar slot range clamps the
DP's feasible window); an onset never crosses a bar boundary, so `bar` /
`beat_in_bar` never need resyncing.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import anthropic

from app.config import settings
from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.geometric_snap import snap_lane
from app.pipeline.llm_util import call_messages_with_refusal_retry

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# 1/48 of a whole note = 12 slots per quarter-note beat. Matches
# `src/midi/from_midi.ts::gridDivision`'s default; the frontend snap
# becomes a no-op on MIDI produced by this stage. Threaded as a parameter
# (default here) so the grid density is not hard-assumed downstream.
SLOTS_PER_BEAT = 12

# Match band for the geometric snap: the farthest (in slots) an onset may
# be pulled to reach a free slot. Beyond it, the onset is left off-grid.
# ±2 slots ≈ 83 ms at 120 BPM, so rejection is rare; it's the only control
# on off-grid promotion now that the cross-instrument cluster pull is gone.
_MATCH_BAND = 2

# Off-grid penalty for the DP: strictly worse than any in-band placement
# (cost <= band^2), so an onset goes off-grid only when no in-band slot is
# free. `(band + 1) ** 2` per the design spec.
_OFF_GRID_PENALTY = float((_MATCH_BAND + 1) ** 2)

# Maximum |shift| accepted from the LLM. Clamped server-side regardless
# of what the model returns; the model isn't a re-quantiser, it's a
# jitter corrector.
_MAX_LLM_SHIFT = 2

# Haiku model id for this stage. Quantisation correction is constrained
# pattern-matching; same tier as the filter stage.
_LLM_MODEL = "claude-haiku-4-5-20251001"

# Per-call token budget. The prompt asks the model to return entries
# ONLY for non-zero shifts (mirror of `filter_llm`'s "rejected_onsets"
# pattern), so the natural response size is proportional to the number
# of jitter corrections needed; not the total onset count. We still
# size the cap from `n_onsets` as defence-in-depth: a model that
# ignores the prompt and emits one entry per onset measured at
# ~13 tokens/entry in the wild (Haiku 4.5: `{"id":1234,"shift":-2},`
# tokenises to ~10–13 tokens depending on id width and field ordering),
# so the per-onset multiplier is set to 16 for headroom. Haiku 4.5
# supports 64K output tokens so the cap stays generous.
_LLM_MAX_TOKENS_PER_ONSET = 16
_LLM_MAX_TOKENS_FLOOR = 8192
_LLM_MAX_TOKENS_OVERHEAD = 1024


def _max_tokens_for(n_onsets: int) -> int:
    """Headroom-aware token budget for the forced-tool quantise response."""
    return max(
        _LLM_MAX_TOKENS_FLOOR,
        n_onsets * _LLM_MAX_TOKENS_PER_ONSET + _LLM_MAX_TOKENS_OVERHEAD,
    )

_QUANTISE_TOOL: dict[str, Any] = {
    "name": "shift_onsets",
    "description": (
        "Return ONLY the onsets that need to move; omit any onset that "
        "should stay where it is. Each entry is a signed integer "
        "1/48-slot shift (negative = earlier; positive = later). Use "
        "surrounding musical context across instruments to decide which "
        "onsets to shift. Bounded |shift| <= 2; anything larger will be "
        "clamped. An empty `shifts` array is the correct answer when "
        "every onset is already correctly placed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "shifts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "minimum": 0},
                        "shift": {
                            "type": "integer",
                            "minimum": -_MAX_LLM_SHIFT,
                            "maximum": _MAX_LLM_SHIFT,
                        },
                    },
                    "required": ["id", "shift"],
                    "additionalProperties": False,
                },
                "description": (
                    "Onsets to shift; identified by the `#N` id shown in "
                    "the prompt. Include ONLY onsets that need to move — "
                    "omit any onset that should stay where it is. Empty "
                    "array means nothing needs shifting. Unknown ids are "
                    "ignored; the server clamps shift to the allowed range."
                ),
            },
        },
        "required": ["shifts"],
        "additionalProperties": False,
    },
}


# ---------- Public API ----------

def quantise_kept_onsets(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    *,
    use_llm: bool = True,
    slots_per_beat: int = SLOTS_PER_BEAT,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run both passes (geometric snap + optional LLM) in order.

    Mutates `kept_by_pitch` candidates in place: placed onsets get
    `quantised_time` set to their exact slot time and `quantised_shift_slots`
    to the integer shift applied; band-rejected onsets get `off_grid = True`
    and keep `quantised_time = None`. Original `time` / `beat_in_bar`
    are left untouched.

    `slots_per_beat` is the grid density (default `SLOTS_PER_BEAT`); the
    single knob for the slot resolution, not hard-assumed across helpers.

    Returns a debug summary suitable for persisting to
    `quantise/shifts.json`. The summary is best-effort: an LLM failure
    degrades to "geometric only" rather than aborting the request.
    """
    geometric_shifts = _geometric_snap(
        kept_by_pitch, structure, slots_per_beat=slots_per_beat
    )
    llm_shifts: dict[tuple[str, int], int] = {}
    llm_status = "skipped"
    if use_llm:
        if cancel_event is not None and cancel_event.is_set():
            llm_status = "cancelled"
        else:
            try:
                llm_shifts = _llm_residual_pass(
                    kept_by_pitch, structure, slots_per_beat=slots_per_beat
                )
                llm_status = "ok"
            except Exception as exc:
                log.warning(
                    "quantise: LLM residual pass failed (%s); "
                    "keeping geometric-only result", exc,
                )
                llm_status = f"error: {exc}"
            else:
                _apply_llm_shifts(
                    kept_by_pitch, structure, llm_shifts,
                    slots_per_beat=slots_per_beat,
                )

    return _build_summary(
        kept_by_pitch=kept_by_pitch,
        geometric_shifts=geometric_shifts,
        llm_shifts=llm_shifts,
        llm_status=llm_status,
        slots_per_beat=slots_per_beat,
    )


# ---------- Geometric pass ----------

def _geometric_snap(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    *,
    slots_per_beat: int,
    band: int = _MATCH_BAND,
    off_grid_penalty: float = _OFF_GRID_PENALTY,
) -> dict[tuple[str, int], int]:
    """Per-lane, per-bar monotonic-injective slot assignment.

    Runs `geometric_snap.snap_lane` on each (lane, bar) group. Placed
    onsets get `quantised_time` set to their exact slot time and
    `quantised_shift_slots` to the integer shift from their natural slot;
    band-rejected onsets get `off_grid = True` and keep `quantised_time =
    None` (so the MIDI emitter falls back to their raw `time`, which the
    frontend then records as the note's sub-slot `offset`).

    Returns `{(pitch, idx): shift_slots}` for the non-zero shifts (for the
    debug summary). Mutates candidates in place.
    """
    shifts: dict[tuple[str, int], int] = {}
    if not structure.bars:
        return shifts

    for pitch, cands in kept_by_pitch.items():
        # Group this lane's in-range onsets by bar; out-of-range onsets
        # (bar < 0) aren't quantised and wouldn't render anyway.
        by_bar: dict[int, list[int]] = defaultdict(list)
        for idx, c in enumerate(cands):
            bar_idx = int(c.bar)
            if 0 <= bar_idx < len(structure.bars):
                by_bar[bar_idx].append(idx)

        for bar_idx, idxs in by_bar.items():
            bar = structure.bars[bar_idx]
            num_beats = max(int(bar.time_signature[0]), 1)
            max_slot = num_beats * slots_per_beat - 1
            span = float(bar.end_time) - float(bar.start_time)
            if span <= 0 or max_slot < 0:
                continue
            slot_span = span / (num_beats * slots_per_beat)

            # naturals must be ascending by time; sort the bar's onsets.
            idxs.sort(key=lambda i, _c=cands: _c[i].time)
            naturals = [
                (float(cands[i].beat_in_bar) - 1.0) * slots_per_beat for i in idxs
            ]
            assigned = snap_lane(
                naturals,
                band=band,
                off_grid_penalty=off_grid_penalty,
                min_slot=0,
                max_slot=max_slot,
            )
            for idx, nat, slot in zip(idxs, naturals, assigned, strict=True):
                c = cands[idx]
                if slot is None:
                    c.off_grid = True
                    c.quantised_time = None
                    c.quantised_shift_slots = None
                    continue
                c.off_grid = False
                c.quantised_time = float(bar.start_time) + slot * slot_span
                delta = slot - round(nat)
                c.quantised_shift_slots = delta
                if delta != 0:
                    shifts[(pitch, idx)] = delta

    return shifts


# ---------- LLM residual pass ----------

def _llm_residual_pass(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
) -> dict[tuple[str, int], int]:
    """One Haiku call across all on-grid kept onsets. Returns the
    (pitch, idx_in_pitch_list) -> shift_slots map the model proposed
    (clamped to ±_MAX_LLM_SHIFT, unknown ids dropped). Off-grid onsets are
    excluded from the shift-target set; their position is geometric truth.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot call the quantise LLM."
        )

    indexed = _index_for_llm(kept_by_pitch, structure, slots_per_beat=slots_per_beat)
    if not indexed:
        return {}

    bar_blocks = _format_for_llm(indexed, structure, slots_per_beat=slots_per_beat)
    initial_sig = structure.initial_time_signature
    prompt = (
        _load_prompt_template()
        .replace("{INITIAL_TEMPO}", f"{structure.initial_tempo:.2f}")
        .replace("{INITIAL_TIME_SIG}", f"{initial_sig[0]}/{initial_sig[1]}")
        .replace("{BAR_COUNT}", str(len(structure.bars)))
        .replace("{ONSET_COUNT}", str(len(indexed)))
        .replace("{SLOTS_PER_BEAT}", str(slots_per_beat))
        .replace("{MAX_SHIFT}", str(_MAX_LLM_SHIFT))
        .replace("{BARS}", bar_blocks)
    )

    max_tokens = _max_tokens_for(len(indexed))
    log.info(
        "Calling quantise LLM model=%s prompt_chars=%d onsets=%d max_tokens=%d",
        _LLM_MODEL, len(prompt), len(indexed), max_tokens,
    )
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = call_messages_with_refusal_retry(
        client,
        {
            "model": _LLM_MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [_QUANTISE_TOOL],
            "tool_choice": {"type": "tool", "name": _QUANTISE_TOOL["name"]},
        },
        base_prompt=prompt,
        purpose="quantise",
    )

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "max_tokens":
        # A forced tool call that hits max_tokens emits unparseable JSON;
        # `block.input` will be empty / partial and we'd silently report
        # "0 shifts" with no other signal. Flag it explicitly so the
        # next time this happens it's obvious from the log.
        log.warning(
            "quantise LLM hit max_tokens (%d) for %d onsets; response will "
            "be truncated and most shifts will be lost. Bump "
            "_LLM_MAX_TOKENS_PER_ONSET in quantise.py.",
            max_tokens, len(indexed),
        )
    raw_shifts = _extract_shifts(response, len(indexed))
    shifts: dict[tuple[str, int], int] = {}
    for llm_id, shift in raw_shifts.items():
        if llm_id < 0 or llm_id >= len(indexed):
            continue
        entry = indexed[llm_id]
        clamped = max(-_MAX_LLM_SHIFT, min(_MAX_LLM_SHIFT, shift))
        if clamped == 0:
            continue
        shifts[(entry.pitch, entry.idx)] = clamped
    log.info(
        "quantise LLM: %d / %d onsets received a non-zero shift",
        len(shifts), len(indexed),
    )
    return shifts


def _apply_llm_shifts(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    llm_shifts: dict[tuple[str, int], int],
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
) -> None:
    """Apply the LLM's per-onset slot shifts, atomically per (lane, bar).

    The geometric snap leaves each lane+bar's on-grid onsets on distinct,
    strictly-increasing slots. The LLM proposes independent ±shifts, which
    applied naively could collide two onsets onto one slot or reorder them.
    So a group's shifts are applied only if the resulting slots stay
    strictly increasing and in-bar; otherwise the group keeps its geometric
    placement (preserving the snap's monotonic-injective invariant).
    Off-grid onsets are never shifted.
    """
    if not llm_shifts:
        return

    for pitch, cands in kept_by_pitch.items():
        # Group on-grid onsets by bar (same shape as `_geometric_snap`).
        by_bar: dict[int, list[int]] = defaultdict(list)
        for idx, c in enumerate(cands):
            if c.off_grid:
                continue
            bar_idx = int(c.bar)
            if 0 <= bar_idx < len(structure.bars):
                by_bar[bar_idx].append(idx)

        for bar_idx, idxs in by_bar.items():
            bar = structure.bars[bar_idx]
            num_beats = max(int(bar.time_signature[0]), 1)
            slot_span = (float(bar.end_time) - float(bar.start_time)) / (
                num_beats * slots_per_beat
            )
            if slot_span <= 0:
                continue
            max_slot = num_beats * slots_per_beat - 1

            idxs.sort(key=lambda i, _c=cands: _c[i].time)
            # Current geometric slot + the LLM's delta -> intended slot.
            # `quantised_time` is set for every placed onset; fall back to
            # the raw `time` snap defensively.
            plan: list[tuple[int, int, int]] = []  # (idx, delta, intended_slot)
            for i in idxs:
                c = cands[i]
                base_time = c.quantised_time if c.quantised_time is not None else c.time
                current_slot = round((base_time - float(bar.start_time)) / slot_span)
                delta = llm_shifts.get((pitch, i), 0)
                plan.append((i, delta, current_slot + delta))

            intended = [p[2] for p in plan]
            in_bounds = all(0 <= s <= max_slot for s in intended)
            increasing = all(
                intended[k] < intended[k + 1] for k in range(len(intended) - 1)
            )
            if not (in_bounds and increasing):
                if any(delta for _, delta, _ in plan):
                    log.info(
                        "quantise: LLM shifts for lane %r bar %d would break "
                        "slot order/injectivity; keeping geometric placement",
                        pitch, bar_idx,
                    )
                continue

            for i, delta, new_slot in plan:
                if delta == 0:
                    continue
                c = cands[i]
                c.quantised_time = float(bar.start_time) + new_slot * slot_span
                c.quantised_shift_slots = (c.quantised_shift_slots or 0) + delta


# ---------- Helpers ----------

def _index_for_llm(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
) -> list[_LlmEntry]:
    """Build a stable LLM-facing list of `(bar, slot, pitch)` entries.

    Ordering: by `(bar, current_slot, pitch)` so a human can cross-read
    the prompt and the response. Out-of-range and off-grid onsets are
    excluded (off-grid onsets are geometric findings, not jitter to
    correct, so they're never offered to the model as shift targets).
    """
    entries: list[_LlmEntry] = []
    for pitch in sorted(kept_by_pitch.keys()):
        for idx, c in enumerate(kept_by_pitch[pitch]):
            if c.off_grid:
                continue
            bar_idx = int(c.bar)
            if bar_idx < 0 or bar_idx >= len(structure.bars):
                continue
            bar = structure.bars[bar_idx]
            current_time = c.quantised_time if c.quantised_time is not None else c.time
            num_beats = max(int(bar.time_signature[0]), 1)
            slot_span = (bar.end_time - bar.start_time) / (num_beats * slots_per_beat)
            if slot_span <= 0:
                continue
            current_slot = round((current_time - bar.start_time) / slot_span)
            current_slot = max(0, min(num_beats * slots_per_beat - 1, current_slot))
            entries.append(_LlmEntry(pitch=pitch, idx=idx, bar=bar_idx, slot=current_slot))
    entries.sort(key=lambda e: (e.bar, e.slot, e.pitch))
    return entries


class _LlmEntry:
    __slots__ = ("pitch", "idx", "bar", "slot")

    def __init__(self, pitch: str, idx: int, bar: int, slot: int) -> None:
        self.pitch = pitch
        self.idx = idx
        self.bar = bar
        self.slot = slot


def _format_for_llm(
    indexed: list[_LlmEntry],
    structure: BeatStructure,
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
) -> str:
    """One block per bar: header + indexed onset rows grouped by slot."""
    by_bar: dict[int, list[tuple[int, _LlmEntry]]] = {}
    for llm_id, e in enumerate(indexed):
        by_bar.setdefault(e.bar, []).append((llm_id, e))

    blocks: list[str] = []
    for bar in structure.bars:
        header = (
            f"Bar {bar.index} "
            f"[{bar.time_signature[0]}/{bar.time_signature[1]}, "
            f"{bar.tempo_bpm:.1f} BPM, feel={bar.feel}]:"
        )
        rows = [header]
        bar_entries = by_bar.get(bar.index, [])
        if not bar_entries:
            rows.append("  (no onsets)")
            blocks.append("\n".join(rows))
            continue
        # Group by slot for compactness.
        by_slot: dict[int, list[tuple[int, _LlmEntry]]] = {}
        for llm_id, e in bar_entries:
            by_slot.setdefault(e.slot, []).append((llm_id, e))
        for slot in sorted(by_slot.keys()):
            slot_entries = by_slot[slot]
            beat_label = _slot_label(slot, slots_per_beat)
            rendered = " ".join(
                f"#{llm_id}({e.pitch})" for llm_id, e in slot_entries
            )
            rows.append(f"  slot {slot:>2} {beat_label}: {rendered}")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def _slot_label(slot: int, slots_per_beat: int) -> str:
    """Human-readable label for a slot, e.g. "(beat 2)" or "(& of 1)".

    Positions are fractions of `slots_per_beat`, so the labels track the
    active grid density rather than a hardcoded 12. Named positions are
    emitted only when they land on an integer slot for this density
    (always so at 12 = 1/48); otherwise a generic fallback names the
    whole-note subdivision.
    """
    beat = slot // slots_per_beat + 1  # 1-indexed beat
    within = slot % slots_per_beat
    if within == 0:
        return f"(beat {beat})"
    if within == slots_per_beat // 2:
        return f"(& of {beat})"
    if within == slots_per_beat // 4:
        return f"(e of {beat})"
    if within == 3 * slots_per_beat // 4:
        return f"(a of {beat})"
    if slots_per_beat % 3 == 0 and within == slots_per_beat // 3:
        return f"(trip-2 of {beat})"
    if slots_per_beat % 3 == 0 and within == 2 * slots_per_beat // 3:
        return f"(trip-3 of {beat})"
    return f"(1/{4 * slots_per_beat} +{within} of {beat})"


def _extract_shifts(
    response: anthropic.types.Message, n: int
) -> dict[int, int]:
    """Pull `shifts` from the forced tool call. Returns `{id: shift}`,
    ignoring entries that aren't well-formed; clamping happens at the
    call site (so the dict here can carry the raw values for logging)."""
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != _QUANTISE_TOOL["name"]:
            continue
        raw = block.input.get("shifts", [])
        if not isinstance(raw, list):
            log.warning(
                "quantise: tool returned non-list shifts (%s); "
                "treating as no shifts", type(raw).__name__,
            )
            return {}
        out: dict[int, int] = {}
        for row in raw:
            if not isinstance(row, dict):
                continue
            try:
                rid = int(row.get("id"))
                shift = int(row.get("shift"))
            except (TypeError, ValueError):
                continue
            if 0 <= rid < n:
                out[rid] = shift
        return out
    log.warning("quantise: no tool_use block in response; no shifts applied")
    return {}


def _load_prompt_template() -> str:
    return (PROMPT_DIR / "quantise_onsets.md").read_text(encoding="utf-8")


def _build_summary(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    geometric_shifts: dict[tuple[str, int], int],
    llm_shifts: dict[tuple[str, int], int],
    llm_status: str,
    slots_per_beat: int = SLOTS_PER_BEAT,
) -> dict[str, Any]:
    """Compact JSON-friendly summary for `quantise/shifts.json`."""
    per_pitch: dict[str, list[dict[str, Any]]] = {}
    off_grid_total = 0
    for pitch, cands in kept_by_pitch.items():
        rows: list[dict[str, Any]] = []
        for idx, c in enumerate(cands):
            if c.off_grid:
                off_grid_total += 1
            geo = geometric_shifts.get((pitch, idx), 0)
            llm = llm_shifts.get((pitch, idx), 0)
            if geo == 0 and llm == 0 and not c.off_grid:
                continue
            rows.append({
                "idx": idx,
                "bar": int(c.bar),
                "beat_in_bar": float(c.beat_in_bar),
                "original_time": float(c.time),
                "quantised_time": (
                    float(c.quantised_time) if c.quantised_time is not None else None
                ),
                "off_grid": bool(c.off_grid),
                "geometric_shift": geo,
                "llm_shift": llm,
                "total_shift": (c.quantised_shift_slots or 0),
            })
        if rows:
            per_pitch[pitch] = rows
    return {
        "geometric_shifted": sum(1 for v in geometric_shifts.values() if v),
        "llm_shifted": sum(1 for v in llm_shifts.values() if v),
        "off_grid": off_grid_total,
        "llm_status": llm_status,
        "match_band": _MATCH_BAND,
        "max_llm_shift": _MAX_LLM_SHIFT,
        "slots_per_beat": slots_per_beat,
        "per_pitch": per_pitch,
    }
