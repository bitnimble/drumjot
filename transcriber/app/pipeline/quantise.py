"""Optional `quantise` pipeline stage: snap kept onsets to a 1/48 grid.

Runs between `filter` and `transcribe`. Two passes:

  1. **Deterministic joint snap.** Each kept onset's `beat_in_bar` is
     rounded to its nearest 1/48 slot. Then onsets across pitches that
     fired close in time (within `_CLUSTER_WINDOW_S`) and snapped to
     *different* slots within the same bar are pulled onto the
     highest-beat-hierarchy slot present in the cluster, so a kick + snare
     that fired ~8 ms apart but rounded to slots 11 and 12 both end up at
     slot 12. Bounded by `_MAX_DETERMINISTIC_SHIFT` slots per onset.

  2. **LLM residual pass** (`quantise_with_llm`). Sends every kept onset
     with its current 1/48 slot to Haiku in one joint call. The model
     returns an integer slot shift per onset, clamped server-side to
     `_MAX_LLM_SHIFT` (= 2). The joint call is intentional: this is
     where *cross-instrument* musical reasoning earns its keep ("the
     kick is on the &, so the snare ~1/48 later belongs on the same &").

Both passes write to `OnsetCandidate.quantised_time` (and add up into
`quantised_shift_slots`), leaving the original `time` / `beat_in_bar`
fields untouched. `onsets_to_midi_bytes` reads `quantised_time` when
present, so per-note provenance still reports the original detector
hit while the rendered MIDI uses the corrected positions.

Shifts that would cross a bar boundary are dropped; the cap (±2 slots
deterministic, ±2 slots LLM, so ±4 total worst-case) makes that
exceedingly rare, and handling bar crossings would require resyncing
`bar` / `beat_in_bar` which we'd rather not do for a debug-friendly
side-channel field.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import anthropic

from app.config import settings
from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.llm_util import call_messages_with_refusal_retry

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# 1/48 of a whole note = 12 slots per quarter-note beat. Matches
# `src/midi/from_midi.ts::gridDivision`'s default; the frontend snap
# becomes a no-op on MIDI produced by this stage.
SLOTS_PER_BEAT = 12

# Cross-instrument cluster window. At 120 BPM one slot is ~41 ms, so
# 60 ms catches typical onset-detector jitter (one slot's worth) without
# being aggressive enough to merge genuinely-separate hits.
_CLUSTER_WINDOW_S = 0.060

# Maximum |shift| the deterministic joint-snap pass will apply. ±2 slots
# is ample for jitter-class corrections; anything bigger is structural
# and belongs to the LLM (or to a human).
_MAX_DETERMINISTIC_SHIFT = 2

# Maximum |shift| accepted from the LLM. Clamped server-side regardless
# of what the model returns; the model isn't a re-quantiser, it's a
# jitter corrector.
_MAX_LLM_SHIFT = 2

# Haiku model id for this stage. Quantisation correction is constrained
# pattern-matching; same tier as the filter stage.
_LLM_MODEL = "claude-haiku-4-5-20251001"

# Conservative per-call token budget. The response is one int per onset
# (+ JSON scaffolding), so even a 1000-onset chart fits easily.
_LLM_MAX_TOKENS = 4096

# Beat-hierarchy weights for `_slot_weight`. Higher = "stronger" slot,
# i.e. the snap target a cluster prefers. Tie-breaking goes downbeat >
# beat > 8th > triplet > 16th > arbitrary 48th; matches what notation
# software does and what most drummers would call the "stronger" position.
_SLOT_DOWNBEAT_WEIGHT = 100
_SLOT_BEAT_WEIGHT = 80
_SLOT_OFFBEAT_8TH_WEIGHT = 60
_SLOT_TRIPLET_WEIGHT = 50
_SLOT_16TH_WEIGHT = 40
_SLOT_48TH_WEIGHT = 10


_QUANTISE_TOOL: dict[str, Any] = {
    "name": "shift_onsets",
    "description": (
        "For each onset shown, return an integer 1/48-slot shift "
        "(negative = earlier, positive = later, 0 = leave it). Use "
        "surrounding musical context across instruments to decide. "
        "Bounded |shift| <= 2; anything larger will be clamped."
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
                    "One entry per onset id shown in the prompt. Unknown / "
                    "missing ids are ignored; the server clamps shift to "
                    "the allowed range."
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
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run both passes (deterministic + optional LLM) in order.

    Mutates `kept_by_pitch` candidates in place: sets `quantised_time`
    to the snap-corrected absolute time and `quantised_shift_slots` to
    the total integer shift applied. Original `time` / `beat_in_bar`
    are left untouched.

    Returns a debug summary suitable for persisting to
    `quantise/shifts.json`. The summary is best-effort: an LLM failure
    degrades to "deterministic only" rather than aborting the request.
    """
    deterministic_shifts = _deterministic_joint_snap(kept_by_pitch, structure)
    llm_shifts: dict[tuple[str, int], int] = {}
    llm_status = "skipped"
    if use_llm:
        if cancel_event is not None and cancel_event.is_set():
            llm_status = "cancelled"
        else:
            try:
                llm_shifts = _llm_residual_pass(kept_by_pitch, structure)
                llm_status = "ok"
            except Exception as exc:
                log.warning(
                    "quantise: LLM residual pass failed (%s); "
                    "keeping deterministic-only result", exc,
                )
                llm_status = f"error: {exc}"
            else:
                _apply_llm_shifts(kept_by_pitch, structure, llm_shifts)

    return _build_summary(
        kept_by_pitch=kept_by_pitch,
        deterministic_shifts=deterministic_shifts,
        llm_shifts=llm_shifts,
        llm_status=llm_status,
    )


# ---------- Deterministic pass ----------

def _deterministic_joint_snap(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
) -> dict[tuple[str, int], int]:
    """Snap each onset to its 1/48 slot, then pull cross-instrument
    clusters onto a common slot (highest beat-hierarchy weight wins).

    Returns `{(pitch, idx_in_pitch_list): shift_slots}` for inspection.
    Mutates each shifted onset's `quantised_time` / `quantised_shift_slots`
    in place.
    """
    if not structure.bars:
        return {}

    # Flat list of (pitch, idx, c, bar, initial_slot, slot_seconds) for
    # in-range onsets only. Out-of-range (bar < 0) onsets aren't
    # quantised; they wouldn't render anyway.
    entries: list[_Entry] = []
    for pitch, cands in kept_by_pitch.items():
        for idx, c in enumerate(cands):
            bar_idx = int(c.bar)
            if bar_idx < 0 or bar_idx >= len(structure.bars):
                continue
            bar = structure.bars[bar_idx]
            slot, slot_seconds = _initial_slot_for(c, bar)
            entries.append(
                _Entry(
                    pitch=pitch,
                    idx=idx,
                    candidate=c,
                    bar_idx=bar_idx,
                    initial_slot=slot,
                    slot_seconds=slot_seconds,
                )
            )

    if not entries:
        return {}

    entries.sort(key=lambda e: e.candidate.time)
    target_slot: dict[int, int] = {}  # id(entry) -> chosen slot
    # First pass: each onset's default target is its own snapped slot.
    for e in entries:
        target_slot[id(e)] = e.initial_slot

    # Cluster across pitches by time proximity. Within a cluster, group
    # by bar (cross-bar onsets in the same window are very unusual but
    # mathematically possible right at a bar boundary; we don't try to
    # merge across bars). For each bar's sub-cluster, pick the highest-
    # hierarchy slot present and pull others onto it; subject to the
    # per-onset deterministic shift cap.
    i = 0
    while i < len(entries):
        j = i + 1
        while (
            j < len(entries)
            and entries[j].candidate.time - entries[j - 1].candidate.time
            <= _CLUSTER_WINDOW_S
        ):
            j += 1
        cluster = entries[i:j]
        i = j
        if len(cluster) < 2:
            continue
        by_bar: dict[int, list[_Entry]] = {}
        for e in cluster:
            by_bar.setdefault(e.bar_idx, []).append(e)
        for bar_idx, sub in by_bar.items():
            if len(sub) < 2:
                continue
            num_beats = int(structure.bars[bar_idx].time_signature[0])
            present_slots = {target_slot[id(e)] for e in sub}
            if len(present_slots) < 2:
                continue  # already aligned
            # Pick the slot with the highest weight. Tie-breaker: the
            # slot that needs the smallest total |shift| across the
            # cluster, then the smaller slot number for stability.
            # Default-arg binding pins loop vars onto the lambda so
            # ruff's B023 stays quiet (lambda is called inside this
            # iteration anyway, so it's defensive-only).
            cluster_slots = [target_slot[id(e)] for e in sub]
            chosen = min(
                present_slots,
                key=lambda slot, nb=num_beats, cs=cluster_slots: (
                    -_slot_weight(slot, nb),
                    sum(abs(slot - s) for s in cs),
                    slot,
                ),
            )
            for e in sub:
                current = target_slot[id(e)]
                delta = chosen - current
                if abs(delta) > _MAX_DETERMINISTIC_SHIFT:
                    continue
                target_slot[id(e)] = chosen

    # Apply: any entry whose target differs from its initial slot gets
    # quantised_time written.
    shifts: dict[tuple[str, int], int] = {}
    for e in entries:
        final = target_slot[id(e)]
        delta = final - e.initial_slot
        if delta == 0:
            continue
        bar = structure.bars[e.bar_idx]
        num_beats = max(int(bar.time_signature[0]), 1)
        slot_span = (bar.end_time - bar.start_time) / (num_beats * SLOTS_PER_BEAT)
        if slot_span <= 0:
            continue
        e.candidate.quantised_time = bar.start_time + final * slot_span
        e.candidate.quantised_shift_slots = delta
        shifts[(e.pitch, e.idx)] = delta

    return shifts


# ---------- LLM residual pass ----------

def _llm_residual_pass(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
) -> dict[tuple[str, int], int]:
    """One Haiku call across all kept onsets. Returns the
    (pitch, idx_in_pitch_list) -> shift_slots map the model proposed
    (clamped to ±_MAX_LLM_SHIFT, unknown ids dropped).
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot call the quantise LLM."
        )

    indexed = _index_for_llm(kept_by_pitch, structure)
    if not indexed:
        return {}

    bar_blocks = _format_for_llm(indexed, structure)
    initial_sig = structure.initial_time_signature
    prompt = (
        _load_prompt_template()
        .replace("{INITIAL_TEMPO}", f"{structure.initial_tempo:.2f}")
        .replace("{INITIAL_TIME_SIG}", f"{initial_sig[0]}/{initial_sig[1]}")
        .replace("{BAR_COUNT}", str(len(structure.bars)))
        .replace("{ONSET_COUNT}", str(len(indexed)))
        .replace("{SLOTS_PER_BEAT}", str(SLOTS_PER_BEAT))
        .replace("{MAX_SHIFT}", str(_MAX_LLM_SHIFT))
        .replace("{BARS}", bar_blocks)
    )

    log.info(
        "Calling quantise LLM model=%s prompt_chars=%d onsets=%d",
        _LLM_MODEL, len(prompt), len(indexed),
    )
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = call_messages_with_refusal_retry(
        client,
        {
            "model": _LLM_MODEL,
            "max_tokens": _LLM_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [_QUANTISE_TOOL],
            "tool_choice": {"type": "tool", "name": _QUANTISE_TOOL["name"]},
        },
        base_prompt=prompt,
        purpose="quantise",
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
) -> None:
    for (pitch, idx), delta in llm_shifts.items():
        cands = kept_by_pitch.get(pitch)
        if cands is None or idx < 0 or idx >= len(cands):
            continue
        c = cands[idx]
        bar_idx = int(c.bar)
        if bar_idx < 0 or bar_idx >= len(structure.bars):
            continue
        bar = structure.bars[bar_idx]
        num_beats = max(int(bar.time_signature[0]), 1)
        slot_span = (bar.end_time - bar.start_time) / (num_beats * SLOTS_PER_BEAT)
        if slot_span <= 0:
            continue
        # Base slot = current quantised slot (deterministic pass may
        # already have moved it). Falls back to the initial snap of `time`
        # for onsets the deterministic pass didn't touch.
        current_time = c.quantised_time if c.quantised_time is not None else c.time
        current_slot = round((current_time - bar.start_time) / slot_span)
        new_slot = current_slot + delta
        max_slot = num_beats * SLOTS_PER_BEAT - 1
        if new_slot < 0 or new_slot > max_slot:
            # Would cross a bar boundary; skip rather than re-bar.
            continue
        c.quantised_time = bar.start_time + new_slot * slot_span
        existing = c.quantised_shift_slots or 0
        c.quantised_shift_slots = existing + delta


# ---------- Helpers ----------

class _Entry:
    __slots__ = ("pitch", "idx", "candidate", "bar_idx", "initial_slot", "slot_seconds")

    def __init__(
        self,
        pitch: str,
        idx: int,
        candidate: OnsetCandidate,
        bar_idx: int,
        initial_slot: int,
        slot_seconds: float,
    ) -> None:
        self.pitch = pitch
        self.idx = idx
        self.candidate = candidate
        self.bar_idx = bar_idx
        self.initial_slot = initial_slot
        self.slot_seconds = slot_seconds


def _initial_slot_for(c: OnsetCandidate, bar: Any) -> tuple[int, float]:
    """Round `c.beat_in_bar` to its nearest 1/48 slot within `bar`.

    Returns `(slot, slot_seconds)` where `slot` is 0..(num_beats*12 - 1)
    and `slot_seconds` is the audio duration of one slot in this bar.
    """
    num_beats = max(int(bar.time_signature[0]), 1)
    max_slot = num_beats * SLOTS_PER_BEAT - 1
    # `beat_in_bar` is 1-indexed: 1.000 = beat 1 (slot 0), 1.500 = slot 6,
    # 2.000 = slot 12, etc. Direct: slot = round((beat - 1) * 12).
    slot = round((float(c.beat_in_bar) - 1.0) * SLOTS_PER_BEAT)
    slot = max(0, min(max_slot, slot))
    span = float(bar.end_time) - float(bar.start_time)
    slot_seconds = span / (num_beats * SLOTS_PER_BEAT) if span > 0 else 0.0
    return slot, slot_seconds


def _slot_weight(slot: int, num_beats: int) -> int:
    """Beat-hierarchy weight for a slot within a bar of `num_beats` beats.

    Generic across meters; doesn't try to be feel-aware (that's the LLM's
    job). Higher = "stronger" slot.
    """
    if slot == 0:
        return _SLOT_DOWNBEAT_WEIGHT
    if slot % SLOTS_PER_BEAT == 0:
        return _SLOT_BEAT_WEIGHT
    if slot % (SLOTS_PER_BEAT // 2) == 0:
        return _SLOT_OFFBEAT_8TH_WEIGHT
    # Triplet positions inside a beat: slots 4 and 8 (== beat * 12 + {4,8}).
    if slot % SLOTS_PER_BEAT in (4, 8):
        return _SLOT_TRIPLET_WEIGHT
    if slot % (SLOTS_PER_BEAT // 4) == 0:
        return _SLOT_16TH_WEIGHT
    return _SLOT_48TH_WEIGHT


def _index_for_llm(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
) -> list[_LlmEntry]:
    """Build a stable LLM-facing list of `(bar, slot, pitch)` entries.

    Ordering: by `(bar, current_slot, pitch)` so a human can cross-read
    the prompt and the response. Out-of-range onsets are excluded.
    """
    entries: list[_LlmEntry] = []
    for pitch in sorted(kept_by_pitch.keys()):
        for idx, c in enumerate(kept_by_pitch[pitch]):
            bar_idx = int(c.bar)
            if bar_idx < 0 or bar_idx >= len(structure.bars):
                continue
            bar = structure.bars[bar_idx]
            current_time = c.quantised_time if c.quantised_time is not None else c.time
            num_beats = max(int(bar.time_signature[0]), 1)
            slot_span = (bar.end_time - bar.start_time) / (num_beats * SLOTS_PER_BEAT)
            if slot_span <= 0:
                continue
            current_slot = round((current_time - bar.start_time) / slot_span)
            current_slot = max(0, min(num_beats * SLOTS_PER_BEAT - 1, current_slot))
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
            beat_label = _slot_label(slot)
            rendered = " ".join(
                f"#{llm_id}({e.pitch})" for llm_id, e in slot_entries
            )
            rows.append(f"  slot {slot:>2} {beat_label}: {rendered}")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def _slot_label(slot: int) -> str:
    """Human-readable label for a 1/48 slot, e.g. "(beat 2)" or "(8th of 1)"."""
    beat = slot // SLOTS_PER_BEAT + 1  # 1-indexed beat
    within = slot % SLOTS_PER_BEAT
    if within == 0:
        return f"(beat {beat})"
    if within == SLOTS_PER_BEAT // 2:
        return f"(& of {beat})"
    if within == SLOTS_PER_BEAT // 4:
        return f"(e of {beat})"
    if within == 3 * SLOTS_PER_BEAT // 4:
        return f"(a of {beat})"
    if within == 4:
        return f"(trip-2 of {beat})"
    if within == 8:
        return f"(trip-3 of {beat})"
    return f"(48th +{within} of {beat})"


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
    deterministic_shifts: dict[tuple[str, int], int],
    llm_shifts: dict[tuple[str, int], int],
    llm_status: str,
) -> dict[str, Any]:
    """Compact JSON-friendly summary for `quantise/shifts.json`."""
    per_pitch: dict[str, list[dict[str, Any]]] = {}
    for pitch, cands in kept_by_pitch.items():
        rows: list[dict[str, Any]] = []
        for idx, c in enumerate(cands):
            det = deterministic_shifts.get((pitch, idx), 0)
            llm = llm_shifts.get((pitch, idx), 0)
            if det == 0 and llm == 0:
                continue
            rows.append({
                "idx": idx,
                "bar": int(c.bar),
                "beat_in_bar": float(c.beat_in_bar),
                "original_time": float(c.time),
                "quantised_time": (
                    float(c.quantised_time) if c.quantised_time is not None else None
                ),
                "deterministic_shift": det,
                "llm_shift": llm,
                "total_shift": (c.quantised_shift_slots or 0),
            })
        if rows:
            per_pitch[pitch] = rows
    return {
        "deterministic_shifted": sum(1 for v in deterministic_shifts.values() if v),
        "llm_shifted": sum(1 for v in llm_shifts.values() if v),
        "llm_status": llm_status,
        "max_deterministic_shift": _MAX_DETERMINISTIC_SHIFT,
        "max_llm_shift": _MAX_LLM_SHIFT,
        "slots_per_beat": SLOTS_PER_BEAT,
        "per_pitch": per_pitch,
    }
