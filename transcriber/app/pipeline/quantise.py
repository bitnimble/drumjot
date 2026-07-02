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
original detector hit is preserved on `c.time`; per-note provenance
uses that for the "Detected" stage.

The per-bar DP's feasible window is bounded to `[0, max_slot]`, but a
forward cross-bar pre-pass in `_geometric_snap` first reassigns any
onset whose natural slot rounds past its bar's last slot (the common
"early downbeat" case: a hit detected just before a downbeat by the beat
tracker still landed in the previous bar). On reassignment `c.bar` /
`c.beat_in_bar` are rewritten to the next bar's slot frame so every
downstream pass (envelope re-snap, musical-grid snap, LLM residual,
MIDI render) operates on the correct bar. Backward cross-bar isn't
needed: the beat tracker guarantees `beat_in_bar >= 1.0`, so naturals
are never negative.

This file is the stable public facade. The implementation is split across
cohesive focused modules; everything is re-exported here so existing call
sites (`from app.pipeline.quantise import …`) are unchanged:

- `quantise_config` - shared constants + the LLM tool schema + token budget.
- `quantise_apply`  - the shared shift applier + slot-geometry helpers.
- `quantise_grid`   - the deterministic musical-grid inference + snap pass.
- `quantise_llm`    - the Haiku residual pass (indexing / windowing /
                      prompt formatting / forced-tool call / extraction).

The geometric snap, the per-note envelope re-snap and the orchestration
(`quantise_kept_onsets` + the debug summary) stay here.
"""
# ruff: noqa: F401  -- this module is a re-export facade; the "unused" imports
# are the point (they keep `from app.pipeline.quantise import …` call sites intact).
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any

from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.envelope import OnsetEnvelope
from app.pipeline.geometric_snap import snap_lane
from app.pipeline.quantise_apply import (
    _apply_llm_shifts,
    _current_slot,
    _resolve_cross_bar_target,
)
from app.pipeline.quantise_config import (
    _CONTEXT_BARS,
    _ENV_RESNAP_DOMINANCE,
    _ENV_RESNAP_FLOOR_FRAC,
    _ENV_RESNAP_TOLERANCE,
    _GRID_COMPLEXITY_PENALTY,
    _GRID_DECISIVE_MARGIN,
    _GRID_MIN_ONSETS,
    _GRID_SNAP_TOLERANCE,
    _LLM_MAX_TOKENS_FLOOR,
    _LLM_MAX_TOKENS_OVERHEAD,
    _LLM_MAX_TOKENS_PER_ONSET,
    _LLM_MODEL,
    _MATCH_BAND,
    _MAX_BARS_PER_WINDOW,
    _MAX_LLM_SHIFT,
    _MAX_PARALLEL_CHUNKS,
    _OFF_GRID_PENALTY,
    _QUANTISE_TOOL,
    _TARGET_ONSETS_PER_WINDOW,
    PROMPT_DIR,
    SLOTS_PER_BEAT,
    _max_tokens_for,
)
from app.pipeline.quantise_grid import (
    _candidate_grids,
    _circular_dist,
    _infer_grid,
    _musical_grid_snap,
    _nearest_grid_slot,
)
from app.pipeline.quantise_llm import (
    _build_windows,
    _call_window,
    _extract_shifts,
    _format_window,
    _index_for_llm,
    _llm_residual_pass,
    _LlmEntry,
    _load_prompt_template,
    _residual_tag,
    _slot_label,
    _Window,
)

log = logging.getLogger(__name__)


# ---------- Public API ----------

def _snapshot_total_shifts(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
) -> dict[tuple[str, int], int]:
    """Snapshot every candidate's current `quantised_shift_slots` so a
    subsequent pass's actual contribution can be recovered by diffing.
    Treats `None` (off-grid / unset) as `0` so the diff is well-defined
    for every candidate index."""
    return {
        (pitch, idx): (c.quantised_shift_slots or 0)
        for pitch, cands in kept_by_pitch.items()
        for idx, c in enumerate(cands)
    }


def _record_pass_contributions(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    snapshot: dict[tuple[str, int], int],
    field: str,
) -> None:
    """Write each candidate's per-pass contribution onto the named field,
    derived as `current_total_shift - snapshot_total_shift`. Off-grid
    candidates are skipped (the pass never touched them; leaves the
    field `None`). Result captures what was *actually* applied, after
    the monotonic-injective guard in `_apply_llm_shifts` ran, a pass
    whose proposed shifts were rejected ends up recorded as `0`, not
    the proposed value."""
    for pitch, cands in kept_by_pitch.items():
        for idx, c in enumerate(cands):
            if c.off_grid:
                continue
            prev = snapshot.get((pitch, idx), 0)
            cur = c.quantised_shift_slots or 0
            setattr(c, field, cur - prev)


def quantise_kept_onsets(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    *,
    use_llm: bool = True,
    use_grid: bool = True,
    envelopes: dict[str, OnsetEnvelope] | None = None,
    slots_per_beat: int = SLOTS_PER_BEAT,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run the quantise passes in order: geometric snap, per-note envelope
    re-snap, deterministic musical-grid snap, then the optional LLM pass.

    Mutates `kept_by_pitch` candidates in place: placed onsets get
    `quantised_time` set to their exact slot time and `quantised_shift_slots`
    to the integer shift applied (summed across passes); band-rejected
    onsets get `off_grid = True` and keep `quantised_time = None`. Original
    `time` / `beat_in_bar` are left untouched.

    `slots_per_beat` is the grid density (default `SLOTS_PER_BEAT`); the
    single knob for the slot resolution, not hard-assumed across helpers.
    `use_grid` / `use_llm` gate the grid and LLM passes independently;
    `envelopes` (per-pitch `OnsetEnvelope`s, keyed to `kept_by_pitch`'s
    pitches) enables the envelope re-snap when supplied.

    Returns a debug summary suitable for persisting to
    `quantise/shifts.json`. The summary is best-effort: an LLM failure
    degrades to "geometric + grid only" rather than aborting the request.
    """
    geometric_shifts = _geometric_snap(
        kept_by_pitch, structure, slots_per_beat=slots_per_beat
    )
    # Record per-pass contributions on each candidate so the per-note
    # debug popup can attribute every quantise shift to its specific
    # pass instead of one collapsed sum. Geometric writes
    # `quantised_shift_slots` directly (no prior pass), so mirroring is
    # straightforward: off-grid → None (couldn't place; pass effectively
    # didn't apply), placed → whatever the pass set. Later passes accumulate
    # into `quantised_shift_slots` via `_apply_llm_shifts`; we snapshot
    # before each one and diff after to capture what was *actually*
    # applied (the monotonic-injective guard inside `_apply_llm_shifts`
    # can reject a whole group, in which case the proposed-shifts dict
    # overstates the contribution).
    for cands in kept_by_pitch.values():
        for c in cands:
            c.geometric_shift_slots = c.quantised_shift_slots

    envelope_shifts: dict[tuple[str, int], int] = {}
    if envelopes:
        snapshot = _snapshot_total_shifts(kept_by_pitch)
        envelope_shifts = _envelope_snap(
            kept_by_pitch, structure, envelopes, slots_per_beat=slots_per_beat
        )
        _record_pass_contributions(kept_by_pitch, snapshot, "envelope_shift_slots")
    grid_shifts: dict[tuple[str, int], int] = {}
    if use_grid:
        snapshot = _snapshot_total_shifts(kept_by_pitch)
        grid_shifts = _musical_grid_snap(
            kept_by_pitch, structure, slots_per_beat=slots_per_beat
        )
        _record_pass_contributions(kept_by_pitch, snapshot, "grid_shift_slots")
    llm_shifts: dict[tuple[str, int], int] = {}
    llm_status = "skipped"
    if use_llm:
        if cancel_event is not None and cancel_event.is_set():
            llm_status = "cancelled"
        else:
            try:
                llm_shifts, llm_status = _llm_residual_pass(
                    kept_by_pitch, structure,
                    slots_per_beat=slots_per_beat,
                    cancel_event=cancel_event,
                )
            except Exception as exc:
                log.warning(
                    "quantise: LLM residual pass failed (%s); "
                    "keeping geometric-only result", exc,
                )
                llm_status = f"error: {exc}"
            else:
                snapshot = _snapshot_total_shifts(kept_by_pitch)
                _apply_llm_shifts(
                    kept_by_pitch, structure, llm_shifts,
                    slots_per_beat=slots_per_beat,
                )
                _record_pass_contributions(kept_by_pitch, snapshot, "llm_shift_slots")

    return _build_summary(
        kept_by_pitch=kept_by_pitch,
        geometric_shifts=geometric_shifts,
        envelope_shifts=envelope_shifts,
        grid_shifts=grid_shifts,
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

    A forward cross-bar pre-pass runs first: any onset whose natural slot
    rounds past its bar's last slot is reassigned to the next bar's
    corresponding slot before the DP groups by bar, so an "early
    downbeat" hit lands on the downbeat instead of being clamped to the
    previous bar's last slot. `c.bar` / `c.beat_in_bar` are rewritten on
    reassignment so every downstream pass operates on the new bar.

    Returns `{(pitch, idx): shift_slots}` for the non-zero shifts (for the
    debug summary). Mutates candidates in place.
    """
    shifts: dict[tuple[str, int], int] = {}
    if not structure.bars:
        return shifts

    # Forward cross-bar reassignment. The beat tracker puts each onset
    # in the bar whose downbeat it has just passed, so a hit detected
    # ~1 slot before the next downbeat lands in the *previous* bar with
    # beat_in_bar ≈ num_beats + small (natural slot ≈ max_slot + 1).
    # Without this pre-pass the per-bar DP clamps it to the last slot of
    # the original bar (musically wrong); with it, the onset is moved
    # into the next bar's slot frame and the DP places it there.
    # Backward overflow can't arise (beat_in_bar >= 1.0 by construction:
    # the beat tracker only assigns an onset to a bar whose downbeat
    # it's already crossed, and `_apply_llm_shifts`'s cross-bar moves
    # always set `beat_in_bar = 1.0 + slot/slots_per_beat ≥ 1.0`).
    # Unlike later passes, the geometric snap is a placement, not a
    # shift, so it doesn't go through `_apply_llm_shifts`, this pre-
    # pass is the only cross-bar logic it needs.
    for cands in kept_by_pitch.values():
        for c in cands:
            bar_idx = int(c.bar)
            if not (0 <= bar_idx < len(structure.bars) - 1):
                continue
            here = structure.bars[bar_idx]
            num_beats_here = max(int(here.time_signature[0]), 1)
            max_slot_here = num_beats_here * slots_per_beat - 1
            natural_here = (float(c.beat_in_bar) - 1.0) * slots_per_beat
            overflow = round(natural_here) - max_slot_here
            if overflow <= 0:
                continue
            # Slot `max_slot+k` in this bar maps to slot `k-1` in the
            # next bar (max_slot+1 is the next bar's downbeat).
            c.bar = bar_idx + 1
            c.beat_in_bar = 1.0 + (overflow - 1) / slots_per_beat

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
                    c.quantised_residual_slots = None
                    continue
                c.off_grid = False
                c.quantised_time = float(bar.start_time) + slot * slot_span
                delta = slot - round(nat)
                c.quantised_shift_slots = delta
                # Signed sub-slot residual: how far the raw natural position
                # sat from its own nearest integer slot, range (-0.5, +0.5].
                c.quantised_residual_slots = float(nat - round(nat))
                if delta != 0:
                    shifts[(pitch, idx)] = delta

    return shifts


# ---------- Per-note envelope re-snap ----------

def _envelope_snap(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    envelopes: dict[str, OnsetEnvelope],
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
) -> dict[tuple[str, int], int]:
    """Re-snap each on-grid onset to the slot whose audio transient it's on.

    For every (lane, bar) group, samples the lane's onset-strength envelope
    in each candidate slot's exclusive time-bin (±half a slot) within
    ±`_ENV_RESNAP_TOLERANCE` of the onset's current slot, and moves the
    onset to the bin holding the strongest transient. Candidate slots
    that fall outside the current bar's range are walked into the
    adjacent bar via `_resolve_cross_bar_target` and sampled there (so a
    boundary hit whose true transient sits on the next bar's downbeat
    can be recovered), with the resulting cross-bar shift carried
    through `_apply_llm_shifts`. A move is taken only when the target
    bin clearly dominates the current slot's bin
    (`_ENV_RESNAP_DOMINANCE`) and clears an absolute floor
    (`_ENV_RESNAP_FLOOR_FRAC × envelope ref`), so a note already on its
    hit stays put and an ambiguous/sustained region isn't disturbed.
    Shifts are applied through `_apply_llm_shifts`, whose
    monotonic-injective + cross-bar occupancy guards keep two onsets
    from being pulled onto the same transient. Returns the applied
    `{(pitch, idx): shift}` map; mutates candidates in place.

    This is the only quantise pass that consults the audio: it catches hits
    whose detected time locked onto the wrong (early) envelope max, which
    the time-and-grid-only passes can't see.
    """
    if not envelopes:
        return {}
    shifts: dict[tuple[str, int], int] = {}
    for pitch, cands in kept_by_pitch.items():
        env = envelopes.get(pitch)
        if env is None:
            continue
        floor = _ENV_RESNAP_FLOOR_FRAC * env.ref
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
            for idx in idxs:
                c = cands[idx]
                base = c.quantised_time if c.quantised_time is not None else c.time
                cur = max(0, min(max_slot, round((base - float(bar.start_time)) / slot_span)))
                best_offset, best_e, cur_e = 0, -1.0, 0.0
                for offset in range(-_ENV_RESNAP_TOLERANCE, _ENV_RESNAP_TOLERANCE + 1):
                    cand_slot = cur + offset
                    # Resolve the candidate slot's bar frame (may walk
                    # into an adjacent bar; tolerance is ±2 so a single
                    # boundary crossing is the worst case in practice).
                    if 0 <= cand_slot <= max_slot:
                        cand_bar = bar
                        cand_in_slot = cand_slot
                        cand_slot_span = slot_span
                    else:
                        dest = _resolve_cross_bar_target(
                            bar_idx, cur, offset, structure, slots_per_beat
                        )
                        if dest is None:
                            continue
                        cand_bar = structure.bars[dest[0]]
                        cand_in_slot = dest[1]
                        cand_num_beats = max(int(cand_bar.time_signature[0]), 1)
                        cand_slot_span = (
                            float(cand_bar.end_time) - float(cand_bar.start_time)
                        ) / (cand_num_beats * slots_per_beat)
                        if cand_slot_span <= 0:
                            continue
                    half = cand_slot_span / 2.0
                    center = float(cand_bar.start_time) + cand_in_slot * cand_slot_span
                    e = env.peak_in(center - half, center + half)
                    if offset == 0:
                        cur_e = e
                    if e > best_e:
                        best_e, best_offset = e, offset
                if best_offset == 0 or best_e < floor:
                    continue
                if best_e < _ENV_RESNAP_DOMINANCE * max(cur_e, 1e-9):
                    continue
                shifts[(pitch, idx)] = best_offset

    if shifts:
        _apply_llm_shifts(
            kept_by_pitch, structure, shifts, slots_per_beat=slots_per_beat
        )
    log.info(
        "quantise envelope: %d onset(s) re-snapped to a stronger transient",
        len(shifts),
    )
    return shifts


# ---------- Summary ----------

def _build_summary(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    geometric_shifts: dict[tuple[str, int], int],
    envelope_shifts: dict[tuple[str, int], int],
    grid_shifts: dict[tuple[str, int], int],
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
            env = envelope_shifts.get((pitch, idx), 0)
            grid = grid_shifts.get((pitch, idx), 0)
            llm = llm_shifts.get((pitch, idx), 0)
            if geo == 0 and env == 0 and grid == 0 and llm == 0 and not c.off_grid:
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
                "residual_slots": (
                    float(c.quantised_residual_slots)
                    if c.quantised_residual_slots is not None else None
                ),
                "geometric_shift": geo,
                "envelope_shift": env,
                "grid_shift": grid,
                "llm_shift": llm,
                "total_shift": (c.quantised_shift_slots or 0),
            })
        if rows:
            per_pitch[pitch] = rows
    return {
        "geometric_shifted": sum(1 for v in geometric_shifts.values() if v),
        "envelope_shifted": sum(1 for v in envelope_shifts.values() if v),
        "grid_shifted": sum(1 for v in grid_shifts.values() if v),
        "llm_shifted": sum(1 for v in llm_shifts.values() if v),
        "off_grid": off_grid_total,
        "llm_status": llm_status,
        "match_band": _MATCH_BAND,
        "max_llm_shift": _MAX_LLM_SHIFT,
        "slots_per_beat": slots_per_beat,
        "per_pitch": per_pitch,
    }
