"""Shared shift-application primitives for the quantise passes.

`_apply_llm_shifts` is the single safety guard every non-geometric pass
(envelope re-snap, musical-grid snap, LLM residual) routes its
`{(pitch, idx): shift}` map through; `_resolve_cross_bar_target` and
`_current_slot` are the slot-geometry helpers it (and the passes) share.
Split out of `quantise.py` so the passes can import the applier without a
cycle back through the orchestrator.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.quantise_config import SLOTS_PER_BEAT

log = logging.getLogger(__name__)


def _apply_llm_shifts(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    llm_shifts: dict[tuple[str, int], int],
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
) -> None:
    """Apply per-onset slot shifts atomically per (lane, source bar).

    Shared by the envelope re-snap, musical-grid snap, and LLM residual
    pass; all three produce a `{(pitch, idx): shift}` map and need the
    same safety guard.

    Each shift's destination is resolved against the song's full bar
    grid via `_resolve_cross_bar_target`, so a shift can move an onset
    across one or more bar boundaries in either direction; the helper
    honours per-bar time-signature changes. In-bar destinations (where
    the source bar still owns the onset after the shift) are validated
    by the original monotonic-injective guard: distinct, strictly
    increasing slots in time-sorted order. Cross-bar destinations are
    validated against a per-pitch `(bar_idx, slot)` occupancy snapshot
    that's updated as each cross-bar move is applied, so two
    simultaneous moves targeting the same destination, or a move into
    a slot some other lane onset already holds, are rejected.
    Whole-group atomicity is preserved: any failure for a (lane, source
    bar) group keeps the entire group on its prior placement.
    Off-grid onsets are never shifted.

    Sorting is by post-snap time (`quantised_time`, falling back to the
    raw detected `time`) rather than the raw detected time alone: after
    a cross-bar move, a candidate's `c.time` still points into the
    source bar's audio window while its `quantised_time` reflects the
    destination bar's frame. A future pass touching the destination
    bar's lane then sees the moved candidate in its correct slot order
    relative to non-moved onsets.
    """
    if not llm_shifts:
        return

    def sort_time(c: OnsetCandidate) -> float:
        return c.quantised_time if c.quantised_time is not None else c.time

    # Per-pitch (bar_idx, slot) occupancy snapshot. Built once before
    # any moves and updated as cross-bar moves apply, so a later move's
    # destination check sees the post-move state.
    occupied: dict[str, set[tuple[int, int]]] = {}
    for pitch, cands in kept_by_pitch.items():
        occ: set[tuple[int, int]] = set()
        for c in cands:
            if c.off_grid:
                continue
            bar_idx = int(c.bar)
            if not (0 <= bar_idx < len(structure.bars)):
                continue
            slot = _current_slot(c, structure.bars[bar_idx], slots_per_beat)
            if slot is not None:
                occ.add((bar_idx, slot))
        occupied[pitch] = occ

    for pitch, cands in kept_by_pitch.items():
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

            idxs.sort(key=lambda i, _c=cands: sort_time(_c[i]))
            # plan[k] = (idx, delta, src_slot, dest_bar_idx, dest_slot).
            # dest_bar_idx == bar_idx for in-bar moves; otherwise the
            # shift walks across one or more bar boundaries.
            plan: list[tuple[int, int, int, int, int]] = []
            walk_failed = False
            for i in idxs:
                c = cands[i]
                base_time = c.quantised_time if c.quantised_time is not None else c.time
                current_slot = round((base_time - float(bar.start_time)) / slot_span)
                delta = llm_shifts.get((pitch, i), 0)
                target_abs = current_slot + delta
                if 0 <= target_abs <= max_slot:
                    plan.append((i, delta, current_slot, bar_idx, target_abs))
                else:
                    dest = _resolve_cross_bar_target(
                        bar_idx, current_slot, delta, structure, slots_per_beat
                    )
                    if dest is None:
                        # Walks off the song; reject the whole group so
                        # we don't silently swallow part of it.
                        walk_failed = True
                        break
                    plan.append((i, delta, current_slot, dest[0], dest[1]))
            if walk_failed:
                log.info(
                    "quantise: shift for lane %r bar %d walks off the song "
                    "end; keeping prior placement", pitch, bar_idx,
                )
                continue

            # In-bar invariant: among onsets that stay in this bar after
            # the shift, slots must remain strictly increasing in the
            # time-sorted order. Cross-bar moves vacate their source slot
            # and are excluded from this check.
            in_bar_intended = [p[4] for p in plan if p[3] == bar_idx]
            in_bar_ok = all(
                in_bar_intended[k] < in_bar_intended[k + 1]
                for k in range(len(in_bar_intended) - 1)
            )

            # Cross-bar invariant: each destination must be free in the
            # per-pitch occupancy (which already accounts for moves
            # applied earlier in this pass), and no two cross-bar moves
            # within this group may target the same destination.
            cross_bar_ok = True
            seen_destinations: set[tuple[int, int]] = set()
            for _i, _delta, _src_slot, dest_bar, dest_slot in plan:
                if dest_bar == bar_idx:
                    continue
                key = (dest_bar, dest_slot)
                if key in seen_destinations or key in occupied[pitch]:
                    cross_bar_ok = False
                    break
                seen_destinations.add(key)

            if not (in_bar_ok and cross_bar_ok):
                if any(delta for _i, delta, _ss, _db, _ds in plan):
                    log.info(
                        "quantise: shifts for lane %r bar %d would break "
                        "slot order/injectivity; keeping prior placement",
                        pitch, bar_idx,
                    )
                continue

            for i, delta, src_slot, dest_bar, dest_slot in plan:
                if delta == 0:
                    continue
                c = cands[i]
                if dest_bar == bar_idx:
                    c.quantised_time = float(bar.start_time) + dest_slot * slot_span
                else:
                    dest_bar_obj = structure.bars[dest_bar]
                    dest_num_beats = max(int(dest_bar_obj.time_signature[0]), 1)
                    dest_slot_span = (
                        float(dest_bar_obj.end_time) - float(dest_bar_obj.start_time)
                    ) / (dest_num_beats * slots_per_beat)
                    c.bar = dest_bar
                    c.beat_in_bar = 1.0 + dest_slot / slots_per_beat
                    c.quantised_time = (
                        float(dest_bar_obj.start_time) + dest_slot * dest_slot_span
                    )
                    occupied[pitch].discard((bar_idx, src_slot))
                    occupied[pitch].add((dest_bar, dest_slot))
                c.quantised_shift_slots = (c.quantised_shift_slots or 0) + delta


def _resolve_cross_bar_target(
    src_bar_idx: int,
    src_slot: int,
    shift: int,
    structure: BeatStructure,
    slots_per_beat: int,
) -> tuple[int, int] | None:
    """Walk an integer slot shift across bar boundaries.

    Returns `(dest_bar_idx, dest_slot)` with `dest_slot` in
    `[0, num_slots_in(dest_bar) - 1]`, or `None` if the shift walks off
    the song (no further source/destination bar exists in the walked
    direction). Multi-bar walks are supported so any pass can propose
    an arbitrary shift; in practice all current passes bound their
    output to ±2 slots so a single boundary crossing is the worst case.

    Time-signature changes between bars are honoured: each iteration
    reads the current bar's slot count and consumes/restores that many
    slots when walking past its boundary.
    """
    if shift == 0:
        return (src_bar_idx, src_slot)
    bar_idx = src_bar_idx
    abs_slot = src_slot + shift
    while True:
        if not (0 <= bar_idx < len(structure.bars)):
            return None
        bar = structure.bars[bar_idx]
        num_slots = max(int(bar.time_signature[0]), 1) * slots_per_beat
        if 0 <= abs_slot < num_slots:
            return (bar_idx, abs_slot)
        if abs_slot < 0:
            prev_idx = bar_idx - 1
            if prev_idx < 0:
                return None
            prev = structure.bars[prev_idx]
            prev_num_slots = max(int(prev.time_signature[0]), 1) * slots_per_beat
            abs_slot += prev_num_slots
            bar_idx = prev_idx
        else:
            abs_slot -= num_slots
            bar_idx += 1
            if bar_idx >= len(structure.bars):
                return None


def _current_slot(
    c: OnsetCandidate, bar: Any, slots_per_beat: int
) -> int | None:
    """The onset's current integer slot within its bar (post-snap), or None.

    Uses `quantised_time` when set (the canonical post-snap position),
    falling back to the raw `time`. Returns None for a degenerate bar.
    """
    num_beats = max(int(bar.time_signature[0]), 1)
    slot_span = (float(bar.end_time) - float(bar.start_time)) / (
        num_beats * slots_per_beat
    )
    if slot_span <= 0:
        return None
    base_time = c.quantised_time if c.quantised_time is not None else c.time
    slot = round((base_time - float(bar.start_time)) / slot_span)
    return max(0, min(num_beats * slots_per_beat - 1, slot))
