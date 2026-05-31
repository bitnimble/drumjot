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
"""
from __future__ import annotations

import contextvars
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anthropic

from app.config import settings
from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.envelope import OnsetEnvelope
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

# The LLM residual pass is split into windows of consecutive bars that run
# concurrently. Splitting (a) shrinks per-call latency and lets windows
# overlap on the wire, and (b) keeps each prompt small enough that Haiku
# reasons sharply over it instead of skimming a 1000+-onset wall. Onsets
# never cross a bar boundary, so windows always break on bar boundaries.
#
# A window accumulates consecutive (onset-bearing) bars until either the
# onset target or the bar-span cap is reached, whichever comes first. The
# onset target is a SOFT cap: a single dense bar that exceeds it alone
# still becomes its own window (we never split a bar). Each window also
# renders +/-_CONTEXT_BARS neighbour bars read-only so groove continuity
# across the window seam is visible without those bars being shiftable.
_TARGET_ONSETS_PER_WINDOW = 150
_MAX_BARS_PER_WINDOW = 8       # max bar-index span of a window's core bars
_CONTEXT_BARS = 1              # read-only neighbour bars rendered per side
_MAX_PARALLEL_CHUNKS = 8       # cap on concurrent Anthropic requests

# Deterministic musical-grid pass (runs between the geometric snap and the
# LLM residual pass). It infers, per (lane, bar), which subdivision grid
# the surrounding rhythm is using, then snaps onsets onto that grid. Unlike
# the geometric snap (which reasons purely from audio timing), this pass
# uses the *population* of onsets to recover the slot a hit musically
# belongs on, including the case where a performer played a consistent full
# slot off the beat and rounded cleanly onto the wrong slot. Tuplet/swing
# safety is structural: a note is only judged off-grid relative to a grid
# its own lane voted for, so genuine triplets/shuffle/poly-rhythm survive.
_GRID_MIN_ONSETS = 4            # min onsets for a (lane, bar) to vote a grid
_GRID_COMPLEXITY_PENALTY = 0.5  # Occam cost per grid slot (favours simpler)
_GRID_DECISIVE_MARGIN = 0.3     # winner must beat runner-up by this (else skip)
_GRID_SNAP_TOLERANCE = 1        # max |slot shift| this pass will apply

# Per-note envelope re-snap (runs right after the geometric snap). Every
# other quantise pass trusts the onset *time* and never looks at the audio,
# so a hit whose detection locked onto the wrong (early) envelope max stays
# misplaced, on a real transient, just the wrong one. This pass samples the
# lane's onset-strength envelope in each candidate slot's time-bin and moves
# the note to the bin holding the strongest transient, but only when that
# bin's energy clearly dominates the current slot's (so it never nudges a
# note that's already on its hit, and stays put when the audio is ambiguous).
# Bounded to ±tolerance slots and applied through the monotonic-injective
# guard, so it can't collide or reorder onsets.
_ENV_RESNAP_TOLERANCE = 2       # max |slot shift| this pass will apply
_ENV_RESNAP_DOMINANCE = 2.0     # target bin must be >this × the current bin
_ENV_RESNAP_FLOOR_FRAC = 0.15   # target bin must clear this × envelope ref


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
    # Backward overflow can't arise (beat_in_bar >= 1.0 by construction).
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
    onset to the bin holding the strongest transient. A move is taken only
    when that bin clearly dominates the current slot's bin
    (`_ENV_RESNAP_DOMINANCE`) and clears an absolute floor
    (`_ENV_RESNAP_FLOOR_FRAC × envelope ref`), so a note already on its hit
    stays put and an ambiguous/sustained region isn't disturbed. Shifts are
    applied through `_apply_llm_shifts`, whose monotonic-injective guard
    keeps two onsets from being pulled onto the same transient. Returns the
    applied `{(pitch, idx): shift}` map; mutates candidates in place.

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
            half = slot_span / 2.0
            for idx in idxs:
                c = cands[idx]
                base = c.quantised_time if c.quantised_time is not None else c.time
                cur = max(0, min(max_slot, round((base - float(bar.start_time)) / slot_span)))
                best_slot, best_e, cur_e = cur, -1.0, 0.0
                for cand in range(cur - _ENV_RESNAP_TOLERANCE, cur + _ENV_RESNAP_TOLERANCE + 1):
                    if cand < 0 or cand > max_slot:
                        continue
                    center = float(bar.start_time) + cand * slot_span
                    e = env.peak_in(center - half, center + half)
                    if cand == cur:
                        cur_e = e
                    if e > best_e:
                        best_e, best_slot = e, cand
                if best_slot == cur or best_e < floor:
                    continue
                if best_e < _ENV_RESNAP_DOMINANCE * max(cur_e, 1e-9):
                    continue
                shifts[(pitch, idx)] = best_slot - cur

    if shifts:
        _apply_llm_shifts(
            kept_by_pitch, structure, shifts, slots_per_beat=slots_per_beat
        )
    log.info(
        "quantise envelope: %d onset(s) re-snapped to a stronger transient",
        len(shifts),
    )
    return shifts


# ---------- LLM residual pass ----------

def _llm_residual_pass(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
    cancel_event: threading.Event | None = None,
) -> tuple[dict[tuple[str, int], int], str]:
    """Haiku residual pass over parallel windows of consecutive bars.

    Returns `(shifts, status)`, where `shifts` is the merged
    (pitch, idx_in_pitch_list) -> shift_slots map the model proposed
    (clamped to ±_MAX_LLM_SHIFT, unknown ids dropped) and `status` is a
    short summary suitable for `quantise/shifts.json` ("ok",
    "partial: F/N windows failed", "error: ...", or "cancelled").

    The on-grid onsets are split into windows (`_build_windows`); each
    window is one forced-tool call carrying its core bars (shiftable,
    `#id`-tagged) plus ±_CONTEXT_BARS read-only neighbour bars. Windows
    run concurrently. A window that raises degrades only its own bars to
    geometric placement, the rest keep their LLM shifts. Off-grid onsets
    are never offered as shift targets; their position is geometric truth.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot call the quantise LLM."
        )

    indexed = _index_for_llm(kept_by_pitch, structure, slots_per_beat=slots_per_beat)
    if not indexed:
        return {}, "ok"

    windows = _build_windows(
        indexed, structure,
        target_onsets=_TARGET_ONSETS_PER_WINDOW,
        max_bars=_MAX_BARS_PER_WINDOW,
        context_bars=_CONTEXT_BARS,
    )
    template = _load_prompt_template()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    log.info(
        "quantise LLM: %d onsets across %d bars -> %d window(s), "
        "<=%d concurrent",
        len(indexed), len(structure.bars), len(windows), _MAX_PARALLEL_CHUNKS,
    )

    shifts: dict[tuple[str, int], int] = {}
    failed = 0
    cancelled = False
    max_workers = max(1, min(_MAX_PARALLEL_CHUNKS, len(windows)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for wi, window in enumerate(windows):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            # Submit through a copy of the submitting thread's context so
            # contextvars propagate into the pool workers;
            # ThreadPoolExecutor does NOT copy them. This carries the
            # request id (for log correlation) plus the debug-sink /
            # run-log contextvars; copy_context() runs here, on the
            # submitting thread, which is already under the pipeline's
            # asyncio.to_thread context and so holds those values.
            futures[pool.submit(
                contextvars.copy_context().run,
                _call_window, client, template, structure, window, wi,
                slots_per_beat, cancel_event,
            )] = wi
        for fut in as_completed(futures):
            wi = futures[fut]
            try:
                window_shifts = fut.result()
            except Exception as exc:
                failed += 1
                log.warning(
                    "quantise LLM: window %d failed (%s); keeping geometric "
                    "placement for its bars", wi, exc,
                )
                continue
            if window_shifts is None:  # window short-circuited on cancel
                cancelled = True
                continue
            shifts.update(window_shifts)

    n = len(windows)
    if cancelled:
        status = "cancelled"
    elif failed == 0:
        status = "ok"
    elif failed >= n:
        status = f"error: all {n} windows failed"
    else:
        status = f"partial: {failed}/{n} windows failed"
    log.info(
        "quantise LLM: %d onsets received a non-zero shift (status=%s)",
        len(shifts), status,
    )
    return shifts, status


def _build_windows(
    indexed: list[_LlmEntry],
    structure: BeatStructure,
    *,
    target_onsets: int,
    max_bars: int,
    context_bars: int,
) -> list[_Window]:
    """Partition the globally-sorted `indexed` entries into bar windows.

    Walks the onset-bearing bars in ascending order, accumulating them
    into a window until adding the next bar would exceed `target_onsets`
    (a soft cap, a lone dense bar may exceed it) or push the window's
    bar-index span past `max_bars`. Each window also picks up the
    ±`context_bars` neighbour bars (clamped to the score) as read-only
    context. Local `#id`s are assigned to core-bar onsets in global sort
    order so a window's response maps cleanly back to (pitch, idx).
    """
    by_bar: dict[int, list[tuple[int, _LlmEntry]]] = {}
    for gid, e in enumerate(indexed):
        by_bar.setdefault(e.bar, []).append((gid, e))
    core_bars_sorted = sorted(by_bar.keys())

    groups: list[list[int]] = []
    cur: list[int] = []
    cur_onsets = 0
    for bar_idx in core_bars_sorted:
        n = len(by_bar[bar_idx])
        if cur and (
            cur_onsets + n > target_onsets or bar_idx - cur[0] >= max_bars
        ):
            groups.append(cur)
            cur, cur_onsets = [], 0
        cur.append(bar_idx)
        cur_onsets += n
    if cur:
        groups.append(cur)

    last_bar = len(structure.bars) - 1
    windows: list[_Window] = []
    for core in groups:
        core_set = set(core)
        lo = max(0, core[0] - context_bars)
        hi = min(last_bar, core[-1] + context_bars)
        # Local id assignment in global sort order (core bars only).
        local_to_global: list[tuple[str, int]] = []
        gid_to_lid: dict[int, int] = {}
        for bar_idx in core:
            for gid, e in by_bar[bar_idx]:
                gid_to_lid[gid] = len(local_to_global)
                local_to_global.append((e.pitch, e.idx))
        windows.append(_Window(
            core_set=core_set,
            render_bars=list(range(lo, hi + 1)),
            by_bar=by_bar,
            gid_to_lid=gid_to_lid,
            local_to_global=local_to_global,
        ))
    return windows


def _call_window(
    client: anthropic.Anthropic,
    template: str,
    structure: BeatStructure,
    window: _Window,
    window_index: int,
    slots_per_beat: int,
    cancel_event: threading.Event | None,
) -> dict[tuple[str, int], int] | None:
    """Run one window's forced-tool call; return its (pitch, idx) -> shift.

    Returns None if `cancel_event` is set before the call is made (the
    caller treats this as cancelled, not failed). Raises on API error so
    the caller can degrade just this window's bars to geometric.
    """
    if cancel_event is not None and cancel_event.is_set():
        return None

    n_local = len(window.local_to_global)
    bar_blocks = _format_window(structure, window, slots_per_beat=slots_per_beat)
    initial_sig = structure.initial_time_signature
    prompt = (
        template
        .replace("{INITIAL_TEMPO}", f"{structure.initial_tempo:.2f}")
        .replace("{INITIAL_TIME_SIG}", f"{initial_sig[0]}/{initial_sig[1]}")
        .replace("{BAR_COUNT}", str(len(window.render_bars)))
        .replace("{ONSET_COUNT}", str(n_local))
        .replace("{SLOTS_PER_BEAT}", str(slots_per_beat))
        .replace("{MAX_SHIFT}", str(_MAX_LLM_SHIFT))
        .replace("{BARS}", bar_blocks)
    )

    max_tokens = _max_tokens_for(n_local)
    log.info(
        "quantise LLM window %d: prompt_chars=%d shiftable_onsets=%d "
        "max_tokens=%d", window_index, len(prompt), n_local, max_tokens,
    )
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
        purpose=f"quantise_w{window_index:02d}",
    )

    if getattr(response, "stop_reason", None) == "max_tokens":
        # A forced tool call that hits max_tokens emits unparseable JSON;
        # `block.input` will be empty / partial and we'd silently report
        # "0 shifts". Flag it so it's obvious from the log next time.
        log.warning(
            "quantise LLM window %d hit max_tokens (%d) for %d onsets; "
            "response will be truncated and most shifts lost. Bump "
            "_LLM_MAX_TOKENS_PER_ONSET in quantise.py.",
            window_index, max_tokens, n_local,
        )
    raw_shifts = _extract_shifts(response, n_local)
    out: dict[tuple[str, int], int] = {}
    for local_id, shift in raw_shifts.items():
        if local_id < 0 or local_id >= n_local:
            continue
        clamped = max(-_MAX_LLM_SHIFT, min(_MAX_LLM_SHIFT, shift))
        if clamped == 0:
            continue
        out[window.local_to_global[local_id]] = clamped
    return out


def _apply_llm_shifts(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    llm_shifts: dict[tuple[str, int], int],
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
) -> None:
    """Apply per-onset slot shifts atomically per (lane, bar).

    Shared by the LLM residual pass and the deterministic musical-grid
    pass; both produce a `{(pitch, idx): shift}` map and need the same
    safety guard. The geometric snap leaves each lane+bar's on-grid onsets
    on distinct, strictly-increasing slots. A pass proposes independent
    ±shifts, which
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
                        "quantise: shifts for lane %r bar %d would break "
                        "slot order/injectivity; keeping prior placement",
                        pitch, bar_idx,
                    )
                continue

            for i, delta, new_slot in plan:
                if delta == 0:
                    continue
                c = cands[i]
                c.quantised_time = float(bar.start_time) + new_slot * slot_span
                c.quantised_shift_slots = (c.quantised_shift_slots or 0) + delta


# ---------- Deterministic musical-grid pass ----------

def _musical_grid_snap(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
) -> dict[tuple[str, int], int]:
    """Snap on-grid onsets onto the subdivision grid their rhythm implies.

    For each (lane, bar) it infers the best-fitting candidate grid from the
    *population* of that lane's onsets (`_infer_grid`); a lane too sparse to
    vote falls back to the bar aggregate, then the song aggregate. Each
    onset is then nudged to its grid's nearest slot, bounded to
    ±`_GRID_SNAP_TOLERANCE`. The shifts are applied through
    `_apply_llm_shifts`, which enforces the per-(lane, bar)
    monotonic-injective guard, so a grid snap can never collide or reorder
    onsets. Returns the applied `{(pitch, idx): shift}` map for the debug
    summary; mutates candidates in place.

    Tuplet/swing safety is structural: an onset is only ever moved toward a
    grid its own lane (or, for sparse lanes, the surrounding population)
    voted for, so genuine triplets, shuffle and cross-limb poly-rhythm are
    preserved rather than squared.
    """
    if not structure.bars:
        return {}
    grids = _candidate_grids(slots_per_beat)
    if not grids:
        return {}

    # Current slot of every on-grid onset, grouped for voting and snapping.
    lane_bar_members: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    bar_folded: dict[int, list[int]] = defaultdict(list)
    song_folded: list[int] = []
    for pitch, cands in kept_by_pitch.items():
        for idx, c in enumerate(cands):
            if c.off_grid:
                continue
            bar_idx = int(c.bar)
            if bar_idx < 0 or bar_idx >= len(structure.bars):
                continue
            slot = _current_slot(c, structure.bars[bar_idx], slots_per_beat)
            if slot is None:
                continue
            lane_bar_members[(pitch, bar_idx)].append((idx, slot))
            bar_folded[bar_idx].append(slot % slots_per_beat)
            song_folded.append(slot % slots_per_beat)

    song_grid = _infer_grid(song_folded, grids, slots_per_beat)
    bar_grids = {
        b: _infer_grid(folded, grids, slots_per_beat)
        for b, folded in bar_folded.items()
    }

    shifts: dict[tuple[str, int], int] = {}
    grid_tally: dict[str, int] = defaultdict(int)  # inferred-grid distribution
    for (pitch, bar_idx), members in lane_bar_members.items():
        lane_folded = [slot % slots_per_beat for _idx, slot in members]
        grid = (
            _infer_grid(lane_folded, grids, slots_per_beat)
            or bar_grids.get(bar_idx)
            or song_grid
        )
        grid_tally["deferred" if grid is None else grid[0]] += 1
        if grid is None:
            continue
        _name, positions = grid
        bar = structure.bars[bar_idx]
        max_slot = max(int(bar.time_signature[0]), 1) * slots_per_beat - 1
        for idx, slot in members:
            target = _nearest_grid_slot(slot, positions, slots_per_beat)
            shift = target - slot
            if shift == 0 or abs(shift) > _GRID_SNAP_TOLERANCE:
                continue
            if not (0 <= target <= max_slot):
                continue
            shifts[(pitch, idx)] = shift

    if shifts:
        _apply_llm_shifts(
            kept_by_pitch, structure, shifts, slots_per_beat=slots_per_beat
        )
    log.info(
        "quantise grid: %d onset(s) proposed a grid shift across %d "
        "(lane, bar) group(s); inferred %s",
        len(shifts), len(lane_bar_members),
        dict(sorted(grid_tally.items())),
    )
    return shifts


def _candidate_grids(slots_per_beat: int) -> list[tuple[str, tuple[int, ...]]]:
    """The subdivision grids we test, as per-beat slot sets (mod beat).

    Only grids whose positions land on integer slots at this density are
    included, so a coarse grid stays usable when finer subdivisions don't
    divide evenly. Positions are sorted for deterministic nearest-slot
    tie-breaking. Triplet slots are deliberately disjoint from straight
    16th slots: that disjointness is what lets the inference tell a stray
    triplet-position hit in a straight lane from a real triplet.
    """
    s = slots_per_beat
    specs: list[tuple[str, list[float]]] = [
        ("quarter", [0]),
        ("straight_8", [0, s / 2]),
        ("straight_16", [0, s / 4, s / 2, 3 * s / 4]),
        ("triplet_8", [0, s / 3, 2 * s / 3]),
        ("triplet_16", [0, s / 6, s / 3, s / 2, 2 * s / 3, 5 * s / 6]),
        ("swing_8", [0, 2 * s / 3]),
    ]
    grids: list[tuple[str, tuple[int, ...]]] = []
    for name, positions in specs:
        if all(abs(p - round(p)) < 1e-9 for p in positions):
            grids.append((name, tuple(sorted({int(round(p)) for p in positions}))))
    return grids


def _infer_grid(
    folded: list[int],
    grids: list[tuple[str, tuple[int, ...]]],
    slots_per_beat: int,
) -> tuple[str, tuple[int, ...]] | None:
    """Pick the candidate grid the folded per-beat positions best fit.

    Cost per grid = mean squared (circular) slot-distance from each onset to
    the nearest grid slot + an Occam penalty proportional to the grid's slot
    count, so a denser grid only wins when it fits materially better.
    Returns None when there's too little evidence (`< _GRID_MIN_ONSETS`) or
    no grid wins by a decisive margin (the lane/bar is ambiguous; defer it).
    """
    if len(folded) < _GRID_MIN_ONSETS:
        return None
    scored: list[tuple[float, str, tuple[int, ...]]] = []
    for name, positions in grids:
        ssd = sum(
            _circular_dist(p, positions, slots_per_beat) ** 2 for p in folded
        )
        cost = ssd / len(folded) + _GRID_COMPLEXITY_PENALTY * len(positions)
        scored.append((cost, name, positions))
    scored.sort(key=lambda t: t[0])
    best = scored[0]
    if len(scored) > 1 and (scored[1][0] - best[0]) < _GRID_DECISIVE_MARGIN:
        return None
    return (best[1], best[2])


def _circular_dist(pos: int, positions: tuple[int, ...], slots_per_beat: int) -> int:
    """Min distance from `pos` to any slot in `positions`, wrapping the beat.

    A hit at slot 11 (of 12) is 1 away from the next downbeat (slot 0/12),
    not 11; the beat is cyclic, so distances wrap modulo `slots_per_beat`.
    """
    best = slots_per_beat
    for g in positions:
        d = abs(pos - g)
        best = min(best, d, slots_per_beat - d)
    return best


def _nearest_grid_slot(
    slot: int, positions: tuple[int, ...], slots_per_beat: int
) -> int:
    """Absolute slot of the nearest grid position to `slot`, beat-cyclic.

    Considers each grid position and its neighbouring-beat images so a hit
    just before a beat can snap forward onto the next downbeat.
    """
    folded = slot % slots_per_beat
    best_img = folded
    best_d: int | None = None
    for g in positions:
        for img in (g - slots_per_beat, g, g + slots_per_beat):
            d = abs(folded - img)
            if best_d is None or d < best_d:
                best_d, best_img = d, img
    return slot + (best_img - folded)


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
            entries.append(_LlmEntry(
                pitch=pitch, idx=idx, bar=bar_idx, slot=current_slot,
                residual=c.quantised_residual_slots,
            ))
    entries.sort(key=lambda e: (e.bar, e.slot, e.pitch))
    return entries


class _LlmEntry:
    __slots__ = ("pitch", "idx", "bar", "slot", "residual")

    def __init__(
        self, pitch: str, idx: int, bar: int, slot: int,
        residual: float | None = None,
    ) -> None:
        self.pitch = pitch
        self.idx = idx
        self.bar = bar
        self.slot = slot
        # Geometric snap's signed sub-slot residual (range (-0.5, +0.5]),
        # surfaced to the model as a near-miss hint. None when unknown.
        self.residual = residual


class _Window:
    """One LLM call's worth of bars.

    `core_set` holds the bar indices whose onsets are shiftable (rendered
    with `#id`s); `render_bars` is the full ordered span to print (core +
    read-only context bars). `gid_to_lid` maps a global onset id (its
    position in the shared `indexed` list) to this window's local id;
    `local_to_global` is the inverse for shiftable onsets only, mapping a
    local id back to its `(pitch, idx)`. `by_bar` is the shared
    global-id-keyed grouping (read-only here).
    """
    __slots__ = ("core_set", "render_bars", "by_bar", "gid_to_lid", "local_to_global")

    def __init__(
        self,
        core_set: set[int],
        render_bars: list[int],
        by_bar: dict[int, list[tuple[int, _LlmEntry]]],
        gid_to_lid: dict[int, int],
        local_to_global: list[tuple[str, int]],
    ) -> None:
        self.core_set = core_set
        self.render_bars = render_bars
        self.by_bar = by_bar
        self.gid_to_lid = gid_to_lid
        self.local_to_global = local_to_global


def _format_window(
    structure: BeatStructure,
    window: _Window,
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
) -> str:
    """Render a window's bars: one block per bar, onset rows grouped by slot.

    Core bars carry shiftable onsets tagged with their window-local `#id`.
    Context bars are tagged `[context - read-only]` and their onsets are
    rendered WITHOUT an id, so the model can see the surrounding groove
    but cannot (and is told not to) shift them.
    """
    blocks: list[str] = []
    for bar_idx in window.render_bars:
        bar = structure.bars[bar_idx]
        is_core = bar_idx in window.core_set
        tag = "" if is_core else " [context - read-only]"
        header = (
            f"Bar {bar.index} "
            f"[{bar.time_signature[0]}/{bar.time_signature[1]}, "
            f"{bar.tempo_bpm:.1f} BPM, feel={bar.feel}]{tag}:"
        )
        rows = [header]
        bar_entries = window.by_bar.get(bar_idx, [])
        if not bar_entries:
            rows.append("  (no onsets)")
            blocks.append("\n".join(rows))
            continue
        # Group by slot for compactness.
        by_slot: dict[int, list[tuple[int, _LlmEntry]]] = {}
        for gid, e in bar_entries:
            by_slot.setdefault(e.slot, []).append((gid, e))
        for slot in sorted(by_slot.keys()):
            slot_entries = by_slot[slot]
            beat_label = _slot_label(slot, slots_per_beat)
            if is_core:
                rendered = " ".join(
                    f"#{window.gid_to_lid[gid]}({e.pitch}{_residual_tag(e.residual)})"
                    for gid, e in slot_entries
                )
            else:
                rendered = " ".join(f"({e.pitch})" for _gid, e in slot_entries)
            rows.append(f"  slot {slot:>2} {beat_label}: {rendered}")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def _residual_tag(residual: float | None) -> str:
    """Compact near-miss hint for a core onset, e.g. " r+0.45".

    Only emitted for a notable sub-slot residual (|r| >= 0.25); a small or
    absent residual is left unannotated. Surfaces how far the raw audio sat
    from the slot so the model knows which rounds were coin-flips. NOT a
    correctness signal on its own (a systematic full-slot offset rounds
    cleanly, residual ~ 0), the prompt says to weigh musical context, not r.
    """
    if residual is None or abs(residual) < 0.25:
        return ""
    return f" r{residual:+.2f}"


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
