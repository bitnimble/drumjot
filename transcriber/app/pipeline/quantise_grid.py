"""Deterministic musical-grid snap: infer each (lane, bar)'s subdivision
grid from its onset population, then snap onsets onto it.

Runs between the geometric snap and the LLM residual pass. Unlike the
geometric snap (which reasons purely from audio timing), this pass uses
the population of onsets to recover the slot a hit musically belongs on.
Split out of `quantise.py` as one cohesive pass.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.quantise_apply import _apply_llm_shifts, _current_slot
from app.pipeline.quantise_config import (
    _GRID_COMPLEXITY_PENALTY,
    _GRID_DECISIVE_MARGIN,
    _GRID_MIN_ONSETS,
    _GRID_SNAP_TOLERANCE,
    SLOTS_PER_BEAT,
)

log = logging.getLogger(__name__)


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
        for idx, slot in members:
            target = _nearest_grid_slot(slot, positions, slots_per_beat)
            shift = target - slot
            if shift == 0 or abs(shift) > _GRID_SNAP_TOLERANCE:
                continue
            # `_apply_llm_shifts` resolves the destination bar/slot when
            # `target` lies outside this bar's range, so cross-bar moves
            # (e.g. an onset on the bar's last slot snapping forward to
            # the next bar's downbeat) are emitted as ordinary shifts
            # and gated by the shared occupancy check there.
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
