"""Pure geometric onset-snap: per-lane monotonic-injective slot assignment.

Replaces the deterministic nearest-slot snap + cross-instrument cluster
pull in `quantise.py` with a single-pass exact dynamic program. Given a
lane's onsets as unrounded fractional slot positions (ascending by time),
assign each to an integer slot such that:

  - assignments are strictly increasing (monotonic + injective: no two
    onsets in a lane share a slot, and detected order is preserved),
  - total squared slot-distance is minimised,
  - an onset with no affordable in-band slot is left off-grid (None),
    charged `off_grid_penalty`.

The DP is exact (Bellman-optimal over all monotonic-injective
assignments): one forward sweep filling the cost table, one backward
sweep over backpointers to reconstruct. O(n · W) where W = 2·band + 1 is
the per-onset feasible-window size. No I/O; tested with synthetic slot
lists. See `docs/superpowers/specs/2026-05-29-geometric-quantise-design.md`
§6.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

# A DP state / action value: an integer slot, or None (off-grid, or the
# empty "nothing placed yet" floor).
_State: TypeAlias = "int | None"

# Tie-break weight: when two placements have equal squared cost, prefer
# the one closer to the onset's natural position. Far smaller than the
# squared-distance gap between any two distinct integer slots, so it never
# overrides a genuine cost difference.
_EPS = 1e-9

# The off-grid state is keyed by None in the per-onset cost/back tables;
# a placed onset is keyed by its integer slot.
_OFF: None = None


def snap_lane(
    naturals: Sequence[float],
    band: int,
    off_grid_penalty: float,
    min_slot: int | None = None,
    max_slot: int | None = None,
) -> list[int | None]:
    """Assign each onset to an integer slot (or None = off-grid).

    `naturals` are the onsets' unrounded fractional slot positions,
    ascending by time (ties allowed). `band` bounds each onset's feasible
    slots to `round(natural) ± band`; `min_slot` / `max_slot` (when given)
    additionally clamp the feasible window to a valid range (e.g. a bar's
    `[0, slots_per_bar - 1]`), so a clamped-empty window forces that onset
    off-grid. `off_grid_penalty` is the cost of leaving an onset off-grid;
    an onset goes off-grid only when no feasible slot is both free
    (injectivity) and cheaper than the penalty.
    """
    n = len(naturals)
    if n == 0:
        return []

    def cost(natural: float, slot: int) -> float:
        d = natural - slot
        return d * d + _EPS * abs(d)

    def window(natural: float) -> range:
        c = round(natural)
        lo = c - band
        hi = c + band
        if min_slot is not None:
            lo = max(lo, min_slot)
        if max_slot is not None:
            hi = min(hi, max_slot)
        return range(lo, hi + 1)

    # The DP state after onset i is the HIGHEST slot placed among onsets
    # 0..i (None = nothing placed yet). Carrying the running max, rather
    # than just "this onset's slot", is what keeps monotonicity + injectivity
    # intact across off-grid onsets: an off-grid onset leaves the floor
    # unchanged, so a later placed onset still has to clear the last real
    # placement two or more steps back.
    #
    # back[i] maps each state of onset i to (predecessor_state, action),
    # where action is the slot this onset was placed at, or None if it
    # went off-grid.
    back: list[dict[_State, tuple[_State, _State]]] = []

    # ---- Row 0 ----
    prev: dict[_State, float] = {}
    b0: dict[_State, tuple[_State, _State]] = {}
    for s in window(naturals[0]):
        prev[s] = cost(naturals[0], s)
        b0[s] = (None, s)  # placed at s, from the empty floor
    prev[_OFF] = off_grid_penalty
    b0[_OFF] = (None, _OFF)  # off-grid, nothing placed
    back.append(b0)

    # ---- Rows 1..n-1 ----
    for i in range(1, n):
        nat = naturals[i]
        prev_none = prev.get(None, float("inf"))  # "nothing placed" floor
        prev_slots = sorted(k for k in prev if k is not None)

        cur: dict[_State, float] = {}
        cur_back: dict[_State, tuple[_State, _State]] = {}

        # PLACE onset i at slot s: predecessor must be a lower placed slot
        # (s' < s) or the empty floor (None). Sweep candidate slots
        # ascending, advancing a running prefix-min over the previous
        # placed slots; the empty floor is always an eligible predecessor.
        ptr = 0
        run_min = float("inf")
        run_arg: int | None = None
        for s in window(nat):
            while ptr < len(prev_slots) and prev_slots[ptr] < s:
                c = prev[prev_slots[ptr]]
                if c < run_min:
                    run_min, run_arg = c, prev_slots[ptr]
                ptr += 1
            if prev_none <= run_min:
                base, arg = prev_none, None
            else:
                base, arg = run_min, run_arg
            cand = base + cost(nat, s)
            if s not in cur or cand < cur[s]:
                cur[s] = cand
                cur_back[s] = (arg, s)

        # OFF onset i: floor unchanged, every previous state carries
        # forward at +penalty. Kept only when it beats a PLACE landing on
        # the same state value.
        for pl, c in prev.items():
            cand = c + off_grid_penalty
            if pl not in cur or cand < cur[pl]:
                cur[pl] = cand
                cur_back[pl] = (pl, _OFF)

        prev = cur
        back.append(cur_back)

    # ---- Terminal: cheapest final state, then walk backpointers. ----
    final_state: int | None = None
    final_cost = float("inf")
    for state, c in prev.items():
        if c < final_cost:
            final_cost, final_state = c, state

    assignment: list[int | None] = [None] * n
    state = final_state
    for i in range(n - 1, -1, -1):
        prev_state, action = back[i][state]
        assignment[i] = action
        state = prev_state
    return assignment
