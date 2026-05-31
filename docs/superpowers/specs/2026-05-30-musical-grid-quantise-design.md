# Musical-grid deterministic quantise pass (spec)

## 1. Problem

The geometric snap (`pipeline/quantise.py::_geometric_snap`) is excellent
at matching **the performer's actual playing**: every onset lands on the
1/48 slot closest to where the detector heard it. But the performer is
human. A hit *meant* for one slot can be played a hair past the midpoint
and round onto the neighbouring slot; or be played a consistent full
1/48 off the metronome and land *cleanly* on the wrong slot. Nothing that
reasons purely from audio timing can recover this: the audio genuinely is
closer to the wrong slot.

Today only the LLM residual pass corrects these. It works, but it's the
expensive, non-deterministic backstop. Most of these errors are obvious
from the surrounding rhythm and can be fixed deterministically, cheaply,
and reproducibly, leaving the LLM for the genuinely ambiguous and
cross-instrument calls.

The hard constraint: **do not break legitimate off-straight rhythm**
(tuplets, swing, intentional poly-rhythm across limbs). A correction pass
that squares a triplet groove is worse than no pass.

## 2. Goals

- A deterministic pass between the geometric snap and the LLM residual
  pass that pulls onsets onto the musical grid the surrounding rhythm
  implies, including the full-slot systematic-offset case.
- **Per-lane** grid inference, so simultaneous multi-rhythm across limbs
  (straight hats over a shuffle ride, a triplet kick figure under
  straight 16ths) is preserved, not flattened.
- Tuplet/swing safety by construction: a note is only ever judged "wrong"
  relative to a grid the surrounding **population voted for**, never in
  isolation.
- Carry the geometric snap's sub-slot residual forward as a first-class
  field so both this pass and the LLM stage can use it as information.

## 3. Non-goals (and explicitly deferred)

- **No residual gating.** We do NOT restrict the pass to "coin-flip"
  rounds (|residual| ≈ 0.5). A performer consistently a full slot off the
  beat rounds *cleanly* (residual ≈ 0) onto the wrong slot, so a residual
  gate would skip exactly those systematic offsets. The pass considers
  every on-grid onset and decides on grid membership + support, not on how
  borderline the round was. The residual is informational only.
- **No sub-bar / intra-bar feel changes in v1.** Grid inference is
  per-(lane, bar). A bar that is straight for three beats then a triplet
  fill on beat 4 is inferred as a single grid per lane. This is a known
  limitation (see §8); the real fix is hierarchical per-beat or
  change-point segmentation, recorded here as **deferred future work**,
  not built now.
- **No change to the LLM pass's structure.** It still operates on slots;
  it gains residual annotations in its prompt and inherits cleaner input.
- **No new DSL surface syntax.** This pass only adjusts `quantised_time` /
  `quantised_shift_slots`, same as the existing passes.

## 4. New model field: sub-slot residual

`OnsetCandidate` gains:

```python
# Signed sub-slot residual from the geometric snap: the fractional part
# of the onset's natural slot position, i.e. how far (in slots, range
# (-0.5, +0.5]) the raw detector timing sat from the integer slot it was
# rounded to. + = the hit was late of its slot, - = early. None for
# off-grid onsets (no rounded slot) and for onsets the quantise stage
# never processed. Informational: the musical-grid pass and the LLM
# residual pass read it to understand how confident each round was; it is
# NOT a gate.
quantised_residual_slots: float | None = None
```

Set in `_geometric_snap`: it already computes `naturals[i]` (the
unrounded fractional slot) and `delta = slot - round(nat)`. The residual
is `nat - round(nat)`. Stored on placed onsets only; left `None` for
band-rejected (off-grid) onsets, consistent with how `quantised_time`
stays `None` there.

It threads through `filter` → `quantise` like the existing
`quantised_shift_slots` (both are inspection/provenance fields, untouched
by intermediate stages).

## 5. The grid model

At the default 12 slots/beat the musically meaningful per-beat positions
(slot mod 12) are:

| Grid              | Allowed slots (mod 12) |
|-------------------|------------------------|
| quarter           | {0}                    |
| straight 8th      | {0, 6}                 |
| straight 16th     | {0, 3, 6, 9}           |
| 8th triplet       | {0, 4, 8}              |
| 16th triplet      | {0, 2, 4, 6, 8, 10}    |
| swing 8th (2:1)   | {0, 8}                 |

(The set is derived from `slots_per_beat`, not hardcoded to 12, so it
tracks grid density like the rest of the stage.)

A "grid" is one of these per-beat patterns, tiled across the bar's beats.
Triplet positions (4, 8) are deliberately disjoint from 16th positions
(3, 6, 9), that disjointness is what lets us tell a stray triplet-slot
note in a straight lane from a real triplet.

## 6. Inference + snap

Within the `quantise` stage, a new pass `_musical_grid_snap` runs
**after** `_geometric_snap` and **before** `_llm_residual_pass`.

### 6.1 Per-(lane, bar) grid inference

For each (lane, bar) group of on-grid onsets:

1. Fold each onset to its per-beat position (`slot % slots_per_beat`).
2. Score every candidate grid by sum-of-squared-distance from each folded
   position to the nearest allowed slot in that grid, plus an **Occam
   complexity penalty** proportional to the grid's slot count (so a denser
   grid only wins when it fits *materially* better, a few 16th notes
   don't get promoted to 16th-triplets on noise).
3. The lowest-cost grid wins, but only if it wins by a **decisive margin**
   over the next candidate. If no grid is decisive, the lane+bar is
   "ambiguous" → snap nothing, defer to the LLM.

### 6.2 Sparse-lane fallback (the hierarchy)

A lane with too few onsets in a bar to vote (e.g. a crash hit twice)
cannot infer its own grid. Fall back, in order:

1. the lane's own per-bar vote (if onset count ≥ a min-evidence threshold),
2. else the **bar aggregate** grid (all lanes folded together),
3. else the **song aggregate** grid (every onset, every bar).

The aggregates are **unweighted**, every lane contributes equally. We
deliberately do NOT anchor on the hi-hat/ride: songs with a syncopated
hat over an otherwise straight kick/snare are common, and weighting the
hat would drag the aggregate off the rhythm the rest of the kit is
playing. Per-lane voting already gives a syncopated hat its own grid; the
aggregate only exists to cover lanes too sparse to vote.

### 6.3 Snap decision

For each on-grid onset, find the nearest allowed slot of its inferred grid.
Snap it there iff:

- the move is within the snap tolerance (±1 slot to start; a knob), AND
- the resulting per-(lane, bar) slots stay **strictly increasing and
  in-bar**, reuse the exact monotonic-injective guard
  `_apply_llm_shifts` already enforces, applied atomically per group.

No residual gate (per §3). The residual may be used only as a *tiebreaker*
when two grid slots are equidistant (snap toward the side the raw timing
leaned).

Onsets already on their inferred grid don't move. Off-grid (band-rejected)
onsets are never touched; their position is geometric truth.

### 6.4 Output

Mutates candidates in place exactly like the other passes: sets
`quantised_time` to the new slot time and accumulates into
`quantised_shift_slots`. The debug summary (`quantise/shifts.json`) gains
a `grid_shifted` count and a per-note `grid_shift` / `residual_slots`
field, and the runner log reports the geometric/grid/LLM shift counts
side by side. The pass also logs the inferred-grid distribution across all
(lane, bar) groups (e.g. `{straight_16: 12, triplet_8: 2, deferred: 3}`)
so a human can see *what* it decided and how often it deferred.

**As-built note:** the per-(lane, bar) inferred grid is surfaced as that
aggregate log line rather than a per-group record in `shifts.json`. If
per-group provenance proves needed during evaluation, promote it to a
field then.

## 7. Effect on the LLM residual pass

- The LLM receives cleaner input, most obvious rounding errors are
  already fixed, so it can focus on ambiguous and cross-instrument calls.
- Onsets are annotated with their residual in the prompt, e.g.
  `#7(s) [raw timing +0.45 slot]`, so the model knows which hits were
  near-misses versus confidently placed. (Prompt already instructs the
  model that the snap is audio-only and to trust musical context; the
  residual makes that concrete.)
- The inferred grid per (lane, bar) MAY also be surfaced to the LLM as
  context ("this lane/bar reads as straight 16ths"), optional, evaluate
  whether it helps or over-anchors.

## 8. Tuplet safety, why this can't square a real tuplet

Every judgement is relative to a grid the **population voted for**:

- A lane genuinely playing triplets votes the triplet grid (lowest cost),
  so its notes snap *toward* triplet slots, preserved, not squared.
- A mostly-straight lane with one stray note on a triplet slot votes
  straight; the stray loses the vote and snaps to the straight grid. This
  is the sibling-consistency idea, realised as per-lane voting.
- Simultaneous multi-limb poly-rhythm survives because grids are inferred
  **per lane**, never shared across lanes.

**Known limitation (per-bar):** a *legitimate short tuplet burst inside an
otherwise straight bar+lane* (e.g. a 3-note triplet fill on beat 4 of a
straight-16th hat lane) would lose the bar-level vote and be squared. Two
mitigations in v1: the decisive-margin requirement (a few triplet notes
won't flip a strongly-straight bar, but they also won't be protected) and
the LLM backstop (it sees the residual annotations and can re-spread
them). The real fix is **sub-bar inference**, hierarchical per-beat grids
with the bar grid as a prior, or change-point segmentation along the
onset stream. **Deferred to a future iteration; recorded here so it isn't
lost.**

## 9. Pipeline placement

```
filter → quantise[ geometric snap → MUSICAL GRID SNAP (new) → LLM residual ] → transcribe
```

All three sub-passes mutate the same candidates in place; ordering is
geometric (audio truth) → grid (deterministic musical correction) → LLM
(judgement). Each later pass sees the previous pass's `quantised_time`.

## 10. Testing

Pure, no-LLM unit tests alongside the existing `tests/test_quantise.py`:

- grid inference picks straight-16th for a 0/3/6/9 lane, 8th-triplet for a
  0/4/8 lane, and returns "ambiguous" for a lane that fits neither
  decisively;
- a stray slot-4 note in an otherwise straight-16th lane snaps to slot 3;
- a genuine 0/4/8 triplet lane is left untouched (no squaring);
- two lanes in the same bar with different grids (straight hats + triplet
  kick) each keep their own grid, the poly-rhythm case;
- the monotonic-injective guard rejects a grid snap that would collide or
  reorder onsets, same as `_apply_llm_shifts`;
- a sparse lane (1–2 onsets) inherits the bar/song aggregate grid;
- `_geometric_snap` records `quantised_residual_slots` with the right sign
  and leaves it `None` for off-grid onsets.

## 11. Open questions / knobs

- Snap tolerance (±1 slot to start) and the decisive-margin threshold are
  empirical, tune against real transcriptions.
- Occam penalty weight: how much better must a denser grid fit before it
  wins. Start conservative (bias toward simpler grids).
- Min-evidence threshold for a lane to vote its own grid vs inherit the
  aggregate.
- Whether to surface the inferred grid (not just the residual) to the LLM.
