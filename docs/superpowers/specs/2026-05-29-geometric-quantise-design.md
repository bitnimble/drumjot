# Geometric onset quantisation + sub-slot note offsets (spec)

Status: design, approved 2026-05-29. Not yet implemented.

Sibling reference: `research/midi-audio-alignment-score.md` (the
monotonic-injective matching idea this borrows) and
`transcriber/app/pipeline/quantise.py` (the stage this replaces the
deterministic half of).

## 1. Purpose

Replace the current naive nearest-1/48-slot snap (plus cross-instrument
cluster pull) in the `quantise` stage with a **per-lane,
monotonic-injective, globally-optimal geometric snap**. The new snap:

- never piles two onsets in the same lane onto one slot (injectivity),
- preserves detected order (monotonicity),
- **rejects** onsets that sit too far from any feasible grid slot,
  leaving them *off-grid* at their detected position rather than forcing
  a wrong snap (swing, ghost flams, push/pull feel survive),
- feeds the existing LLM residual pass unchanged (the LLM still does
  cross-instrument musical reasoning on the on-grid hits).

Off-grid hits are carried to the frontend as a new **sub-slot timing
offset** on the note, so the rendered score and playback reflect the
note's true position instead of snapping it onto the grid.

Two motivating constraints drove the scope:

1. **The DSL must not assume 48 slots per bar.** 48 (1/48-of-whole-note,
   12 per quarter) is a *current MIDI-loader implementation detail*; it
   may grow later. The grid density becomes dynamic, carried as
   metadata, read by every consumer instead of hardcoded.
2. **Notes need sub-slot positions.** The DSL gains a per-note `offset`
   (in milliseconds) so a hit can render and play between grid slots.

## 2. Non-goals

- **No** DSL *surface syntax* for `offset` in v1. It is set
  programmatically by loaders (`from_midi`, the pipeline). Hand-authoring
  syntax (e.g. `n[+12ms]`) is a later step if needed.
- **No** change to the LLM residual pass's *structure*, it operates on
  whatever the geometric snap produced, exactly as today.
- **No** RLRR offset support. RLRR's grid is 1/16; offsets are dropped on
  export (with an aggregate log) and ignored on import.
- **No** outer optimizer / ICP loop. The geometric snap is a single-pass
  exact DP, not an iterative fit (contrast the alignment doc's correction
  stage, which iterates because its correspondence changes with the
  warp; here correspondence is fixed input).

## 3. Architecture: three layered pillars

Built bottom-up; each is testable in isolation.

- **Pillar A, dynamic grid density.** The DSL core (`jot.ts`) is already
  grid-agnostic (positions come from element weights, not slot indices).
  The leak is downstream: `note_position.ts`, the playback offset slider,
  score-debug strings, and `quantise.py` all hardcode 12-per-quarter /
  48-per-bar. Fix: carry the loader's chosen density on
  `globalMetadata.gridDivision` (default 48) and have consumers read it
  via a single helper.
- **Pillar B, `Note.offset` (sub-slot positioning).** New optional
  `Note.offset?: number` in **signed milliseconds**. The note stays
  anchored to its slot; `offset` is a per-note timing nudge applied by
  the renderer (horizontal shift) and playback (scheduling shift), and
  round-tripped through MIDI.
- **Pillar C, geometric DP + band rejection.** Replaces
  `_deterministic_joint_snap`. Produces the off-grid findings that Pillar
  B carries.

Layering: A first (can't sanely add offsets while 48 leaks everywhere),
B on top of A (offset is in ms / beats, independent of grid density), C
on top of B (C produces the data B carries).

## 4. Pillar A, dynamic grid density

### 4.1 DSL change

Add to `Metadata` (`src/dsl.ts`):

```ts
gridDivision?: number;  // loader-chosen grid density (1/N of a whole note); default 48
```

Purely advisory, it does not change how the DSL is interpreted. The slot
count for a bar is `gridDivision × count / unit` (so 48 for 4/4, 36 for
3/4 or 6/8).

### 4.2 Helper

New `src/dsl/grid.ts` (~15 lines):

```ts
export const DEFAULT_GRID_DIVISION = 48;
export function gridDivisionFor(jot: Jot): number {
  return jot.globalMetadata.gridDivision ?? DEFAULT_GRID_DIVISION;
}
export function slotsPerQuarter(jot: Jot): number {
  return gridDivisionFor(jot) / 4;
}
```

### 4.3 Audit list (mechanical; no behavior change)

- `src/note_position.ts`, `SUBDIVISIONS_PER_QUARTER = 12` becomes a
  constructor arg (`slotsPerQuarter`) supplied from the jot.
- `src/jot_view/playback.tsx:149-154`, the `/48` drum-offset slider
  labels its denominator from `gridDivisionFor(jot)`; underlying integer
  slot-shift state is renamed after the active grid.
- `src/jot_view/toolbar.tsx:743`, the "1/48 dotted grid" tooltip text is
  interpolated from the active grid.
- `src/jot_view/score.tsx`, the "1/48" comments/strings become "1/N";
  `gridTicks` math (≈ line 859) reads `gridDivisionFor(jot)`.
- `src/jot_view/store.ts:1961, 2664`, comments + constants derived from
  active grid.
- `src/midi/from_midi.ts`, keeps `gridDivision: 48` as the load-time
  default but **writes it onto `globalMetadata.gridDivision`**.
- `src/midi/to_midi.ts`, reads `globalMetadata.gridDivision` for
  round-trip fidelity. `TICKS_PER_BEAT = 480` is unaffected (a MIDI-file
  concept, separate from grid density).
- `transcriber/app/pipeline/quantise.py`, `SLOTS_PER_BEAT = 12` becomes
  a parameter threaded from a single config knob (default 12).
- Tests (`src/midi/__tests__/midi.test.ts:162,270`,
  `transcriber/tests/test_quantise.py`), parametrize on grid density;
  keep 48 as the default-case assertion.

## 5. Pillar B, `Note.offset` (milliseconds, signed)

### 5.1 DSL change

Add to `Note` (`src/dsl.ts`):

```ts
offset?: number;  // signed ms relative to the note's natural slot position
```

Semantics: a note at slot `s` with local tempo `bpm` plays at
`slot_time(s) + offset / 1000` seconds. Rests get no offset.
Simultaneities are unaffected (an offset on one inner note is a flam-like
spread, which is fine).

### 5.2 Consumers

- **`to_midi.ts` (emit):**

  ```ts
  const baseTick = barStartTick + slotIdx * gridTicks;
  let offsetTicks = 0;
  if (note.offset !== undefined) {
    const bpm = bpmAtTick(baseTick, tempoChanges);
    const ticksPerMs = ticksPerBeat * bpm / 60_000;
    offsetTicks = Math.round(note.offset * ticksPerMs);
  }
  const tick = baseTick + offsetTicks;
  ```

  Backward compatible: absent `offset`, behavior is unchanged.

- **`from_midi.ts` (read):** after the existing nearest-slot snap, compute
  the residual ms from the unrounded tick. If `|residualMs| >= TOLERANCE_MS`
  (default 5, a load-time `FromMidiOptions` knob), set `note.offset`; else
  leave it off (so round-tripped hand-authored MIDI doesn't acquire
  spurious offsets).

  ```ts
  const slotTick = barStartTick + slotIdx * gridTicks;
  const residualTicks = origTick - slotTick;
  const bpm = bpmAtTick(origTick, tempoChanges);
  const msPerTick = (60_000 / bpm) / ticksPerBeat;
  const residualMs = residualTicks * msPerTick;
  if (Math.abs(residualMs) >= TOLERANCE_MS) note.offset = residualMs;
  ```

- **Renderer (`src/jot_view/score.tsx`):** horizontal shift only, no
  badge, no glyph variation.

  ```ts
  const msPerSlot = (60_000 / localBpm) / slotsPerQuarter(jot);
  const offsetPx = (note.offset ?? 0) / msPerSlot * slotWidthPx;
  x += offsetPx;
  ```

- **Playback (`src/jot_view/playback.tsx` / scheduler):** offset is
  already ms, so it adds directly with no tempo lookup.

  ```ts
  const scheduleSec = slotTimeSec + (note.offset ?? 0) / 1000;
  ```

  Composes additively with the global `/48` drum-offset slider (which
  lives at the slot-grid layer; `note.offset` is per-note). No conflict.

- **Selection popup debug details (`src/note_position.ts`):** the *only*
  place the offset is surfaced numerically. Add:

  ```ts
  formatOffset(): string | null {
    return this.offsetMs === undefined
      ? null
      : `${this.offsetMs >= 0 ? '+' : ''}${this.offsetMs.toFixed(1)} ms`;
  }
  ```

  Thread `offsetMs?: number` through `NotePosition`'s constructor input;
  the popup site passes `note.offset`. The default `toString()` picks it
  up on the existing `.filter(Boolean).join(' · ')` chain.

### 5.3 Unchanged

- **RLRR** (`src/rlrr/...`): drop offsets on export with one aggregated
  `console.warn`; ignore on import.
- **Linter**: offset doesn't affect structural validity; no new rules.
- **Parser / DSL surface syntax**: none in v1.

## 6. Pillar C, geometric DP + band rejection

Replaces `_deterministic_joint_snap`. The LLM residual pass is unchanged.

### 6.1 Per-lane, single-pass exact DP

Inputs per lane (from `kept_by_pitch`): in-range onsets sorted by time,
each with a `natural_i` = its **unrounded fractional slot position**
`(beat_in_bar − 1) × slots_per_beat` (NOT rounded, the fractional part
is what makes the quadratic cost discriminate sub-slot distances and
drives the tie-break). The integer feasible window is
`[round(natural_i) − B, round(natural_i) + B]` clamped to the onset's
own bar.

**State** (onsets 1-indexed, processed in time order):
- `dp[i][s]` = min total cost placing onsets 1..i with onset i at slot
  `s ∈ [natural_i − B, natural_i + B]`.
- `dp[i][off]` = min total cost with onset i off-grid.

**Transitions:**

```
dp[i][s]   = cost(i, s) + min( min over s' < s of dp[i-1][s'], dp[i-1][off] )
dp[i][off] = penalty_off_grid + min( min over any s' of dp[i-1][s'], dp[i-1][off] )
```

`s' < s` (strict, on integer slots) enforces monotonicity AND per-lane
injectivity simultaneously.

**Cost:** `cost(i, s) = (natural_i − s) ** 2`, quadratic, biases against
large shifts.

**Off-grid penalty:** `penalty_off_grid = (B + 1) ** 2`, strictly worse
than any feasible on-grid placement, so an onset goes off-grid only when
no feasible slot remains (band-rejected by neighbours, or natural
position outside the band of every slot).

**Tie-break:** equal `dp[i][s]` → prefer `s` closer to `natural_i`,
encoded as `+ ε × |natural_i − s|` (ε ≪ any genuine cost difference).

**Complexity:** O(N · B) per lane with a prefix-min running tally
(transitions only read `min dp[i−1][s' < s]`). Single forward sweep to
fill, single backward sweep (backpointers) to reconstruct from
`argmin over {s, off} of dp[N][·]`. Provably global-optimal over all
monotonic-injective assignments (Bellman). **Zero iterations.**

### 6.2 Module

New `transcriber/app/pipeline/geometric_snap.py`, pure (numbers in,
numbers out, no I/O), mirroring `score.py`'s purity contract. Exposes
`snap_lane(natural_slots, slot_range, B, off_grid_penalty) -> list[int | None]`
(None = off-grid).

`quantise.py` becomes a thin orchestrator: build per-lane natural-slot
lists, call `snap_lane`, write `quantised_time` / `quantised_shift_slots`
/ `off_grid` back onto the `OnsetCandidate`s, then run the unchanged LLM
residual pass.

### 6.3 Output fields

- `OnsetCandidate.quantised_time: float | None`, unchanged; None = use
  raw `time`.
- `OnsetCandidate.quantised_shift_slots: int | None`, unchanged; signed
  integer for on-grid hits.
- **New** `OnsetCandidate.off_grid: bool = False`, distinguishes "not
  snapped because no shift needed" from "deliberately not snapped (band
  rejected)".

The MIDI emitter (`onsets_midi.py`) emits the raw-`time`-derived tick for
`off_grid` hits (no slot rounding). The frontend's slot-snap then yields
a `> 5 ms` residual and writes `note.offset`.

### 6.4 Default B

`B = 2` (parity with today's `_MAX_DETERMINISTIC_SHIFT`). Off-grid
penalty `(B+1)² = 9`. At 120 BPM, 2 slots ≈ 83 ms, so band rejection is
rare in v1, intended. Revisit after Phase 4 evidence (band rejection is
now the *only* thing controlling off-grid promotion, so it may want a
tighter cap, e.g. `B = 1`).

### 6.5 What goes away / stays

Goes: `_CLUSTER_WINDOW_S`, `_slot_weight`, cluster-pull logic,
cross-instrument `present_slots`, the cross-pitch flat-list sort.

Stays: `_initial_slot_for` (reused for natural slot), all LLM helpers
(`_index_for_llm`, `_format_for_llm`, `_extract_shifts`,
`_llm_residual_pass`, `_apply_llm_shifts`). Onsets with `off_grid = True`
are presented to the LLM as informational context but excluded from the
shift-target set.

## 7. Build order

### Phase 1, Pillar A (grid dynamism; no behavior change)

1. `gridDivision?: number` on `Metadata`.
2. `src/dsl/grid.ts` + tests.
3. Refactor `note_position.ts` to take `slotsPerQuarter` arg; update call
   sites.
4. `from_midi.ts` writes `globalMetadata.gridDivision`; `to_midi.ts` reads
   it.
5. Re-derive UI constants (`score.tsx`, `playback.tsx`, `toolbar.tsx`,
   `store.ts`) from `gridDivisionFor(jot)`; interpolate tooltip strings.
6. `quantise.py` `SLOTS_PER_BEAT` → parameter (default 12).
7. Parametrize a `midi.test.ts` variant at `gridDivision = 96`.

Gate: full suite green; `fromMidi → toMidi` round-trip stable.

### Phase 2, Pillar B (Note.offset; not yet populated from loader)

1. `offset?: number` on `Note`.
2. `to_midi.ts` honors offset on emit.
3. `score.tsx` renders horizontal shift.
4. `playback.tsx` applies offset to scheduling.
5. `NotePosition.offsetMs` + `formatOffset()`; thread from popup.
6. RLRR drops offset on export (aggregate log).
7. Tests: hand-authored Jot with `offset = +12.3` round-trips, renders
   shifted, plays shifted.

Gate: hand-authored offset fixture renders + plays correctly; round-trip
stable.

### Phase 3, Pillar C algorithm (B = ∞, no band rejection)

1. `geometric_snap.py` + `test_geometric_snap.py` (injectivity,
   monotonicity, tie-break, single onset, all-feasible, slot
   contention).
2. Refactor `_deterministic_joint_snap` → `_geometric_snap` in
   `quantise.py`.
3. Retire cluster-pull test cases; add per-lane injectivity assertions.

Gate: synthetic per-lane inputs produce expected assignments; pipeline
tests pass.

### Phase 4, Band rejection end-to-end

1. `off_grid: bool = False` on `OnsetCandidate`.
2. Set `B = 2`, penalty 9; LLM `_index_for_llm` skips off-grid hits.
3. `onsets_midi.py` emits raw-time tick for off-grid hits.
4. `from_midi.ts` (offset-aware from Phase 2) reads them → `note.offset`.
5. Test: synthetic onset 3 slots out → `off_grid = True`,
   `quantised_time = None`, raw `time` survives.

Gate: a known-tricky real track (swing, ghost flams) renders with visible
shifts on off-grid hits; the LLM doesn't re-snap them.

### Phase 5, Bench LLM-on vs LLM-off (decision step, no code)

1. Surface the existing `use_llm` arg as a `quantise_use_llm` form param.
2. Run the same audio twice (LLM on / off); diff `quantise/shifts.json`
   summaries and rendered MIDIs.
3. Decide: drop the LLM stage, keep it, or investigate consistent
   disagreement.

## 8. Testing strategy

| Phase | Primary verification |
|---|---|
| 1 | Existing suite + parametrized `gridDivision=96` variants |
| 2 | Hand-authored offset fixture round-trip + render |
| 3 | Unit tests on `geometric_snap.py`; retired cluster-pull tests |
| 4 | Synthetic off-grid onset + real-track manual eyeballing |
| 5 | Pipeline diff between LLM-on / LLM-off |

## 9. Risks / open questions

- **5 ms `from_midi` tolerance:** too tight → spurious offsets from
  integer-tick round-trip noise; too loose → real off-grid hits get
  snapped away. Revisit after Phase 4.
- **`B = 2` default:** same value as the old deterministic cap, but the
  *meaning* changed (old: cluster-pull cap; new: feasibility band). Band
  rejection might want `B = 1`. Re-evaluate after Phase 4.
- **`gridDivisionFor` fallback to 48:** good for backward compatibility,
  but masks a future bug where the field is forgotten. Consider a strict
  mode asserting presence after a deprecation window.

## 10. As-built notes (deviations from the plan)

Implemented 2026-05-29. Four places where the build diverged from §4–§7
after the code revealed the plan's premise was off:

- **`to_midi.ts` needs no `gridDivision` read.** §4.3 had `to_midi` read
  `globalMetadata.gridDivision` "for round-trip fidelity". In fact the
  writer derives every note's tick from the rendered *layout* (element
  beat-fraction × bar ticks), which already encodes the grid, so a
  `gridDivision = 96` jot round-trips bit-for-bit with no `to_midi`
  change. Proven by `midi.test.ts` "round-trips note ticks at a denser
  (1/96) grid", which passed before `to_midi` was touched.
- **RLRR *applies* the offset; it doesn't drop it.** §5.3 said to drop
  offsets on RLRR export (on the assumption RLRR is 1/16-grid-quantised).
  RLRR event times are actually real-time **seconds**, so the offset
  composes exactly like playback, a swung hit charts at the time it
  plays. `jot_to_rlrr.ts` adds `offset/1000` to the event time. (Import
  still snaps to RLRR's 1/16 grid, so the offset is lost on the way back
  in, matching "ignore on import".)
- **`onsets_midi.py` needs no change.** §6.3 listed a `onsets_midi` edit
  to emit the raw-time tick for off-grid hits. The emitter already falls
  back to raw `time` whenever `quantised_time is None`, so leaving
  off-grid onsets' `quantised_time = None` (which the geometric snap
  does) emits the raw tick for free.
- **Phases 3 + 4 shipped together; snap is per-(lane, bar), not
  cross-bar.** The DP runs per (lane, bar) with the feasible window
  clamped to the bar's `[0, slots_per_bar − 1]` (new `min_slot`/`max_slot`
  args on `snap_lane`), preserving the old "no bar crossing" guarantee.
  Consequently band rejection is a *contention / bar-edge* phenomenon
  (more onsets than free in-band slots), not "an onset 3 slots from any
  grid line", every fractional position rounds to a slot ≤ 0.5 away, so
  a lone onset is never rejected. Placed onsets now get `quantised_time`
  set to their exact slot time **even at zero shift** (so the frontend
  sees ~0 residual and adds no spurious offset); only off-grid onsets keep
  `None`. The `quantise/shifts.json` summary keys were renamed
  (`deterministic_* → geometric_*`) and gained an `off_grid` count.

**Phase 5 status:** the code half is done; `quantise_use_llm` is a
request option (`PipelineOptions.quantise_use_llm`) and a form param on
both `/transcribe` and `/transcribe/resume`, threaded to
`quantise_kept_onsets(use_llm=…)`. The actual A/B (run real audio twice,
diff `shifts.json` + MIDIs, decide the LLM's fate) is a manual
data-collection step requiring audio fixtures + Anthropic budget; not run
here.

**Not yet verified in-browser:** the score render shift (§5.2, Pillar B)
applies `note.offset` to the `--note-beat` CSS var via `BarTimingsContext`
+ `msOffsetToBeats` (the conversion is unit-tested). There is no React
DOM test harness, so the *visual* shift hasn't been confirmed in a
running browser, worth an eyeball before final sign-off.
