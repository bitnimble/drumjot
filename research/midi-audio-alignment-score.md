# MIDI ↔ drum-audio onset-alignment score (spec)

Status: design, reviewed (subagent pass May 2026), not yet implemented.
Sibling reference: `benchmarks/core/score.py` (the existing binary
3-class onset F1) and `app/pipeline/adtof_onsets.py` (the audio onset
detector this reuses).

## 1. Purpose

Given **(a)** a drum stem (or the per-instrument stems / `onsets.json`
the pipeline already derives from it) and **(b)** an *externally
sourced* MIDI file, produce a scalar **0–100** for "how well do the
MIDI's onsets line up with the audio's onsets," plus the diagnostics
and a *corrected* MIDI.

The MIDI is **external** (a third-party chart, a downloaded `.mid`),
never our own `prediction.mid`. That independence is what makes reusing
the pipeline's own ADTOF onsets as the audio reference legitimate; the
circularity that would poison QA of our own predictions does not apply
here.

Two distinct deliverables come out of one machine:

1. **Score**; the rigid 0–100 alignment number (this doc's core).
2. **Correction**; a separate, *downstream* stage that warps the MIDI
   to best fit the audio and emits a cleaned `.mid`. It consumes the
   score as its objective but never changes how the score is defined.

## 2. Non-goals

- **Not** a transcription-quality metric for our own pipeline. That is
  `benchmarks/core/score.py` against dataset ground truth.
- **No** symbolic/musical judgement (no "is this the right groove"); purely onset timing + presence.
- **No** new dependencies. `mido`, `mir_eval`, `librosa`, `scipy`,
  `numpy` are all already in `pyproject.toml`; audio onsets come from
  the existing `detect_onsets_adtof`.
- The scorer does **not** separate audio itself. Its audio input is
  per-instrument onset times; either the live pipeline's
  `onsets_by_pitch` or a cached `onsets.json` from a debug folder.

## 3. The one invariant that defines everything: the score is RIGID

The score is the **objective function of the correction optimizer**.
That single fact forces every other choice:

- **No pre-alignment is baked into scoring.** We do *not* remove a
  global offset or tempo error before measuring. If we did, the score
  would be constant w.r.t. offset and the correction loop would have a
  flat objective with nothing to climb. Offset, tempo error, and
  per-note jitter must *all* register as cost, measured at the notes'
  current positions.
- **The score is one function evaluated at two points in the loop**:
  pre-correction (offset counts against you → low) and post-correction
  (residual only → high). These are not two metrics.
- **The match band decides *who pairs with whom*; it must NOT decide
  *how much credit they get*.** Correspondence is established with a
  ±50 ms monotonic-injective match, but credit is a *soft* Gaussian
  kernel with **no flat zone**; it keeps charging for every
  millisecond of error out to the band edge. A binary in-tolerance
  match (what `mir_eval.onset.f_measure` does) would make the objective
  a step function: flat plateaus, no gradient, optimum is "good enough"
  rather than "centered." That is why this scorer does **not** reuse
  `score.py`'s binary F1 as the loop objective.
- **Climbable only *inside* the ±B capture region.** Be precise: the
  score is smooth and strictly improving as offset shrinks *once pairs
  are within the band*, but with free gaps it is **flat zero** when no
  MIDI onset is within ±B of any audio onset (large initial offset), and
  it steps by ~`e^-2` per pair at the band edge. So a continuous-gradient
  optimizer on the raw score would stall on the outer plateau. This is
  *why tier-0 cross-correlation (§8) is mandatory, not optional*: it
  pulls the MIDI inside ±B first, where the soft score is then a usable
  objective. Tiers 1–2 fit on *frozen* correspondences, sidestepping the
  band-edge discontinuity.

## 4. Onset representation (per lane)

**Lanes.** Score per drum lane, matching the ADTOF lanes the audio
detector exposes (`app/pipeline/adtof_onsets.py::_LANE_FOR_PITCH`):

| Lane | Audio source | MIDI side (GM fold) |
|---|---|---|
| `k` kick | ADTOF BD lane | 35, 36 |
| `s` snare | ADTOF SD lane | 37, 38, 40 |
| `t` toms | ADTOF TT lane | 41, 43, 45, 47, 48, 50 |
| `h` hi-hat | ADTOF HH lane (closed+open) | 42, 44, 46 |
| `cy` cymbals | ADTOF CY+RD lane (ride+crash **merged**) | 49, 51, 52, 53, 55, 57, 59 |

> **Limitation, stated up front:** the audio side merges ride and crash
> into one cymbal lane (the ADTOF model has no separate ride/crash
> class; see the `d`/`c` note in `adtof_onsets.py`). So the MIDI's
> ride and crash are folded into one `cy` lane for scoring too. A MIDI
> that gets ride-vs-crass *identity* wrong but timing right still scores
> well on `cy`. This matches the audio's actual resolving power; don't
> over-claim cymbal identity accuracy.

**The GM→lane fold is sourced from `src/midi/gm.ts::GM_PERCUSSION`, the
canonical and *current* mapping. NOT the stale 3-class
`benchmarks/core/classes.py::GM_PITCH_TO_CLASS`.** `gm.ts` is richer and
maintained: it covers note 53 (ride bell → ride), splits toms across
letters `f`/`t` (both fold to the toms lane), maps hi-hat variants
(42 closed / 44 pedal / 46 open) to `h`, and has a deterministic
fallback allocator (`allocatePitchesForMidi`) for non-GM notes. The
folds above are exactly its GM-note → DSL-pitch table, then DSL pitches
grouped into the 5 audio lanes (`k`, `s`, toms `f`+`t`→`t`, `h`,
crash `c`+ride `d`→`cy`). Notes that `gm.ts` maps to `p` (clap) or `b`
(tambourine/cowbell) have **no ADTOF lane** and are dropped (counted as
`unmapped_midi_notes`). See "parser-language decision" in §11 for
whether the Python table is a hand-port of `gm.ts` (with a drift-guard
test) or a bun bridge that reuses it directly.

**Audio onsets.** `list[(time, strength)]` per lane. **The two audio
sources are NOT in the same lane space and must each be normalized into
the 5 lanes above:**

- **Live detector** ; `detect_onsets_adtof(stem_path, pitch,
  drum_stem_path=…)`. Cymbals are read with `pitch="c"` off ADTOF lane 4
  (the merged `CY+RD`), so ride/crash already arrive merged ; matches the
  `cy` model directly.
- **Cached `onsets.json`** ; written by the `onsets` stage *after* the
  `cymbal_split` and `hihat_split` stages, so its keys are
  `k, s, t, h, H, d, c` ; ride (`d`) and crash (`c`) are **already
  split**, and there is a separate synthetic open-hat lane `H`. The
  loader **must fold `d`+`c` → `cy` and `H` → `h`** to reach the 5-lane
  space. Each value is a list of objects, not bare times:
  `{"time": float, "strength": float, "bar": int, "beat_in_bar": float,
  "quantised_time": float|null, "quantised_shift_slots": int|null}`
  (`OnsetCandidate.model_dump()`). **Use `time`, never `quantised_time`**
  ; quantised times are grid-snapped and would corrupt a timing
  reference.

Strengths are available and may later weight the cost; v1 ignores them
(treats all onsets equally).

> **Reference-set bias (document, don't silently absorb):** the
> pipeline's onsets are tuned high-recall ("detect hot, the LLM prunes")
> and contain phantom hits the downstream filter is *meant* to remove.
> Scoring against the un-pruned `onsets.json` therefore penalizes a clean
> MIDI on *recall* (`TPQ/J`) for not matching audio onsets that aren't
> real. Offer `filter/kept_onsets.json` (the LLM-pruned set) as an
> alternative audio reference, and state which set a given score used.

**MIDI onsets, output is per-lane onset times in RAW SECONDS,
un-quantized.** The correctness reference is `src/midi/from_midi.ts`
(current and maintained), *not* `benchmarks/core/midi_events.py` (stale,
buggy). Two lessons carry over from the TS, one does not:

- **(carry over) Whole-file tempo-aware timing.** `from_midi.ts` merges
  *all* tracks into one absolute-tick timeline and sorts before reading
  tempo (`[A10]`/`[A4]`), so a conductor track's `setTempo` applies to a
  separate drum track. The Python equivalent is trivial and is the
  *entire* fix for the multi-track bug: iterate **`for msg in
  mido.MidiFile(path)`**, which yields merged, tempo-aware messages with
  **`msg.time` already in seconds**; accumulate it. Do **NOT** iterate
  `mid.tracks` (per-track, raw ticks, default-120-BPM); that is exactly
  the `_events_from_mido` bug, which is invisible on our single-track
  `prediction.mid` but mis-times type-1 multi-track external MIDI (the
  target input).
- **(carry over) GM mapping** from `gm.ts` (§4 above).
- **(do NOT carry over) quantization.** `from_midi.ts` snaps onsets to a
  1/48 grid to build a musical `Jot` (`[A2]`) and never emits per-note
  seconds. We need the opposite: raw, un-snapped onset seconds; grid
  quantization would destroy the very timing signal we score. So we do
  not call `fromMidi`; we reuse only its parse-and-map approach.

Take note-ons (velocity > 0), fold GM pitch → lane. **Channel policy:**
prefer channel 9 (GM drums); if a file has *no* channel-9 note-ons, fall
back to all channels rather than silently returning an empty set (a
non-channel-9 drum chart must not be indistinguishable from "totally
misaligned" ; surface "no drum channel found" if neither yields
drum-mapped notes). `from_midi.ts` makes the drum channel configurable
(`drumChannel`, default 10/idx 9) but has no all-channel fallback; add
one here, since external charts are less disciplined than our own
output. Notes whose GM pitch maps to no lane are dropped (and counted;
see §7 diagnostics, "unmapped MIDI notes").

## 5. The scoring function

Per lane, with MIDI onset times `M = [m_1 < … < m_I]` and audio onset
times `A = [a_1 < … < a_J]`:

**Match reward** (soft, no flat zone):

```
reward(i, j) = exp( -(m_i - a_j)^2 / (2 σ²) )   if |m_i - a_j| ≤ B
             = disallowed                        otherwise
```

**Monotonic, injective, bounded matching** via a DP that maximizes
total reward over an order-preserving partial matching (each onset used
at most once):

```
dp[i][j] = max(
    dp[i-1][j],                       # MIDI i unmatched  (insertion / extra)
    dp[i][j-1],                       # audio j unmatched (deletion / miss)
    dp[i-1][j-1] + reward(i, j)       # match i↔j         (only if within band B)
)
dp[0][*] = dp[*][0] = 0
TPQ = dp[I][J]                        # total matched quality
```

This is bounded DTW / Needleman–Wunsch with a soft substitution and
free gaps. The monotonicity + injectivity is what makes a *structurally
wrong* MIDI expensive: it runs out of distinct, order-preserving
targets within ±B and the leftovers fall through as insertions/
deletions. Restrict the DP to the band (`|m_i − a_j| ≤ B`) for
`O(N · band/gap)` cost.

**Per-lane soft P/R/F1:**

```
soft_precision = TPQ / I        # extra MIDI notes drag this down
soft_recall    = TPQ / J        # missed audio onsets drag this down
soft_f1        = 2·P·R / (P+R)   # := 0 when P+R == 0  (guard the 0/0)
```

`reward ∈ (0,1]` so `TPQ ≤ min(I,J)` and `P,R,F1 ∈ [0,1]`. Edge cases,
mirroring `score.py`: a lane empty on **both** sides is skipped; a lane
with onsets on only **one** side scores `f1 = 0`; and when `TPQ = 0`
(nothing within band) `P = R = 0` → **define `soft_f1 = 0`** (don't let
`0/0` NaN through). `f1_weighted`'s denominator (Σ audio counts) can be
0 when no scored lane has audio onsets → fall back to 0, as
`score.py:132` does.

> **Monotonic vs bipartite (false-penalty mode to be aware of):**
> `mir_eval` uses order-free optimal bipartite matching; this DP forces
> *monotonic* matching. They diverge when onsets cross within the band
> (MIDI A→B but audio B'→A'): bipartite matches both, the monotonic DP
> can keep only one and the other falls through as ins+del. So a MIDI
> with a couple of *locally reordered* hits (flam/grace-note ordering)
> scores slightly worse here than `mir_eval` would. This is an accepted
> cost of the structural-error sensitivity ; but it also means the §10.5
> "report `mir_eval` for comparability" numbers will differ
> *systematically* from the soft score even on good MIDI (different
> correspondence), which is expected, not a bug.

**Aggregate → 0–100.** Two roll-ups, both reported (as `score.py`
does):

- `f1_macro` = mean of per-lane `soft_f1`.
- `f1_weighted` = per-lane `soft_f1` weighted by that lane's **audio**
  onset count `J` (busy lanes count more).

Headline score = `round(100 · f1_weighted)`. A `class_weights` override
(e.g. kick/snare > hats) is a config knob, default uniform.

**Why the coverage term stays even though the band handles timing:** in
a *saturated* lane (real 16th hats everywhere, MIDI also dense but
rhythmically wrong) a feasible monotonic match with small residuals can
still exist; injectivity raises the floor but doesn't guarantee a bad
score there. The P/R coverage term is what catches density/count
mismatch in that regime. Timing-residual and coverage cover each
other's blind spots; keep both. (In practice the timing term still
bites until onset spacing approaches `2B` ; at ≤ ~100 ms spacing, i.e.
faster than ~16ths at 150 BPM, a wrong-but-saturated lane is where the
soft score is least trustworthy and coverage carries the weight.)

## 6. Defaults

| Param | Default | Notes |
|---|---|---|
| `B` (match band) | 50 ms | correspondence gate; matches the field's onset tolerance |
| `σ` (kernel) | 25 ms | `B = 2σ` → edge reward ≈ `e^-2` ≈ 0.14; tunable |
| `class_weights` | uniform | optional per-lane weights |
| per-note nudge cap | 50 ms | §8 correction only |
| offset search bound | ±2 bars (≈ wide) | §8 correction only |

`B` and `σ` are separate knobs on purpose (§3): `B` is correspondence,
`σ` is credit.

> **Tune `σ` against the detector, not just `B`.** ADTOF on isolated
> stems smears transients by tens of ms (its activations are OOD; see
> `adtof_onsets.py`), even after the ±window audio-envelope refine. If
> the detector's own timing jitter is ~30 ms, a *perfect* MIDI scores
> visibly < 100 and `score_corrected` plateaus below 100 for reasons
> that are the reference's error, not the MIDI's. Calibrate `σ` to the
> detector's empirical timing variance rather than defaulting `B = 2σ`
> blindly.

## 7. Outputs (interface)

```
AlignmentResult:
  score: int                      # 0–100, = round(100 · f1_weighted), pre-correction
  f1_macro: float
  f1_weighted: float
  per_lane: {lane: {soft_f1, soft_precision, soft_recall, n_midi, n_audio}}
  # diagnostics (populated by the correction stage, §8):
  offset_sec: float | None        # global shift the corrector applied
  tempo_ratio: float | None       # affine slope a (1.0 = no tempo error)
  nudge_energy_sec: float | None  # Σ|per-note correction|; a quality signal in itself
  score_corrected: int | None     # rigid score re-evaluated at corrected positions
  unmapped_midi_notes: int        # GM pitches that folded to no lane
```

A large `offset_sec`, `|tempo_ratio − 1|`, or `nudge_energy_sec` is its
own red flag ("this MIDI was authored at the wrong tempo / start
reference") even after correction has fixed it.

## 8. Correction stage (separate, downstream)

Independent of scoring. Finds a warp of the MIDI that minimizes the
(rigid) score, emits a cleaned `.mid`, and reports what it had to do.
The score is **measured before per-note nudging** at each evaluation
point, so per-note freedom can never inflate the reported number.

**ICP-style alternation** (the agreed structure): freeze correspondence
→ optimize the warp to convergence → re-solve correspondence once →
repeat until correspondence stops changing. Smooth within each round,
adaptive across rounds.

Tiered warp family, coarse→fine (each tier initializes the next, which
keeps the flexible tiers out of local minima):

0. **Coarse global offset**; cross-correlate summed per-lane onset
   impulse trains (MIDI vs audio) over the offset bound; argmax lag.
   Threshold-free, gets within the band in one shot (this is the only
   place the continuous-envelope/cross-correlation idea is used).
   Rasterize the impulse trains to a grid of **≤ 10 ms** bins so the
   argmax reliably lands inside ±B; the search is cheap (a few thousand
   lags over ±2 bars).
1. **Offset + affine tempo**; `t' = a·t + b`. With the current DTW
   correspondence, robust (Huber) least-squares fit on matched pairs.
   Recovers "MIDI authored at slightly wrong BPM, walks out of sync."
   **Bound `a ∈ [0.5, 2.0]` and require ≥ 3 matched pairs** before
   trusting the fit ; an unbounded slope (or a 1-pair, under-determined
   fit) lets tier-1 collapse/expand the MIDI onto accidental
   coincidences and quietly inflate the re-solved score. This bound is
   what actually keeps the post-correction score honest (the ±50 ms
   tier-2 cap does nothing to constrain `a`); a large `|a − 1|` is the
   §7 red-flag diagnostic, not a free pass.
2. **Per-note bounded nudge**; freeze correspondence; move each matched
   MIDI note toward its audio onset, **capped at ±50 ms** of total
   displacement (from its tier-1 position) and **preserving monotonic
   order + no two notes on one onset** (injective). Re-solve DTW; repeat.

The ±50 ms cap and no-overlap constraint are what stop a bad MIDI from
being snapped to a perfect score: a note genuinely 200 ms off can only
be pulled 50 ms (150 ms residual remains), and notes can't pile onto a
single onset; a sloppy MIDI runs out of feasible distinct targets and
its score stays low.

The corrector reports `offset_sec = b`, `tempo_ratio = a`,
`nudge_energy_sec = Σ|tier-2 displacement|`, and the corrected MIDI.
Stiffness lives entirely here (which tiers you enable); the score itself
has no stiffness knob.

## 9. Module layout

```
benchmarks/core/
  alignment.py      # PURE: soft-cost monotonic DTW + per-lane soft P/R/F1
                    # + roll-up. No I/O, no logging; pytest-friendly,
                    # mirrors score.py's purity contract.
  correction.py     # ICP alternation, tiered warp, ±50ms/no-overlap.
                    # Pure given onset lists in; returns warp + corrected times.
  lane_fold.py      # GM pitch -> 5-lane fold (or extend classes.py).
benchmarks/
  align_midi.py     # CLI: --midi PATH --onsets onsets.json | --stems DIR
                    #      [--correct] [--out corrected.mid]. Resolves audio
                    #      onsets (load onsets.json or call detect_onsets_adtof),
                    #      parses MIDI, scores, optionally corrects, prints JSON.
transcriber/tests/  # (the repo's Python tests live here, NOT benchmarks/tests/)
  test_alignment_score.py       # synthetic onset lists -> known scores
  test_alignment_correction.py  # known offset/tempo -> recovered to ~0 residual
```

`alignment.py` and `correction.py` stay pure (numbers in, numbers out)
so they test with synthetic lists and never need audio. The CLI is the
only piece that touches `detect_onsets_adtof` / files. **Keep the
`detect_onsets_adtof` import lazy** (inside the `--stems` branch):
`benchmarks/` has no existing `app` import, and `detect_onsets_adtof`
drags in torch + adtof_pytorch ; the `onsets.json` path must stay
audio-free and heavy-import-free.

## 10. Build order

1. `lane_fold.py` + the GM→lane table ported from `src/midi/gm.ts`
   (+ drift-guard test, §11 option A). Plus the `mido`-based
   MIDI→onset-seconds reader (`for msg in MidiFile(...)`, raw seconds,
   no quantization, channel-9-with-fallback).
2. `alignment.py`: the DP + soft P/R/F1 + roll-up. Tests with synthetic
   lists: perfect match → 100; uniform 20 ms shift → high but < 100
   (kernel charges for it); extra/missing notes → P/R drop; saturated
   wrong-rhythm lane → coverage term bites.
3. `align_midi.py` CLI wired to `onsets.json` first (no audio needed),
   then to live `detect_onsets_adtof`.
4. `correction.py`: tier 0 (offset) → verify recovers a known injected
   offset; tier 1 (affine) → recovers injected tempo error; tier 2
   (per-note ICP) → cap + no-overlap honored; `score_corrected` > `score`.
5. Optional: report `mir_eval.onset.f_measure` at corrected positions
   alongside, for comparability with the existing benchmark numbers.

## 11. Open questions / deferred

- **Strength-weighted reward**: should a strong audio onset missed by
  the MIDI cost more than a ghost hat? Deferred; v1 unweighted.
- **Cross-lane shared clock in correction**: tiers 0–1 fit one global
  `(a, b)` across all lanes (drums share a clock); tier 2 nudges per
  lane. Confirm that split is what we want vs a fully joint solve.
- **Score-vs-corrected as the headline**: do we surface the
  pre-correction score, the corrected score, or both as the headline?
  Spec currently makes `score` pre-correction and `score_corrected`
  the post.
- **Endpoint**: v1 is a CLI/offline tool. Exposing it as a service
  endpoint is a later step if needed.
- **Parser-language decision (needs a call before build).** `src/midi/`
  + `gm.ts` are the correct, current MIDI logic, but the transcriber is
  now **pure Python**; the bun bridge (`tools/jot_to_onsets.ts` etc.)
  that AGENTS.md §5.2 describes was **removed in the May 2026 purge**;
  there are no `.ts` files in `transcriber/` and no bun in the live
  pipeline. So the two real options are:
  - **(A) Python port (recommended).** Implement the MIDI→onset-seconds
    reader in Python with `mido` (already a dep), porting the `gm.ts`
    GM→lane table. Address the staleness concern not with language but
    with a **drift-guard test**: a fixture `.mid` whose expected per-lane
    onset seconds were generated from the TS path, asserted in
    `tests/test_alignment_score.py`, so a future `gm.ts` change that
    isn't mirrored fails CI. Keeps the offline tool self-contained, no
    toolchain resurrection.
  - **(B) Bun bridge.** Reintroduce a `tools/midi_to_onsets.ts` that
    reuses `gm.ts` directly and shell out from Python (the old §5.2
    pattern). True single-source-of-truth, but re-adds bun + node_modules
    to the transcriber image/dev loop that was just deleted; a heavy
    price for one offline tool.
  Recommendation: **A**, unless we expect `gm.ts` to churn a lot. Either
  way, the GM source of truth is `gm.ts`, never the stale Python 3-class
  fold.
```
