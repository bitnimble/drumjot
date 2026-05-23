# Design notes for a future AI-assisted MIDI → Jot converter

Status: **research only** — captured at the moment the legacy DSL-output
transcribe endpoint and the librosa onset backend were deleted (May 2026).
The backend now produces only MIDI; the deterministic
`src/midi/from_midi.ts` is what currently turns that MIDI into a Jot.

That deterministic pass is fine for the obvious shape — quantize each
onset to a 16th grid, look up the kit pitch by GM note, assemble bars —
but several decisions a notation engine actually needs are higher-order
reasoning problems the deterministic path cannot solve:

- **Tuplet / feel detection per bar** (straight16 vs triplet vs shuffle
  vs swing). A run of three near-equal hits inside one beat could be
  triplets, very tight 32nds, or sloppy 16ths; the right call depends
  on the surrounding groove and tempo.
- **Pattern factoring** (`[Groove=(...)]` + references). Identifying
  repeating bars or sub-bar motifs and rewriting them as named patterns
  is the difference between a one-page chart and a sixteen-page wall of
  notes.
- **Hand voicing for ambiguous polyphony.** Which hits go on the same
  staff line / get joined with `+`, which go in a second voice via
  `||`, and which become a `(A) + (B)` polyrhythm group when the
  subdivisions don't share a clean grid.
- **Sticking inference** (`@stick`). Pure cross-hand notation needs the
  full kit context.
- **Dynamics / accent grouping.** Velocities turn into `:a` / `:g` /
  per-note `vol` only after a pass that groups numerically close hits
  into musical accent levels.
- **Bar-level musical edits** the deterministic quantizer can't make:
  collapsing the same-bar-as-the-bar-before pattern into a `*N`,
  picking up that the kick on `2.5e` is musically the `&` of beat 2 in
  a triplet bar, etc.

This document captures the techniques the deleted DSL-output pipeline
used so a future agent can pull whichever pieces are useful for an
AI-assisted MIDI → Jot pass — without having to re-derive them from the
git history. Nothing here is currently live code. All file references
point at `git log` paths in the commit that removed them.

---

## 1. Per-instrument LLM calls, deterministic merge

The single most important architectural choice. Instead of one LLM call
that emits the full multi-voice Jot, the transcribe stage made **one
small call per drum pitch**, each emitting a single monophonic line
(one pitch letter or `.` per position — no `+`, no `||`, no metadata
block). A deterministic merge then assembled them into one Jot.

Why this was good:

- An autoregressive model maintains one coherent monophonic line far
  more reliably than it interleaves several. Column-merging all limbs
  into one sequence is the worst case for both accuracy and token cost.
- Each call's prompt is tiny → calls run in parallel
  (`instrument_concurrency`), best-of-K applied per instrument scored
  on that pitch's own onset F1, errors isolated and independently
  re-scorable, failures of one pitch don't poison the rest.
- The merge logic stays deterministic (and tested locally without an
  LLM): hands joined with `+` on the *minimal common* grid; hands whose
  only common grid is an LCM blow-up of genuinely different subdivision
  families (e.g. straight 8ths vs triplets) become `(A) + (B)`
  polyrhythm groups; kick goes to the second `||` voice. Code:
  `src/recompose.ts` + the `tools/recompose_jot.ts` bridge.

What it sacrificed (worth knowing for a successor design):

- **No cross-instrument musical context.** Each per-pitch call sees
  only its own onsets + the shared beat frame, so it can't use groove
  context ("the kick is on the `&` so this hi-hat gap is intentional")
  to disambiguate. The mitigation was designed but not built: feed each
  call a read-only summary of other instruments' onset positions
  (compact `pitch beat_in_bar` strings per bar — same shape the filter
  prompt's "others:" block uses today). That's the first lever to try
  if an AI MIDI → Jot pass needs more context.
- **No cross-instrument pattern factoring or sticking generation.**
  Both are inherently multi-voice operations; per-instrument
  monophonic calls have nothing to factor and no hands to alternate.
  Candidates for a v2 that runs *after* the per-instrument pass:
  a deterministic identical-consecutive-bar → `*N` / pattern pass, and
  a deterministic alternating-sticking pass, both on the recomposed Jot.

A future AI MIDI → Jot pipeline could keep the per-instrument shape
(operating on per-pitch MIDI note lists instead of audio-derived
onsets) and add a separate **whole-chart pass** with the merged
notation as input for the cross-instrument decisions (pattern
factoring, sticking, voicing of ambiguous polyphony, accent grouping).

## 2. Per-instrument prompt + monophonic spec subset

Per-instrument calls were given the canonical `SPEC.md` *with the
"Global simultaneity" section stripped at load time* (the `||` operator
plus the reserved-characters table row) — so the LLM never saw `||` in
its grammar and couldn't emit it accidentally. `SPEC.md` on disk was
deliberately untouched; the strip happened in-memory only
(`llm._load_spec_subset()` was the relevant ~30 lines).

The prompt body per call:

- the canonical (stripped) SPEC
- monophonic few-shots (`prompts/examples_instrument.md`)
- the global frame: initial tempo, initial time sig, total bar count
- per-bar listings, one bar per block, of *this pitch's* onsets:
  `Bar 0 [4/4, 120.0 BPM, feel=straight16]:` then
  `  h: (1.000, 7.2) (1.500, 6.8) (2.000, 7.0) ...`
  with `(beat_in_bar, strength)` tuples.
- a single optional parse-error retry hint (the loop gave one more
  shot on parse failure).

The deleted prompt template was at `transcriber/prompts/transcribe_instrument.md`.

For a MIDI input the equivalent payload would be per-pitch lists of
`(beat_in_bar, velocity)` tuples, with the same bar-headered structure.
The MIDI's bar/time-sig/tempo events already give the frame for free —
no separate beat tracker is needed.

## 3. F1-gated multi-level refinement loop

The deleted `pipeline/refine.py` ran the initial transcription through
up to five passes, each gated by a deterministic score:

```
LINT → MACRO → STRUCTURE → ONSETS → VELOCITY
```

The pattern is general and worth keeping in mind for the AI MIDI → Jot
case (with the obvious caveat that some levels are no longer relevant —
ONSETS/VELOCITY were scored against audio stems, which an AI MIDI → Jot
pass doesn't have).

For each level except LINT, one iteration was:

1. Compute a typed issue list (`pipeline/diff.py`): one of
   `missing_onset`, `extra_onset`, `velocity_mismatch`, `tempo_mismatch`,
   each carrying `(pitch, time, confidence, notes, expected_X,
   current_X)`.
2. Triage with a cheap critic LLM (Haiku via Anthropic tool-use, so
   structured output without JSON-from-text parsing) — `pipeline/critic.py`.
   Caps at `max_issues=25` going into the generator. Critic can be
   disabled by setting `CRITIC_MODEL=""`, falling back to deterministic
   confidence ranking.
3. Ask the expensive generator (Opus) to revise the Jot with the issues
   as evidence — `pipeline/refine.py::_generator_revise`.
4. Validate the new Jot parses (retry once on parse error).
5. Score the new Jot against the source via `mir_eval` (`pipeline/score.py`).
6. **Accept only if score strictly improves**, otherwise stop the
   level entirely.

The accept-gate is what made the loop monotone-improving: a bogus issue
list can never make the Jot worse, because a rejection is a no-op.

For the AI MIDI → Jot case the score would not be onset F1 (the input
already is MIDI; quantization is what we're choosing). A better gate
might be:

- **Edit-distance / round-trip fidelity:** convert the new Jot back to
  MIDI via `to_midi.ts` and compare against the input MIDI under some
  tolerance. Acceptance = the round-trip diff is ≤ the previous best's.
- **Pattern-coverage score for STRUCTURE-like passes:** total DSL byte
  count after pattern factoring, weighted by the number of `[Pattern]`
  references — a longer chart with few patterns is "worse" than a
  shorter chart with many.
- **Tuplet-correctness score for a FEEL pass:** count of notes whose
  position is dyadic under the chosen feel divided by the total. A
  triplet pass that puts hits at perfect 1/3 positions wins over one
  that put them at near-1/3.

## 4. LINT pass: surgical per-segment patches

The LINT pass was structurally different and worth preserving as a
pattern. It ran first (cheap deterministic diagnostics, not a diff
against audio) and **patched per segment** rather than asking for a
whole-Jot rewrite.

Sequence:

1. Run the deterministic lint (`pipeline/lint.py` + the bun bridge
   `tools/lint_jot.ts`).
2. Group diagnostics into per-voice "segments" — one or more adjacent
   bars in a single voice that contain at least one diagnostic, with
   ±`LINT_CONTEXT_BARS` of read-only context.
3. For each segment, ask the LLM (`prompts/refine_lint_segment.md`) to
   rewrite that segment's text only — the prompt sees the segment plus
   its diagnostics plus the affected bars' audio onsets, NOT the whole
   chart. The model returns the patched segment text.
4. Apply patches **right-to-left** so earlier byte offsets stay valid
   for the cascade.
5. Re-parse + re-lint the cascaded result; accept if the error count
   dropped.

The blast-radius limit was the whole point: token cost stayed low, one
LLM mistake couldn't perturb unrelated bars, and the accept-gate ran
against a deterministic diagnostic count (no scoring runs needed).

This shape transplants cleanly to an AI MIDI → Jot lint pass — the
diagnostics would just be different (instrument-mapping mismatches,
mid-bar tempo/time changes, malformed `(...)_N` groups, etc.).

## 5. Best-of-K per instrument

The transcribe stage supported best-of-K sampling: K candidates per
drum pitch, each scored on that pitch's own onset F1, best kept.
Implementation lived in `pipeline/llm.py::transcribe_instrument_best_of_k`.

Two pieces are worth carrying forward:

- **Temperature schedule.** First sample always greedy (T=0). Later
  samples ramp diversity: `[0.0, 0.4, 0.7]` for the first three, then
  `0.7 + 0.05·n`. Opus 4.7 ignores temperature internally (extended
  thinking models do their own RL) but the API still rejects the
  override on some models — the loop retries once with `temperature=None`
  on `anthropic.BadRequestError` mentioning temperature. Keep that
  retry path for any Opus generation.
- **Per-instrument scoring is the unit.** A chart-wide best-of-K would
  pick the best chart, which is dominated by whichever instrument has
  the most onsets. Scoring each pitch independently and picking the
  best per pitch gave better aggregate F1 in practice.

For AI MIDI → Jot, "best-of-K per instrument" stays equally sensible
if the per-instrument shape is preserved.

## 6. Pattern-aware issue suppression (the "this isn't a missed hit" trick)

In the ONSETS pass, `pipeline/diff.py::_is_subpulse_flicker` suppressed
`missing_onset` issues on **repetitive pitches** (`{"h", "d"}`) when
the unmatched source onset sat between two predicted hits about one
local pulse apart **and** was weaker than the median kept hit. The
intuition: if the LLM deliberately thinned a 1/16 hi-hat stream to a
1/8 pattern and one of the dropped weak hits is "missing", forcing it
back in re-introduces the very 16th spam the thinning removed. Kick /
snare were excluded from suppression — a missing kick is almost always
real.

Also reframed in this session: the ONSETS prompt
(`prompts/refine_onsets.md`) was rewritten to tell the model that
flagged issues are *evidence it may overrule on musical grounds*, not
commands. The old "address every issue, don't selectively fix a subset"
phrasing made the loop fight the transcription. Combined with the F1
accept-gate ("the LLM rejecting a bogus issue is a safe no-op, not a
regression") this stopped the refinement from drifting into the noise.

A successor pass over Jot output (e.g. an AI pattern-factoring level)
should inherit the same framing: issues as evidence, accept-gate, and
domain-specific suppression rules for known false-positive shapes.

## 7. Bun bridge for DSL manipulation

The Python service never owned a DSL parser, formatter, lint engine,
or recomposer. Every DSL operation shelled out to a TypeScript bridge
running under bun:

- `tools/jot_to_onsets.ts` — DSL → JSON list of `(pitch, time, vel)`
- `tools/recompose_jot.ts` — merge per-instrument fragments → Jot
- `tools/lint_jot.ts` — DSL → list of diagnostics with bar ranges
- `tools/format_jot.ts` — canonical formatting of a Jot

The Python wrappers (`pipeline/jot_extract.py`, `pipeline/recompose.py`,
`pipeline/lint.py`, `pipeline/format.py`) handled the subprocess + JSON
plumbing. The Docker build context = repo root (not `transcriber/`) so
the same `src/` parser code that runs in the browser also ran inside
the bridge.

This single-source-of-truth principle is non-negotiable for any future
AI MIDI → Jot work that mutates DSL: do not write a second parser in
Python. Reuse `src/parser/` via a bun bridge, or — if the Python side
needs richer manipulation — call out to a small TS helper exactly as
above. The full set of touchpoints (Docker build context, synthesised
`/app/tsconfig.json`, JSON-in/JSON-out contract) is preserved in
`Dockerfile` lines 96–118 and the `RECOMPOSE_JOT_TOOL` /
`JOT_TO_ONSETS_TOOL` env vars.

## 8. Beat-relative coordinates `(bar, beat_in_bar)`

The transcribe stage operated in `(bar, beat_in_bar)` space throughout
— never a fixed 1/16 grid, never an LCM grid. Triplets became a
property of intra-beat fractions (0.000 / 0.333 / 0.667); tempo /
time-sig changes worked naturally because each beat had its own
absolute time anchor; tolerances stayed musically sensible at any
tempo. The implementation lived in `pipeline/beats.py` and that part
is still very much alive — it feeds the ADTOF onsets path today.

For AI MIDI → Jot the input MIDI already encodes ticks-per-quarter and
a tempo map, so `(bar, beat_in_bar)` falls out of MIDI parsing directly.
Use the same coordinate system end-to-end.

## 9. Resume-from-stage debugging

The deleted DSL pathway sat inside the named-stage pipeline runner
(`pipeline/runner.py`) and benefited massively from
`POST /transcribe/resume` — you could iterate on the LLM stages alone
without paying the 30–60 s separation cost on every change. That part
of the runner is still alive (it's how filter-mode resume works today),
and any AI MIDI → Jot stage added later should slot in as a new
`Stage` in the same enum + a new `_do_<stage>` body, so resume works
for it too.

## 10. What NOT to bring back

Some of the design choices in the deleted code were specific to the
"LLM produces full DSL from audio onsets" problem and should not be
reused as-is for AI MIDI → Jot:

- **The whole-chart `transcribe.md` prompt** (`prompts/transcribe.md`)
  + its multi-voice few-shots (`prompts/examples.md`). These predated
  the per-instrument refactor and were already dead code at delete
  time. The per-instrument prompts are the model.
- **`pipeline/diff.py::diff_velocities` and `score.py`'s velocity
  scoring.** Both used per-stem audio for ground truth, which an AI
  MIDI → Jot pass doesn't have. Velocity in MIDI is exact — there's no
  diff signal to chase.
- **The `||`-stripping spec subset.** Only made sense because per-
  instrument calls had to never produce `||`. A whole-chart AI
  MIDI → Jot pass needs the full grammar.
- **The STRUCTURE level's audio-independent issue** (the
  `structure_refactor_hint`). The deleted code's STRUCTURE pass was
  essentially a no-op — a placeholder note telling the LLM "look for
  patterns" with no actual diff. A real pattern-factoring pass needs a
  proper deterministic detector (bar similarity, motif extraction) as
  its diff signal.

---

## Where to find the deleted code

Everything described above was in the working tree as of the commit
*before* "delete legacy DSL transcribe + librosa onset backends" (or
whatever message the cleanup commit ends up with).

Files removed in that commit:

- `transcriber/app/pipeline/llm.py` — per-instrument transcribe
- `transcriber/app/pipeline/refine.py` — multi-level refinement loop
- `transcriber/app/pipeline/recompose.py` — Python wrapper for the merge
- `transcriber/app/pipeline/diff.py` — typed issue detectors
- `transcriber/app/pipeline/score.py` — mir_eval F1 scoring
- `transcriber/app/pipeline/critic.py` — Haiku triage
- `transcriber/app/pipeline/lint.py` — Python wrapper for lint bridge
- `transcriber/app/pipeline/jot_extract.py` — Python wrapper for the onset bridge
- `transcriber/app/pipeline/format.py` — Python wrapper for the format bridge
- `transcriber/app/pipeline/onsets.py` — librosa detector
- `transcriber/tools/jot_to_onsets.ts` — DSL → onsets bridge
- `transcriber/tools/recompose_jot.ts` — merge bridge
- `transcriber/tools/lint_jot.ts` — lint bridge
- `transcriber/tools/format_jot.ts` — formatter bridge
- `transcriber/prompts/transcribe.md` + `transcribe_instrument.md`
- `transcriber/prompts/examples.md` + `examples_instrument.md`
- `transcriber/prompts/critic.md`
- `transcriber/prompts/refine_*.md` (lint_segment / macro / structure / onsets / velocity)
- `src/recompose.ts` + `src/recompose.test.ts`

`git show <cleanup-commit>:transcriber/app/pipeline/refine.py` is the
fastest way to read the multi-level loop as it last ran. Similar for
each file above.
