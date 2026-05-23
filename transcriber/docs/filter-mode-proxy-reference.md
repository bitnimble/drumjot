# Filter-mode refine via a proxy reference (design note — not implemented)

Status: **deferred.** v1 of the `transcribe_mode=filter` pathway ships
with refinement disabled. This note captures the design for adding an
in-loop refine gate later, so it can be picked up without re-deriving
the reasoning. Read alongside the filter-pathway plan.

## Background: why the existing gate doesn't work for a filter

The `dsl` pathway's refine loop accepts a revised transcription only if
its onset F1 improves. `transcriber/app/pipeline/score.py::score_jot`
computes that F1 with `mir_eval` (note-level, onset-timing-only, 0.05 s
tolerance, per-pitch mean), and `refine.py` passes the **detected
onsets** (`ctx.onsets_by_pitch`, the librosa output `D`) as the
reference.

For an end-to-end DSL transcription that gate is fine — the prediction
can diverge from `D` in both directions. For a *filter* it is
degenerate. The filter keeps a subset `K ⊆ D` with original onset times
verbatim, so under `mir_eval` matching every kept onset matches its
identical twin in `D`:

- precision = |K| / |K| = 1.0 (always)
- recall    = |K| / |D|
- **F1 = 2·|K| / (|K| + |D|)**, strictly increasing in |K|

So F1-vs-`D` is maximised by keeping everything (`K = D`). It cannot
distinguish a correct rejection (an artifact) from an incorrect one (a
real note) — both just shrink |K| and lower the score identically. An
F1-gated loop on this signal reverts every filter action and converges
to the no-filter baseline.

The gate that *would* be correct scores against ground-truth real
onsets `R` (removing an artifact raises F1, removing a real note lowers
it). But `R` does not exist at service time — the live `/transcribe`
service has only the audio and `D`. The benchmark has `R`, but it
scores the service's *final output* after the fact; the in-service
refine loop cannot read it.

## The idea: gate against a proxy reference R̂

Construct, deterministically and with no LLM and no ground truth, a
**cleaned subset / estimate** `R̂` of the detected onsets that
approximates `R` better than raw `D` does. Then keep the existing
"accept iteration only if F1 improves" loop structure, simply pointing
the reference at `R̂` instead of `D`.

This restores a useful gradient because `R̂` is built by a method
*independent of the LLM's filtering choice*:

- LLM drops an onset **not in R̂** → precision-vs-R̂ ↑, recall
  unchanged → F1 ↑ (correctly rewards killing a likely artifact).
- LLM drops an onset **in R̂** → recall-vs-R̂ ↓ → F1 ↓ (correctly
  penalises killing a likely-real note).

The gate is useful exactly to the degree `R̂` correlates with `R`
better than `D` does.

## Constructing R̂

`R̂` is an intersection of cheap, independent signals computed from
inputs the onset stage already has (per-stem audio, `onsets_by_pitch`,
`BeatStructure`):

1. **Detector consensus (primary).** Run a second, *stricter* onset
   pass per stem — different params from the permissive detector in
   `app/pipeline/onsets.py` (which uses `pre_max=post_max=wait=3`).
   Keep onsets that survive *both* the permissive and the strict pass.
   Artifacts tend to disappear under strict settings; genuine
   transients survive both. This is the strongest single signal and the
   most natural fit for the codebase.
2. **Cross-stem bleed exclusion.** Drop an onset in stem X that
   coincides (±~5–10 ms) with a substantially stronger onset in another
   stem — probable bleed, so exclude from `R̂`.
3. **Per-stem strength floor.** Drop onsets far below the stem's own
   strength distribution (e.g. below a low percentile) — likely
   detector noise.

Deliberately **not** used: beat-grid proximity. Gating toward grid
positions re-bakes the quantization assumption the filter pathway
exists to remove, and would penalise genuine off-grid playing.

`R̂` = onsets that pass (1) AND are not excluded by (2) AND clear (3).

## Implementation footprint

- New module (e.g. `app/pipeline/proxy_reference.py`) computing `R̂`
  from `(per_instrument_stems, onsets_by_pitch, structure)`. One extra
  librosa onset pass per stem + threshold/coincidence logic. CPU-cheap,
  no LLM, no network.
- Filter-mode refine loop: a critic→re-filter loop (Haiku critic via
  the existing tool-use channel flags suspect keeps/drops; the filter
  LLM re-decides), with each iteration **gated by F1 against `R̂`**
  reusing `score.py::score_jot`'s `mir_eval` machinery essentially
  unchanged — only the reference set is swapped.
- Reuse `refine.py`'s existing accept-if-improved iteration structure;
  the loop *shape* barely changes, only the reference source.
- Wire under the existing `--refine/--lint` flags for
  `transcribe_mode=filter`.

## Risks and validation

- **A biased proxy steers the filter wrong.** `R̂` has its own
  errors. If its strength floor or strict detector systematically
  excludes a class of genuine onsets (soft/ghost notes, deliberately
  off-grid hits), the gate will *reward* the LLM for deleting exactly
  those real notes — degrading true F1 in the expressive cases this
  project most cares about. A bad proxy makes results worse, not merely
  unimproved. Tune `R̂`'s thresholds conservatively (favour recall of
  `R̂` over its precision) so the gate rarely punishes a correct keep.
- **In-loop validation is impossible.** Whether `R̂` helps can only be
  judged by the post-hoc benchmark against real `R`, with vs. without
  the proxy gate — the same A/B needed to justify any refine variant.
  Treat `R̂`'s parameters as benchmark-tuned hyperparameters.
- **Heuristic creep.** This reintroduces hand-tuned deterministic rules
  (second-detector params, coincidence window, percentile floor) — some
  tension with the "LLM does only the judgement" framing of the
  pathway. Keep the rule set minimal and documented here.

## Decision recorded

Ship v1 filter-only with refine off; measure filter-vs-`dsl` F1 first.
Only invest in this proxy-reference refine if (a) the core filter bet
demonstrably wins and (b) error analysis shows residual false positives
the LLM's single pass leaves behind that a gated second look could
remove.
