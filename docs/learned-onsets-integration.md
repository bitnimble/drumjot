# Substituting the trained onset model into the transcriber

How the `training/` frozen-MERT drum-onset model slots into the transcriber
pipeline in place of the ADTOF Frame_RNN backend, what had to change, and how
the post-model stages (peak-picker + the rest) compare between the two
implementations.

Status: wired (`app/pipeline/learned_onsets.py`, gated by
`PipelineOptions.use_learned_onsets` + `learned_onsets_checkpoint`). Not yet the
default backend; flip it on per-request once a full-kit checkpoint exists.

## How the model plugs in

The pipeline already has the seam: at the `onsets` stage, `use_learned_onsets`
swaps `detect_onsets_adtof` (per stem) for `detect_all_pitches_learned`, and
because the learned model already separates ride/crash + the hat articulations
*and* carries tuned per-lane thresholds, the runner **skips the cymbal split,
the hi-hat split, and the per-instrument filter LLM** for it (`runner._do_onsets`
/ `_do_filter`). The learned model is its own per-class classifier; those ADTOF
post-stages would be redundant or actively harmful on top of it.

What it consumes: the **per-instrument stems** (`ctx.per_instrument_stems`,
k/s/h/c/t) that the `stems_per` stage already produces. What it emits:
`OnsetCandidate`s keyed by DSL pitch via the injective `inference.LANE_TO_PITCH`
(every trained class keeps a distinct pitch + GM note; folding for display
happens later in `src/midi/from_midi.ts`).

### What had to change from the spike

The original spike ran the model **once on the merged drum stem** with a
**single-pass** MERT encode. Both are wrong for production:

1. **Per-stem, not merged.** The model is trained, threshold-tuned, and
   evaluated **per stem** (`sota_eval._predict_perstem` + `enst.PERSTEM_TO_LANES`:
   run on each isolated stem, keep only that stem's owned lanes). That per-stem
   isolation is the architecture, not an eval artifact, the published F-numbers
   exclude cross-lane bleed. Running once on the merged drum stem would leak
   bleed the model never saw at tuning time. Fixed: `detect_all_pitches_learned`
   now takes `per_instrument_stems` and loops `PERSTEM_TO_LANES`.
2. **Windowed encode.** Switched from `inference.lane_probs` (one full-song
   pass, MERT attention is O(n²)) to `inference.stitched_probs` (overlapping
   ~30 s chunks, centre-crop stitched), the same path `inference.transcribe`
   and the eval harness use, so the transcriber's onsets match the scored
   numbers.
3. **GPU + single encoder.** Loads the checkpoint on CUDA when available and
   builds one `MertEncoder`, reused across all stems (the spike rebuilt MERT
   per call and ran on CPU).

### Checkpoint format requirement

`learned_onsets_checkpoint` must be a **run directory** with `model.pt` +
`meta.json` (the `meta.json` carries `lanes`, `encoder`/`layer`/`fps`, head
shape, and the **tuned per-lane `thresholds`**), i.e. what `checkpoint.save`
writes. The A/B / overnight experiment checkpoints are bare `.pt` state-dicts
with no `meta.json`; those won't load here. **The full-kit checkpoint must be
saved via `checkpoint.save` (with tuned thresholds) to be deployable.**

### Known follow-ups (not blockers)

- `amplitude` is left `None`, so velocity falls back to `strength`. A real
  per-onset amplitude would have to be read off each per-stem audio (the ADTOF
  backend's `_peak_amplitude` / `_bloom_amplitude`); deferred.
- `refine_audio` (audio-domain time snap) is wired but **off by default**, see
  the picker comparison below.

## Peak-picker comparison

Both backends share the **same core algorithm** (`drumjot_dsp.peakpick`:
`find_peaks` height + min-distance + prominence, then the decay-reset pass) and
the **same per-lane min-distance / prominence / decay-reset values**, the
training table (`metrics.LANE_PEAK_PARAMS`) was deliberately matched to the
transcriber settings:

| lane group | min-dist | prominence | decay-reset (frac/floor) |
|---|---|---|---|
| clean (k, s, ss, t) | 20 ms | 0.10 | off |
| hat (hc, hp, ho / `h`) | 50 ms | 0.10 | 0.6 / 0.05 |
| cymbal (rd, cr / `c`,`d`) | 70 ms | 0.20 | 0.6 / 0.05 |

So the peak-pick *shape* is already identical. The differences are in the
**threshold strategy** and the **ADTOF-only pre/post-processing**:

| stage | ADTOF backend (`detect_onsets_adtof`) | Learned backend (training picker) | keep for learned? |
|---|---|---|---|
| **height threshold** | `0.10` fixed (clean); **adaptive** `max(floor, 0.5·p95)` for noisy h/c/d (floors 0.22 / 0.12) | **tuned per-lane** fixed thresholds from `meta["thresholds"]` (val-optimised per lane) | **tuned**; adaptive would override calibrated operating points |
| input **median-normalise** | yes (separator gain variance vs a fixed threshold) | no | **no**. MERT normalises internally; thresholds are calibrated |
| **audio time-refine** (`_refine_peak_times_audio`) | yes, ±30 ms snap to onset-strength max | **off by default** (wired, opt-in) | **maybe**, see below |
| hi-hat **audio supplement** (union librosa onsets) | yes (band-limit starves ADTOF's HH activation) | no | **no**, dedicated hc/hp/ho heads + 6–20 kHz high-band feature replace this crutch |
| **amplitude floor** (hat + cymbal) | yes (drop near-silent phantoms) | no | **no**, tuned threshold + prominence + decay-reset already gate phantoms |
| **crash-shadow filter** | yes (drop decay re-triggers) | no | **no**, decay-reset + the model not firing on tails covers this |

### Why the ADTOF machinery should *not* carry over

Every ADTOF-only stage exists to compensate for ADTOF being **out-of-distribution
on isolated stems**: it was trained on full mixes, so on a separated stem its
activation scale drifts (→ adaptive threshold + median-normalise), its HH lane
is starved by the ~14 kHz band-limit (→ audio supplement), and it fires phantoms
on decay tails / silence (→ amplitude floor + crash-shadow). The learned model
has none of those failure modes: it is trained **on the per-stem distribution**,
has a dedicated head per articulation, ingests a 6–20 kHz high-band block built
for hat sizzle, and ships **per-lane thresholds tuned on held-out audio**. Layering
the OOD corrections on top would, at best, be dead weight and, at worst (the
adaptive threshold), discard the calibrated operating points the model was tuned
to.

### The one stage worth revisiting: audio time-refine

`_refine_peak_times_audio` snaps each peak to the nearest audio onset-strength
maximum within ±30 ms. The learned model's targets are onset-centred, but the
MERT grid is **75 fps (~13 ms/frame)**, so raw peak times are quantised to ~13 ms.
The refine step would upgrade those to transient-honest (~1 ms) times, plausibly
a net win for playback feel and likely F-neutral. It's wired (`refine_audio=True`)
but **off by default** so the transcriber's scored output matches the eval
harness exactly until an A/B confirms it's F1-neutral on the eval sets.

## Recommendation

**Keep the training-side picker for the learned backend** (shared core +
`LANE_PEAK_PARAMS` + the checkpoint's tuned per-lane thresholds), which is what
the wired code does. Do **not** port the ADTOF OOD-correction stages. Treat the
audio time-refine as the single optional add-on, validate it F1-neutral on
ENST/MDB via `sota_eval`, then enable it by default if it helps timing without
costing F.

Net: the two pickers already converge on one algorithm and one per-lane
parameter table; the only real fork is **adaptive (ADTOF, OOD) vs tuned-fixed
(learned, in-distribution) thresholds**, and the learned model's tuned
thresholds are the correct choice precisely because it is in-distribution.
