# Beat-tracker A/B: madmom vs Beat Transformer

**Date:** 2026-06-30
**Goal:** A deterministic, paired head-to-head between the two beat-tracking
front-ends so we can rule one out and delete the loser's code path.

## Background / why this is needed

The `BEATS` stage (`transcriber/app/pipeline/beats.py`) supports two beat
trackers, switched by `settings.beat_tracker` (`"madmom"` default |
`"beat_transformer"`, env `BEAT_TRACKER`):

- **madmom path** (`_madmom_beats`): `RNNDownBeatProcessor` produces beat/
  downbeat activations.
- **beat_transformer path** (`_beat_transformer_beats`): the vendored neural
  net (`app/vendor/beat_transformer/`) produces activations, plus a tempo head
  that narrows the DBN search to ±15%.

**Key structural fact:** both front-ends converge on the *same* madmom DBN
decoder (`_decode_activations` → `DBNDownBeatTrackingProcessor`). So:

- Dropping **Beat Transformer** is the big cleanup: removes
  `app/vendor/beat_transformer/`, `app/pipeline/beat_transformer.py`, the
  checkpoint download, GPU park/unpark coordination, mel extraction, and the
  ±15% tempo-window narrowing. madmom stays (it's the default and owns the DBN).
- Dropping the **madmom RNN front-end** is the small cleanup: delete
  `_madmom_beats`; madmom itself stays because the DBN needs it.

There is **no beat-accuracy harness today**, the existing
`benchmarks/run_benchmark.py` measures end-to-end *onset* F1 via the HTTP API,
not beat tracking. This spec adds one.

This is **purely additive eval scaffolding**: no production code changes.

## What we measure

For each clip, run **both** front-ends through the real production grid and
score the final `BeatStructure`:

```
tracker activations (madmom RNN | Beat Transformer)
  → madmom DBN (_decode_activations, identical for both)
  → align_beats_to_onsets       (gated; align-onsets per --onsets mode)
  → _finalize_bar_tempos
  → BeatStructure   ──score──▶  mir_eval
```

The alignment **gate stays live**: if `MIN_ALIGN_COVERAGE` isn't met the grid
is left unaligned, and that counts against whichever tracker produced it. No
skipping on the gate.

We drive `analyze_beats` twice per clip with `beat_tracker` overridden (via a
settings override in-process). Both runs see the **identical align-onset list**
per clip, so the paired delta is fair regardless of onset mode.

## Ground truth

From each clip's E-GMD MIDI (synced to the audio):

- beat & downbeat times from the MIDI tempo-map + time-signature meta
  (`pretty_midi.get_beats()`/`get_downbeats()`, or an equivalent built on the
  existing `core/midi_events.py` MIDI reader if `pretty_midi` isn't in the
  venv, confirm at implementation time, don't add a dep unprompted).
- tempo reference from the same.

**Load-time sanity gate:** GT downbeats must land near strong audio onsets;
clips whose MIDI↔audio phase is grossly mismatched are dropped as bad data
(logged with the clip id, never silently).

## Sample selection (deterministic, stratified)

Driven by `e-gmd-v1.0.0.csv` (`bpm`, `time_signature`, `duration` columns), no
RNG. Reuse / extend `loaders/egmd.py`.

- **Filter:** clips ≥ 8 bars (from `duration` × `bpm` × time-sig).
- **Tempo bands:** `<90 / 90–120 / 120–150 / >150` BPM; even quota per band,
  deterministic ordering within a band (sort by `track_id`).
- **Time-sig quota:** explicitly reserve a slice for non-4/4 (3/4, 6/8, 5/4,
  7/8, …), spread across tempo bands where the data allows.
- **Target N = 96 ≈ 72× 4/4 + 24× non-4/4.**
- The exact selected clip-id list is saved to the output dir so a re-run is
  bit-identical.

## Onset modes (`--onsets`)

How the align-onset list fed to `align_beats_to_onsets` is produced. Both
trackers in a single run get the **same** list per clip.

- **`adtof`**, truthful production path: ADTOF onset detection on the clip
  audio, pooled across all 5 lanes (exactly `_do_beats`). GPU-bound.
  **Implemented but not run now** (GPU saturated by an ongoing training run).
- **`synthetic`**, **CPU-only, runnable now.** Build align-onsets from the
  clip's per-lane MIDI note onsets, degrade *deterministically* (per-clip seed
  = stable hash of `track_id`) to emulate an imperfect detector, then pool
  across lanes (= what ADTOF pooling approximates):
  - **Recall, drop 15%** of true onsets → recall ≈ 0.85.
  - **Precision (uncorrelated FP), add ≈ 10%** of kept count at uniform-random
    times (≥ 30 ms from any real onset), velocities sampled from the clip's own
    velocity distribution.
  - **Precision (stem bleed, correlated FP), for ≈ 12%** of onsets inject a
    "bleed" copy sourced from a *spectrally-similar* lane, at the source hit's
    time ± a small jitter (~15 ms) with attenuated velocity (× 0.3–0.6).
    Similarity groups follow the GM-class map: cymbals/metals
    {hi-hat, ride, crash} and membranes {kick, snare, toms} (fall back to the
    3-class KD/SD/HH grouping if finer lanes aren't exposed). This is the
    confounder that directly stresses the median-offset aligner: a ghost near a
    real beat can become the "strongest nearby onset" and bias the global
    offset, where uniform FPs mostly cannot.
  - Net ≈ recall 0.85 / precision 0.82 / F1 ≈ 0.83, the ADTOF-on-drums
    ballpark. All three ratios are named constants at the top of the module,
    easily retuned.
- **`gt`**, zero degradation; a "perfect aligner" ceiling for context.

In `synthetic`/`gt` (CPU) mode, Beat Transformer inference also runs **on CPU**
(GPU busy), with `OMP_NUM_THREADS=8` (local box is a 12-thread 7800X3D). Slower
but fine for 96 short clips, and CPU makes BT trivially deterministic.

## Metrics & reporting

- **Primary:** downbeat F-measure (drives time-sig + bar drift), **AMLt**, and
  **tempo** (MAE BPM, within-4% / within-8% hit-rate). BPM is the only signal
  the pipeline has for tempo, so tempo + AMLt are decisive.
- **Secondary diagnostic:** beat F-measure (70 ms), CMLt. Beat F is secondary
  because the onset model carries fine phase downstream.

Reported as:

- Per-tracker aggregate + **paired delta** (same clips) with a deterministic
  Wilcoxon signed-rank.
- **Breakouts: per tempo band, and 4/4 vs non-4/4** (non-4/4 is the real
  downbeat stress test).
- Outputs under `transcriber/benchmarks/out/beat_ab/`:
  `selected_clips.txt`, `per_clip.jsonl` (every score, both trackers),
  `summary.md` (tables + verdict + cleanup recommendation).

## Determinism

- madmom RNN + DBN + mir_eval are deterministic.
- Beat Transformer: `torch.use_deterministic_algorithms(True)`, cuDNN
  deterministic, fixed seed, `eval()` + `no_grad`, pinned device. Both trackers
  run on the same machine/device in one invocation.
- Sample selection and synthetic-onset degradation are seeded off stable hashes
  (no `random` without a fixed seed, no time-based seeds).

## Code shape

- `transcriber/benchmarks/beat_ab.py`, CLI entry (`--onsets`, `--limit`,
  `--output-dir`, `--seed-band-quota` etc.), selection, per-clip A/B loop,
  scoring, report writing.
- `transcriber/benchmarks/egmd_beats.py` (or extend `loaders/egmd.py`), GT
  beat/downbeat loader + the load-time sanity gate.
- Reuse: `mir_eval` (already a dep), `loaders/egmd.py` metadata read, the
  runner's ADTOF-onset helper (for `adtof` mode), `app/pipeline/beats.py`
  internals (`analyze_beats` with a `beat_tracker` override).

## Run plan

1. Land the harness; run **`--onsets synthetic`** now (CPU) over N=96.
2. When the GPU frees up, run **`--onsets adtof`** to confirm the synthetic
   verdict holds on the truthful path.
3. Record the run in `training/RESULTS.md` (BT inference is a GPU run under
   `adtof`; record both modes for reproducibility) per the RESULTS.md rule.
4. Verdict → delete the losing front-end and its now-dead support code.

## Non-goals

- Not changing the production beat pipeline behaviour.
- Not retraining or fine-tuning Beat Transformer.
- Not building a general beat-tracking leaderboard; this is a one-shot
  decision harness (though it stays in-tree and re-runnable).
