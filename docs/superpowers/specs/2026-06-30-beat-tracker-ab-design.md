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

From each clip's E-GMD MIDI (synced to the audio), built on `mido` (already a
dep; no `pretty_midi`):

- beat & downbeat times from the MIDI tempo-map + the CSV's `time_signature`
  (authoritative meter). Beats step at the meter's beat unit (`4/denominator`
  quarter notes); downbeats land every `numerator` beats; anchored at tick 0.
- **tempo reference from the MIDI tempo map** (60e6/`set_tempo` = quarter-note
  BPM, the same unit the trackers report), *not* the CSV `bpm` (which may be
  eighth-relative for x/8 meters). Verified: E-GMD MIDI carries the real
  `set_tempo` (e.g. 138.0, matching the CSV).

**Load-time sanity gate:** scored on GT **downbeats** (bar starts), not every
beat: a drum onset must land within 80 ms of ≥ 50 % of downbeats, else the clip
is dropped (logged, never silently). Downbeat-based because a compound 6/8 grid
has a beat on every eighth, which a drummer rarely all strikes, that tanked
beat-level coverage and wrongly dropped non-4/4 clips; bar starts almost always
carry a hit, so the gate is meter-robust.

## Sample selection (deterministic, stratified)

Driven by `e-gmd-v1.0.0.csv` (`bpm`, `time_signature`, `duration`, `split`
columns), no RNG.

- **Root resolution** mirrors training: `--root` → `$DRUMJOT_EGMD` →
  `training/data_paths.toml` (`egmd`) → the documented codebox default
  (`/codebox-workspace/datasets/e-gmd-v1.0.0`) → in-tree benchmarks dir.
- **Split = `all` (default).** E-GMD's `test` split is **100 % 4/4** (5289
  clips); the only non-4/4 meters (3/4, 6/8, 5/4, 5/8, ~516 clips) live in
  train/validation. madmom and Beat Transformer are **pretrained** (never
  trained on E-GMD), so sampling across splits has no leakage and is the only
  way to fill the non-4/4 quota.
- **Filter:** clips ≥ 8 bars (from `duration` × `bpm` × time-sig).
- **Tempo bands:** `<90 / 90–120 / 120–150 / >150` BPM; round-robin quota per
  band, deterministic ordering within a band (sort by `track_id`).
- **Time-sig quota:** reserve a slice for non-4/4, stratified across tempo
  bands where the data allows.
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
- **Compound-meter caveat:** downbeat F is meter-fair (it only matches bar
  starts, regardless of beat subdivision), so it's a sound primary even on
  6/8 & 5/8. AMLt is **not** fair there: the GT grid is at the eighth pulse,
  so a tracker reporting the dotted-quarter pulse is a ×3 metrical level, which
  AMLt rejects. Read AMLt on x/8 with caution; 3/4 & 5/4 are quarter-based and
  unaffected. (First-run sighting: a 6/8 clip scored madmom dbF 0.97 / BT 0.00; a real BT bar-finding failure, not the subdivision artifact.)

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

---

## Results (2026-06-30, synthetic onsets, N=96 → 77 scored, 19 dropped)

Three-way, scored on the Drumjot-relevant metrics (medians; data is bimodal).
Beat This! (ISMIR 2024, MIT, DBN-free) added as a third arm.

| metric | madmom | beat_transformer | **beat_this** |
|---|---|---|---|
| **downbeat_f** (overall) | 0.41 | 0.97 | **1.00** |
| **bar_len_ok** (overall) | 0.00 | 1.00 | **1.00** |
| downbeat_f · 4/4 (53) | 0.23 | 0.97 | **1.00** |
| downbeat_f · 3/4 (8) | 0.42 | 0.46 | **0.99** |
| downbeat_f · 6/8 (6) | 0.97 | 0.00 | **0.98** |
| downbeat_f · 5/8 (4) | 0.00 | 0.66 | **0.94** |
| downbeat_f · 5/4 (6) | 0.82 | 0.98* | 0.61 |
| downbeat_f · 120-150 band | 0.22 | 0.00 | **1.00** |

**Verdict: Beat This! wins decisively** on bar alignment + bar grouping —
overall, on 4/4 (the ~99% case), and on 3/4 / 5/8 / 6/8. It's the only
tracker that survives BT's catastrophic bands (120-150 BPM) and meters
(6/8) *and* madmom's (5/8, fast tempo). madmom's 4/4 numbers are partly
GT-phase-deflated but it's clearly the weakest on bar phase regardless.

**One caveat:** 5/4 (6 clips) is Beat This!'s weak spot (bar_len_ok 0.00 —
it mis-groups the 5). BT's 5/4 "0.98" is a small-sample fluke (its beat_f
0.22 / amlt 0.00 there contradict it). 5/4 is the rarest meter (43 in all
of E-GMD) and hand-fixable in the UI.

**Implication:** Beat This! can replace **both** madmom and Beat Transformer
— delete `_madmom_beats`, the whole `beat_transformer.py` + vendored model
+ checkpoint + GPU-park + tempo-narrowing, and the shared madmom DBN — and
derive tempo from its beats. madmom drops out entirely (it only stayed for
the DBN). Tempo (octave-corrected) is a wash across all three.

**Still to confirm before adopting in production:**
- Validate on a real full-song **separated drum stem** (E-GMD is clean solo
  drums; Beat This! trained on full mixes). The harness's `adtof` mode (GPU)
  is the closest proxy; run it when the GPU frees up.
- Beat This! barely uses `align_onsets` (uniform offset only), so the
  synthetic-vs-adtof onset mode matters far less for it than for the DBN
  path — the bar-alignment wins are intrinsic.
