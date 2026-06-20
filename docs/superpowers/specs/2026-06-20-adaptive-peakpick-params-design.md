# Adaptive per-song peak-picking parameters for the learned onset model

**Date:** 2026-06-20
**Status:** Design approved, pre-implementation
**Scope:** `training/` only (the learned MERT+heads model and its eval harness). Does **not** touch the existing ADTOF transcriber pipeline.

## Problem

The learned drum-onset model (`training/`) turns per-lane frame-wise activation
curves into discrete onsets with a peak-picker
(`metrics.py` → `drumjot_dsp.peakpick.pick_peaks`). The peak-picker is governed
by per-lane parameters, **threshold, prominence, min_distance_s,
decay_reset_frac, decay_reset_floor**, and today every one of these is a fixed
constant. The per-lane *threshold* is "tuned", but `tune_thresholds`
(`train.py:204-239`) picks a **single value per lane for the entire validation
set**. That is the "optimal overall, not per-song" behaviour we want to fix.

Different kits and recordings sound very different, and gain/timbre vary wildly
song-to-song, especially after source separation. A threshold/prominence that
is ideal for a bright, hot-gain recording is wrong for a dark, quiet one. The
prize is the gap between **one global parameter per lane** and **the best
parameter for *this* song**.

## Approach (selected)

**Decoupled post-hoc parameter predictor.** A lightweight artifact that sits
*after* the frozen MERT+heads model and *before* the peak-picker:

```
audio → stems → frozen MERT+heads → activation curves
      → [feature extractor → param predictor → per-lane params]
      → pick_onsets_lane → onsets
```

The onset model is **not** retrained. We learn, offline against ground truth
(strategy "a"), a mapping from label-free signal features to per-song peakpick
parameters, then consume it label-free at inference (strategy "c").

Rejected alternatives:
- **End-to-end sequence predictor** (CNN/GRU over the raw activation curve):
  more expressive, data-hungrier, slower to debug, overfit risk. Held in reserve
  if the decoupled regressor's captured gap plateaus short of the ceiling.
- **Deterministic-only analytic rules:** kept as the *baseline* and as *input
  features*, but not the primary mechanism, it under-serves the coupled params
  (prominence × decay_reset on cymbals).

## Output space

The full `pick_peaks` per-lane parameter surface:

| Param | Lanes | Notes |
|---|---|---|
| `threshold` | all 9 | Most gain/timbre-sensitive; the existing tuned knob. |
| `prominence` | all 9 | Noise-floor-sensitive. |
| `min_distance_s` | all 9 | Semi-physical (fastest playable rate); regularize toward defaults when F1-flat. |
| `decay_reset_frac` | sustained only (hp, ho, rd, cr; hc) | Decay/ring-sensitive. |
| `decay_reset_floor` | sustained only | Decay/ring-sensitive. |

The crash-shadow and hi-hat-tail filters mentioned in the original discussion
live in the *transcriber* (`hihat_split.py`, `cymbal_split.py`), **not** in this
model's eval path, and are out of scope.

## Components, `training/drumjot_training/parampred/`

### `oracle.py`
Given one song's per-lane activation curve + ground-truth onsets, sweep the
peakpick params and return the per-lane param vector that maximizes onset-F1
(±50 ms tolerance, reusing `metrics.py`), plus the achieved F1 (the per-song
ceiling).
- Coordinate-ascent seeded from today's global-tuned values; a few passes.
- Fine grid on `threshold`, coarser on the rest.
- When the F1 surface is flat across a param's range (common for
  `min_distance_s` / `decay_reset_*`), return the value **closest to the current
  default** rather than an arbitrary maximizer, keeps labels well-conditioned.
- Results cached per (song, augmentation-variant).

### `features.py`
Per-song-per-lane feature vector, two groups:
- **Activation-curve** (per lane): noise floor (low percentile of the curve),
  peak-height percentiles (50/75/90/max), candidate-peak count at a low probe
  threshold, top-peak/median ratio, autocorrelation peak at the beat period, and
  the deterministic knee-threshold (from `baseline.py`).
- **Audio** (per stem, shared by that stem's lanes): spectral
  centroid/rolloff/flatness/bandwidth, crest factor, RMS percentiles, high-band
  (6–20 kHz) log-mel energy (cymbal-relevant).

### `baseline.py`
Deterministic analytic rules, `threshold = histogram knee`,
`prominence ∝ noise σ`, etc. Two roles: (1) the baseline the learned model must
beat, (2) input features to the regressor.

### `regressor.py`
Per-lane model mapping the feature vector → the 5-param vector.
- **Default: sklearn `HistGradientBoostingRegressor`**, one regressor per
  (lane, param). No new dependency, scikit-learn already ships transitively via
  librosa. Robust on small data, fast to iterate, interpretable feature
  importances.
- Fallback (deferred): a tiny torch MLP, or the end-to-end sequence model, if we
  later want the raw curve fed in directly.
- Artifact saved alongside the onset checkpoint (joblib).

### `dataset.py`
Orchestrates: corpus → frozen-model inference → {`features`, `oracle` labels} →
cached table (npz/parquet). Augmentation hooks live here.

## Data flow

**Build (offline):** for each labeled song (and each augmented variant): load
audio → (augment) → stems → frozen MERT+heads → activation curves →
`features` + `oracle` → one dataset row. Train the regressor on the **train**
split (E-GMD / STAR / ENST / MDB at clip level); hold **ParaDB** out as the
test set.

**Inference (label-free):** audio → stems → frozen model → curves → `features`
→ `regressor` → per-lane params → `pick_onsets_lane`.

**Split discipline:** the predictor is trained only on train-split oracles and
evaluated on held-out ParaDB. Never train the predictor on a test song's oracle.

## Augmentation

Applied at audio level *before* the frozen model, on **train songs only**. All
transforms are **onset-preserving** (they do not move onset times), so the
ground-truth labels stay valid and the oracle simply re-sweeps on the new
activation curve, that is the core trick that lets one labeled song become many
training rows.

Augmentation set:
- **Gain** (the residual after median-normalize).
- **Parametric-EQ tilt**, kit brightness (dark ↔ bright).
- **Room IR / reverb**, drives `decay_reset_*` and cymbal `prominence`.
- **Dynamic-range compression**, reshapes the peak-height histogram.
- **Additive / separation-artifact noise**, bleed and musical noise.
- **Lossy-codec round-trip**, 128 kbps and 256 kbps MP3 (or Opus/AAC at matched
  perceptual quality for faster encode). Band-limits the signal and adds codec
  pre-echo, shifting the activation noise floor, a realistic real-world
  degradation.

~4–8 variants per song. **Excluded from v1:** time-stretch / pitch-shift (they
move onsets and would need label remapping). Cost lands on re-running the frozen
model per variant (acceptable per stakeholder).

## Eval integration

Extend `eval_paradb.py` with a `--param-predictor <artifact>` flag. For each
song, report **three F1 columns per lane**:
1. **Current**, global-tuned threshold (today's behaviour).
2. **Predicted**, per-song params from the predictor.
3. **Oracle**, per-song best (the ceiling).

Headline metric: **fraction of the oracle gap captured** = (Predicted −
Current) / (Oracle − Current), per lane and aggregate.

Safety rail: if predicted params lose to global-tuned on aggregate, fall back to
global-tuned.

## Build order

1. **Oracle-gap harness**, run on ParaDB to confirm the prize is real and see
   which lanes/params carry it. Gates the rest of the work.
2. **Feature extractor + deterministic baseline**, measure the gap captured by
   analytic rules alone.
3. **Learned regressor**, measure the gap captured.
4. **Augmentation**, widen the distribution, re-run the frozen model, retrain,
   re-measure.

## Testing & risks

**Tests:**
- Oracle F1 ≥ global-tuned F1 by construction (per lane).
- Feature-extractor output shapes/finiteness.
- Deterministic baseline monotonicity (higher noise floor → higher threshold).
- Regressor round-trips on synthetic feature→param data.

**Risks / mitigations:**
- *Predictor overfits the augmentation distribution* → ParaDB is un-augmented
  real audio, an honest held-out test.
- *Flat-F1 regions make `decay_reset_*` / `min_distance_s` oracle labels
  ambiguous* → label regularization toward current defaults (see `oracle.py`).
- *Leaking test oracles into training* → strict train/test split discipline.

## Out of scope (v1)

- Retraining or fine-tuning the MERT+heads onset model.
- The transcriber's crash-shadow / hi-hat-tail filters.
- Time-stretch / pitch-shift augmentation.
- The end-to-end sequence predictor (held in reserve).
