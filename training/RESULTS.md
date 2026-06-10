# Drum-onset model, results log

Tracks the learned drum-onset model (frozen MERT + per-lane heads) over time:
each run's setup, the numbers, and *why* things changed. Newest entries on top.
Two metrics matter, in order:

- **ParaDB onset-F1**, real, hand-charted songs run through our *own* separation
  (the deployment domain). This is the number that counts.
- **STAR held-out val**, in-domain *synthetic* (`re_synthesized_drum`) audio.
  Useful for convergence/regression checks but **optimistic** (no synth→real gap),
  so treat it as a training signal, not ground truth.

Scoring is `mir_eval` onset-F1 at ±50 ms (`metrics.onset_f1`).

---

## Current setup (2026-06-09)

**Architecture.** Frozen **MERT-v1-330M** encoder (sr 24 kHz, **layer 10**, ~75 fps,
1024-dim), features cached to disk (fp16). On top, **11 independent per-lane heads**
(`MultiLaneHeads`): each a **2-layer bidirectional GRU, hidden 128**, + linear →
one onset logit/frame. Separate heads (not one shared multi-output head) to avoid
inter-lane negative transfer; shared context comes from the frozen features.
Lanes: `k s ss t hc hp ho rd cr mc mp`.

**Training data.** STAR `re_synthesized_drum` stems (whole-kit, **synthetic**,
label-accurate). STAR ships only full-kit stems (no per-instrument audio); labels
are ADT-generated. The model's input domain is therefore *separated drum audio*.

**Targets / loss / optim.** Gaussian onset bumps (σ 1.5 frames, peak 1.0, combined
by max). Loss: **pos-weighted BCE** (default); CenterNet penalty-reduced **focal**
available via `--loss focal` (A/B pending). Optimizer: **AdamW** (wd 0.01) +
**warmup→cosine** schedule.

**Peak picker** (`drumjot_dsp.peakpick`, shared with the transcriber).
`find_peaks` (height + per-lane min-distance + prominence) then a decay-reset pass
(collapses one sustained ring read as a stream). Per-lane params
(`metrics.LANE_PEAK_PARAMS`):

| class | lanes | min-dist | prominence | decay-reset |
|---|---|---|---|---|
| clean | k s ss t mp | 20 ms | 0.10 | off |
| hat | hc hp ho | 50 ms | 0.10 | 0.6 / 0.05 |
| cymbal | rd cr mc | 70 ms | 0.20 | 0.6 / 0.05 |

Per-lane heights are tuned on val. Global GT↔audio offset = **median nearest-peak**
(robust; replaced argmax-of-support which overshot on dense drums).

**ParaDB eval methodology.** Reconstruct the song from the chart's audio tracks
(sum; add drums only if the song is drumless) → **our** separation: BS-Roformer
drum stem → MDX23C 5-stem drumsep (kick/snare/hi-hat/cymbals/toms) → run the model
on each **isolated stem**, keep only that stem's matching lanes, count the rest as
cross-instrument **leakage** (a hallucination metric). Adaptive hat/cymbal folding
per map (split if the chart distinguishes them, else fold to a parent). `mp`/`mc`
scored only when the chart charts that percussion.

---

## ParaDB onset-F1 (6 maps: AllIWanted, Aint_it_Fun, Death_of_a_Bachelor, Kaikai_Kitan, That's_What_You_Get, Kyouran_Hey_Kids; per-instrument isolation; new per-lane picker)

### star_stem_full_v1, 6035 train clips, 40 epochs (2026-06-09)

| lane | F1 | P | R | maps | fold |
|---|---|---|---|---|---|
| k | 0.907 | 0.869 | 0.958 | 6 | |
| s | 0.806 | 0.878 | 0.778 | 6 | |
| t | 0.660 | 0.675 | 0.669 | 6 | |
| ho | 0.702 | 0.624 | 0.802 | 2 | split |
| cr | 0.417 | 0.595 | 0.447 | 4 | split |
| hc | 0.297 | 0.222 | 0.582 | 2 | split |
| cym | 0.238 | 0.161 | 0.453 | 2 | folded |
| rd | 0.178 | 0.236 | 0.242 | 4 | split |
| mc | 0.092 | 0.062 | 0.618 | 2 | split |
| h (hats) | 0.455 | 0.308 | 0.919 | 4 | folded |
| mp | 0.000 | 0.000 | 0.000 | 1 | (no mp stem) |

Profile: kick/snare strong; hard lanes are **high-recall / low-precision**
(hats folded R 0.92 / P 0.31; mc R 0.62 / P 0.06); aligned with the
"catch-everything-then-filter" strategy. Ride/cymbals remain weak.

### star_balanced_stem_v1, 448 train clips, 20 epochs (2026-06-09)

| lane | F1 | P | R | maps | fold |
|---|---|---|---|---|---|
| k | 0.914 | 0.881 | 0.957 | 6 | |
| s | 0.826 | 0.878 | 0.803 | 6 | |
| t | 0.639 | 0.684 | 0.617 | 6 | |
| ho | 0.787 | 0.783 | 0.793 | 2 | split |
| cr | 0.562 | 0.626 | 0.578 | 4 | split |
| hc | 0.238 | 0.158 | 0.701 | 2 | split |
| cym | 0.191 | 0.122 | 0.442 | 2 | folded |
| rd | 0.210 | 0.156 | 0.570 | 4 | split |
| mc | 0.073 | 0.044 | 0.647 | 2 | split |
| h (hats) | 0.535 | 0.397 | 0.873 | 4 | folded |
| mp | 0.000 | 0.000 | 0.000 | 1 | (no mp stem) |

### star_balanced_stem_v2, 1000 train clips, ~80 epochs, from scratch (2026-06-10)

The "bigger balanced, from scratch" run (AdamW + warmup→cosine; no warm-start, so
a clean balance-vs-natural comparison to full_v1). STAR-val plateaued ~epoch 60.

| lane | F1 | P | R | maps | fold |
|---|---|---|---|---|---|
| k | 0.906 | 0.868 | 0.956 | 6 | |
| s | 0.851 | 0.881 | 0.846 | 6 | |
| t | 0.663 | 0.681 | 0.668 | 6 | |
| hc | 0.314 | 0.252 | 0.643 | 2 | split |
| ho | 0.662 | 0.573 | 0.784 | 2 | split |
| h (hats) | 0.519 | 0.385 | 0.853 | 4 | folded |
| cr | 0.449 | 0.679 | 0.420 | 4 | split |
| cym | 0.291 | 0.250 | 0.349 | 2 | folded |
| rd | 0.028 | 0.023 | 0.037 | 4 | split |
| mc | 0.000 | 0.000 | 0.000 | 2 | split |
| mp | 0.000 | 0.000 | 0.000 | 1 | (no mp stem) |

**Verdict, scaling the balanced set helped the clean lanes, not cymbals.** Best
of all three runs on snare (0.851), toms (0.663), closed-hat (0.314), folded-cym
(0.291); ~tied on kick. But ride/crash/open-hat are flat-to-worse vs v1, so the
three synthetic runs (448-warm / 6035-natural / 1000-balanced) have **converged on
the clean lanes and all stall on cymbals**, strong evidence the cymbal ceiling is
the **synthetic→real domain gap, not data quantity/balance** (this is what the
separation-aware experiment targets). Caveat: `rd=0.028` is mostly a **threshold
artifact**, v2 tuned ride to thr 0.10 on STAR-val's 4 ride clips (v1 used 0.80),
which floods the real cymbals stem (P 0.023); the model isn't necessarily worse at
ride, its rare-lane threshold is mis-calibrated by the tiny val (`mc=0` likewise:
thr 0.5 off 13 val onsets). → a rare-lane threshold floor would recover these.

### A/B verdict: more data did NOT win on real audio

Same 6 maps, same picker, `full_v1 − balanced_v1`:

| lane | balanced_v1 | full_v1 | Δ |
|---|---|---|---|
| cr | **0.562** | 0.417 | **−0.145** |
| ho | **0.787** | 0.702 | −0.085 |
| h (hats) | **0.535** | 0.455 | −0.080 |
| rd | **0.210** | 0.178 | −0.032 |
| s | **0.826** | 0.806 | −0.020 |
| k | 0.914 | 0.907 | −0.007 |
| mc | 0.073 | 0.092 | +0.019 |
| t | 0.639 | 0.660 | +0.021 |
| cym | 0.191 | 0.238 | +0.047 |
| hc | 0.238 | 0.297 | +0.059 |

Read carefully, it is **not** "worse overall." The two are ~equal on the common
lanes (kick/snare ~tie; toms & closed-hat actually *better* on full); full_v1 is
only worse on the **rare** lanes: crash −0.145, open-hat −0.085, folded-hats
−0.080. Those are exactly the lanes that are sparse in the natural STAR
distribution and that the balanced extract over-covered. So 13× more *natural*
clips mostly adds already-saturated common-lane data; the rare cymbals stay rare.
Textbook class imbalance: scale on the common classes doesn't help the rare ones.
(STAR val said the opposite, full looked better, a reminder that in-domain
synthetic val can mislead; trust ParaDB.)

**Confounds, treat this as suggestive, not airtight:**
- The gap is **not a threshold artifact**: both checkpoints used identical ParaDB
  thresholds on the gap lanes (cr/ho/hc all 0.8), so it's real model behaviour.
- BUT `balanced_v1` was **warm-started (`--resume`) from `star_stemtest`** + 20
  epochs, while `full_v1` was **40 epochs from scratch**, so "448 clips" undersells
  what balanced actually saw (star_stemtest's data + the 448). The distribution
  effect and the warm-start can't be cleanly separated here.
- ParaDB cymbal lanes come from only **2–4 maps** → wide error bars.

**Tentative takeaway: class balance helps the rare lanes more than raw natural-
distribution volume, but confirm before trusting it.** Clean test = train a
*balanced* set and a *natural* set from the **same** start (both from scratch, or
both fine-tuned), same epoch budget, and compare. Productive next move regardless:
a **larger balanced** set (rare-lane-rich subset drawn from the full 6k), which
gets scale *and* cymbal coverage.

### Cross-instrument leakage (promiscuity, onsets discarded as wrong-lane)

Per-stem leak % (leaked / (matched+leaked)), same 6 maps:

| stem | balanced_v1 | full_v1 | Δ |
|---|---|---|---|
| h (hi-hat) | 35.4% | **26.7%** | **−8.7** |
| k | 20.6% | 32.1% | +11.5 |
| s | 52.5% | **73.6%** | +21.1 |
| c | 47.4% | 62.6% | +15.2 |
| t | 61.9% | 66.7% | +4.8 |

More data **improved the well-represented confusion** (hi-hat → cymbal more than
halved: hat→ride 18.5%→6.0%, hat→any-cymbal 24.7%→10.5% of kept hat onsets), i.e. mutual exclusivity *did* sharpen where the extra data had examples. But it
**worsened the noisy catch-all lanes**: the full model sprays far more pedal-hat
`hp` (snare-stem → hp 632→2504, 4×) and misc-perc `mp` (the two weakest lanes:
STAR-val F1 0.47 and ~noise; `mp` has no stem at all). The natural distribution
has more, and noisier ADT-labelled, `hp`/`mp`, so 13× of it taught the model to
fire those defaults harder, swamping the hi-hat gain. (All discarded by isolation,
so it's a promiscuity diagnostic, not final F1.) → **`mp` is a candidate to drop
from the model** (no stem, pure garbage-attractor).

---

## STAR held-out val (tuned thresholds; in-domain synthetic, optimistic)

> NB the two checkpoints used **different val sets** (the balanced run's val is
> rare-lane-rich; the full run's is the natural STAR split with very few
> ride/misc clips), so cross-checkpoint STAR comparison is muddy, use ParaDB
> for that. Listed for per-checkpoint convergence reference.

| lane | balanced_v1 (old picker) | full_v1 (old picker) | full_v1 (new picker) |
|---|---|---|---|
| k | 0.994 | 0.995 | 0.992 |
| s | 0.989 | 0.985 | 0.980 |
| ss | 0.451 | 0.490 | 0.507 |
| t | 0.744 | 0.714 | 0.710 |
| hc | 0.808 | 0.845 | 0.840 |
| hp | 0.434 | 0.464 | 0.474 |
| ho | 0.767 | 0.794 | 0.797 |
| rd | 0.644 | 0.428¹ | 0.457¹ |
| cr | 0.677 | 0.732 | 0.728 |
| mc | 0.350 | 0.684¹ | 0.704¹ |
| mp | 0.402 | 0.569 | 0.585 |
| **macro** | **0.66** | **0.70** | **0.71** |

¹ full_v1's val has only ~4 clips for ride/misc → those F1s are high-variance,
don't read into them. (The during-training `val_macro_f1` of ~0.32 is at the
fixed 0.5 threshold; the ~0.70 above is after per-lane threshold tuning.)

---

## Findings & changelog (what moved the needle)

- **Drum stems ≫ full mix** (early mix-vs-stem test): training on isolated drum
  audio beat the full mix on every lane → the model trains on (and is fed at
  inference) *separated* drums, never the song mix.
- **Class balance may matter more than raw volume for rare lanes (unconfirmed).**
  STAR val *suggested* more data helped (balanced 448 → full 6035, macro 0.66 →
  0.70), but on the real-audio A/B the full natural-distribution run is ~equal on
  common lanes and *worse* on the rare ones (crash −0.145, open-hat, hats), the
  lanes the balanced extract over-covered and that stay sparse in natural STAR no
  matter the size. Same thresholds, so it's real behaviour, not tuning. BUT
  confounded: balanced was warm-started from `star_stemtest`, full was from
  scratch, and the cymbal lanes come from only 2–4 ParaDB maps. So: a real lead,
  not a proven law, confirm with a same-start balanced-vs-natural A/B. (Also a
  reminder that in-domain synthetic STAR val can point the wrong way; trust ParaDB.)
- **The deterministic envelope filter is a no-op.** Across every checkpoint, map,
  and picker, gating onsets on the onset-strength envelope gives dF ≈ 0, the
  false onsets sit on real transients (a crash *is* a transient), so an
  energy gate can't tell ride from crash. Precision must come from elsewhere.
- **Per-instrument isolation** (vs scoring the combined drum stem) discards
  cross-instrument hallucination (e.g. hi-hat→ride leakage) and helps confused
  lanes (toms, ride), but the extra MDX23C split adds artifacts that *hurt* the
  clean lanes (kick/snare) vs running the model on the whole BS-Roformer drum
  stem. → a **hybrid** (clean lanes from the full drum stem, cymbals from the
  isolated cymbal stem) is the likely best of both; not yet built.
- **Per-lane peak picker** (this entry) replaced flat `find_peaks(0.5, 30 ms)`.
  Near-neutral on clean STAR val (nothing to suppress), and it lets the tuned
  thresholds drop (more recall) without losing precision; the payoff should
  show on messy real audio. Shared via `drumjot_dsp` so transcriber + trainer
  can't drift.
- **Persistent weak spot: ride / cymbals** (and `mp` is structurally undetectable
  in the per-instrument eval, no misc-perc stem). Causes: synthetic→real domain
  gap (trained on clean synthetic stems, evaluated on real separator output) +
  within-cymbal ride↔crash confusion the picker can't fix. Likely next levers:
  train on `original_drum` (real isolated kit) or separator output; focal loss;
  better encoder (MuQ/MusicFM A/B).
- **Infra:** AdamW + warmup→cosine (was plain Adam, constant LR); focal loss
  selectable; cu130 torch (CUDA-13 driver match); global offset via median
  nearest-peak.
