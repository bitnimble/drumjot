# Drum-onset model, results log

Tracks the learned drum-onset model (frozen MERT + per-lane heads) over time:
each run's setup, the numbers, and *why* things changed. Newest entries on top.
Code/architecture changes (and how to flip each one back) are tracked
separately in [CHANGELOG.md](CHANGELOG.md).
Two metrics matter, in order:

- **ParaDB onset-F1**, real, hand-charted songs run through our *own* separation
  (the deployment domain). This is the number that counts.
- **STAR held-out val**, in-domain *synthetic* (`re_synthesized_drum`) audio.
  Useful for convergence/regression checks but **optimistic** (no synth→real gap),
  so treat it as a training signal, not ground truth.

Scoring is `mir_eval` onset-F1 at ±50 ms (`metrics.onset_f1`).

---

## Per-lane F1 progress (consolidated, newest columns right within each group)

One row per lane, one column per run condition, so each lane's trajectory reads
across. **Columns are only comparable *within* an eval-domain group** (the three
`||`-separated blocks below use different eval sets, real ParaDB vs synthetic
STAR-val vs separated pooled per-stem; so a 0.96 in PS is not "better" than a
0.18 in PDB; compare down a column and along a row *within* a block). Detailed
per-run tables (with precision/recall, leakage, confounds) stay in the sections
below; this is the at-a-glance index. Legend under the table.

| lane | PDB·full | PDB·bal | PDB·bal2 || SV·bal | SV·full || PS·trial | PS·sweep |
|---|---|---|---|---|---|---|---|---|
| k  | 0.907 | 0.914 | 0.906 || 0.994 | 0.992 || 0.964 | 0.97 |
| s  | 0.806 | 0.826 | 0.851 || 0.989 | 0.980 || 0.815 | 0.84 |
| ss | –     | –     | –     || 0.451 | 0.507 || 0.481 | 0.55 |
| t  | 0.660 | 0.639 | 0.663 || 0.744 | 0.710 || 0.759 | 0.76 |
| hc | 0.297 | 0.238 | 0.314 || 0.808 | 0.840 || 0.601 | 0.59 |
| hp | –     | –     | –     || 0.434 | 0.474 || 0.378 | 0.36 |
| ho | 0.702 | 0.787 | 0.662 || 0.767 | 0.797 || 0.661 | 0.62 |
| rd | 0.178 | 0.210 | 0.028 || 0.644 | 0.457¹| 0.535 | 0.51 |
| cr | 0.417 | 0.562 | 0.449 || 0.677 | 0.728 || 0.561 | 0.54 |
| mc | 0.092 | 0.073 | 0.000 || 0.350 | 0.704¹| 0.392 | 0.45 |
| mp | –     | –     | –     || 0.402 | 0.585 || ✗     | ✗    |

**Columns**
- **PDB** = ParaDB, real hand-charted songs through our own separation (the metric
  that counts), per-instrument isolation + new per-lane picker.
  - `full` = star_stem_full_v1, 6035 train clips, 40 ep, natural dist., scratch.
  - `bal` = star_balanced_stem_v1, 448 clips, 20 ep, warm-started.
  - `bal2` = star_balanced_stem_v2, 1000 clips, ~80 ep, scratch.
- **SV** = STAR held-out val (synthetic, in-domain, tuned thresholds; optimistic).
  - `bal` = balanced_v1 (old picker); `full` = full_v1 (new picker).
- **PS** = pooled per-stem val (star+enst+egmd sep stems, deployment-domain proxy).
  - `trial` = first trial, cap 30, 15 ep, layer 10 (single config).
  - `sweep` = per-stem layer sweep, cap 30, 35 ep, 2 seeds, each lane at its **best
    layer** (see sweep section for the full lane×layer matrix).

`–` = lane not scored in that run (ParaDB tables fold/omit ss/hp/mp; ParaDB also
has folded `h`/`cym` rows not shown here). `✗` = `mp` removed from the model
(2026-06, garbage-attractor lane). ¹ few-clip, high-variance (full_v1's val has
~4 ride/misc clips); don't read into it.

**Reading the rows.** Kick/snare/toms saturate everywhere. The cymbal lanes
(rd/cr/mc) and closed-hat are strong on synthetic SV but collapse on real PDB, the synthetic→real gap. The PS columns are the separation-aware bet: ride
**0.51–0.54** and crash **0.54–0.56** on separated stems vs PDB ride 0.03–0.21,
the cymbal recovery the per-stem direction targets.

---

## MuQ vs MERT encoder A/B (2026-06-13)

**Clean A/B**, both encoders fresh through the full per-stem pipeline (not the bare
probe), identical settings: pooled star+enst+egmd, `--pool-cap 60`, 2 seeds, 45
epochs, high-band + aux + sibling. MuQ swept layers 1/4/7/10/12 (it exposes only 13
hidden states, NOT 24 as its w2v2 config implies), MERT 1/4/7/10/13. Best layer per
lane (mean over 2 seeds):

| | k | s | ss | t | hc | hp | ho | rd | cr | mc | macro |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **MERT** | 0.97 | 0.83 | 0.50 | 0.79 | 0.65 | 0.38 | 0.68 | 0.44 | 0.62 | 0.42 | **0.63** |
| **MuQ** | 0.90 | 0.75 | 0.47 | 0.69 | 0.35 | 0.25 | 0.62 | 0.21 | 0.51 | 0.38 | **0.51** |
| Δ | +.07 | +.08 | +.03 | +.10 | **+.30** | +.13 | +.06 | **+.23** | +.11 | +.04 | +.12 |

**Verdict: MERT wins every lane; keep MERT.** MuQ is decisively worse for drum
onsets (macro 0.51 vs 0.63, −19%), collapsing on the hard fine-timing/timbre lanes
(closed-hat −0.30, ride −0.23). Two coherent causes, both matching priors:
- **25 fps hurts onset precision** (hats/cymbals worst), as expected vs MERT's 75.
- **MuQ's onset signal lives in its EARLY layers** (s/ss/t/hc/hp/ho all peak at L1,
  declining with depth), whereas MERT's depth helps (hats/cymbals peak L7-L13). MuQ's
  deeper layers lean semantic (its MARBLE strength) rather than acoustic; onsets are
  an acoustic, fine-time task, MERT's wheelhouse. MuQ's MARBLE lead does NOT transfer.

The fresh MERT here reproduces the recorded cap-30 sweep within noise (hc 0.65 vs
0.59, cr 0.62 vs 0.54, t 0.79 vs 0.76), confirming the A/B is sound. **Decision:
stay on MERT.** (License is moot: both CC-BY-NC. MuQ remains an option only if a
future need is semantic, not onset.)

## Soft-argmax onset timing spike (2026-06-13)

Tested whether sub-frame **soft-argmax** decoding (re-time each detected peak as the
prob-weighted centroid of its ±1-frame lobe) recovers MuQ's 25 fps timing loss.
Same trained model decoded two ways (identical peaks, only the time differs), MuQ
hc@L1 + cr@L4, cap 60 / 2 seeds / 45 ep (reproduces the overnight MuQ: hard@±50 ms
hc 0.351, cr 0.507). Scored at three tolerances:

| lane | ±50 ms | ±25 ms | ±10 ms |
|---|---|---|---|
| hc (dense) | −0.001 | +0.001 | **−0.011** |
| cr (sparse) | +0.001 | +0.010 | **+0.011** |

**Conclusion: ~no-op at our ±50 ms tolerance; tiny (+0.01) gain only on sparse/
sustained lanes at tighter tolerances; slightly NEGATIVE on dense lanes.** As
theory predicts, it's precision-only (the note-density ceiling stays frame-rate
bound) and there's nothing to win at ±50 ms because 25 fps quantization (±20 ms) is
already inside tolerance. Dense hats get *worse* at ±10 ms: neighbouring 16th-note
lobes (~2.5 frames apart at 25 fps) contaminate the centroid. **Don't ship as-is**
(would need an isolated-peak gate); it does not rescue MuQ (its deficit is
representational, not sub-tolerance timing). Useful only if we ever score tighter /
care about groove feel, and then sparse lanes only.

## MuQ cymbal classification probe (idea-A, 2026-06-13)

Tests the two-stage hypothesis: given a KNOWN onset, can MuQ out-CLASSIFY MERT on
cymbal type (ride/crash/misc + reject)? Proper setup: full features (encoder +
high-band) + a 2-layer BiGRU over a 40 ms-pre .. 500 ms-post window (attack + ring),
mean-pooled -> softmax; identical examples/head/seed, only the encoder differs.
Cymbal stem, cap 60. MERT@L13 vs MuQ at L4 (detection-best) / L7 / L10 (deeper,
where its semantic/timbre strength should live).

| config | acc | ride F1 | crash F1 | mc F1 | reject F1 | crash→ride |
|---|---|---|---|---|---|---|
| **MERT@L13** | 0.851 | 0.834 | **0.554** | 0.262 | 0.924 | **0.244** |
| MuQ@L4 | 0.846 | 0.824 | 0.514 | 0.181 | 0.923 | 0.466 |
| MuQ@L7 | 0.861 | 0.852 | 0.442 | 0.133 | 0.937 | 0.433 |
| MuQ@L10 | 0.704 | 0.716 | 0.453 | 0.359 | 0.738 | 0.448 |

**MuQ ruled out for drums, at every layer.** On the call that matters (crash vs
ride) MERT wins everywhere: best crash F1 (0.554) and **half** the crash→ride
confusion (0.244 vs MuQ's 0.43-0.47). MuQ@L7's higher *overall* acc is a mirage,
it comes from the easy classes (ride + reject) while crash gets worse; aggregate
accuracy rewards MuQ's ride-bias. Deeper (L10) collapses (reject recall 0.63: 883
rejects called "ride"), confirming MuQ's deep layers shed the acoustic onset signal
as they go semantic. Across detection (A/B macro 0.63 vs 0.51), timing (soft-argmax
no-op) and classification (here), MuQ never beats MERT for drums. **Stay on MERT.**

**Encoder-agnostic by-products:** (1) the two-stage structure works, reject F1
~0.92-0.94, so "high-recall propose -> classify-with-reject" is viable (use MERT as
the classifier). (2) Crash/ride is an INTRINSIC ceiling: even MERT given the onset
hits only crash F1 0.55 / 24% crash→ride, and crash is data-starved (802 train vs
ride 2500, mc 180). Levers for cymbals are more crash/mc data, the sub-6 kHz `cym`
timbre block (built, never A/B'd), or better separation, NOT the encoder.

## Two-stage vs per-frame: CEILING gate (2026-06-13)

**Question.** Would a two-stage arch (high-recall onset *proposer* -> per-onset
*classifier*) beat the current per-frame detector? First gate: the **ceiling** --
give the proposer GROUND-TRUTH onset times (perfect recall) and see if the best
case clears the baseline. If not, the arch is dead cheaply.

**Setup** (`tmp_e2e_ceiling_k7m2.py`, cymbals + hats stems, ceiling-gate). Both
arms on the IDENTICAL pooled split (star+enst+egmd sep, cap 60: 180 train / 173 val
per stem), SAME cached MERT features (c@L10, h@L7, high-band), scored mir_eval
onset-F1 @ ±50 ms, per-lane tuned thresholds. Restricted to each stem's 3 lanes.
- **per-frame** = existing `MultiLaneHeads` (per-lane BiGRU, aux + sibling), 45 ep.
- **two-stage** = proposer = GT union onsets clustered @20 ms (perfect recall);
  classifier = shared 2-layer BiGRU over a 40 ms-pre..500 ms-post window,
  mean-pooled, MULTI-LABEL sigmoids; ±a cross-stem sibling-coincidence vector
  (other-stem kit lanes onset within ±30 ms, from full-kit onsets).

| lane | per-frame | 2stage | 2stage+sib | best Δ |
|---|---|---|---|---|
| rd | 0.344 | 0.830 | **0.846** | **+0.502** |
| cr | 0.469 | 0.685 | **0.727** | +0.258 |
| mc | 0.298 | 0.413 | **0.484** | +0.186 |
| hc | 0.657 | **0.793** | 0.769 | +0.136 |
| hp | 0.312 | **0.478** | 0.465 | +0.167 |
| ho | 0.690 | **0.828** | 0.806 | +0.138 |

**Verdict: ceiling PASSES on every lane, hugely on the detection-limited ones
(ride +0.50, crash +0.26, hp +0.17).** The arch has large headroom -> proceed to
the realistic proposer. But read the gap correctly:
- **It is mostly perfect-RECALL headroom, not realized.** The two-stage was handed
  the onset times; ride's +0.50 says the per-frame model's ride failure is
  dominantly a *detection/recall* problem, fixable IF a proposer hits ~100% recall.
  The realistic-proposer stage is now the whole ballgame; this gap collapses toward
  the proposer's actual recall.
- **Two confounds inflate the gap (both pessimise the baseline):** (1) the per-frame
  cymbal baseline **overfit** -- val macro peaked ~0.60 @ep2, decayed to 0.36 @ep44,
  and we score the final epoch (no best-checkpoint). Best-epoch baseline would be
  higher. (2) single seed. Fix both in the realistic round (early-stop/best-ckpt,
  2 seeds).
- **Sibling conditioning is a cymbal-specific win** (cr +0.04, mc +0.07, rd +0.02)
  and a slight hat *loss* (hc/ho −0.02): cross-stem coincidence helps crashes (kick
  under crash) but adds noise to steady-state hats. Keep it cymbal-side.

Next: build a class-agnostic high-recall onset proposer, swap it for the GT
proposer, re-score; the ceiling→realistic drop is the real number.

**Junk-rejection probe (precision side, same settings).** The ceiling assumed not
just perfect recall but perfect candidate *precision* (no junk). This probe
(`tmp_junk_probe_w4r.py`) keeps perfect recall but injects synthetic JUNK
candidates -- random non-onset times (≥120 ms from any onset), label all-zeros --
into BOTH train and val at 1:1 with true onsets (≈50% proposer precision), to test
whether the multi-label classifier can reject them. No-junk columns reproduce the
ceiling EXACTLY (same seed), so it's directly comparable. Each stem at its better
sib setting (cymbals +sib, hats no-sib):

| lane | per-frame | 2stage no-junk | 2stage +junk | junk cost |
|---|---|---|---|---|
| rd | 0.344 | 0.846 | 0.713 | **−0.133** |
| cr | 0.469 | 0.727 | 0.716 | −0.011 |
| mc | 0.298 | 0.484 | 0.502 | +0.018 |
| hc | 0.657 | 0.793 | 0.756 | −0.037 |
| hp | 0.312 | 0.478 | 0.435 | −0.043 |
| ho | 0.690 | 0.828 | 0.769 | −0.059 |

Junk rejection rate (all lanes below thr): cymbals 0.91 no-sib / 0.85 +sib; hats
0.94 / 0.91.
- **The advantage survives 50%-precision junk** -- every lane still clears per-frame
  (ride +0.37, crash +0.25, hats +0.06–0.12). The headroom isn't a perfect-recall
  mirage; it tolerates heavy false-positive load.
- **Ride is the precision canary** (−0.13): dense/steady, historically the #1
  garbage-attractor, so it over-fires on junk. Crash barely moves (−0.01), distinct
  enough that junk rarely reads as crash.
- **Sib trades rejection for classification**: +sib lowers junk reject rate (junk
  under a kick/snare nudges the cymbal heads) yet still nets higher cymbal F1.
- **Caveat:** random junk (silence/ring-tail) is far easier to reject than a real
  proposer's transient-like FPs, so these costs are optimistic FLOORS (ride worst).
  Still passes, so the gate holds; the real proposer remains the deciding test.

**hp-as-output ablation (shared trunk, 2026-06-13).** Now that the classifier
shares one BiGRU trunk across hc/hp/ho (vs the old per-lane heads that walled lanes
off), does hp's output head drag hc/ho? Controlled (`tmp_hp_abl_q3z.py`): identical
candidates/windows, only the hp output node + its loss toggled. Removing hp:
hc 0.808→0.822 (+0.014), ho 0.805→0.810 (+0.005) -- a slight drag, within
single-seed noise. No meaningful negative transfer; keeping hp costs the hats
~nothing (hp itself stays poor, 0.372).

**REALISTIC two-stage vs per-frame -- the deciding test (2026-06-13).** Replaced
the perfect GT proposer with a real one (`tmp_e2e_full_z9x.py`): a class-agnostic
1-lane "any-drum onset" head trained on the union target (keep_best), peak-picked
at a recall-tuned threshold -> real candidates WITH real false positives, fed to
the multi-label classifier. Baseline fixed (keep_best + 2 seeds) to kill the
ceiling run's overfit/single-seed confounds. Same split/features/metric.

| lane | per-frame (fixed) | 2stage (real) | ceiling | real Δ | vs ceiling |
|---|---|---|---|---|---|
| rd | 0.603 | 0.558 | 0.846 | −0.045 | −0.288 |
| cr | 0.647 | 0.527 | 0.727 | −0.120 | −0.200 |
| mc | 0.334 | 0.482 | 0.484 | +0.149 | −0.002 |
| hc | 0.653 | 0.685 | 0.793 | +0.032 | −0.108 |
| hp | 0.455 | 0.313 | 0.478 | −0.142 | −0.165 |
| ho | 0.706 | 0.753 | 0.828 | +0.047 | −0.075 |

Proposer op points (val): cymbals recall 0.83 / prec 0.70; hats recall 0.90 / 0.82.

**Verdict: two-stage does NOT beat per-frame at this scale.** It loses on ride,
crash, hp; the only wins are marginal (hc +0.03, ho +0.05, within seed noise) plus
noisy mc. The ceiling's huge gap was a mirage, two causes now exposed:
1. **The ceiling baseline was overfit.** keep_best + 2 seeds lifted per-frame
   enormously: ride 0.34→0.60, crash 0.47→0.65. Most of the "+0.50 ride headroom"
   was a broken baseline, not an arch win. (The confound flagged in the ceiling
   entry -- caught before believing it.)
2. **Proposer recall caps the two-stage.** Held-out recall 0.83/0.90 (not 100%);
   the "vs ceiling" column IS that loss (ride −0.29, crash −0.20). The bottleneck
   just MOVED from the per-frame detector's recall to the proposer's recall, both
   equally data-limited at cap 60 (cymbal proposer overfit: train 0.995 vs val
   0.826).

Separating detection from classification is theoretically sound (the ceiling shows
classification-given-onset CAN be excellent) but unrealizable here: a high-recall
proposer needs the same data the per-frame model already uses, so the split buys
nothing and the simpler per-frame model wins. **Drop the two-stage direction;** the
real lever stays shared across both archs -- more cymbal/crash data + better
separation to lift the recall ceiling. (Caveat: cap 60; a cap-150+ re-test could
check if two-stage scales differently, but the burden of proof is on it and it's
currently losing.)

## Per-stem pooled MERT layer sweep (2026-06-12)

**Setup.** `scripts/perstem_layer_sweep.py` over pooled per-stem examples from all
three SEPARATION-AWARE trees (star_balanced_sep + enst-sep + egmd-sep), full
pipeline (high-band block + aux ring-activity + sibling weighting), `--pool-cap 30`
balanced (450 train / 865 val per-stem windows), layers {1,4,7,10,13}, 2 seeds, 35
epochs. One MERT forward/clip caches all layers. Goal: lock the best encoder layer
per lane on the deployment (per-stem) domain. STAR-val F1, ±std over 2 seeds.

| lane | L1 | L4 | L7 | L10 | L13 | best |
|---|---|---|---|---|---|---|
| k  | 0.97 | 0.97 | 0.97 | 0.96 | 0.97 | flat |
| s  | 0.83 | 0.84 | 0.83 | 0.83 | 0.84 | flat |
| t  | 0.76 | 0.74 | 0.73 | 0.75 | 0.76 | flat |
| hc | 0.57 | 0.58 | 0.59 | 0.57 | 0.58 | ~L7 (flat) |
| hp | 0.36 | 0.32 | 0.36 | 0.35 | 0.35 | ~L7 (rare/noisy) |
| ss | 0.55 | 0.52 | 0.42 | 0.42 | 0.41 | **L1** (monotonic ↓) |
| ho | 0.62 | 0.62 | 0.59 | 0.54 | 0.60 | **L1/L4** (early) |
| rd | 0.42 | 0.49 | 0.47 | 0.51 | 0.49 | **L10** |
| cr | 0.50 | 0.43 | 0.51 | 0.54 | 0.50 | **L10** |
| mc | 0.38 | 0.41 | 0.44 | 0.30 | 0.45 | L13 (53 onsets, noisy) |

**Findings.**
- **Most lanes are layer-insensitive** (k/s/t/hc/hp flat within ±0.01–0.04). Layer
  choice barely matters for them.
- **Real per-lane preferences:** side-stick wants **L1** (0.55→0.41, the largest
  effect), open-hat wants **early** (L1/L4 > L10), and **ride+crash peak at L10**, *deeper* than the full-mix sweep found (L4/L7); isolating the cymbal stem shifts
  the optimum.
- **Per-stem ≫ full-mix on cymbals**: ride 0.51 vs ~0.25 full-mix, the per-stem
  direction is the cymbal win, more than layer choice.

**Recommended per-STEM layers** (stems are encoded per-instrument): s→**L1** (ss
+0.14, s indifferent), c→**L10** (rd/cr peak), h→**L7** (hc/hp; ho ~flat L1–L7),
k/t→any (L7). Layer-concat's gain over this is mostly ss + cymbals (most lanes
flat), so single-layer-per-stem captures most of it at 1× input width.

**Caveats.** cap=30 / 2 seeds, flat lanes genuinely flat; ss/ride/crash clear the
noise; mc/hp shaky (few onsets). Worth a higher-cap confirmation of s→L1 / c→L10.

**Layer-concat probe (hi-hat stem only), cap-30 first pass.** Concatenating the hat
lanes' top layers, `[MERT_L1 | MERT_L7 | high-band]` (2064-d), on 90 train / 150
val, 1 seed, batch 4: every hat lane got *worse* (hc 0.46 vs 0.59, hp 0.11 vs 0.36,
ho 0.39 vs 0.62). But this was **data-starved** (see the controlled run next, which
overturns the "concat hurts" reading).

**Controlled hat concat-vs-single (cap-150, 2026-06-12).** Matched A/B to remove
the cap-30 confounds: pooled per-stem h-stem, `--pool-cap 150` (450 train / 120 val,
~5× the data), 2 seeds, 35 epochs, IDENTICAL windows/val. Four arms; concat-h256
uses batch 2 (2064-d into a 256 head OOMs at 4 on 6 GB), the others batch 4.

| lane | L1 (h128) | L7 (h128) | concat L1+L7 (h128) | concat L1+L7 (h256) |
|---|---|---|---|---|
| hc | 0.658 | 0.653 | 0.665 | **0.691** |
| hp | 0.373 | 0.378 | **0.391** | 0.374 |
| ho | 0.665 | **0.680** | 0.679 | 0.687 |

**Findings.**
- **More data was the real story.** All cap-150 single-layer arms (0.65–0.68) sit
  far above the cap-30 baselines (hc ~0.57–0.59, ho ~0.62), +0.05–0.08. The earlier
  "concat hurts" was the 90-clip starvation, not the concat.
- **Layer-concat is a wash, not a win.** At matched head/batch, concat-h128 vs
  best single: hc +0.007, hp +0.013, ho −0.001, all within noise (±std ≤0.016).
  Confirms the recommendation: **pick one best layer per lane; don't concat** (the
  flat-across-layers profile means the layers are redundant, not complementary).
- **Capacity (head 256) helped hc** (+0.026 over concat-h128, ~2σ) and ho slightly,
  hp not, a hint the head is mildly capacity-limited for closed-hat. **Confounded**
  by batch 2 (forced by GPU memory) and by having no single-layer-h256 arm, so it's
  suggestive, not conclusive: a clean head-size A/B (best-layer, h128/256/384, same
  batch) is the follow-up, and it speaks to the "is the RNN too small" question more
  than concat does.

**Full-band log-mel vs high-band input (2026-06-13).** Tested N2N's input recipe,
replace the targeted 6-20 kHz high-band block with a 128-bin FULL-band log-mel
(`[MERT_layer | mel128]`, replace not augment), same per-stem sweep (cap 30, layers
1/4/7/10/13, 2 seeds, full pipeline). Best layer per lane, mel vs the hb16 sweep:

| | k | s | ss | t | hc | hp | ho | rd | cr | mc |
|---|---|---|---|---|---|---|---|---|---|---|
| mel-128 | 0.97 | 0.85 | 0.55 | 0.77 | 0.59 | 0.36 | 0.62 | 0.54 | 0.54 | 0.47 |
| hb16 | 0.97 | 0.84 | 0.55 | 0.76 | 0.59 | 0.36 | 0.62 | 0.51 | 0.54 | 0.45 |

**Wash.** Full-band mel tracks the narrow high-band almost exactly; only a marginal
cymbal edge (rd +0.03, mc +0.02, ~2σ) at **8× the width** (128 vs 16 dims). The
high-band already carries the discriminative cymbal info and MERT covers the
low/mid mel adds, so **keep hb16**; mel is a *slight* cymbal lever if ever needed.

## Per-stem pooled pipeline, first trial (2026-06-12)

**Setup.** First end-to-end run of pooled per-stem training (`train.py --dataset
pooled`): per-stem examples pooled from all three SEPARATION-AWARE trees
(star_balanced_sep + enst-sep + egmd-sep), `--pool-cap 30 --pool-balance` (450
train / 865 val per-stem windows), full pipeline (high-band + aux ring-activity +
sibling weighting), default encoder **layer 10**, 15 epochs, 1 seed. Eval is the
pooled per-stem val set (each clip = one isolated instrument's stem, scored only on
its own lanes). This is the single-config baseline the layer sweep above builds on.

**Held-out per-lane F1** (tuned thresholds, pooled per-stem val):

| lane | k | s | t | ho | hc | cr | rd | ss | mc | hp |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.964 | 0.815 | 0.759 | 0.661 | 0.601 | 0.561 | 0.535 | 0.481 | 0.392 | 0.378 |
| onsets | 4869 | 4342 | 1312 | 1882 | 4418 | 457 | 1732 | 1057 | 53 | 765 |

**Findings.**
- **The pooled per-stem pipeline works**: kick/snare/toms strong (0.96/0.82/0.76),
  and the historically weak cymbals recover hard, **ride 0.535** and **crash 0.561**
  here vs full-mix ParaDB ride ~0.18, crash ~0.42 (and STAR-val ride 0.43). The
  per-stem (isolated-instrument) direction is the cymbal win, exactly what the
  separation-aware experiment targeted.
- **The shared per-lane picker earns its place on separated audio.** Bare
  peak-pick vs the `drumjot_dsp` deterministic picker on this val: +0.082 ride,
  +0.089 misc-cym, +0.059 side-stick, +0.057 closed-hat, +0.049 crash, +0.034
  open-hat (precision-driven). Contrast the near-no-op (dF ≈ 0) on *clean* STAR
  val, the picker pays off where the audio is messy (real separator output), which
  is the deployment domain.
- **Exposed the `val_macro_f1` bug.** During-training val read ~0.057 despite kick
  at 0.96, because the old metric averaged over all 10 lanes including the ~9
  empty-reference lanes per per-stem clip. Fixed afterward (now counts only lanes
  with reference onsets); see CHANGELOG / `mean_f1`.

**Caveats.** cap=30, 15 epochs (short), 1 seed, layer 10 (pre-sweep default; the
sweep later found cymbals prefer L10 anyway, side-stick L1). Pooled val mixes the
three sources' per-stem clips, so it's a deployment-domain proxy, not ParaDB.

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
