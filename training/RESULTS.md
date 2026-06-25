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

## 2026-06-25 · ROOT CAUSE of the "broken cymbals": `--lanes` built the model without `lane_names` (one-line bug). Snap helps, gate hurts.

**The bug.** `train.py` main built `MultiLaneHeads(...)` **without `lane_names=cfg.lanes`**, so the
model had all 8 global `LANES` heads in `LANES` order while targets / loss / pos_weight / sibling /
aux-activity / val are indexed in **`cfg.lanes`** order. With any `--lanes` subset the orderings
diverge → the selected lanes get **zero gradient**, stay at random init (~0.5 sigmoid on silence,
~0 onset-F1). `--lanes hc,ho,rd,cr` left all four cymbal/hat heads untrained; the A/B v2
(`--lanes k,s,t,hc,ho,rd,cr`) left **rd/cr** untrained. **This, not snap/gate, h128, data, or
paradb (all chased + exonerated); is the entire cymbal regression.** Introduced by `a498466`
(--lanes); fixed in `f02dea9` (`lane_names=cfg.lanes` + a `train_loop` guard asserting
`model.lane_names == cfg.lanes`). **⇒ the cymbal numbers of `ab2_prev` / `ab2_paradb` (A/B v2) are
invalid, both were `--lanes`-bugged (rd/cr never trained).** Pre-bug checkpoints (≤06-21, no
`--lanes`) were always healthy; a historical silence sweep confirmed the break is 06-25-only.

**Confirmation, C3 3-way (h128, cap100; the smallest/fastest config, all-broken pre-fix).**
Silence baseline (sigmoid on 8 s of zeros) + MDB-Drums onset-F1 (23 tracks, `current` thresholds):

| arm (post-fix) | hc | ho | rd | cr | silence rd / cr |
|---|---|---|---|---|---|
| pre-fix (any) |, |, | **0.004** | **0.000** | **0.44 / 0.49** (untrained) |
| raw (no snap/no gate) | 0.665 | 0.328 | 0.486 | 0.332 | 0.000 / 0.004 |
| snap (aligned, gate off) | **0.714** | 0.266 | **0.581** | **0.509** | 0.000 / 0.005 |
| snapgate (aligned + gate) | 0.626 | 0.214 | 0.548 | 0.462 | 0.000 / 0.002 |

rd/cr recovered ~0 → 0.49–0.58 / 0.33–0.51, comparable to the prior healthy 5-lane c3000/h256
checkpoint (rd 0.679 / cr 0.515) at 1/30 the data.

**Recipe signal (directional, cap100 is tiny, confirm at scale):** offline **snap (aligned
onsets) HELPS** sharp-attack lanes (rd +0.10, cr +0.18, hc +0.05 vs raw; slightly hurts soft
open-hat, ho −0.06). The **label-quality gate HURTS** every lane (snapgate < snap), matching the
relative-floor over-drop concern (`label_support_percentile=60` collapses to the noise floor on
sparse cymbal windows). ⇒ lean **aligned-onsets ON + `--label-min-support 0`** (gate off).

**Reproduce (per arm; local RTX 3080):**
```
MODELS_DIR=/codebox-workspace/drumjot/models-cache \
DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep DRUMJOT_PARADB=/codebox-workspace/datasets/paradb-sep \
DRUMJOT_ALIGNED_ONSETS=<raw:/dev/null | snap:/codebox-workspace/datasets/_onsets_aligned.json> \
OMP_NUM_THREADS=8 PYTHONPATH=training:dsp python3 -m drumjot_training.train \
  --dataset pooled --pool-sources star,enst,egmd,paradb --pool-cap 100 --pool-val-cap 30 --pool-balance \
  --pool-cache /codebox-workspace/datasets/_cache_mert_pooled --head-hidden 128 --lanes hc,ho,rd,cr \
  --epochs 30 --es-min-epochs 12 --no-keep-best --early-stop --no-filter-report --seed 0 \
  --label-min-support <raw,snap:0 | snapgate:0.95> --out <dir>
```
Other params = train.py defaults @ `f02dea9`: loss=bce (focal none), pos_weight cap 50,
sib_pos_weight 3.0, aux_act_weight 0.5 (SUSTAINED_LANES ho/rd/cr), sigma_frames 1.5,
label_support_percentile 60, label_support_window_s 0.04, encoder MERT-v1-330M layer 10 @75 fps,
high_band on (in_dim 1040), bf16 autocast + TF32 (Ampere). Eval: `eval_mdb.py --checkpoint <dir>
--lanes hc,ho,rd,cr` (MDB has only h/c stems); silence = 8 s zeros → sigmoid mean.

---

## 2026-06-21 · MDB cross-check: ParaDB param gains DON'T fully generalize

Ran the dist<=0.10 predictor (`param_predictor_a2md.joblib`, the best ParaDB
artifact) on **MDB-Drums (23 tracks)**, an independent real-domain test set, to
check whether the hybrid routing (read off 6 ParaDB songs) is a real pattern or
overfit. `eval_mdb.py` now prints the same predict + hybrid columns.

| captured of oracle gap | ParaDB (6) | **MDB (23)** |
|---|---|---|
| determ self-cal | strong (hc +59%) | **-26%** |
| learned predict | +0.017 F1 | **-0.006 F1** |
| hybrid | +0.024 F1 | **+0.002 F1** |

Per-lane, the routing splits into robust vs overfit:

| lane | source | ParaDB | MDB | verdict |
|---|---|---|---|---|
| hc | determ | +59% | +11% | robust |
| cr | learned | +64% | +13% | robust |
| ho | learned | +24% | +4.5% | robust (weak) |
| **rd** | learned | +108% | **-40%** | **overfit -> dropped** |

**Honest read:** the headline ParaDB hybrid (+0.024) was partly fit to those 6
songs; on independent MDB the predictor is net-neutral (-0.006) and the hybrid is
barely positive (+0.002). **Ride was the blowup**, learned +108% on ParaDB (4
songs; predict even *beat* the oracle = overfit) but -40% on MDB; it's the thinnest
lane everywhere (26-30 A2MD train rows, <=7 test songs). Only `hc->determ` +
`cr->learned` (+`ho` weakly) capture gap on BOTH sets.

**Action:** revised `DEFAULT_ROUTING` to put **ride on the global rail** (no
adaptation), cross-validated on both sets. With ride dropped the hybrid is small
but positive on both (ParaDB ~+0.018, MDB ~+0.009). Net: adaptive params are a
**real but modest** effect, NOT the big win the single ParaDB run suggested; a
deploy decision now rides on whether ~+0.01-0.02 F1 on hat/crash is worth the
machinery. Artifact + log: `param_predictor_a2md.joblib`, `eval_mdb_hybrid.log`.

---

## 2026-06-21 · More real data (A2MD dist0p20) HURTS, quality > quantity

Ran dist0p20 separation on the 3080 (537 songs, 2.7x the dist<=0.10 set), rebuilt
the identity corpus (1,310 rows, 0.95 support gate), retrained, re-evaluated. First
comparison under the **deterministic chart pick** (`rlrr.pick_hardest`; the
`current`/`determ`/`oracle` columns are now byte-identical across runs, so these
deltas are real, not eval noise):

| corpus | predict | hybrid |
|---|---|---|
| dist<=0.10 (197 songs, 477 rows) | **+0.017** | **+0.024** |
| dist<=0.20 (537 songs, 1,310 rows) | +0.002 | +0.009 |

More data made it **worse**, concentrated exactly in the lanes dist0p20 labels are
sloppiest on: **crash +64%->+27%**, **open-hat +24%->−44%** (flipped harmful); ride
still good but lower (+108%->+64%); closed-hat ~flat (determ-routed). The 0.95
support gate drops the *worst* dist0p20 labels but the survivors still carry enough
timing noise to teach slightly-wrong oracle params. **Quality beats quantity for
the param corpus: the dist<=0.10 predictor (`param_predictor_a2md.joblib`) is the
best artifact so far.** This is the same "closeness to the clean signal matters"
lesson as the synth dilution + the augmentation-hurts result, a third time.

**Stricter gate partially recovers, but dist0p20 still loses.** Rebuilt the
537-corpus at `--min-support 0.98` (CPU-only re-gate, probs cached; dropped only
1,310->1,251 rows), retrained, re-evaled:

| corpus / gate | predict | hybrid | cr pred% | rd pred% |
|---|---|---|---|---|
| dist<=0.10, 0.95 (197) | **+0.017** | **+0.024** | +64% | +108% |
| dist<=0.20, 0.95 (537) | +0.002 | +0.009 | +27% | +64% |
| dist<=0.20, 0.98 (537) | +0.010 | +0.016 | +53% | +95% |

So the gate IS a real lever (0.98 recovered ~half: crash 27%->53%, ride 64%->95%),
but **even strictly gated, dist0p20 stays below clean dist<=0.10.** Since the 537
set = the 197 clean songs + 340 dist0p20, and it loses to 197 alone, the dist0p20
additions are net-negative regardless of gate -- their residual is *timing* noise
(slightly-off MIDI), which support (an onset-presence check) only partly catches.

**Verdict:** keep the dist<=0.10 predictor (`param_predictor_a2md.joblib`,
+0.017/+0.024) as the param artifact. To grow it, get more *low*-dist (clean) real
songs, not higher-dist ones; a higher gate just converges back toward the dist<=0.10
subset. Artifacts: `a2md_corpus_id_v2{,_s98}.npz`,
`param_predictor_a2md_v2{,_s98}.joblib`, `eval_a2md_{197,537}_det.log`,
`eval_a2md_537_s98.log`.

---

## 2026-06-21 · Adaptive params, take 2: REAL-domain corpus (A2MD) flips it positive

Follow-up to the negative result below. That predictor was trained on the model's
own *synthetic* training stems (in-domain, overconfident curves); the diagnosis
was synth→real domain mismatch, and the fix was "build the corpus from curves that
look like deployment -- real separated stems" (next-lever (a)). Did exactly that:
197 real A2MD songs (dist≤0.10) run through our own separation, full-song MIDI drum
onsets (channel-9-only), per-stem corpus built via the new **fresh-encode path**
(`build_param_dataset_perstem.py` now plans + encodes stems not in the MERT cache).
**Identity only, no augmentation**, to isolate the real-data variable -- 477 rows,
5 lanes, 188 songs, ~13 min on the 1660. The 0.95 label-support gate + onset snap
runs over every window.

Re-eval on the same held-out ParaDB, per-lane **% of that lane's oracle gap
captured** (negative = actively worse than today's global params):

| predictor (corpus) | hc | ho | rd | cr | **mean captured** |
|---|---|---|---|---|---|
| synth-only (28,645 rows) | −66% | +19% | −23% | +23% | **−0.002 F1** |
| **A2MD-only (477 rows)** | −12% | +24% | **+89%** | **+63%** | **+0.016 F1** |
| union, synth+A2MD (29,122) | −51% | +25% | +25% | +31% | **+0.004 F1** |
| determ self-cal (no training) | **+59%** | −169% | +6% | −61% |, |

(oracle gap +0.042 F1; `current`/`determ`/`oracle` columns byte-identical across
all three runs -- only the predictor changed.)

**Takeaways:**
1. **Real-domain data is the missing ingredient.** A2MD-only (+0.016) beats
   synth-only (−0.002) despite **60× fewer rows** -- your call that it was a
   data-diversity problem, not a dead end, was right.
2. **Blending raw dilutes it.** The union (+0.004) ≈ synth-only: 477 real rows are
   1.6% of the synthetic pool and get swamped. Real data must be **isolated or
   heavily upweighted**, not poured into the synthetic corpus.
3. **Cymbals are where real data wins most** -- ride −23%→**+89%**, crash
   +23%→**+63%**. Synthetic STAR cymbals are clean/uniform; real A2MD cymbal timbre
   is the diversity the predictor needed. (hp/rd corpus is thin at 33/30 rows, so
   ride's +89% is encouraging but noisy.)
4. **Closed-hat stays a deterministic-self-cal job.** determ +59% beats every
   learned predictor (all net-negative on hc). The two methods are complementary.
5. **Deployable policy = hybrid, per lane:** determ for **hc**; learned-on-real
   (A2MD) for **rd / cr / ho**. Best-of-each capture ≈ hc 59% + rd 89% + cr 63% +
   ho 24%, far more of the +0.042 gap than any single method.

**Hybrid picker, measured** (`parampred/hybrid.py`, `HybridParamPicker` +
`DEFAULT_ROUTING`; wired into `eval_paradb --oracle-report`). Routing each lane to
its winner and scoring on the same ParaDB gap table:

| | determ-only | learned-only (A2MD) | **HYBRID** |
|---|---|---|---|
| mean captured of +0.042 gap | −0.022 F1 | +0.016 F1 | **+0.023 F1 (55%)** |

Per-lane the hybrid takes hc 60% (determ), ho 22% / rd 88% / cr 63% (learned) --
it beats *both* single methods because no single source wins every lane (determ is
net-negative alone, dragged down by ho/cr; learned alone leaves hc on the table).
Caveat: the routing was read off these 6 ParaDB songs, so re-validate it on the
larger dist0p20 A2MD corpus + MDB before trusting it as a constant.

**Augmentation on real audio HURTS (don't do it).** Rebuilt the A2MD corpus with
the +4 onset-preserving variants (gain/EQ/reverb/compression/noise/codec) -> 2,385
rows (5x identity), retrained, re-evaluated:

| A2MD predictor | mean captured | hybrid captured |
|---|---|---|
| identity-only (477 rows) | **+0.016** | **+0.023** |
| +augmented (2,385 rows) | +0.006 | +0.014 |

Augmenting *already-real, already-separated* stems pushes them OFF the deployment
manifold (ParaDB is real + separated + un-augmented), so the off-distribution rows
dilute the pure-real signal -- the same lesson as the synth+A2MD union, restated:
**corpus value tracks closeness to the deployment distribution, and identity real
data is closest.** Clearest on the stable `ho` lane (identical current 0.630 /
oracle 0.675 across runs): predict 0.640 -> 0.619, i.e. augmentation flipped it
from +0.010 to −0.011. So grow the corpus with **more real songs, not augmented
copies.** Augmentation is presumably still right for the synthetic ADT datasets (it
moves them *toward* realism); it's wrong for already-real audio.

> **Eval-variance caveat (ROOT-CAUSED + FIXED):** the `rd` current/oracle moved
> between runs (0.174->0.085) because `Kaikai_Kitan.zip` ships TWO charts --
> `_Expert` (1640 onsets) and `_Hard` (1516) -- that TIE at `complexity=4`, and
> `_pick_rlrr`'s `max(charts, key=complexity)` then resolved the tie by unstable
> `rglob` order (fresh temp-dir extraction each run), so it parsed a different chart
> each time. Fixed: `rlrr.pick_hardest` now breaks ties by filename difficulty
> (expert>hard>medium>easy) then path, so it deterministically picks `_Expert`. The
> dense lanes (`hc`/`ho`/`cr`) were already stable, so the conclusions above hold;
> single-run pre-fix `rd` numbers were noisy. Re-run evals will all use `_Expert`.

**Next:** grow the real corpus -- the other dist buckets (0.00 tight + 0.20 with
the support gate dropping bad lanes) and more A2MD songs (dist0p20 separation
running on the 3080); then the hybrid picker + a deploy safety-rail, and fix eval
determinism. Artifacts at `checkpoints/ovn3080/mixed_c3000_h256_s1/`:
`a2md_corpus_id.npz` (use this, not `_aug`), `param_predictor_a2md.joblib`,
`param_predictor_synth_a2md.joblib`, `param_predictor_a2md_aug.joblib`,
`eval_{synth,a2md,synth_a2md,a2md_aug}.log`.

---

## 2026-06-21 · Adaptive params: trained predictor does NOT beat global (negative)

Full pipeline ran end-to-end overnight on the hat+cymbal checkpoint: built a
**28,645-row** corpus from STAR/ENST/E-GMD per-stem (cache-aware -- identity free
from the training MERT cache, +4 onset-preserving augmented variants per window,
all 3,768 h/c stems, ~5.5 h on the 1660), trained the per-lane HistGBR predictor,
evaluated on held-out ParaDB.

**The predictor does not capture the oracle gap -- it's ~neutral-to-slightly-worse
than today's global params:**

| lane | current | predicted | oracle | captured |
|---|---|---|---|---|
| hc | 0.502 | 0.476 | 0.542 | **-66%** |
| ho | 0.630 | 0.638 | 0.675 | +19% |
| rd | 0.174 | 0.169 | 0.198 | -23% |
| cr | 0.373 | 0.386 | 0.433 | +23% |

Mean captured **-0.002 F1** (oracle gap reconfirmed +0.042). It helps crash +
open-hat slightly but *hurts* closed-hat and ride, netting neutral. The deploy
safety-rail (fall back to global when predicted loses on aggregate) would here
just revert to global.

**Diagnosis (primary suspect): train/eval distribution mismatch.** The corpus is
built on the model's own TRAINING stems, where its activation curves are
in-domain / overconfident / clean (params barely matter), whereas ParaDB is
out-of-domain real *separated* stems with messy, low-confidence curves (where
params matter most). The learned feature->param mapping doesn't transfer.
Augmentation covered gain/EQ/codec/reverb/noise but NOT separation artifacts or
the in-domain->OOD confidence gap. Secondary: per-30 s-window corpus vs full-song
eval granularity. The +0.042 ceiling is real; a learned cross-song predictor
trained on in-domain curves isn't the way to bank it.

**Next levers:** (a) build the corpus from curves that look like deployment --
real separated stems (run more real songs through our separation; needs labels
for the oracle) or at least the model's HELD-OUT splits, not its training data;
(b) per-song *self-calibrating* params at inference (e.g. threshold at the knee
of THIS song's peak-height histogram -- the deterministic baseline.py path), which
needs no cross-song transfer; (c) domain-invariant features. Artifacts kept at
`checkpoints/ovn3080/mixed_c3000_h256_s1/` (param_corpus.npz, param_predictor.joblib).

**Held-out check (STAR train vs test/val, free via cache):** confirms the model
is less accurate on unseen stems (current F1 drops every lane) and the hat oracle
gap ~doubles on held-out (hc +0.046->+0.096, ho +0.028->+0.062), BUT even held-out
*synthetic* is far easier than real ParaDB (oracle F1 0.57-0.80 vs 0.20-0.54) and
the gap STRUCTURE differs -- ride's oracle gap is +0.15 on STAR vs +0.02 on ParaDB
(model-limited). So the synthetic->real gap dwarfs train->held-out; a predictor
trained on synthetic (even held-out) won't transfer, and the big synthetic ride
gap is exactly what taught it to over-adjust ride on ParaDB.

**Deterministic self-calibration (knee threshold from each song's OWN curve, NO
training) -- the first real win.** ParaDB:

| lane | current | determ | predict | oracle | det captured |
|---|---|---|---|---|---|
| hc | 0.507 | **0.530** | 0.478 | 0.549 | **+54%** |
| ho | 0.630 | 0.554 | 0.638 | 0.675 | -169% |
| rd | 0.085 | 0.087 | 0.088 | 0.109 | +7% |
| cr | 0.375 | 0.337 | 0.388 | 0.435 | -62% |

Self-calibration **captures 54% of the closed-hat gap (0.507->0.530), beating both
global and the learned predictor** -- but HURTS the sustained lanes (ho/cr).
Mechanism: the knee rule needs a bimodal peak-height histogram -- clean for sharp
percussive `hc`, smeared for ringing `ho`/`cr`/`rd`. **Shippable now: apply
self-cal to `hc` only (free, +0.023 F1), keep global elsewhere.** Next: a
ring-aware threshold rule for the sustained lanes (small 6-song ParaDB sample, so
re-confirm as the test set grows). Added as the `determ` column in
`eval_paradb.py --oracle-report`.

**MDB-Drums real-domain re-check (`eval_mdb.py`, 23 MedleyDB tracks) -- the
deterministic win does NOT replicate, and per-song adaptation is not robust.**

| lane | current | determ | predict | oracle | det% | pred% |
|---|---|---|---|---|---|---|
| hc | 0.641 | 0.652 | 0.667 | 0.733 | +11% | +28% |
| hp | 0.353 | 0.332 | 0.352 | 0.379 | -80% | -4% |
| ho | 0.595 | 0.572 | 0.623 | 0.695 | -24% | +28% |
| rd | 0.679 | 0.656 | 0.708 | 0.743 | -36% | +46% |
| cr | 0.515 | 0.509 | 0.538 | 0.668 | -4% | +15% |

The two real datasets give **opposite verdicts**: on ParaDB deterministic self-cal
won closed-hat (+54%) and the learned predictor failed; on MDB deterministic mostly
HURTS (mean -26%) while the predictor mostly WINS (+15..+46%). Cause = curve
cleanliness: ParaDB (commercial mix -> our separation) is messy (knee finds a real
split; synthetic-trained predictor is OOD); MDB (cleaner MedleyDB-derived stems) is
model-confident (global already near-optimal; knee over-adjusts; predictor back
in-distribution). **Per-song peak-pick adaptation -- learned or deterministic --
does not robustly beat well-tuned global thresholds on real audio; its sign flips
with audio difficulty. Do not ship a blanket per-song-params change.**

**Key positive diagnostic:** ride is **0.679 on MDB vs 0.085 on ParaDB** (same
model/lane). The model CAN detect ride when the separated stem is clean; ParaDB
ride dies in the commercial-mix separation. → the lever for the hard cymbal lanes
is **separation quality**, not the peak-picker or the onset model.

---

## 2026-06-20 · Adaptive per-song peak-pick params: oracle-gap gate (hat+cymbal ckpt)

First run of the new `parampred` gap gate (`eval_paradb.py --oracle-report`) on the
A/B hat+cymbal checkpoint `ovn3080/mixed_c3000_h256_s1/loss_ab_mixed.pt`
(lanes `hc/hp/ho/rd/cr`, in_dim 1040, tuned thr `hc .4 / hp .7 / ho .8 / rd .1 /
cr .7`), 6 ParaDB maps, cached stems. The gate sweeps **all 5 peakpick params per
song per lane** against GT and reports the ceiling vs today's global params:

| lane | current (global) | per-song oracle | gap | songs |
|---|---|---|---|---|
| hc | 0.507 | 0.549 | **+0.041** | 6 |
| ho | 0.630 | 0.675 | **+0.045** | 2 |
| rd | 0.085 | 0.109 | **+0.023** | 4 |
| cr | 0.375 | 0.435 | **+0.060** | 6 |

**Mean oracle gap +0.042 F1.** Read:
- The per-song-param **ceiling is modest** (~+0.04 mean), concentrated in **crash
  (+0.060)** and the hats. A learned predictor captures only a *fraction* of a
  ceiling, so the realizable win here is small.
- **Ride is model-limited, not param-limited:** at 0.085 F1 the oracle barely
  lifts it (+0.023). No threshold rescues a lane the model isn't detecting, the
  ride fix is the model/training, not adaptive params (consistent with the
  long-standing ride/cymbal weak-spot entry below).
- **Caveat, these 6 maps are clean** (support ≈1.0, offsets ≤34 ms). Per-song
  params earn their keep on degraded / oddly-mixed real audio, which the
  augmentation pipeline simulates but this test set doesn't contain. So this gate
  is plausibly a *lower bound* on the real-world gap. Open question: measure the
  oracle gap **under augmentation** (build_param_dataset on these songs, identity
  vs augmented rows) before committing to the full corpus+train.

**Augmented-gap follow-up (same checkpoint, 6 songs × 4 onset-preserving aug
variants, 90 rows).** The clean-test-set caveat was right: degradation widens the
oracle gap, and **almost all of it is closed hi-hat**.

| lane | identity gap | augmented gap | widening |
|---|---|---|---|
| hc | +0.040 | **+0.148** | **+0.109** |
| cr | +0.060 | +0.066 | +0.006 |
| ho | +0.045 | +0.047 | +0.002 |
| rd | +0.024 | +0.030 | +0.006 |
| mean | +0.042 | +0.073 | +0.031 |

`hc` global-param F1 *collapses* under degradation (0.502→0.400) while the
per-song oracle holds (0.548) → adaptive params recover **+0.148** on degraded
closed-hat. Crash is a steady ~+0.06 clean or degraded. Open-hat is small (high
base F1); ride stays model-limited. → **Build the predictor; expect the win in
`hc` + `cr`.** Caveat: training it needs labeled hat+cymbal data that ISN'T the
ParaDB test set (extend build_param_dataset to the E-GMD/STAR/ENST loaders, or
more .rlrr-format songs). Identity rows reproduced the gate's +0.042 exactly
(internal consistency check).

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

## ADTOF vs our cym+HH checkpoint, apples-to-apples (2026-06-20)

Head-to-head of the **deployed ADTOF backend** vs **our `loss_ab_mixed.pt`**
(cym+HH checkpoint: lanes hc/hp/ho/rd/cr) on the *identical* test suite, via
`sota_eval.py --backend {adtof,learned}`: same per-stem audio (enst-sep /
mdb-sep), same GT, same fold5, same `mir_eval` ±50 ms, same track set. Each
detector runs at **its own deployed config** (ADTOF = adaptive threshold +
audio-refine + hihat audio-supplement + amplitude floor + crash-shadow; learned
= the shared training picker at the checkpoint's tuned per-lane thresholds), i.e.
"what we'd actually ship", not a stripped comparison. Pooled (micro) F:

| set | class | ADTOF (R / P / F) | learned (R / P / F) | winner |
|---|---|---|---|---|
| **MDB** (pristine) | HH | 0.971 / 0.615 / **0.753** | 0.936 / 0.588 / 0.722 | ADTOF +0.031 |
| MDB | CY | 0.757 / 0.824 / **0.789** | 0.866 / 0.645 / 0.740 | ADTOF +0.049 |
| MDB | AVG | **0.771** | 0.731 | ADTOF |
| **ENST** (learned tuned here*) | HH | 0.937 / 0.780 / **0.851** | 0.915 / 0.640 / 0.753 | ADTOF +0.098 |
| ENST | CY | 0.839 / 0.648 / 0.731 | 0.784 / 0.731 / **0.756** | learned +0.025 |
| ENST | AVG | **0.791** | 0.755 | ADTOF |

*ENST drummer_3 was in our val/threshold-tuning pool → mildly optimistic for the
learned model; MDB is pristine for both, ADTOF was never tuned on either.

**Verdict: ADTOF currently wins.** It takes HH on both sets (decisively on ENST)
and CY on the pristine MDB; the learned model only edges CY on the set it was
tuned on. The cym+HH checkpoint is **not yet ready to replace ADTOF.**

**Root cause = precision, not recall.** In every cell the learned model has
**higher recall but lower precision** (it over-fires): HH P 0.588/0.640 vs ADTOF
0.615/0.780; CY P 0.645/0.731 vs 0.824/0.648. The fixed tuned thresholds don't
transfer to unseen audio as well as ADTOF's *adaptive, self-calibrating* per-stem
threshold + amplitude/shadow gates do. Levers: re-tune thresholds toward
precision on a held-out set, add a learned-side energy floor or a small
adaptive-threshold term, and the full-kit checkpoint (more data) should help.
Re-run after the full-kit train. cf. [[ab-test-tuned-thresholds]].

Harness: `sota_eval.py` gained `--backend adtof` (`_predict_perstem_adtof`, keys
ADTOF h→hc, c→rd so fold5 scores it identically). JSONs at
`/codebox-workspace/cmp_{enst,mdb}_{adtof,learned}.json`.

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

**Verdict: MERT wins every lane; keep MERT.** (The MuQ encoder pathway was
**removed from the codebase on 2026-06-14** on the strength of this; see
CHANGELOG #7.) MuQ is decisively worse for drum onsets (macro 0.51 vs 0.63, −19%),
collapsing on the hard fine-timing/timbre lanes (closed-hat −0.30, ride −0.23).
Two coherent causes, both matching priors:
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

> Erratum: in this cap-60 realistic run the cymbal classifier's sibling vector was
> a silent no-op (fed `onsets_by_lane`, which carries only the stem's own lanes, so
> the cross-stem vec was all-zeros), i.e. the cymbal arm was effectively no-sib.
> Per the ceiling run, real sib adds ~+0.02–0.04 to cymbals -- not enough to flip
> the crash deficit (−0.12). Fixed for the cap-300 re-run (use the full-kit
> `weight_onsets`); conclusion unchanged.

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

**cap-300 (5x data) confirmation -- two-stage still loses (2026-06-14).** Re-ran the
realistic test at `--pool-cap 300` (802 train / 173 val per stem; val identical to
cap-60 so directly comparable) with ALL fixes: cross-stem sib live, F1 counts
zero-candidate clips as misses, baseline keep_best + 2 seeds, single-pass
preallocated stage-2 arrays (the cap-300 OOM fix). Per-lane onset-F1:

| lane | per-frame | 2stage | real Δ | cap-60 Δ |
|---|---|---|---|---|
| rd | 0.597 | 0.602 | +0.005 (tie) | −0.045 |
| cr | 0.696 | 0.600 | **−0.096 (lose)** | −0.120 |
| mc | 0.405 | 0.479 | +0.073 (noisy) | +0.149 |
| hc | 0.741 | 0.750 | +0.009 (tie) | +0.136 |
| hp | 0.414 | 0.373 | **−0.041 (lose)** | +0.167 |
| ho | 0.763 | 0.735 | **−0.028 (lose)** | +0.138 |

Proposer val recall: cymbals 0.860, hats 0.919 (barely up from cap-60's 0.826/0.904).
**Verdict holds and hardens: two-stage does NOT beat per-frame at 5x.** It ties on
the easy lanes (ride, hc) and LOSES on crash, hp, ho; only noisy mc favors it. The
cap-60 hat "wins" (hc/ho +0.13) were the F1-inflation bug + an under-trained
final-epoch baseline -- with the fair keep_best baseline, **more data helped
per-frame MORE than two-stage** (per-frame hc 0.65->0.74, ho 0.71->0.76), widening
the gap in per-frame's favor. Two-stage is decisively not worth the added pipeline.
**Drop it; stay single-stage per-frame.**

**High-band frame-alignment check -- center=True is fine (2026-06-14).** A review
flagged that the 6-20 kHz high-band block (librosa `center=True`) might sit ~1-2
frames off the MERT conv frames it's concatenated to. Measured directly (synthetic
clicks through `highband_from_wave` + `MertEncoder.encode`, cross-correlated vs the
`round(t*fps)` label grid): **MERT-minus-HB offset = -0.43 frames (-5.8 ms)**, below
the ~0.5-frame "aligned" bar. An A/B training the cymbal detector on center=True vs
center=False high-band confirmed no gain (rd −0.013, cr −0.011, mc −0.004, all within
noise -- center=False marginally *worse*). **Non-issue; keep center=True, no cache
bump.** (`tmp_hb_align_check_v5k.py` / `tmp_cym_center_ab.py`.)

**`cym` sub-6 kHz timbre block A/B -- no benefit (2026-06-14).** Tested the
ride-ping-vs-crash-wash block (`embeddings.cym_features`: low_mid ratio, low-band
crest, flatness + 2 smoothed copies) that was built but never A/B'd. Per-stem
cymbals + hats, cap 150, 2 seeds, per-lane keep_best; reuse cached [MERT|HB],
append the cym block ([MERT|HB|CYM]) for the +cym arm, train the per-frame detector
both ways:

| lane | baseline | +cym | Δ |
|---|---|---|---|
| rd | 0.662 | 0.647 | −0.015 |
| cr | 0.657 | 0.670 | +0.013 |
| mc | 0.466 | 0.449 | −0.017 |
| hc | 0.718 | 0.711 | −0.007 |
| hp | 0.485 | 0.474 | −0.011 |
| ho | 0.723 | 0.731 | +0.008 |

**Wash -- every delta is within seed noise (±0.02); crash's +0.013 is cancelled by
ride/mc (cymbal sum 1.785 vs 1.766, net −0.019).** The hand-crafted ride/crash
timbre cue adds nothing over MERT + the high-band block at this scale. (NB:
baselines here, e.g. ride 0.662, sit above the cap-300 run's per-frame ride 0.597
-- partly the new per-lane keep_best, partly cap/seed variance; not directly
comparable.)

**cym A/B on a CYMBAL-BALANCED set -- still no benefit (2026-06-14).** To rule out
"crash was too starved for cym to matter," curated a same-size set (450 train/stem)
from the FULL pool via greedy "fill the rarest lane" selection, lifting crash from
~1:12 vs ride (natural) to ~1:3 (train cymbal onsets: ride 65k / crash 22k / mc 22k;
hats hc 68k / hp 52k / ho 50k), then re-ran base vs +cym:

| lane | baseline | +cym | Δ | train onsets |
|---|---|---|---|---|
| rd | 0.682 | 0.637 | −0.044 | 65k |
| cr | 0.645 | 0.645 | −0.000 | 22k |
| mc | 0.339 | 0.375 | +0.036 | 22k |
| hc | 0.709 | 0.711 | +0.002 | 68k |
| hp | 0.428 | 0.420 | −0.008 | 52k |
| ho | 0.723 | 0.728 | +0.005 | 50k |

**Crash gets exactly 0.000 from cym even with 4x the crash representation, ride is
hurt (−0.044). Conclusive: the cym block is not useful -> REMOVED from the codebase
(2026-06-14), same as MuQ and dropped-neg.** Second
finding: **balancing crash did NOT move baseline crash F1** (0.645 balanced vs 0.657
natural -- flat), so crash F1 is NOT crash-volume-starved. The cymbal ceiling is
**separation quality / inherent ride-crash acoustic overlap**, not data quantity --
updates the "more crash data" lever (likely won't help; separation is the lever).

**dropped-neg A/B -- doesn't help, defaulted OFF (2026-06-14).** A/B of
`use_dropped_neg` (feed the `x` ghost lane's onsets to the loss as hard negatives
for every output lane). Real pipeline (CachedClips carry the negative/weight
targets; only the loss flag differs), per-stem cymbals+hats, cap 150, 2 seeds,
keep_best, tuned thresholds. ON vs OFF (Δ = on − off):

| lane | F1 off | F1 on | ΔF1 | P off | P on | ΔP |
|---|---|---|---|---|---|---|
| rd | 0.655 | 0.625 | −0.030 | 0.634 | 0.645 | +0.012 |
| cr | 0.641 | 0.618 | −0.023 | 0.671 | 0.637 | −0.035 |
| mc | 0.490 | 0.492 | +0.003 | 0.550 | 0.501 | −0.048 |
| hc | 0.718 | 0.722 | +0.004 | 0.717 | 0.726 | +0.009 |
| hp | 0.485 | 0.481 | −0.004 | 0.459 | 0.422 | −0.037 |
| ho | 0.736 | 0.734 | −0.003 | 0.824 | 0.816 | −0.008 |

The feature was meant to raise PRECISION on leak-prone lanes; instead precision
mostly *drops* (cr −0.035, mc −0.048, hp −0.037) and it costs ride/crash **F1**
(−0.030/−0.023, the targeted lanes). Everything else is noise. **Removed entirely
from the codebase (2026-06-14)** -- per-stem separation already strips the aux perc,
so the hard negatives mostly land on silent frames and over-suppress real cymbal
attacks near residual bleed (right idea, wrong pipeline stage). See CHANGELOG #6.

**keep_best re-baseline: per-lane vs global vs final (2026-06-14).** From the
per-epoch per-lane val-F1 curves (`history["vf1_<lane>"]`), UNTUNED (0.5 thr) so
absolute values are low; the comparison across the three is the point:

| lane | final | global | per-lane | pl−glob | pl−final |
|---|---|---|---|---|---|
| rd | 0.519 | 0.616 | 0.620 | +0.004 | +0.102 |
| cr | 0.581 | 0.528 | 0.616 | **+0.089** | +0.036 |
| mc | 0.301 | 0.425 | 0.487 | **+0.062** | +0.186 |
| hc | 0.705 | 0.715 | 0.720 | +0.006 | +0.015 |
| hp | 0.323 | 0.445 | 0.469 | +0.025 | +0.147 |
| ho | 0.721 | 0.722 | 0.730 | +0.008 | +0.009 |

**Per-lane keep_best is validated**: over the old global-best it adds real F1 where
a lane peaks off the macro (crash **+0.089**, mc +0.062, hp +0.025); over
final-epoch it's huge on the overfitters (mc +0.186, hp +0.147, ride +0.102).
Note crash benefits most from per-lane -> the cap-300 runs (global keep_best)
understated crash; doesn't change the two-stage verdict but worth knowing.

## Phase 1: head-capacity gate + convergence (2026-06-14)

Testing the "we're capacity/under-fit-bound, not data-bound" hypothesis: does a
bigger per-lane head lift the cymbal/hat ceiling? Cymbals+hats per-stem pool
(restricted to lanes hc/hp/ho/rd/cr/mc, 6 heads, full cross-stem sibling
weighting preserved), pooled star+enst+egmd, per-lane keep_best, batch held
constant across arms (the prior h256 confound was a GPU-forced batch-2). Slow on
the 1660 Super: h512 ~691s/epoch.

**Width A/B, cap-100, 12 epochs, seed 0 (h128 vs h512), tuned F1:**

| lane | h128 | h512 | Δ |
|---|---|---|---|
| hc | 0.627 | 0.628 | +0.001 |
| ho | 0.679 | 0.681 | +0.002 |
| rd | 0.549 | 0.518 | -0.031 |
| cr | 0.628 | 0.681 | +0.053 |
| mc | 0.352 | 0.333 | -0.019 |

Cymbal sum flat (1.529 vs 1.532). **But inconclusive: both arms were
UNDER-TRAINED at 12 epochs**, and the per-epoch curves showed it (the value of
tracing them) -- ho/cr still climbing at epoch 11 for both; the lone crash
"+0.053" was a same-epoch-6 eval spike (noise), not capacity. h512 just trains
slower (rd still climbing @11, hc collapsed at epoch 1) without overtaking h128.

**Convergence run: h128, 40 epochs, cap-100, enlarged 4x-windowed val (346->700
windows):** the bigger val **erased the crash lucky-spike** -- crash is now a
smooth curve converging to a stable **~0.52 plateau** (epochs 25-39 all
0.51-0.54); the old single-window `cr 0.628` was a small-val artifact. All lanes
**converge by ~epoch 25-30** (hc ~0.60, ho ~0.615, cr ~0.52 dead-flat from ~25).
rd/mc/hp **overfit** (peak mid-run, decay; rd 0.51@19 -> 0.44@39, mc 0.32@24 ->
0.27) so per-lane keep_best is essential. Final per-lane-keep_best tuned baseline:
hc 0.606 / hp 0.427 / ho 0.658 / rd 0.519 / cr 0.539 / mc 0.345.

**Takeaways:** (1) 12 epochs under-tests; **~30 is the real budget** at cap-100.
(2) ~~A fair width test needs >=30 epochs both arms; the flat 12-ep hint + cost
deprioritize pure width.~~ **RETRACTED (2026-06-16) -- the width A/B is invalid:
cap-100 is data-bound, so it cannot test capacity at all (see Phase 2). The fix
is not "more epochs," it's "data-UNbound arms."** (3) **Use the convergence-run
numbers as the honest cap-100 baseline** (NOT the spiky 12-ep ones). (4) Enlarged
val is a keeper -> full windowing (now default) gives it automatically. (5) Next:
the **data-scale axis** (full windowing x cap), the actual untapped-data test,
cheap at h128.

## Phase 2: data-scale axis (h128) (2026-06-16)

**Setup.** Identical to Phase 1's convergence run (cym+hat pool hc/hp/ho/rd/cr/mc,
same ~700-window val, per-lane keep_best, early-stop: es_min 20, |slope|<0.002,
jitter<0.015, 8-ep window, 80-ep cap), h128, batch 8, 8 workers. The **only**
knob is `--pool-cap`, now in WINDOW units. Caps: 1000 / 3000 / 0(full). Tuned
per-lane keep_best F1.

**Data-scale progression (tuned per-lane keep_best F1):**

| cap | train win | stop ep | hc | hp | ho | rd | cr | mc | **cymMac** | allMac |
|---|---|---|---|---|---|---|---|---|---|---|
| 100^1 | ~clip | 40(fix) | 0.606 | 0.427 | 0.658 | 0.519 | 0.539 | 0.345 | **0.468** | 0.516 |
| 500 | 1519 | 58 | 0.689 | 0.470 | 0.663 | 0.583 | 0.629 | 0.463 | **0.558** | 0.583 |
| 750 | 2008 | 69 | 0.691 | 0.478 | 0.638 | 0.590 | 0.625 | 0.450 | **0.555** | 0.579 |
| 1000 | 2563 | 69 | 0.689 | 0.478 | 0.664 | 0.623 | 0.620 | 0.504 | **0.582** | 0.596 |
| 3000 | 6559 | 62 | 0.702 | 0.507 | 0.661 | 0.607 | 0.656 | 0.509 | **0.591** | 0.607 |
| 0(full) | 15952 | (on 3080) | tbd | tbd | tbd | tbd | tbd | tbd | **tbd** | tbd |

cymMac slope: 100→500 **+0.090**, 500→750 **-0.003**, 750→1000 **+0.027**,
1000→3000 **+0.009**. The 500/750 points are indistinguishable (single-seed
noise ~±0.01-0.015); the climb to plateau happens ~750→1000, flat after.
**Knee ~cap-1000**; finer resolution is below the single-seed noise floor (would
need 2-3 seeds/point). NOTE: these caps predate the `mc`-lane removal, so cymMac
is 3-lane (rd+cr+mc); future 9-lane runs report 2-lane (rd+cr).

^1 Phase-1 convergence run; cap-100 was CLIP-unit (pre window-cap), so the cap
*number* isn't 1:1 on the windows axis, but the val is identical (~700 windows)
and the lift is unambiguous.

**Per-lane dynamics (cap-1000 curve).** Two regimes -> per-lane keep_best is
load-bearing: late monotonic risers hc(best@68) / ho(@62); early-peak-then-decay
(overfit) rd(@12) / cr(@15) / hp(@13); mid-late mc(@48). Early-stop @69 = global
flatness (risers plateaued, overfitters bottomed out), NOT all-at-peak.

**Headline: at h128 the cymbal ceiling is DATA-bound, not capacity-bound.** The
big lift is cap-100→cap-1000 (cym macro +0.114; mc +0.159, rd +0.104). The rise
persists up to ~cap-1000 (cap-500→1000 still +0.024 > noise) then flattens
(cap-1000→3000 only +0.009). **So h128's data-saturation knee is ~cap-1000**
(2563 train windows): below it, more data helps; above it, gains are within
single-seed noise / internal reshuffling. Note the hats already plateau by
cap-500 (hc/hp/ho flat across 500→1000); the cap-500→1000 gain is almost entirely
cymbals (rd +0.040, mc +0.041). Sub-1000 search bottomed out at the noise floor:
cap-500 (0.558) and cap-750 (0.555) are indistinguishable, the climb to plateau
is ~750→1000. cap-0(full) on the 3080 to confirm the top (E-GMD mix-shift caveat).

**Seed noise band (2-seed, for judging the future h512 width test).** Ran a 2nd
seed at cap-1000 + cap-3000 (seed 1; NOTE it's 9-lane post-mc-removal vs seed 0's
10-lane, so |Δ| conflates seed + the sibling-weighting change on ho/cr/rd). Per-
lane |Δ| over the 5 shared cym+hat lanes:
- **cap-1000: mean 0.017, max 0.040** (cr 0.040, ho 0.026 the movers; stable lanes
  hc/hp/rd 0.003-0.009).
- **cap-3000: mean 0.007, max 0.014** -- markedly tighter.
Takeaway: **noise shrinks with data** (more data -> more stable runs), and at the
plateau (cap-3000) the band is ~0.01. So a future h512-vs-h128 win must clear
~0.02 at cap-3000 (or ~0.04 at cap-1000) to be real. The data-scale per-point
deltas (500→750→1000) sit inside the cap-1000 band, confirming "knee ~cap-1000"
is the resolution limit without more seeds.

### Why this invalidates the Phase-1 width A/B

A data-bound operating point **cannot** test capacity. If data is the binding
constraint, width can't express itself -- neither arm has enough data to fill
even h128's capacity, let alone h512's. Worse, it's anti-informative: larger
models are more data-hungry, so at cap-100 h512 was *more* starved than h128
(trained slower, hc collapsed@1, rd still climbing@11). The flat/slightly-negative
h512 result is the expected artifact of that confound, not evidence about its
ceiling. Phase 1 measured "which model is more starved," not "which has more
useful capacity."

### Corrected width test: bracket h128's saturation

Width is only interpretable where the *smaller* model has saturated but data is
still being added:
1. Map h128's data curve (cap-1000/3000/0) to find two adjacent caps that tie
   (e.g. cap-3000 ~= cap-0) -> h128's taper / saturation bracket.
2. Run h512 at those two bracketing scales. If h512 climbs across the bracket
   where h128 was flat, it converts the extra data into accuracy h128 structurally
   can't -> width IS the lever. If h512 also flatlines, width is dead.
3. h512 is ~691s/ep on the 1660 -> this test wants the 3080.

Caveat: if h128 is still climbing at cap-0 (full = all data we have), it hasn't
saturated and the width test is confounded by data availability -- note the
limitation rather than over-read it.

## Matched-LR width A/B: h128 vs h512 at cap-3000 (2026-06-18)

**Setup.** The corrected width test above, on the cym+hat 5-lane set at cap-3000
(h128's saturation bracket), batch 8, **matched schedule across arms** (lr 3e-4,
warmup 500, seed 1, early-stop, 80-epoch cap), per-lane keep_best. h512 on the
3080; h128 (matched-LR control) + h256 (intermediate) on the 1660. Tuned per-lane
F1 on the pooled per-stem val:

| arm | params | hc | hp | ho | rd | cr | macro | cym(rd,cr) |
|---|---|---|---|---|---|---|---|---|
| h128 | 6.0M | 0.708 | 0.504 | 0.679 | 0.591 | 0.640 | 0.624 | 0.616 |
| h256 | ~16M | 0.716 | 0.521 | 0.697 | **0.643** | 0.623 | **0.640** | 0.633 |
| h512 | 47.5M | 0.722 | 0.507 | 0.689 | 0.631 | 0.638 | 0.637 | 0.635 |
| Δ128→512 | 8x | +.014 | +.003 | +.010 | **+.040** | **−.002** | +.013 | +.019 |

**h256 completes the curve (2026-06-20).** It's the sweet spot: best macro (0.640)
and best ride (0.643), both edging out h512 -- macro climbs 128→256 (+0.016) then
plateaus 256→512 (−0.003). Crucially **crash is FLAT across all three widths**
(0.640 / 0.623 / 0.638, within the ~0.02 band) -- width does nothing for crash.
The cymbal gain is entirely ride, and it saturates by h256.

**LR-confound control.** h128 at the matched lr 3e-4 ≈ h128 at the old default lr
1e-3 (hc .707 hp .510 ho .671 rd .593 cr .658) within noise (biggest Δ cr −0.018)
→ the warmup/LR schedule is NOT a confound; the A/B is clean.

**Verdict: capacity is NOT the cymbal bottleneck.** An 8× bigger head buys +0.013
macro; the only material per-lane gain is **ride +0.040**, and **crash is a tie
(−0.002)**. The rd gain is fragile too -- h512's rd peak was an early (ep5)
keep_best snapshot that later drifted to ~0.52. Single seed; deltas near the
cap-3000 noise band (~0.02). With Phase 2 (data volume doesn't move crash) this
rules out both capacity AND volume → the cymbal ceiling is intrinsic (features /
labels / separation / decision), not model size. The h256 curve confirms it:
crash is flat across 128/256/512 → width-invariant. **This is now directly
testable:** the onset-alignment work (2026-06-19) found ~28% of crash labels
mistimed/false (mostly lane-mislabels) → the aligned-onset retrain
(`--aligned-onsets`) is the test of whether the intrinsic crash ceiling is the
LABELS. h256 is the width to use for it (best cym at no extra crash cost).

## Cymbal feature-separability probe (2026-06-18)

**Setup.** `scripts/cymbal_feature_probe.py`: a convex linear probe (multinomial
logreg) over FROZEN features AT ground-truth onset frames, decoupled from the
GRU / recall / peak-pick / threshold -- isolates whether ride/crash (and hats)
are *linearly decodable* from the features. Same cap-3000 cym+hat per-stem cache
as the sweep (425,362 train / 24,032 val onsets; rd/cr 133,165 / 5,646). Compares
MERT(1024) / MERT+HB(1040) / HB-only(16) slices × single-frame / 8-frame
post-onset pool. 3080.

**Ride-vs-crash balanced accuracy (headline):**

| variant | MERT | MERT+HB | HB-only |
|---|---|---|---|
| single | 0.823 | 0.819 | 0.633 |
| pooled | 0.826 | **0.836** | 0.759 |

**5-way lane recall (pooled, MERT+HB):** hc 0.86, hp 0.49, **ho 0.34**, rd 0.84, cr 0.77.

**Findings.**
- **Ride/crash IS linearly separable (~0.84)** -- well above the ~0.6
  "info-not-in-features" floor. The encoder is not the main ride/crash problem.
- **MERT alone separates them (0.823); the high-band block barely helps** (HB-only
  weaker, 0.63-0.76). The distinction lives in MERT's sub-12 kHz content, NOT the
  >12 kHz sizzle → "richer HF" is not the cymbal lever (matches the mel-128 wash).
  Pooling the tail helps cymbals (+0.02; +0.13 for HB) -- cymbal identity is in
  the sustain.
- **The feature-limited lane is the HAT, not the cymbal.** Open-hat recall 0.34 /
  pedal 0.49 (vs rd 0.84, cr 0.77); both confuse with closed-hat. Pooling *hurts*
  the 5-way (0.715 single → 0.660 pooled): hats are attack-defined, cymbals
  tail-defined.

**Interpretation (corrected).** 0.84 linear separability is a REAL ~16% ride/crash
confusion floor (crash→ride 18.6%, ride→crash 14.3%) -- NOT "features fine, purely
downstream." Caveats both ways: (1) it's the LINEAR floor, a nonlinear head over
more layers/context could push below ~16%; (2) the old two-stage's 24% crash→ride
is ~5 pts WORSE than this floor, so some confusion IS recoverable downstream; (3)
the probe is handed the true onset, so it says nothing about RECALL -- end-to-end
ride/crash F1 (~0.59/0.64) is confusion AND the ~0.83 proposer recall compounding.
Going BELOW the ~16% confusion floor needs better features OR audio-level
ride/crash separation (split stems sidestep the disambiguation). Next:
recall-vs-confusion decomposition of the end-to-end ride/crash F1.

## Ride/crash recall-vs-confusion decomposition (2026-06-18)

End-to-end h128 cap-3000 (s1, the same head, not the linear probe), each matched
ride/crash onset bucketed as **hit** (right lane), **confused** (predicted the
*other* cymbal), or **missed** (no prediction in tolerance). 3080;
`cymbal_recall_confusion.json`.

| lane | n | hit | confused | missed | false_pos | recall | prec | F1 |
|------|---|-----|----------|--------|-----------|--------|------|-----|
| rd | 4415 | **84.2%** | **0.7%** | 15.1% | 954 | 0.842 | 0.796 | 0.818 |
| cr | 1365 | **55.1%** | **17.3%** | 27.6% | 451 | 0.551 | 0.625 | 0.586 |

**This overturns the "confusion-bound ceiling" reading.** Confusion is NOT the
dominant error mode and it is wildly **asymmetric**, not the symmetric ~16% the
linear probe implied:

- **Ride loses almost nothing to confusion (0.7%).** Its entire shortfall is
  *missed detection* (15.1%), a recall problem, not a discrimination problem.
- **Crash confusion is real (17.3%)** but its *bigger* loss is also missed
  detection (27.6%). Missed > confused for both lanes (ride 95% of error is
  misses; crash 61%).
- The asymmetry is a **majority-class default**: ride outnumbers crash 3.2:1 in
  val, so the head defaults to "ride" when unsure → true crashes leak to ride
  (17.3%) while rides almost never leak to crash (0.7%). The probe's "~16% floor"
  was symmetric because the probe was class-balanced per onset; the real head is not.

**Implication for levers.** The biggest headroom for *both* cymbals is **recall
(missed onsets)**, not ride/crash disambiguation, opposite to the probe's steer.
Detection-side levers (threshold/peak-pick recall, class re-balancing to stop the
ride default, focal loss, real-audio domain match) should beat any
confusion-targeted work. Confusion-side work (better features, split stems) only
meaningfully helps **crash**, and only after its 27.6% miss rate is addressed.
Precision is the other drag on crash (0.625; 451 false-pos vs 752 hits).

## Ride/crash miss typing + picker sweep (2026-06-18)

Follow-up to the decomposition: *why* are the cymbal onsets missed, and how much
is recoverable with the **picker alone** (no retrain) vs needs new training.
Reuses `h128_cymhat_s1.pt` -- one head forward over the warm cache, then pure
numpy. Each own-lane miss bucketed by inspecting this lane's activation around it:
`dead` (peak <max(0.1,0.5*thr) -> model sees nothing), `subthreshold` (a bump,
below the tuned height), `merge`/`decay`/`prominence` (clears height but a single
picker constraint drops it). `cymbal_miss_typing.json`. Tuned thr: rd 0.6, cr 0.8.

| lane | recall | misses | dead | subthr | merge | decay | prom | picker-recoverable |
|------|--------|--------|------|--------|-------|-------|------|--------------------|
| rd | 0.841 | 700 | **43.7%** | 26.7% | 2.6% | 15.6% | 3.6% | 48.4% |
| cr | 0.551 | 613 | **74.6%** | 20.9% | 2.3% | 0.8% | -- | 24.0% |

Picker grid-sweep (min-dist x decay-reset x prominence x thr-scale), best point
holding precision >= current:

| lane | config | R | P | F1 |
|------|--------|---|---|-----|
| rd current | md .070 dr .6 | 0.841 | 0.714 | 0.773 |
| rd **best** | md .040 dr .0 | **0.866** | 0.717 | 0.784 |
| cr current | md .070 dr .6 | 0.551 | 0.608 | 0.578 |
| cr best | md .070 dr .0 | 0.554 | 0.608 | 0.580 |

**Verdict: the picker is NOT the lever; under-activation is.** The dominant miss
type is `dead` (the head's activation is ~0 at the true onset) -- 43.7% for ride,
**74.6% for crash**. That's a TRAINING problem, not post-processing.

- **No clean picker win (corrected).** Per-axis: **min-distance is INERT** --
  0.070/0.050/0.040/0.030 give identical R/P/F1 (the 70 ms limit isn't binding;
  the merge bucket is only 18 onsets). The entire ride +2.5 pt
  (0.841->0.866 R, 0.773->0.784 F1) comes from **turning the decay-reset filter
  OFF** (the `if decay_reset_frac>0` gate is skipped), and dr=0.3 is actually
  *worse* than the current 0.6 (R 0.722 -- a stricter reset drops more). Decay-off
  is exactly the change most likely to **regress on real audio**: that filter
  stops one sustained cymbal/open-hat ring being read as a stream of false
  onsets, and clean in-domain val under-represents sustain. So it is NOT a free
  ship -- it needs ParaDB/real-audio validation first.
- **Crash recall is not picker-limited at all** (every axis ~0; best dr=0.0 is
  0.551->0.554). 95.5% of crash misses are dead+subthreshold -- the head
  under-fires on crash, and F1-tuning pushed its threshold to 0.8 (vs ride 0.6),
  trading recall for precision.
- **`subthreshold` (ride 26.7%, crash 20.9%) is recoverable only by lowering the
  threshold, a bad trade**: ride thr*0.85 buys +1.5 pt R for -4 pt P; crash
  thr*0.85 buys +4.4 pt R for -6.4 pt P. A softer form of the `dead` under-firing.
- **dead is worst on the rare class** (crash 74.6% vs ride 43.7%; crash 1365 vs
  ride 4415 onsets) -> class imbalance strongly implicated. The ONLY real lever
  is the retrain: **class re-balancing / inverse-freq or focal loss** (raise crash
  activation so peaks clear), then real-audio domain match. The picker/threshold
  knobs are spent.

## Crash label-quality audit -- labels are fine (2026-06-19)

Tested the obvious alternative to "head under-fires": maybe the crash LABELS are
wrong (no transient at the labelled time -> a target the head can't/shouldn't
fire on). Audited all 19,592 crash onsets (train+val, per-instrument crash stems)
two ways. `cymbal_label_audit.py`; `cymbal_label_audit_canonical.json`.

Note: the onset cleaner (`forced_align` snap + `clean.support_score` support gate)
EXISTS but is **only wired into `eval_paradb.py`, NOT the training pool** --
`build_specs` feeds raw onsets. So the model trains uncleaned. But:

| source | onsets | canonical %unsupp | hand-rolled %suspect |
|--------|--------|-------------------|----------------------|
| enst | 1,333 | 0.3% | 8.1% |
| egmd | 1,499 | 3.2% | 31.2% |
| star | 16,760 | 1.2% | 31.4% |
| ALL | 19,592 | **1.3%** | 29.8% |

- **Canonical gate** (`forced_align.onset_envelope` + `postfilter` pct-60 floor +
  `align_lane` +/-30 ms -- the ParaDB cleaner's own criterion): **1.3%
  unsupported.** Wiring the cleaner into training would discard ~1% -> negligible.
- **Hand-rolled gate** (peak vs clip-MAX, rel>=0.15 OR local-SNR>=3): ~30%. The
  gap is threshold strictness, not a contradiction: the pct-60 floor sits at the
  near-silence level on a sparse crash stem (medFloor ~0.13), so canonical flags
  only DEAD-SILENCE labels; the relative test additionally flags SOFT / ring-tail
  crashes -- which are *real* onsets, just quiet, so flagging them was wrong.

**Conclusion: the labels are essentially correct (1.3% truly unsupported).** The
dead crashes are real-but-WEAK onsets (soft / on a ring) the head under-fires on,
NOT mislabels. Rules out the data-cleaning detour and re-confirms the under-firing
diagnosis -> focal / oversample (up-weighting hard/weak positives) is the lever.
egmd's 3.2% lines up with its known MIDI<->audio drift but is small.

## Onset SNAP + FILTER: dataset-wide label alignment (2026-06-19/20)

Follow-up that OVERTURNS the "labels are fine" read above. The canonical support
gate was lenient (1.3%); a stricter relative gate flagged ~30% of crash labels,
and the user's eyes+ears on 100 snippets CONFIRMED those are real defects -- not
soft crashes but **mistimed** labels (the label sits off the audio onset) and
**false** labels (no onset there). A "shot in the dark" lane-coincidence check
nailed the kind: of the false crash labels, **88% coincide with ANOTHER
instrument** (open-hat / kick / snare / tom) = lane-mislabels, not pure ghosts.

Built an audio-referenced cleaner (`cymbal_snap_redraw.py` decision fns +
`align_dataset_onsets.py`): per stem, find real onsets (onset-strength local maxima
passing a transient test), then per label SNAP to the nearest real onset within a
per-lane window, else DISCARD (classified ghost vs wrong-lane). Ran over **all
2.62M onsets / 9,420 perstem stems** (star+enst+egmd) -> reversible
`_onsets_aligned.json` (originals untouched). Opt-in via `--aligned-onsets`.

| lane | labels | %snap | %disc | | lane | labels | %snap | %disc |
|------|--------|-------|-------|-|------|--------|-------|-------|
| hc | 252k | 93.4% | 6.6% | | s  | 543k | 83.2% | 16.8% |
| ss | 114k | 91.9% | 8.1% | | t  | 262k | 81.5% | 18.5% |
| k  | 623k | 91.3% | 8.7% | | cr | 28k  | 71.9% | 28.1% |
| hp | 343k | 90.0% | 10.0%| | rd | 369k | 71.2% | 28.8% |
| ho | 86k  | 87.9% | 12.1%| | ALL| 2.62M| 85.6% | 14.4% |

Discards overall: 122k ghost / 257k wrong-lane (**68% wrong-lane**). Cymbals (cr/rd
~28%, mostly wrong-lane) are by far the noisiest -> a large share of the cymbal
"recall gap" is plausibly mislabeled/mistimed targets, not model under-firing.

**Validated:** crash snap+filter eyeballed (000's bogus 3.0s label discarded as
wrong-lane, real 2.6s kept; 016/005 clean). Per-lane samples in
`/codebox-workspace/align_validate/`: snap looks good on ALL lanes; clean lanes
barely touched (kick k_00 = 15 snap/0 discard).
**CAVEAT:** the FILTER is only validated on crash (no soft ghost-notes). On
snare/ride/tom the higher discard MAY drop real soft hits -- review the per-lane
images before training on filtered non-crash lanes. The SNAP alone is the safe
universal win; a snap-only artifact (no discard) is the conservative default.

### Prominence update (2026-06-20)

The numbers above used a height/SNR floor that dropped soft-but-real rides. Per
user review (a barely-audible ride at the noise floor still had a real local
peak), switched the transient test to a **PROMINENCE gate** (peak rise above its
local baseline >= 5% of clip-max -- the transcriber-picker approach; calibration
showed background noise peaks ~3.3% prominence, so 5% sits just above noise). Also
coarsened the detection hop 64->256 (~4x less RAM, identical result). Validated on
rd_03: floor 4 snap/13 discard -> prominence **11 snap/7 discard** (soft rides
kept). Regenerated both JSONs (8 shards, ~13 min each). Final per-lane discard:

| lane | floor->prom | | lane | floor->prom |
|------|-------------|-|------|-------------|
| cr | 28.1% -> **9.1%** | | k  | 8.7% -> **22.5%** |
| rd | 28.8% -> **13.7%** | | s  | 16.8% -> **31.7%** |
| ho | 12.1% -> 7.9% | | t  | 18.5% -> 25.3% |
| hc | 6.6% -> 6.7% | | ss | 8.1% -> 18.2% |
| hp | 10.0% -> 16.2% | | ALL | 14.4% -> 20.3% |

**Prominence fixed the cymbals (cr/rd discard ~halved -- soft rides preserved) but
made kick/snare/tom WORSE.** Cause: prominence relative to CLIP-MAX is too strict
on high-dynamic-range lanes -- a snare's loud backbeats set the max, so real ghost
notes fall below 5% and get dropped (32% of snares). Ride has uniform dynamics so
5% works. ALL rose to 20.3% because the big-count k/s lanes dominate.

**Net:** the **cym+hat pool (hc/hp/ho/rd/cr) is well-handled (7-16%)** -> the
`_onsets_aligned.json` is READY for the h256 crash retrain (the payoff test of
"is the intrinsic crash ceiling the labels?"). For a FULL-KIT run, kick/snare/tom
need a lower / per-lane prominence (to keep ghost notes) or snap-only -- TODO.

### Aligned-onset retrain: h128 cap-1000 (2026-06-19) -- crash WINS

The payoff test, run small to land in ~1 h with a clean prior comparison: the
**raw cap-1000 h128** baseline (Phase-2 line, the data-scale knee) re-run
**identically but on `_onsets_aligned.json`** (snap+prominence-filter labels),
same cym+hat pool, batch 8, 30-ep cap, per-lane keep_best, seed 1.
`aligned_h128_cap1000_s1.json`.

| labels | hc | hp | ho | rd | cr | cym(rd,cr) |
|---|---|---|---|---|---|---|
| raw cap-1000 | 0.689 | 0.478 | 0.664 | 0.623 | 0.620 | 0.622 |
| **aligned** | 0.688 | 0.428 | 0.666 | 0.633 | **0.669** | **0.651** |
| Δ | −.001 | **−.050** | +.002 | +.010 | **+.049** | +.029 |

**Crash clears the bar.** cr **+0.049** is just past the ~0.04 single-seed
cap-1000 noise band -- the first lever that has moved crash at all (it was FLAT
across width *and* data volume; see the width A/B and Phase 2). Aligning the
labels lifts crash where 8× capacity and 3× data did nothing → **a real share of
the crash ceiling was mistimed/mislabeled targets, not model under-firing.** This
is the direct confirmation the under-firing diagnosis was missing: with the labels
on the audio onset, the head fires. Ride +0.010 (within noise); hats flat.

**But hp REGRESSED −0.050.** Pedal-hat is the casualty of the prominence FILTER --
the clip-max-relative gate (calibrated on crash) over-drops soft foot-chicks the
same way it over-drops snare/tom ghost notes (16.2% hp discard). The aligned set
is a net win *for crash* but is **not safe for hp/kick/snare/tom as-is** -- the
per-lane / lower prominence TODO above is now load-bearing, or use snap-only for
those lanes.

**Caveats.** (1) Single seed; cr +0.049 only just clears the noise band -- wants a
2nd seed. (2) **Aligned val is cleaner GT** (its onsets were also snapped/filtered)
so part of the lift is easier matching, not pure model gain -- the unbiased confirm
is **ParaDB** (independent GT, never touched by our snapper). (3) cym+hat-only pool,
not full-kit. **Next:** repeat at h256 (the best-cym width), add a seed, and run
ParaDB on the aligned checkpoint to kill the val-cleanliness confound.

### Suspect-label recheck: are the aligned labels actually clean? (2026-06-19)

The snap+filter discards (cr 9.1% / rd 13.7%) and the original hand-rolled suspect
audit (~30% crash, commit b8ffe8f) measure DIFFERENT things, so the small discard
didn't prove the surviving labels are good. `cymbal_suspect_recheck.py` re-runs the
**original strict gate VERBATIM** (rel>=0.15 of clip-max OR local-SNR>=3 at the
labelled time) over both the raw and the snapped+filtered onsets (one envelope per
stem, scored twice), then decomposes each surviving suspect into **wrong** (a
sibling cymbal onset within +/-50 ms = rd<->cr mislabel), **dead** (snr<1.5 = no
transient, separation drop / cross-kit mislabel), or **soft** (weak-but-real, keep).
Sanity: raw crash ALL = **29.8%**, reproducing the original audit exactly.

| src/lane | raw susp | aligned susp | wrong% | dead% | soft% |
|---|---|---|---|---|---|
| enst/cr (real) | 8.1% | **6.0%** | 0.0% | 1.1% | 4.9% |
| enst/rd | 13.1% | **9.0%** | 0.0% | 0.6% | 8.3% |
| egmd/cr (drift) | 31.2% | **22.9%** | 0.8% | 0.9% | 21.2% |
| egmd/rd | 37.3% | **20.5%** | 0.0% | 0.5% | 20.0% |
| star/cr (synth) | 31.4% | **27.6%** | 2.8% | 3.5% | 21.3% |
| star/rd | 34.8% | **28.2%** | 1.8% | 4.5% | 21.9% |
| **ALL/cr** | 29.8% | **25.7%** | **2.4%** | **3.2%** | 20.1% |
| **ALL/rd** | 36.2% | **21.8%** | **0.4%** | **1.3%** | 20.1% |

**The aligned labels are good targets.** The strict gate's ~26%/22% residual is
**overwhelmingly soft-but-real** (20.1% of all onsets both lanes) -- the quiet hits
we deliberately kept by loosening to the prominence gate. The genuinely-bad share
is small: **crash 5.6% (2.4 wrong + 3.2 dead), ride 1.7%**. Reads:
- **Snap fixed real timing error where it existed:** egmd (MIDI<->audio drift) ride
  raw-suspect 37.3% -> 20.5% (-16.8), and its residual is ~all soft (<=0.9% bad).
- **Real hand-labeled data (enst) is essentially clean** -- 0% rd<->cr mislabels,
  ~1% dead, the rest soft.
- **star carries the most bad** (2-3% wrong, 3.5-4.5% dead) = synthetic separation
  drops + rd/cr acoustic overlap; still soft-dominated.
- `dead` is an **upper bound** on bad: a crash label parked on a kick/snare
  (cross-kit mislabel, invisible to the within-stem rd<->cr check) shows up as a
  no-transient `dead` in the cymbal stem, so the 3.2% crash-dead already absorbs
  most cross-kit bleed. Cross-KIT bleed can't be measured directly here (per-stem
  files carry only cymbal lanes).

**Implication:** the +0.049 crash retrain was NOT built on junk targets; the
premise holds. Optional further tightening = drop the wrong+dead buckets (cr 5.6%)
before training, but soft-dominance says the current `_onsets_aligned.json` is fine.

### Loss A/B on aligned labels: focal / oversample (2026-06-19) -- focal is a RIDE lever, crash wash

The two pending levers (`--loss focal`, crash-oversample), finally run -- on the
ALIGNED labels (so focal isn't fighting label noise). `cymbal_loss_ab.py`, h128
cap-1000 30-ep seed-1, all arms in-harness with the SAME scoring (the
`cymbal_recall_confusion` decompose: hit/confuse/miss + R/P/F1 + the `dead`-rate
that miss-typing flagged). `cymbal_loss_ab_aligned_h128_cap1000.json`.

NB this decompose-F1 uses a different matcher than the head-capacity keep_best F1,
so these numbers are NOT comparable to the +0.049 retrain table above -- compare
ONLY across rows here. `bce` = the clean BCE-on-aligned control; `baseline` = the
old raw-trained cap-3000 ckpt (confounded by 3x data + raw labels, an anchor only).

| arm | rd R | rd P | rd F1 | rd dead | cr R | cr P | cr F1 | cr dead |
|---|---|---|---|---|---|---|---|---|
| baseline (raw cap-3000) | 0.823 | 0.716 | 0.766 | 40.1% | 0.585 | 0.605 | 0.595 | 72.6% |
| **bce** (aligned ctrl) | 0.771 | 0.763 | 0.767 | 31.3% | 0.562 | 0.584 | 0.573 | 84.4% |
| **focal** | **0.888** | 0.785 | **0.833** | 48.5% | 0.460 | **0.734** | 0.565 | 69.2% |
| crash_oversample | 0.803 | **0.816** | 0.810 | 26.3% | 0.533 | 0.605 | 0.567 | 77.9% |

**Focal is a strong RIDE win and a crash wash.**
- **Ride:** F1 0.767 -> **0.833** (+0.066), recall 0.771 -> **0.888** (+0.117) vs
  the bce control. A real, large ride gain -- the standout result.
- **Crash:** focal DID do what miss-typing predicted -- woke up activation
  (dead-rate 84.4% -> 69.2%) and crash precision jumped (0.584 -> **0.734**). But
  crash **recall collapsed** (0.562 -> 0.460) and confusion rose (0.188 -> 0.236),
  so crash **F1 is flat** (0.573 -> 0.565). Focal reshapes crash into a
  high-precision / low-recall lane; it does NOT lift crash.
- **crash-oversample: no win** -- marginal ride precision (+0.05 P), crash flat
  (F1 0.567, recall down 0.029). Duping 71 crash stems 2x didn't help.

**Verdict: neither loss lever breaks the crash ceiling** (crash F1 ~0.57 across all
three aligned arms). This **overturns the miss-typing steer** ("focal/oversample is
the crash lever"): focal raises crash *activation* exactly as predicted, but the
extra firing is low-quality (precision/recall trade, more rd<->cr confusion), not
recall. **The only lever that has moved crash remains label alignment** (+0.049).
The unexpected payoff is RIDE: focal is the first clear ride lever -- worth a 2nd
seed + ParaDB confirm, and folding `--loss focal` into the main cym+hat recipe if
it holds (watch the threshold recalibration: focal tuned to cr 0.4 / rd 0.3 vs bce
0.5 / 0.5). Single seed; decompose-F1; cym+hat pool.

#### Picker retest: was focal's crash recall picker-suppressed? (2026-06-19)

The A/B used the production cymbal picker (min-dist 70 ms, prominence 0.20, decay-
reset, + tuned height). To separate "model can't recall crash" from "picker too
strict for focal's calibration", `cymbal_picker_retest.py` re-scores the saved
keep_best checkpoints (NO retrain) with the height threshold x0.5 and cym
prominence 0.20 -> 0.10. The `orig` rows reproduce the A/B exactly (sanity OK).

| arm | cr R orig->mod | cr P orig->mod | cr F1 orig->mod |
|---|---|---|---|
| bce | 0.562 -> 0.600 | 0.584 -> 0.482 | 0.573 -> 0.535 |
| focal | 0.460 -> **0.614** | 0.734 -> 0.492 | 0.565 -> 0.546 |
| crash_oversample | 0.533 -> 0.597 | 0.605 -> 0.488 | 0.567 -> 0.537 |
| baseline | 0.585 -> 0.661 | 0.605 -> 0.393 | 0.595 -> 0.493 |

**Two reads.** (1) The permissive picker is a recall-for-precision trade in every
arm and lane -- net F1 DROPS throughout (halving height AND dropping prominence
overshoots; FPs rise faster than TPs). So there's no free picker win here. (2) BUT
**focal's crash recall was genuinely picker-suppressed**: loosened, focal reaches
crash recall **0.614 -- the highest of ANY config in the study** (vs 0.460 under the
production picker), and still holds the best precision at that recall (cr P 0.492 vs
baseline 0.393). So focal DID raise crash activation (confirming the dead-rate
84%->69% drop); the strict prominence-0.20 + high-threshold picker was discarding
those firings. focal's apparent "crash recall collapse" is partly a picker-
calibration mismatch, not pure model failure.

**Caveat / next:** this changed two knobs at once and overshot recall, so it does
NOT establish the F1-optimal point at prominence 0.1. The proper test is a height
**re-tune** (sweep to the F1 optimum per arm) at prominence 0.1, not a blind halve.

#### Proper height re-tune at prominence 0.1 vs 0.2 (2026-06-19)

`cymbal_thresh_sweep.py`: for each saved checkpoint, cache the cym activations once
(one model pass), then sweep the height threshold over a grid (0.05..0.90) at BOTH
prominence 0.20 and 0.10 -- same grid, only prominence differs -- and report each
lane's F1-OPTIMAL point. **Bug caught + fixed first:** the initial pass counted a
lane's firing on ALL clips, penalizing precision for cross-stem leakage (rd firing
on a crash/kick stem). The project convention (and `cymbal_recall_confusion.
decompose`) scores a lane ONLY on clips that carry its ground truth; fixed to skip
no-ref clips (FP was ~2x inflated, which had skewed thresholds high). Post-fix the
absolute F1s line up with the A/B decompose.

| arm | rd best (thr/F1) | cr best (thr/F1) | prom0.1-vs-0.2 ΔF1 |
|---|---|---|---|
| baseline (raw cap-3000) | 0.80 / 0.780 | 0.90 / **0.616** | rd 0 / cr 0 |
| bce | 0.80 / 0.780 | 0.75 / 0.585 | rd -.003 / cr -.000 |
| focal | 0.40 / **0.845** | 0.30 / 0.586 | rd -.000 / cr +.000 |
| crash_oversample | 0.85 / 0.814 | 0.80 / 0.570 | rd +.000 / cr -.003 |

**(1) Prominence is NOT a lever.** At the F1-optimal threshold, 0.10 == 0.20 within
+/-0.003 (noise) for every arm/lane. The picker-retest's F1 drop was entirely the
blind height-halve, not the prominence change.

**(2) Focal does NOT win crash even at its F1-optimum** (cr 0.586 ~= bce 0.585). Its
picker-suppressed recall (0.614 reachable) costs too much precision; the F1-optimum
sits back at recall 0.538. So "recall is recoverable" is true but does not convert
to crash F1. **Alignment remains the only crash F1 mover.** Among matched cap-1000
aligned arms crash is flat ~0.57-0.59; baseline's 0.616 is the cap-3000 data-scale
edge, not a loss effect.

**(3) Focal's RIDE win is robust to picker tuning:** rd 0.845 vs bce 0.780 =
**+0.065** at matched-optimal thresholds (focal's optimum is a notably lower height,
0.40 vs 0.80 -- focal outputs sit lower, as expected). This is the keeper result
from the loss A/B: fold `--loss focal` into the cym+hat recipe (pending 2nd seed +
ParaDB), and it needs its own threshold re-tune (don't inherit bce's heights).

#### Per-lane loss pick -> per-lane loss is now supported (2026-06-19)

Since the heads are INDEPENDENT (each lane its own OnsetHead over frozen features),
the best model uses, per lane, whichever loss trained the better head -- no
cross-lane interaction. `cymbal_lane_loss_pick.py` re-scores the saved focal & bce
checkpoints (no retrain), per lane at its F1-optimal height + production prominence:

| lane | bce F1 (thr) | focal F1 (thr) | Δ(f-b) | winner |
|---|---|---|---|---|
| hc | 0.820 (0.90) | **0.850** (0.45) | +0.030 | **focal** |
| hp | 0.633 (0.90) | 0.629 (0.35) | -0.004 | bce |
| ho | 0.724 (0.90) | 0.713 (0.40) | -0.010 | bce |
| rd | 0.780 (0.80) | **0.845** (0.40) | +0.066 | **focal** |
| cr | 0.585 (0.75) | 0.586 (0.30) | +0.001 | bce |

**Focal wins TWO lanes: closed-hat (+0.030) and ride (+0.066)** -- the hc win was
hidden in the untuned keep_best (focal needs a far lower threshold, 0.45 vs 0.90).
hp/ho/cr keep bce. Data-driven map: **`--focal-lanes hc,rd`**. (Every focal lane's
optimum is ~half bce's height -> per-lane threshold tuning is mandatory.)

**Implemented:** `train_loop(focal_lanes=...)` (train.py) computes focal on those
lanes' logits + BCE on the rest and sums them; independent heads make this exactly
a per-head loss choice. Exposed as the `mixed` arm in `cymbal_loss_ab.py`
(`--focal-lanes`, default `hc,rd`).

**Mixed run validated (focal hc,rd + bce rest; h128 cap-1000 aligned).** Per-lane
F1-optimal of the single mixed checkpoint vs the pure arms:

| lane | bce | focal-only | **mixed** | mixed vs bce |
|---|---|---|---|---|
| rd | 0.780 | 0.845 | **0.853** | **+0.073** |
| hc | 0.820 | 0.850 | 0.833 | +0.013 |
| cr | 0.585 | 0.586 | 0.594 | +0.009 |
| hp | 0.633 | 0.629 | 0.622 | -0.011 |
| ho | 0.724 | 0.713 | 0.714 | -0.010 |

**Net win over pure BCE, driven by ride (+0.073) + modest closed-hat (+0.013);**
the BCE lanes are within single-seed run-to-run noise (+/-0.01). Notably **rd
EXCEEDED focal-only** (0.853 vs 0.845) -- concentrating focal on hc+rd raised those
heads' effective LR and helped ride. **hc UNDERSHOT focal-only** (0.833 vs 0.850) --
the predicted normalization caveat biting closed-hat. So:
- **Deployable option A (mixed checkpoint):** one coherent run, rd best-in-study.
- **Deployable option B (graft):** drop `loss_ab_focal.pt`'s `heads.hc`+`heads.rd`
  into `loss_ab_bce.pt` (independent modules, zero training) -> the EXACT per-lane
  bests (hc 0.850, rd 0.845, hp/ho/cr at bce). Marginally better overall (recovers
  hc's 0.017); mixed wins rd by 0.008. Both single-seed; differences mostly within
  the cap-1000 noise band. **Next:** 2nd seed + ParaDB to pick between them and
  confirm the ride/hc wins hold out-of-domain.

## SOTA-comparable eval: ENST + MDB, 5-class mir_eval (2026-06-20)

First apples-to-apples comparison to published ADT. `sota_eval.py` folds our 9
lanes -> KD/SD/HH/TT/CY, scores onset F at +/-50 ms (pooled micro, the dataset
convention) on the standard benchmarks; `eval_gt_cleanliness.py` first confirmed
the GT is clean (so we're not scored down on phantom labels). Prediction = our
DEPLOYMENT path (per-instrument stems via BS-Roformer->MDX23C, keep-own-lanes,
deterministic picker). Checkpoint = the cym+hat **mixed** (loss_ab_mixed, h128
cap-1000 aligned, focal hc+rd) -> only HH+CY scorable (no kick/snare/tom heads;
full-kit checkpoint pending). RBMA skipped (no free audio).

| set | cond. | HH F | CY F | notes |
|---|---|---|---|---|
| ENST (12 takes) | drummer_3 phrases, enst-sep | **0.848** | 0.569 | HH P 0.77 / CY P 0.41 |
| **MDB (23, pristine)** | full_mix -> our sep | **0.722** | **0.740** | HH P 0.59 / CY P 0.65 |

(pooled micro; HH = hc+hp+ho, CY = rd+cr.)

**GT cleanliness (strict dead-label probe, peak <1% track peak within +/-50 ms vs
the dataset's own mix):** ENST 0.18% dead (30/16694, ~all closed-hat in minus-one
takes); **MDB 0.00% (0/7962)**. Both eval sets are clean -> the scores are real,
and since dead labels only cost recall (already ~0.9+), the cymbal weakness is
genuine model over-firing (precision), not bad GT.

**Reads.**
- **Hi-hat is SOTA-band on ENST (0.848)**; drops to 0.722 on MDB, entirely
  PRECISION (0.77 -> 0.59, recall stays 0.94). MDB is full-song -> full separation,
  which leaves more hat-like artifacts; ENST is cleaner drum-focused audio.
- **Cymbals: 0.74 on pristine MDB** (better than ENST's 0.57, precision 0.41->0.65)
  -- competitive-to-strong for the hardest ADT class (cymbal sub-scores are ~0.3-0.6
  even at SOTA). recall 0.87, precision 0.65.
- MDB AVG(HH,CY) 0.731 pooled. Published MDB 5-class is ~0.85-0.89 but kick/snare-
  dominated; the HH/cymbal sub-scores are where we're competing, and we're in range.

**Framing (this is a SYSTEM comparison, not a model-in-isolation one).** We run the
deployed pipeline: separate (BS-Roformer->MDX23C) -> per-instrument stems -> model
-> keep each stem's owned lanes. The per-stem isolation in the eval is NOT a
measurement artifact -- it is exactly what production does (a prediction of lane X
on stem Y!=X is dropped in deployment too). So the eval faithfully measures the
deployable system, and the strong hat/cymbal PRECISION is the intended payoff of
separating first: the separator removes the cross-instrument confounders (the
hi-hat a mix-input model misfires as a ride) BEFORE the lane decision. SOTA
(ADTOF/Vogl-CRNN) transcribes the raw mix; comparing the two is a fair
system-vs-system comparison (audio in -> onsets out), which is the comparison that
matters for a deployable tool. Running our head on the raw mix (no isolation) is a
config we'd never ship, so it's not the relevant baseline.

**Per-class vs SOTA (ADTOF Fig.5, Vogl-CRNN, mir_eval +/-50 ms, micro; cymbals = CY+RD):**

| | HH SOTA | HH ours | CY SOTA | CY ours |
|---|---|---|---|---|
| ENST | 0.77-0.83 | **0.848** | 0.28-0.72 | 0.569 |
| MDB  | 0.70-0.78 | 0.722 | 0.38-0.72 | **0.740** |

Hi-hat matches/beats SOTA (ENST 0.848 > best 0.83); cymbals competitive-to-better
(MDB 0.740 >= SOTA best 0.72). SOTA CY+RD spans 0.28-0.72 by training config; cymbals
are the hardest ADT class everywhere (sparsity), which is where we hold up well.

**Dependencies / non-issues.** System performance is contingent on separation
quality (the separator is a load-bearing component) -- a real property to
characterize, though on MDB no track actually failed. The **0.00-scoring tracks are
EMPTY-GT, not failures**: 80sRock/Beatles/Shadows have 0 hi-hat onsets, Country1/
Disco/Hendrix/Rock/Rockabilly have 0 cymbal onsets (verified from the annotations);
they're correctly excluded from the per-track mean. ENST drummer_3 was in the
val/threshold pool (mildly optimistic); MDB is fully pristine (never
train/val/threshold) = the honest read. Full 5-class row (KD/SD/TT) needs a full-kit
checkpoint (training pending) -- train on snap-only onsets, not the cym-tuned filter.

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
