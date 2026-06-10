# Drum-onset model, change log

Companion to [RESULTS.md](RESULTS.md) (which tracks *numbers*; this tracks
*changes*). Purpose: when accuracy moves and we need to backtrack, this is the
list of what to flip, one entry per change, newest first, each with **scope**
(what it affects), **default**, **how to disable/revert**, and **interactions**.

Scope legend:
- **train**, changes what a NEW training run learns (old checkpoints unaffected).
- **infer/eval**, changes how ANY checkpoint (old or new) is decoded/scored.
- **data**, changes the training data itself.
- **diag/infra**, no model effect.

---

## TO TEST, not yet run / no results reported back (pick up here)

Everything below this line is BUILT + unit-tested but has **no training run or
ParaDB numbers yet**. When results land, move the item into a dated section
below and record numbers in [RESULTS.md](RESULTS.md).

**Division of labour** (don't mix them up):
- **TRAIN on the 3080 box.** `cd ~/code/drumjot && git pull && (cd training && uv sync)`
  first (the `drumjot-dsp` path dep + cu130 wheels). `DRUMJOT_STAR` points at
  whatever dir you rsync'd to (examples use `~/datasets/...`).
- **EVAL runs in the codebox/sandbox, NOT the 3080**, `eval_paradb.py` needs
  the separator models (`MODELS_DIR`) + the ParaDB map zips at
  `/codebox-workspace/paradb`, which only live there. So when a run finishes,
  **push the checkpoint to `/codebox-workspace/checkpoints/<name>/` and ask the
  agent to eval it on ParaDB**; the `eval_paradb.py --maps-dir
  /codebox-workspace/paradb --checkpoint <dir>` invocation is the agent's job,
  not a 3080 command.

All v3 defaults (high-band, aux-activity, sibling-weighting, threshold-floor,
stitching, 10 lanes) are **ON automatically**; no flags needed to get them.

### 1. v3 architecture on separation-aware data, THE headline run
Dataset `star_balanced_sep` (our separator's output; build finishing now).
First run carrying the whole 2026-06-10 stack. Full-drum mode:
```bash
DRUMJOT_STAR=~/datasets/star_balanced_sep uv run python -m drumjot_training.train \
  --dataset star --train-clips 1000 --val-clips 48 --epochs 80 \
  --batch-size 16 --num-workers 8 --out ~/checkpoints/star_sep_v3
```
First epoch re-encodes the cache (~1 h; new `hb16` key + `.rings.json`), then
~3 min/epoch. Watch **val_macro_f1** for the peak (not train_loss). Then push
`~/checkpoints/star_sep_v3` to `/codebox-workspace/checkpoints/` and have the
agent eval on ParaDB (compare vs `star_balanced_stem_v2`). **Caveat:** many
variables moved at once (separated audio + all v3 changes); if it REGRESSES,
disambiguate by rerunning this exact train command on clean
`~/datasets/star_balanced1k_stem` (isolates "v3 code" from "separated audio").

### 2. Per-stem variant (option, not replacement)
Same data, one example per (song, stem); ~5× examples/epoch so fewer epochs:
```bash
DRUMJOT_STAR=~/datasets/star_balanced_sep uv run python -m drumjot_training.train \
  --dataset star_perstem --train-clips 1000 --val-clips 48 --epochs 40 \
  --batch-size 16 --num-workers 8 --out ~/checkpoints/star_sep_perstem_v3
```
Then push + eval as in #1. Hypothesis: surfaces cymbals (isolation) better than
full-drum; clean lanes may be slightly worse. Sibling weighting sees full
onsets here, so it should suppress on-stem bleed.

### 3. Layer-sweep probe (diagnostic, ~1–2 h, run anytime)
Answers "is layer 10 right for hats/cymbals, or is the >12 kHz info just absent"
(→ justifies per-lane layers / learned mix, or confirms high-band is the fix):
```bash
uv run python scripts/layer_sweep.py ~/datasets/star_balanced1k_stem \
  --layers 1,4,7,10,13,16,19,22 --clips 200 --val-clips 24 --epochs 15
```
Reads the per-lane F1 × layer matrix it prints; paste it back here.

### 4. Focal-loss A/B (built 2026-06-09, never A/B'd)
Cheap arm once a baseline exists, same data, `--loss focal`:
```bash
DRUMJOT_STAR=~/datasets/star_balanced_sep uv run python -m drumjot_training.train \
  --dataset star --train-clips 1000 --val-clips 48 --epochs 80 \
  --batch-size 16 --num-workers 8 --loss focal --out ~/checkpoints/star_sep_v3_focal
```

### 5. ENST-Drums (BLOCKED on data)
Loader + `--dataset enst` built but data is request-gated (Télécom Paris) and
**not yet downloaded**. When it lands: (a) VERIFY `enst._ENST_TO_LANE` against
real `annotation/*.txt` labels, (b) optionally run `separate_enst_dataset.py`
for the `sep_drum`/perstem trees, (c) `--dataset enst --enst-mix wet_mix`.

### Open decisions (proposed, not approved, don't action without a nod)
- Sibling λ (8/3) and `aux_act_weight` (0.5) are **untuned guesses**, sweep
  only if v3 underperforms and these are suspected.
- Logit-level joint refiner (cross-lane veto), deferred; only if sibling
  weighting doesn't kill the leakage.
- Commit/push of `training/` + `dsp/`; still uncommitted (user syncs manually).

---

## Kill-switch quick reference (things ON by default)

| change | disable with |
|---|---|
| sibling-aware loss weighting | `--sib-neg-weight 1 --sib-pos-weight 1` |
| aux ring-activity objective | `Config.aux_act_weight = 0` (no CLI flag) |
| rare-lane threshold floor | `Config.rare_thr_floor = 0` |
| high-band spectral block | code revert (embeddings.embed_clip) + retrain, not flag-gated |
| window stitching | code revert (inference.stitched_probs); not flag-gated |
| shared per-lane peak picker | use `metrics.pick_onsets` (bare), eval prints both |
| warmup→cosine LR | `--lr-schedule none` |
| focal loss | already off (`--loss bce` is default) |

---

## 2026-06-10, "v3 architecture" batch (deliberately stacked; first trained model will carry ALL of these)

Decision: stacked on purpose (limited run budget). If the first v3 run loses
accuracy vs `star_balanced_stem_v2`, disambiguate by flipping in this order:
high-band is the most invasive (retrain), sibling/aux/floor are cheap flags.

### `mp` lane removed (11 → 10 lanes)
- **Scope:** train (new checkpoints have 10 heads). Old 11-lane checkpoints
  still load + eval; their `mp` output is dropped at `to_pitch_onsets`.
- **Why:** no per-instrument stem feeds it, ~noise on val, and the top
  cross-instrument leak destination on 4 of 5 stems (a garbage-attractor that
  rewarded firing on anything percussive). See RESULTS.md leakage tables.
- **Revert:** restore `mp` in `lanes.LANES`/`LANE_NAMES`, GM notes 39/54/56,
  STAR `CB/CL/CLP/TB`, rlrr `Tambourine*/Cowbell`, ENST `cb`,
  `inference.LANE_TO_PITCH`, `lanes.CONFUSABLE`.
- **Files:** `lanes.py`, `star.py`, `rlrr.py`, `enst.py`, `inference.py`, `metrics.py`.

### High-band spectral pathway (input 1024 → 1040 dims)
- **Scope:** train + infer (a checkpoint's `meta.in_dim` records whether it was
  trained with it; inference auto-detects, so old 1024-dim checkpoints work).
- **What:** 16 log-mel bands, 6–20 kHz, computed from 44.1 kHz audio at 75 fps
  (hop 588) and appended to the MERT features. Rationale: MERT's 24 kHz input
  = 12 kHz Nyquist, the hat/cymbal sizzle band is discarded before the encoder
  sees anything; no training data can recover missing input bandwidth.
- **Cache:** new cache-key variant `hb16` → all feature caches re-encode once.
- **Disable:** not flag-gated; remove the concat in `embeddings.embed_clip`
  (and the appends in `inference.stitched_probs`/`lane_probs`) and retrain.
  A model trained WITH it cannot run without it (in_dim mismatch) and vice versa.
- **Files:** `embeddings.py` (HB_* consts, `highband_from_wave/features`,
  `cache_key` variant), `inference.py`, `checkpoint.py` (FEAT_DIM default).

### Auxiliary ring-activity objective (Onsets-and-Frames-style)
- **Scope:** train. Each head gained a second linear (`act`); sustained lanes
  (`ho rd cr mc`, `targets.SUSTAINED_LANES`) get a frame-activity BCE at weight
  `Config.aux_act_weight = 0.5`. Targets = per-onset ring spans from the RMS
  envelope (decay to 15% of post-onset peak, capped at next onset / 3 s),
  cached as `<key>.rings.json` beside the feature cache.
- **Why:** open-hat/cymbal identity lives in the tail; a pure onset target
  never shows it to the head.
- **Disable:** `Config.aux_act_weight = 0` (act head still exists, unused).
  Old checkpoints (no `act` weights) load via lenient `checkpoint.load`.
- **Files:** `model.py` (`act`, `forward_all`), `targets.py` (`ring_spans`,
  `spans_to_activity`), `train.py` (`_rings_for_clip`, collate `A`, loss term),
  `checkpoint.py` (meta `aux_activity`, lenient load).

### Sibling-aware loss weighting (hard negatives + co-occurrence reward)
- **Scope:** train. `W = 1 + sib_act·((λp−1)·Y + (λn−1)·(1−Y))`, λn=8, λp=3
  (untuned starting guesses). `lanes.CONFUSABLE` seeded from measured ParaDB
  leakage (e.g. hats are siblings of ride, the #1 leak), not taxonomy.
  In `star_perstem` mode the weighting sees the FULL onsets while targets stay
  stem-restricted, so bleed on an isolated stem is an explicit hard negative.
- **Disable:** `--sib-neg-weight 1 --sib-pos-weight 1`.
- **Files:** `lanes.py` (`CONFUSABLE`, `sibling_matrix`), `config.py`,
  `train.py` (`sibling_weight`, collate `Yw`, loss wiring), perstem spec.

### Rare-lane threshold floor
- **Scope:** infer/eval calibration (tuning step of NEW runs; re-tunes too).
- **What:** lanes with < `Config.rare_lane_min_onsets = 50` val onsets only
  consider thresholds ≥ `Config.rare_thr_floor = 0.3`. Cause: v2 tuned ride to
  0.10 on a 4-clip val and collapsed to F1 0.028 on ParaDB.
- **Disable:** `Config.rare_thr_floor = 0`.
- **Files:** `train.py::tune_thresholds`, `config.py`.

### Window-boundary stitching at inference
- **Scope:** infer/eval, affects ALL checkpoints (old ones too).
- **What:** overlapping 30 s windows (2 s overlap), probability curves
  center-crop-stitched into one global timeline, ONE peak-pick (was:
  independent per-window picks → onsets at window edges can't be local maxima;
  decay-reset reset mid-ring at each seam; ~7 seams on a 200 s song).
- **Revert:** `inference.stitched_probs` (the old per-window generator is in
  git history as `_windowed_probs`).
- **Files:** `inference.py` (`stitched_probs`, `transcribe`, `transcribe_dual`).

### Layer-sweep probe (diagnostic only)
- `scripts/layer_sweep.py`: per-lane F1 vs MERT layer on a small subset,
  one encode pass for all layers (cache variant `""` = MERT-only, no high-band).
  Distinguishes "wrong layer" from "information absent" (the >12 kHz case).

### ENST-Drums loader (option, not default)
- `--dataset enst` + `enst.py`: real acoustic drums, drummer-held-out split,
  `--enst-mix wet_mix|dry_mix|accompaniment` (+ `sep_drum` for the separated
  tree). Label map assembled from the paper. VERIFY against real annotation
  files when the data lands. Data is request-gated (Télécom Paris).

## 2026-06-09/10, separation-aware data (data; option per dataset dir)

- `scripts/separate_star_dataset.py`: STAR mixes → our BS-Roformer drum stem
  (`audio/mix/`, so `--dataset star` trains on it) + MDX23C 5-class stems
  (`audio/perstem/{k,s,h,c,t}/`), labels unchanged (STAR annotations still
  exact). Built `star_balanced_sep` (1000 train + 48 val). Rationale:
  distribution-match training audio to the separator artifacts inference sees.
- `--dataset star_perstem` (option, NOT a replacement for `star`): one example
  per (song, stem), targets restricted to the stem's lanes so the model learns
  silence on bleed; `mp`-less; full onsets ride along for sibling weighting.
- **Files:** `scripts/separate_star_dataset.py`, `star.py`
  (`PERSTEM_TO_LANES`, `perstem_index`, `restricted_onsets`), `train.py`.

## 2026-06-09, eval/infra batch

- **Shared per-lane peak picker** (`drumjot_dsp.peakpick`, new `dsp/` package
  both transcriber + trainer depend on): height + per-lane min-distance
  (clean 20 ms / hat 50 ms / cymbal 70 ms) + prominence + decay-reset.
  Replaced the flat `find_peaks(0.5, 30 ms)`. Scope: infer/eval, all
  checkpoints. Measured ~neutral on clean STAR val (expected; payoff is on
  messy real audio). Params: `metrics.LANE_PEAK_PARAMS`.
- **Envelope support filter removed from reports**; measured dF≈0 across every
  checkpoint/map/picker (false onsets sit on real transients). Reports now show
  **bare pick vs shared picker** (`F_bare/F_pick`) instead; `F_pick` is the
  deployed headline. `postfilter` itself still exists for the transcriber.
- **AdamW (wd 0.01) + warmup→cosine LR** replaced plain Adam + constant LR.
  Disable: `--lr-schedule none`; `--weight-decay 0`. First used by
  `star_balanced_stem_v2`.
- **Focal loss option** (`--loss focal`, CenterNet penalty-reduced; default
  remains pos-weighted BCE). Not yet A/B'd.
- **cu130 torch** (Dockerfile + training/pyproject): CUDA-13 driver match for
  the 3080/WSL box ("named symbol not found" fix). Trainer Dockerfile base
  bumped 12.4.1 → 12.8.0 (cosmetic next to the wheel's bundled runtime).
- **Report tables sorted by F1** (train-side `_report`/`_report_compare`).

## 2026-06-08, ParaDB eval methodology (eval-only; affects all reported numbers)

- Per-instrument isolation eval: model runs on each MDX23C stem, only that
  stem's lanes kept, the rest counted as **leakage** (the hallucination metric).
- `--full-drum` mode (score the whole BS-Roformer drum stem, no isolation).
- Global GT↔audio offset via **median nearest-peak** (replaced
  argmax-of-support, which overshot on dense charts); offsets ≤ ~25 ms not
  applied. Adaptive hat/cymbal folding (split only if the chart distinguishes);
  bimodal-velocity hi-hat open/closed (exactly-two-velocities rule, quieter =
  open); mix reconstruction via raw-sample correlation containment.
- These shift reported numbers vs earlier evals, cross-era comparisons in
  RESULTS.md are only valid within the same methodology block.
