# Thermonuclear review, model trainer + transcriber backend

- **Date:** 2026-07-02
- **Commit reviewed:** `3d77f96` (`main`)
- **Scope:** `training/`, `transcriber/`, `dsp/`; with an emphasis on anything that
  silently produces incorrect results (bad training data, mislabels, train/serve
  skew, misleading metrics, numeric drift in the ONNX rewrite), plus performance
  and structural health.
- **Method:** nine parallel deep-read agents (six over the training + original
  transcriber code, five over the new torch-free numpy/ONNX transcriber rewrite),
  every load-bearing claim re-traced by hand against source. The separation and
  beat/lyrics ONNX ports were additionally **parity-checked empirically against
  real PyTorch in the sandbox** (STFT/iSTFT, band-split, overlap-add, attention,
  decode).

Line numbers are as of `3d77f96`.

---

## Verdict

The transcriber's most dangerous new surface, the ~2100-line numpy/ONNX
reimplementation of BS-Roformer / MDX23C separation and the Beat This! beat
tracker, is **numerically faithful to torch** (mask correlation ≈ 1.0,
STFT/iSTFT/overlap-add match to ~1e-7, beat decode line-for-line vs the reference).
No stem-corrupting or bar-shifting defect was found there.

The real correctness exposure is on the **training side** (a `--input-norm`
propagation gap, a latent lane-mislabel, a silent-data-loss path, a train/serve
GRU skew), one **misleading eval metric**, and a set of **fp16 / robustness /
process** issues around the ONNX serving path; most notably that there is **no
committed torch↔ONNX parity regression test** guarding the shipped models.

Nothing here blocks the current best-model path, but several items are poised to
silently ruin a future experiment (especially anything using `--input-norm`).

---

## Correctness (ranked)

### 1. HIGH (latent), `--input-norm` is silently ignored in four downstream paths

`--input-norm` (commit `52a0493`) peak-normalises the waveform before features.
It is correctly threaded through `train.py`, the torch `inference.py`, and
`embeddings.py` (verified: `cache_key` folds the `_pn` variant token; train↔
inference are byte-consistent; zero-division guarded). It is **dropped in every
auxiliary path**, each failing silently:

- **ONNX serving**, `transcriber/app/pipeline/onset_onnx/np_onsets.py:109,114,125`
  load audio + high-band with no `input_norm`; `load_onnx_onset:143` never checks
  the variant. An `--input-norm` checkpoint served via ONNX runs on un-normalised
  audio → every lane sits off its tuned operating point. The torch reference
  (`inference.py:96-97`) reads `input_norm = meta.get("input_norm", False)` and
  threads it.
- **Param-predictor probs cache**, `training/drumjot_training/parampred/probs_cache.py:25-32`
  `probs_key` omits `input_norm`, and `in_dim` is identical with/without it, so a
  stale non-normalised probability curve is reused under the same key → mislabeled
  oracle targets / `act_*` features.
- **Perstem corpus builder**, `training/scripts/build_param_dataset_perstem.py:129`
  calls `feat_variant(high_band)` without `input_norm` → looks up `hb16` when the
  checkpoint is `hb16_pn` → cache miss → **all identity rows silently skipped**
  (`if feat is not None`), gutting the corpus; and `_encode_window:217` never
  applies `robust_peak_normalize`.
- **Cache pre-warm**, `training/scripts/encode_feature_cache.py:106` same variant
  omission and an `embed_clip` call with no `input_norm` → a full silent re-encode
  instead of cache hits.

Latent today (the current best model has `input_norm` off), but the moment you A/B
input-norm the failure mode is a misleading *"input-norm doesn't help"* conclusion.

**Fix:** thread `input_norm` from `meta`/`cfg` through all four sites; add
`input_norm` to `probs_key`; refuse/warn on a variant mismatch in `load_onnx_onset`.

### 2. HIGH (latent), per-stem lane restriction discarded when the aligned store is active

`training/drumjot_training/train.py:1763-1770` (`_spec`): computes
`keep = set(p2l.get(c.pitch, ()))` (line 1765) but the `al is not None` branch uses
`restricted = _fold_aligned_lanes(al)` (line 1768) and **ignores `keep`**. Safe only
because today's aligned store happens to hold each stem's own lanes. Any store built
with a different/rebuilt `PERSTEM_TO_LANES`, or holding full-kit onsets per stem
path, would label the kick stem with snare/hat/cymbal onsets, the exact inverse of
the isolated-stem bleed-suppression objective, silently.

**Fix:** intersect the folded aligned lanes with `keep`:
`restricted = {ln: (folded[ln] if ln in keep else []) for ln in LANES}`.

### 3. MEDIUM, `materialize` swallows every encode error → trains on a silently decimated set

`training/drumjot_training/train.py` (`materialize`, def at `:790`; the per-window
`except Exception: continue` at ~`:866`) logs and continues on any error (decode
failure, missing `.npy`, transient NFS outage on `/codebox-workspace`, a bad
`--pool-cache` path). A systematic failure yields a near-empty `CachedClips` with
only per-line log noise, training proceeds on a fraction of the data.

**Fix:** count skips and hard-fail past a small drop fraction (e.g. >5% of a source,
or 0 survivors).

### 4. MEDIUM, `f1_weighted` (headline `score_corrected`) is blind to hallucinated extra lanes

`transcriber/app/scoring/alignment.py:180-184`: `f1_weighted` weights each lane by
`n_audio`. A lane the chart populates but the reference has 0 onsets in gets weight
0 → whole-lane over-notation is invisible in the headline number (`f1_macro` catches
it, but it isn't the headline). For ranking/filtering training pairs by notation
faithfulness this masks real regressions.

**Fix:** surface `f1_macro` as the headline, or weight by `max(n_chart, n_audio)`.

### 5. MEDIUM, training runs the BiGRU unpacked over zero-padded batches; inference packs → train/serve skew

`training/drumjot_training/model.py:99` (`forward_all`) always does `self.gru(x)`
with no packing, while `forward` / `stitched_probs` use `pack=True`. In a padded
batch the backward GRU consumes trailing zero frames first, contaminating the hidden
states on the real tail frames, clean at inference, contaminated in training. The
masked loss zeroes pad *frames*, but the real frames near each window end are trained
on context they never see at inference. Bounded (windows are near-uniform, cut at
low-RMS points), cleanly fixable.

**Fix:** give `forward_all` the same `pack` path `forward` already has.

### 6. Lower-severity correctness

- **Sibling weight targets built from unsnapped onsets while targets are snapped**; `train.py:1763-1770` `_spec` returns raw `full` as the sibling-loss weight source
  while `restricted`/targets are the snapped aligned onsets; when the
  `_onsets_aligned_snaponly` store is active, up to ~9-frame (`cr`) misalignment of a
  coarse regional weight. Real but small; only on the aligned-store path.
- **Fixed shuffle seed**, `train.py:1047` `torch.Generator().manual_seed(0)`
  regardless of `--seed` on the non-sampler dataset paths (`shuffle=(train_sampler is
  None)`), so multi-seed runs share batch order → seed-variance CIs are too tight
  (relevant to the noisy `cr` A/B). Fix: seed from `args.seed` (still deterministic
  for resume).
- **Threshold tuned and reported on the same val set**, `train.py:2199`
  (`tune_thresholds(model, val_clips, cfg)` then reported on `val_clips`) and the
  per-epoch curve at `:1186`, in-sample optimism, worst on rare lanes; mitigated by
  the MDB/ParaDB cross-checks but the STAR-val headline is optimistic.
- **fp16 near ~0.1 thresholds (cr/rd/ho)**, `transcriber/app/pipeline/onnx_fp16.py:22`
  (`convert_float_to_float16(keep_io_types=True)`, no `op_block_list`) runs the heads'
  GRU/proj/**calibration `exp()`** in fp16; the calibration amplifies the boundary
  logit error before the sigmoid, so a borderline cymbal/open-hat hit can flip vs the
  fp32/torch reference. The real acceptance test is *per-lane F1 at tuned thresholds*,
  not global correlation. SUSPECTED (empirically corr≥0.99998, but not pinned).
- **Silent CUDA→CPU fallback runs unvalidated fp16 numerics**, `np_onsets.py:79`,
  `separation/np_inference.py:120` catch any session-create failure and retry on the
  CPU EP, which lacks fp16 GRU kernels (late opaque crash) and was never validated
  for fp16 separation. Fix: fall back only when the GPU EP is genuinely unavailable;
  fail loud (or use the fp32/torch path) for fp16 graphs.
- **LLM split/filter calls set no `temperature`**, `hihat_split.py:955`,
  `cymbal_split.py:865`, `filter_llm.py:186`, non-deterministic open/closed,
  ride/crash, and drop decisions in an otherwise deterministic pipeline; breaks A/B
  and debug-bundle replay. Fix: `temperature=0`.
- **Reverb augmentation hardcodes the IR seed**; `parampred/augment.py:52` `np.random.default_rng(0)` → identical room texture on
  every reverb; only decay/wet vary. Cuts the diversity that transform exists for.
- **Cosine LR never reaches its minimum**, `total_steps` counts all batches, but
  `sched.step()` is gated on finite grads (`train.py:1156`) and `--early-stop`
  truncates the decay; the "settles at the LR minimum" comment doesn't hold. Minor.
- **Loss computed inside `autocast()`**, `train.py:1121-1143`. Safe today (PyTorch
  promotes BCE/log_softmax/log to fp32) but fragile to a future non-allowlisted op.

---

## Testing gap (cross-cutting, MEDIUM)

**There is no committed torch↔ONNX parity regression test.** Both ONNX test files
(`tests/test_beat_onnx.py`, `tests/test_onset_onnx.py`) state model parity is
"validated out-of-band"; `separation/np_stft.py:11` and `separation/np_inference.py:10`
cite a "parity test" that **does not exist in the repo**; the one end-to-end ONNX
test (`tests/test_onnx_model_e2e.py`) is skip-gated until the fp16 weights are
provisioned. So a green CI says nothing about whether the shipped ONNX models still
match their torch originals, a future re-export or onnxruntime bump could silently
shift every stem / onset / beat.

The agent-written parity harnesses passed (STFT 2e-5, full MDX stitch 7.5e-8, beat
decode exact). **Commit those** as the guard the docstrings already promise.

---

## Performance

- `parampred/features.py:82-112` recomputes the full STFT **per lane** instead of per
  stem, 2× work on every multi-lane (`s→s,ss` / `h→hc,ho` / `c→rd,cr`) stem, on the
  GPU-feed-bound corpus build. Hoist `audio_features(waveform)` out of the per-lane loop.
- `adtof_onsets.py:~702` `_crash_shadow_filter` is O(n²) (`candidates[:i]` per
  candidate). Pre-existing, ADTOF-only, bounded; sliding window makes it O(n).
- `beats.py:~1370` coarse-offset search is O(n_deltas·n_beats) full `np.interp`
  re-evaluations, it's a cross-correlation of a beat comb against the envelope.
- `targets.ring_spans` recomputes the window RMS envelope 3× with a Python frame loop
  (`train.py:~547`); amortised by the `.rings.json` side-cache but wasteful cold.
- `pipeline/runner.py` reads/STFTs the same drum stems 2-3× per request (beats
  alignment, ADTOF onsets, quantise envelopes), real cost on the HDD-over-NFS box.
  Compute the per-stem onset envelope once, stash it on `PipelineContext`.

---

## Structure / dead code

- **`train.py` (2184 lines / ~8 responsibilities)**, split `losses.py` (the 4 masked
  loss fns), `windowing.py`, `feature_cache.py` (`materialize`/`CachedClips`/index +
  support + rings), `datasets.py` (all `_*_specs` + pooling + `PerSourceResampler`),
  leaving `train.py` = train loop + `main`.
- **`main.py` (1762)**, three near-identical heartbeat-pump streamers
  (`_stream_lyrics_align`/`_stream_score`/`_stream_pipeline`) and two duplicated
  transcribe-endpoint bodies; extract one streamer helper + a
  `PipelineOptions.from_params()`.
- **`beats.py` (1706) / `quantise.py` (1473)**, split detect / meter-recovery /
  smoothing / tempo-segments / alignment (that's where the subtle logic lives).
- **Chunk-schedule + `_normalize`/`_prepare_mix` duplicated verbatim** between
  `separation/runner.py` (torch) and `separation/np_inference.py` (numpy), hand-maintained equivalence; extract the shared schedule math (the missing parity
  test would also guard it).
- **Dead code + false docstrings**, `dedup.py` (`dedup_clips`/`onset_signature`) and
  `clean.filter_by_support` are referenced only by their tests, yet `egmd.py:9` claims
  duplicates are handled "in the cleaning stage." So E-GMD near-duplicate dedup never
  runs (exact same-file leakage is prevented by the CSV split; near-dup is not). Wire
  an onset-signature pass into `_per_source_specs`, or delete the modules and fix the
  docstrings. (`clean.py` itself is *not* dead, `filter_lanes_by_support`/
  `support_score` are used widely.)
- **`np_stft` silently ignores `win_length`**, `separation/np_stft.py:26,33`. Safe
  only because both shipped models set `win_length == n_fft`; a future model with
  `win_length < n_fft` would degrade every stem silently. Add a loud `assert` (or
  implement torch's centre-pad-the-window behaviour).
- **Stringly-typed stage names across the process boundary**; `comms/transcribe_runner.py:35 LIVE_STAGES` is a hand-maintained mirror of
  `pipeline/runner.py STAGE_ORDER`; a rename silently drifts progress fractions. Move
  the `Stage` StrEnum to a torch-free shared module both import.
- **Provision download integrity (pre-existing)**, `pipeline/provision.py:~101-123`
  skip-guard is `st_size > 0`, no `Content-Length`/hash check, so a truncated download
  is trusted forever; plus a shared-`.part` temp name races concurrent writers. Fix:
  verify size (ideally a pinned SHA) before `replace`; unique temp per writer.
  (Capability→asset scoping is **correct and CI-guarded**; the new `trial_perstem`
  loader is **clean**; lane routing, the `(audio_path, restricted, full)` tuple, and
  split integrity all verified + tested.)

---

## Verified correct (checked, not defects)

Recorded so future readers don't re-investigate:

- **Separation forward (numpy/ONNX)**, band-split ordering, complex-mask multiply,
  RoPE attention (softmax axis, 1/√d scale, no causal mask), mask head, and
  stereo/mono axis handling all match torch (empirical mask corr ≈ 1.0). Weights live
  inside the ONNX graph (exported by torch), so no manual transpose/reshape hazards.
- **Separation DSP**, periodic-Hann, reflect-pad, `center=True`, one-sided STFT
  matches `torch.stft` (fwd 2e-5); iSTFT COLA/normalisation matches `torch.istft`
  (~7e-7); both chunk-stitching loops (BS-Roformer per-sample Hamming counter,
  MDX23C constant divisor) match the torch runner to 7.5e-8 with sample-exact length.
- **Beat/lyrics ONNX**, mel frontend (SR/n_fft/hop/mels/norm/`log1p(1000·mel)`),
  50 fps decode (`maximum_filter1d(7)` == `max_pool1d`, threshold >0, dedup, downbeat
  snap-to-nearest-beat), beat-vs-downbeat channel order, lyrics stride/log-softmax/
  `<star>` column, tokenizer/blank resolution, all line-for-line vs the reference.
- **Onset ONNX post-processing**; `activate` (sigmoid + joint ride/crash softmax) is
  unit-tested to 1e-6; the peak-picker, per-lane tuned thresholds, min-distance, and
  windowing are *shared code* with training (`metrics.pick_onsets_lane` +
  `dsp/peakpick`), not a reimplementation; lane→channel mapping is injective.
- **Training internals**; masking (no inversion/double-count), BCE-on-logits (no
  double sigmoid), per-lane `pos_weight`, bf16 AMP (no GradScaler), frozen encoder
  (only heads in the optimizer), MIDI→lane maps, A2MD channel-9 filter, ParaDB
  chart→audio offset sign, `onsets_to_target` Gaussian, cross-dataset SR/hop/frame
  consistency, checkpoint save/load (thresholds in `meta.json`).
- **Provisioning**, capability→asset map complete and correctly scoped (no
  cross-capability pull, no fp16/fp32 mismatch), guarded by
  `test_every_loader_lookup_is_provisioned_by_some_capability`.

---

## Claimed by agents but rejected on verification (do not act on these)

- **"`_smooth_downbeats` decimate drops real beats" (claimed HIGH)**, it's a
  duration-gated heuristic (`beats.py:569`, fires only when the inter-downbeat span is
  genuinely one bar long yet holds ≥2× the beats), removing double-time artifact beats;
  onsets are untouched. Edge-case risk only, not a bug.
- **"Meter-recovery sort/dedup mismatch" (claimed HIGH)**, requires two Beat This!
  beats <50 ms apart; beats are spaced ~one beat-period, so not reachable in practice.
- **"`keep_best` rising-gate misses sharp peaks" (claimed MEDIUM)**, the monotone gate
  is *deliberate*: it exists to reject exactly the early spike-then-collapse
  (open-hat/cymbal at epoch 0) it was accused of missing.
- **Separation / beat / lyrics ONNX numerics**, empirically validated faithful.

---

## Suggested fix order

1. The `--input-norm` cluster (#1), four small fixes; do this before any input-norm
   experiment.
2. Commit the torch↔ONNX parity test (testing gap), cheap, high leverage, guards the
   whole rewrite.
3. `keep` intersect (#2) and `materialize` skip-guard (#3), small, remove silent
   failure modes.
4. `f1_weighted` headline (#4), `temperature=0`, shuffle seed, reverb RNG, one-liners.
5. Performance hotspots (per-lane STFT, coarse-offset cross-correlation).
6. Structural decomposition (`train.py`, `main.py`, `beats.py`/`quantise.py`) as
   opportunity allows.
