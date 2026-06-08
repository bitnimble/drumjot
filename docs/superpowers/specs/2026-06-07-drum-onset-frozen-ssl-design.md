# Drum onset detection via a frozen SSL encoder (feasibility-first design)

**Status:** design / feasibility, pre-implementation
**Date:** 2026-06-07
**Goal:** Test whether a **frozen music self-supervised (SSL) encoder**
(MERT / MusicFM) plus a **small trained head** can do drum **onset
detection** competitively, adopting the modern architecture lineage
(MIROS/N2N) rather than the from-scratch CRNN of `research/HIHAT.md`.

**Scope of v1 (decided):** *onset timing only*, no velocity, no
articulation, feasibility-first. Velocity is a downstream computation
(forced-align, then window the drum stem and take band power), not a
labeling dependency, so it is cleanly deferred (Appendix C). Open-licensed
data only for the feasibility phase (Appendix E covers the real-pop corpus,
which is deferred on both technical and licensing grounds).

**Relationship to existing work:**
- Supersedes `research/HIHAT.md` *architecturally* (frozen SSL encoder vs.
  from-scratch CRNN) and *in scope* (full 5-lane kit vs. hat-only), but is
  a **feasibility probe first**; HIHAT.md's CRNN remains the fallback if the
  frozen-encoder approach does not beat ADTOF. Reuses HIHAT.md's lane
  vocabulary, target-smoothing, peak-pick, and eval discipline.
- Reuses `transcriber/app/scoring/` (the MIDI quality scorer) for the data
  cleaning stage; see §3 for the applicability review the user requested.
- The biggest prize (retire ADTOF, clear the CC-BY-NC-SA blocker in
  `research/MODELS.md`) is the *full-kit* version of this; v1 does not commit
  to it but is the path toward it.

---

## 1. The core thesis (why this might work cheaply)

The 2025 AMT Challenge winner (MIROS) and the current drum SOTA (N2N) share
one load-bearing component: a **frozen music foundation model as the front
end** (MusicFM / MERT), with a small task head behind it. Because the
encoder is frozen, the **trainable parameter count is tiny** (a head of
~0.5–5M params on top of frozen 1024-dim features), so:

- It needs **far less labeled data** than a from-scratch network.
- It fits the **6 GB dev box**: compute embeddings once on the sandbox GPU,
  cache them, train the head cheaply (Appendix B, `research/MODELS.md` for
  the hardware ceiling).
- The "is this approach viable" question can be answered in **days**, on
  **~30 min of clean open-licensed audio**, before any scope commitment.

---

## 2. Phasing (feasibility gates)

Each phase gates the next. Stop if a gate fails and revert to HIHAT.md's
CRNN.

### Phase 0, Smoke test: "does it learn at all?"
**Data:** ~30 min of E-GMD audio (a few dozen clips), cleanest available.
**Goal:** prove the wiring and that the head can extract onsets from frozen
features.
**Milestone ladder (Appendix A has the data-size rationale):**
1. **Overfit one clip**, train loss → ~0 on a single song. If it can't,
   the bug is in data/target/loss wiring, not the model. Minutes to run.
2. **Train loss drops over a few epochs** on ~5–10 clips.
3. **Held-out onset-F1 climbs** above trivial on 2–4 held-out clips.

**Non-negotiables (these decide whether the smoke test tells the truth):**
- Score on **onset-F1 / loss, NOT frame accuracy** (≈99% of frames are
  negative; all-silence scores ~99% frame-accuracy and learns nothing).
- Hold out ≥1 clip from the start; train-only improvement just proves
  memorization.
- Use **clean-label data (E-GMD MIDI-derived)** here precisely to isolate
  the *model* question from the *label-quality* question. Do the data
  cleaning (Phase 1) before introducing any chart-derived labels.

### Phase 1, Data cleaning & label production (§3)
Build a trustworthy training set from open-licensed sources: dedup, quality
score, forced-align (or discard) below threshold. Gates the credibility of
every later number.

### Phase 2, In-domain training & eval
Train the head on the cleaned open-licensed pool. Report **onset-F1 per lane
(±50 ms)** on held-out open data **and** on ENST/MDB/IDMT, **head-to-head
vs. ADTOF**. Decision gate: does frozen-SSL + head beat ADTOF on onset F1?

### Phase 3, Deferred (Appendix F)
Real-pop corpus + synth→real gap; velocity; articulation; full-kit ADTOF
retirement; licensing sign-off. Not in v1.

---

## 3. Data cleaning stage (with quality scoring + forced alignment)

The user has trained on E-GMD before and knows it carries **mislabeled
data, duplicates, and offset timing in places**, "not a huge amount, but it
taints the dataset." This stage exists to catch exactly that. It runs over
**every** training source (E-GMD, STAR, StemGMD) and emits a clean, aligned,
deduped pool plus a quarantine of rejects with reasons.

### 3.0 Applicability of the existing MIDI quality scoring doc (review)

The user asked whether `research/midi-audio-alignment-score.md` (and its v1,
`docs/superpowers/specs/2026-05-30-midi-scoring-utility-design.md`, code in
`transcriber/app/scoring/`) still holds now that forced alignment is on the
table. **Verdict: largely yes, it anticipated this.**

**Still applies as-is:**
- Soft-Gaussian-DTW scoring core; per-lane soft precision/recall/F1; the
  5-lane vocabulary (`k/s/t/h/cy`); the "score *is* the optimizer's
  objective" philosophy; Tier-0 global offset + Tier-1 affine tempo
  correction; the red-flag diagnostics (`offset_sec`, `|tempo_ratio−1|`,
  `unmapped_notes`, channel-9 fallback, score−score_corrected gap).
- The split between `score` (raw) and `score_corrected` (post global align)
  is exactly the right filter primitive: a chart that's *fixably* drifted
  (big offset, otherwise faithful) should be **kept and aligned**, not
  discarded; a chart that's *intrinsically* wrong (low corrected F1) should
  be **discarded**.

**Three changes forced alignment / the new goal introduce:**
1. **Tier-2 per-note alignment becomes first-class.** The v1 spec deferred
   per-note nudging. For *training-label production* (not just filtering) it
   is now the central step: after a chart passes the quality threshold,
   per-note forced alignment (DTW/Viterbi, ±50 ms cap, monotonic, injective)
   snaps each onset onto the true audio onset to produce clean targets. The
   doc already specified its bounds (§8 Tier-2); we promote it from
   "deferred" to "the label step."
2. **Reference timing comes from the onset-strength envelope, not ADTOF.**
   For *training-label* production we don't need ADTOF at all: the **chart**
   supplies which hit and which lane (ground-truth identity/existence), so
   ADTOF's live-pipeline job (figuring out what was played) is already done.
   Forced alignment only needs precise *timing*, and the verified-accurate,
   **license-clean** timing already lives in the pipeline's
   `_refine_peak_times_audio` / `compute_onset_envelope`
   (`app/pipeline/{adtof_onsets,envelope}.py`): a librosa onset-strength
   envelope at ~1.45 ms resolution that snaps each onset onto the true audio
   transient. Reuse it per-lane off the MDX23C stems as the forced-alignment
   timing reference. It is both (a) accurate (the component the data owner
   has verified) and (b) non-circular (we align to the audio signal, not to
   ADTOF's decisions, so the new model isn't trained to merely imitate the
   detector it replaces). For quality *scoring* (chart filtering, §3.2) the
   detector choice is free: ADTOF is fine (NC is not a blocker), but scoring
   against the same per-stem envelope onsets keeps one consistent reference
   across scoring and label timing.
3. **Tighten σ.** The doc sets σ≈25 ms to absorb ADTOF's ~30 ms jitter. The
   envelope reference is sub-2 ms, so σ can shrink considerably, sharpening
   discrimination. Re-calibrate σ to the envelope's empirical timing variance
   (the doc's calibration note still governs; just a far tighter reference).

**Emphasis shift for open data:** E-GMD/STAR/StemGMD labels are MIDI-derived
and mostly sample-accurate, so quality scoring here is less about timing
faithfulness (the real-pop concern the doc was written for) and more about
catching E-GMD's **known flaws**, mislabels, duplicates, the occasional
offset. The same machinery serves both; only the threshold and what we watch
shift.

### 3.1 Dedup (E-GMD has duplicates)
- **Exact dup:** content-hash the audio and the MIDI; drop exact repeats.
- **Near-dup:** audio fingerprint (e.g. chroma/onset-envelope hash) +
  MIDI-onset-sequence similarity; cluster and keep one per cluster.
- **Leakage guard:** dedup **before** the train/val/test split, and split by
  performer/kit/session so near-siblings can't straddle splits.

### 3.2 Quality score every track
Run `transcriber/app/scoring/` (with the §3.0 reference-detector swap) over
every track → `score`, `score_corrected`, per-lane F1, and diagnostics.

### 3.3 Forced-align or discard (the threshold)
- **`score_corrected ≥ T_keep`** → **forced-align** (Tier-2 per-note) and
  admit to the clean pool with corrected targets.
- **`T_discard ≤ score_corrected < T_keep`** → quarantine for optional
  manual review (don't train on it).
- **`score_corrected < T_discard`** → **discard** (intrinsically wrong:
  mislabeled lanes, wrong file pairing, etc.).
- **Red-flag overrides** (discard regardless of score): `|tempo_ratio−1|`
  beyond bounds, `unmapped_notes` above a fraction, channel-9 fallback on a
  source that should have explicit drums, per-lane precision≈0 with many
  chart notes (phantom lane = mislabel).
- `T_keep` / `T_discard` are **calibrated on a hand-labeled audit subset**,
  not guessed; pick them where the score cleanly separates known-good from
  known-bad E-GMD examples the user can point to.

### 3.4 Mislabel detection (beyond the global score)
Per-lane anomaly checks the global roll-up can hide:
- A lane with high chart density but ~0 soft-recall against the reference
  (chart says hits, audio has none) → likely wrong instrument mapping.
- Systematic single-lane offset (one lane drifts, others don't) → channel /
  mapping bug, not a global drift; flag for review.

### 3.5 Outputs of the stage
- `clean_pool/`, admitted tracks with **forced-aligned onset targets** per
  lane, cached log-mel/SSL embeddings, and provenance.
- `quarantine/` and `rejects/`, with the failing metric recorded, so the
  cleaning is auditable and re-runnable.
- A **cleaning report**: counts kept/quarantined/discarded per source, score
  distribution, and the top reject reasons (so we can see *how much* E-GMD
  taint there actually was).

---

## 4. Model & training (Phase 0/2)

> **Update (2026-06-07): lane vocabulary expanded 5 → 11.** Lanes are now
> `k, s, ss, t, hc, hp, ho, rd, cr, mc, mp` (`training/drumjot_training/lanes.py`):
> ride/crash split out of the old `cy`, the three hat articulations
> (closed/pedal/open) split out of the old `h`, side-stick `ss` as its own
> lane, and a sparse tail folded into `mc` (misc cymbals: splash+china+
> ride-bell) and `mp` (misc percussion: cowbell+clap+tambourine). Side-stick
> emits to GM-37 on its own MIDI track but folds onto the snare track as an
> articulation at Jot-load. References to the 5-lane `k/s/t/h/cy` set below
> predate this; the 11-lane set supersedes them. Real-audio overfit on the
> STAR preview reached per-lane F1 = 1.0 on all populated lanes incl. the
> hat split (hp/ho on n=2/1).

- **Encoder (frozen):** **MERT-v1-330M** primary for the probe, HF-native
  (`m-a-p/MERT-v1-330M`), the N2N drum-SOTA choice, **75 Hz** features
  (~13 ms, the best native onset resolution of the candidates), 24-layer
  transformer with acoustic + musical(CQT) teachers; extract an intermediate
  layer (N2N used ~layer 10; tunable). **License CC-BY-NC-4.0, flagged but
  NOT a blocker** (data owner accepts non-commercial; see Appendix E). Note
  NC propagates: a frozen NC encoder caps the *derived head weights* at
  non-commercial even though the training data is clean.
  - **Clean-license alternative: MusicFM** (MIROS's encoder; dual MIT/Apache,
    public weights trained on FMA / Creative-Commons audio). 12-layer
    Conformer + BEST-RQ; MARBLE top-2. Downside: coarser ~25 Hz features
    (~40 ms), but final onset *timing* is snapped to the envelope (§3.0),
    not the encoder grid, so the coarse rate mostly costs the head's native
    resolution, not eval precision. Use as the Phase-2 A/B, or as primary if
    a permissive model is ever wanted.
  - **On the radar: MuQ** (2025) reportedly beats both MERT and MusicFM on
    MARBLE; verify license; candidate if the primary underperforms.
  - Peak-pick in the encoder's frame space or resample to a 100 fps grid to
    match the existing lane/eval conventions.
- **Input: shared frozen encoder on the FULL MIX (default).** One encoder
  pass; the features encode the whole kit (context) and avoid the
  license-unknown separators (`research/MODELS.md`); a frozen MFM is also in
  its pretraining domain on a full mix.
  - **Masked-lane hybrid (targeted, not wholesale):** instruments that get
    buried in the mix (quiet hats, ghost snares) may not be *encoded* at all
    (HIHAT.md reverted hi-hat to the isolated stem for exactly this: a peak
    that isn't there can't be recovered). For those lanes only, additionally
    feed the head the **isolated stem's** encoder features (or stack
    mix+stem). Gate per-lane on the benchmark. Do **not** run every lane on
    its own stem: that loses cross-kit context and re-imports separation
    artifacts (the "phantom crash at bar 5" bleed case in `adtof_onsets.py`).
- **Heads: separate per-lane heads on the shared frozen encoder (decided).**
  One tiny head per lane (`k/s/t/h/cy`), each a 2-layer BiGRU (~128 hidden)
  or a couple of conv+linear layers over the frozen features → a sigmoid
  onset-activation curve; peak-pick (height + min-distance + prominence) with
  per-lane thresholds tuned on val. **Separate heads, not one multi-output
  head**: they capture the specialization of per-instrument models (full
  per-lane capacity; independent thresholds / target smoothing / loss
  weighting / eval-gating; the HIHAT.md per-lane workflow) while the shared
  frozen features retain full-kit context (kick/snare placement, cymbal
  co-occurrence, ring-then-strike). Separate heads also remove the inter-lane
  negative transfer on the overlapping HF lanes (open-hat sizzle vs crash;
  HIHAT.md §6) that a single multi-output head would suffer. Generalizes the
  HIHAT.md output contract to 5 lanes.
  - **Why not N standalone per-stem models:** ignoring compute, they still
    lose cross-kit context and inherit separator artifacts, and the usual
    specialist win (per-instrument input front-end tuning) is moot because
    the encoder is frozen. The frozen-encoder design captures the
    specialization benefit in the cheap heads instead.
- **Targets:** Gaussian/triangular-spread peaks at onset frames (±1–2 frames
  for sample-accurate synth labels; widen for any chart-derived labels).
- **Loss:** per-frame BCE per channel + **focal / positive-weighting** for
  the heavy negative imbalance.
- **Metric:** onset-F1 per lane at ±50 ms (the standard ADT metric), plus a
  confusion view across lanes.
- **Polyphony / CTC (parked):** the data owner is fine with either monophonic
  per-stem or polyphonic full-mix transcription, so polyphony is a non-issue
  (separate per-lane heads each emit one stream anyway). CTC becomes viable
  per-lane in the monophonic case but offers no advantage over a frame-wise
  head for pure onset *timing*; park it for a later metrically-quantized
  output stage (constant-tempo lattice, CTC2-style), not the onset detector.

---

## 5. Open questions / risks

1. **SSL frame rate vs. onset precision.** ~75 Hz (≈13 ms) is fine for
   ±50 ms F1 but may limit a future tight-timing/velocity goal; verify per
   encoder.
2. **Masked instruments on the full mix.** Default is a full-mix shared
   encoder + separate per-lane heads (§4). The open risk is whether the
   frozen encoder *encodes* quiet masked hats / ghost snares at all. If a
   lane's recall is capped, add the isolated-stem feature injection for that
   lane (§4 masked-lane hybrid), gated on the benchmark.
3. **Reference-timing quality for forced alignment.** Labels inherit the
   onset-strength envelope's timing (§3.0), the verified-accurate, clean
   component of the live pipeline. Still validate per-lane on the stems
   (dense hats, bloom-y cymbals): a missing transient makes the
   nearest-local-max snap produce garbage, so the quality score must gate
   which chart hits actually have audio support before snapping.
4. **Threshold calibration depends on a human audit subset** the user must
   provide (known-good and known-bad E-GMD examples).
5. **Cymbal classes.** `cy` merges ride+crash (ADTOF/eval limitation); the
   new model *could* split them, but there's no eval signal until a
   ride/crash-labeled test set exists (mirrors HIHAT.md's articulation-eval
   blocker).
6. **Synth→real gap** is deliberately out of Phase 0–2 (Appendix F).

---

## 6. Immediate next step

Build the **Phase 0 smoke test**: pick ~30 min of clean E-GMD, cache MERT
embeddings on the sandbox GPU, wire the small head + target + onset-F1,
**overfit one clip first**, then watch train loss over ~5–10 clips. That
single result decides whether the rest of this is worth building.

---

# Appendix, context & decisions captured for future revisiting

> The discussion that produced this plan covered more than v1 needs. Parked
> here so we can revisit without re-deriving it.

## A. Minimum-data figures (open-licensed, onset-only, frozen encoder)

The minimum is set by **trainable head capacity + frozen-feature richness**,
not by data source, so it's about the same as for real-pop, but the
**open-data smoke test is cleaner and more trustworthy** because labels are
sample-accurate (no alignment confound). With open data a failed smoke test
unambiguously means the *model/wiring* is wrong.

| Milestone | Real-pop corpus | Open-licensed (E-GMD/STAR/StemGMD) |
|---|---|---|
| Overfit a clip (wiring) | 1–5 songs | a few clips (~minutes) |
| Train loss drops / few epochs | 5–10 songs (~30–40 min) | **~20–30 min** |
| Held-out F1 climbs (in-domain) | 20–40 songs (~1.5–3 h) | ~1–3 h |
| Credible v1 | tens of hours | tens–hundreds h (≈1800 h permissive available) |

Recommendation: run the **first** smoke test on open data, perfect labels,
zero licensing exposure, unlimited supply, then introduce the real-pop
corpus to test the forced-alignment label pipeline as one new variable.

## B. Architecture landscape (the "most modern" survey)

- **Seq2seq encoder–decoder transformer (Whisper-shaped):**
  Hawthorne 2021 → **MT3** → **YourMT3+** (Perceiver-TF encoder) → **MIROS**
  (2025 AMT Challenge winner: **MusicFM** encoder + T5-style multi-decoder,
  RoPE + FlashAttention, recurrent adapter w/ instrument-group embeddings,
  ~370M params, Slakh F≈0.83; *dropped* cross-stem aug, isolating the
  pretrained-encoder contribution).
- **CTC (alignment-free seq output):** strongest standalone is **CTC2**
  (drums, constant-tempo lattice constraint) and CRNN/Conformer+CTC singing.
  CTC's monotonic + conditional-independence assumptions fight **polyphony**
  (simultaneous kick+hat), needs per-lane CTC or a serialized token
  grammar. Now mostly a *component/loss*, not the headline.
- **Diffusion (current drum SOTA):** **Noise-to-Notes (N2N, 2509.21739)**, transformer decoder w/ FiLM + cross-attention, conditional diffusion over
  drum onsets+velocity, **MERT-330M layer-10** features + log-mel, Annealed
  Pseudo-Huber loss. E-GMD 89.7 / MDB 87.9 / IDMT 94.9; big OOD gains.
- **Common factor under both winners:** a **frozen music foundation model
  front end** (MusicFM in MIROS, MERT in N2N). That is the single
  highest-leverage idea and the basis of this plan. Heads differ
  (autoregressive T5 vs. diffusion vs. our simple BiGRU); the encoder thesis
  is shared.
- **Forced alignment / DP (the "when"):** chroma/CQT DTW (synctoolbox),
  HMM-Viterbi forced alignment, neural loose→fine refiners, and the
  Maman 2022 unaligned-supervision EM loop (transcribe ↔ DTW-align to refine
  labels). This is what §3.3's per-note alignment draws on.

## C. Velocity (deferred)

RLRR does **not** encode velocity. Plan: after forced alignment, take a
window on the relevant drum stem around each onset and compute band/onset
power → velocity. Downstream of onsets, not a labeling dependency. Modern
models (N2N, IDM) treat velocity as first-class; fold it into the target
once onsets work.

## D. Dataset landscape

| Source | ~Size | License | Real/synth | Labels | Role |
|---|---|---|---|---|---|
| Real-pop corpus (charts) | ~300–400 h | charts licensed; **audio not** (Appendix E) | real | onset (+derive vel) | deferred gold (Phase 3) |
| STAR Drums | 124 h (~181 GB) | BSD-3 | synth-into-real | 18-class + artic + vel | pretrain realism + clean Phase-0 |
| StemGMD | ~1224 h stems | **verify Zenodo** | synth, isolated | open/closed stems | augmentation engine |
| E-GMD | 444 h | CC-BY-4.0 | e-kit | GM class + vel | primary clean pool (has flaws → §3) |
| ENST / MDB / IDMT | small | mixed | real acoustic | varies | external real test |
| ADTOF | 359 h | CC-BY-NC-SA | real | single HH class | **excluded** (taint + no articulation) |

## E. Licensing, two distinct axes

1. **Training-data license → weight taint** (the `research/MODELS.md`
   concern). CC-BY-NC-SA (ADTOF, madmom weights) ShareAlike-taints derived
   weights. Keep the training pool permissive (STAR/E-GMD CC-BY-4.0;
   StemGMD verify). This is about *dataset* licenses. **Encoder note:** the
   data owner accepts non-commercial, so **MERT-v1-330M's CC-BY-NC-4.0 is
   flagged, not a blocker**. NC propagates to derived weights, so a frozen
   MERT head can't be shipped commercially even on clean data; **MusicFM**
   (MIT/Apache, FMA-trained) is the clean fallback if that ever changes, and
   **MuQ**'s license needs checking.
2. **Input-audio copyright + acquisition** (the real-pop corpus). Licensing
   the *annotations* grants **no rights** to the recordings (two copyrights:
   composition + master). 2025–2026 rulings:
   - **Bartz v. Anthropic (Jun 2025):** training on **lawfully acquired**
     copies is "spectacularly transformative" fair use; **piracy to build
     the corpus is not**; acquisition is decisive.
   - **Kadrey v. Meta (Jun 2025):** training transformative; split from
     Bartz on whether source legitimacy matters (unsettled).
   - **Thomson Reuters v. Ross (Feb 2025):** fair use **rejected**, but the
     tool *competed with* the source and wasn't generative-transformative; a
     transcription tool that doesn't substitute for the recording market is
     distinguishable.
   - **Suno/Udio:** converging on **licensing** (UMG–Udio Oct 2025 royalty
     template; Warner–Suno Nov 2025, framed around **stream-ripping**); Suno
     fair-use SJ hearing **Jul 2026**; indie class actions (Nguyen v. Suno).
   **Implications for the corpus:** (a) **stream-ripping from Spotify/Tidal
   is the most dangerous step** and was specifically penalized, avoid; (b) a
   **transformative, non-generative transcription model whose weights can't
   reproduce recordings**, trained on **lawfully acquired** audio, is a
   strong (not guaranteed) fair-use posture post-Bartz; (c) **never
   distribute the training audio**; (d) **get an IP lawyer before commercial
   distribution.** Cleanest path: ship a model trained on permissive/owned
   audio; use the real-pop corpus for **internal research/eval** only.

## F. Real-pop corpus path (deferred, Phase 3)

When/if pursued: forced-align charts to lawfully-acquired audio (the §3
machinery, real-pop thresholds), measure and close the **synth→real gap**
(STAR / ADT_STR realistic-synthetic augmentation), add velocity (Appendix C)
and articulation, then consider **full-kit** to retire ADTOF entirely (the
licensing prize in `research/MODELS.md`).

## G. Relationship to HIHAT.md

HIHAT.md is a hat-only, from-scratch CRNN on the BS-Roformer drum stem with
open/closed articulation. This plan is broader (5-lane onsets) and modern
(frozen SSL encoder), but **feasibility-gated**: if frozen-SSL doesn't beat
ADTOF (Phase 2 gate), HIHAT.md's CRNN remains the route for the hat lane.
The two share lane vocabulary, target-smoothing, peak-pick, and the
"build the articulation/onset eval set first" discipline (HIHAT.md §8).

## H. Tom pitch splitting (deferred, transcriber-side, NOT training)

The model emits a single merged `t` (toms) lane. Toms are **tonal**, so the
individual tom pitches (high/mid/floor) can be recovered **deterministically
downstream** in the transcriber pipeline, a post-model phase that takes the
`t` onsets and clusters them by **spectral-centroid frequency** (higher
centroid → higher tom) into N tom voices. This stays out of the model
(training keeps toms merged, which also dodges the tom-class sparsity from
the STAR/ADT class-imbalance finding); it's a cheap DSP clustering stage in
the transcriber after onset detection. Noted for later; not in scope here.
