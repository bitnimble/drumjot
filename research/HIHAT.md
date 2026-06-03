# Open/closed hi-hat: custom-trained articulation model (design spec)

**Status:** design, pre-implementation
**Date:** 2026-05-29
**Goal:** Programmatically distinguish open vs closed (vs pedal) hi-hat hits
clearly enough to improve final transcription accuracy, by replacing the
current heuristic+LLM open/closed split with a custom-trained model.

---

## 1. Context and the problem with today's pipeline

> **Update (2026-06-03):** the heuristic hi-hat path described here was
> substantially reworked after this doc was written, hi-hat ADTOF now runs
> on the ISOLATED stem (not the drum mix), with looser gates, an
> audio-domain onset supplement, and an energy floor; the split gained
> `attack_flux` + `lowband_ratio` features (and dropped `flatness`/`centroid`
> from the prompt), a deterministic envelope open/closed guardrail, a
> flux-based open-within-open filter, and a discard-rescue. See
> [../docs/transcriber-pipeline.md](../docs/transcriber-pipeline.md) "Hi-hat
> lane". This raised the heuristic's accuracy, but open/closed remains
> threshold-tuned on limited data and a trained model is still the goal.

Current transcribe path (post-rework):

```
full mix
  → BS-Roformer SW           (drum stem; preserves HF cymbal/hat transients, drums SDR ~14)
  → MDX23C DrumSep           (kick / snare / toms / hi-hat / cymbals stems)
  → ADTOF Frame_RNN          (per-frame onset activations, per ISOLATED stem;
                              hi-hat lane adds an audio-domain onset supplement
                              + energy floor to counter the ~14kHz band-limit)
  → hihat_split.py           (ring-envelope features [late_rms, pre_rms, tail_end_s,
                              attack_flux, lowband_ratio] + ternary LLM classify
                              open/closed/discard + deterministic envelope guardrail
                              + _open_tail_filter + discard-rescue)
  → MIDI → frontend Jot
```

The open/closed decision today is the weakest, most heuristic link in the
chain: librosa features handed to an LLM, plus a physically-motivated
open-tail backstop. It exists because the upstream pieces (ADTOF on a
single-instrument lane) cannot see enough context to make the call.

### Why this is the right thing to replace

Open-vs-closed is **not an onset-instant property; it is a decay/sustain
property.** A closed hat is a 30–80 ms tick; an open hat rings/sizzles for
200 ms – 2 s. The discriminative signal lives in the ~250–500 ms *after* the
strike, not at the transient. Two consequences drive the whole design:

1. A classifier that only inspects a small window around the onset (what the
   DSP features do) works against the grain. We want a model whose temporal
   context spans the tail.
2. **The "open/closed classification" problem and the "phantom sizzle
   re-trigger" problem are the same problem.** A model that sees the tail
   *should* learn that one long ring = *one* open hit, not a train of
   16th-note closed hits. This is the bet, not a guarantee: ADTOF is itself a
   BiGRU CRNN and it still phantom-trains on hat audio (which is exactly why
   `_open_tail_filter` exists). The lever we are actually pulling is **(a)
   in-domain training data + (b) an explicit per-articulation target**, not
   the recurrent architecture alone. We therefore **keep the deterministic
   `_open_tail_filter` as a planned post-filter**, expecting it to fire rarely
   rather than to be deleted (see §7).

---

## 2. Approach (decision)

A **custom frame-wise hi-hat onset + articulation model** (a CRNN), trained
in-domain on drum stems produced by *our own* separation pipeline, emitting
**separate per-frame activation channels for closed-HH and open-HH** (pedal
folded into closed for v1). Peak-pick each channel.

**Scope (decided): hat lane only.** This model **replaces ADTOF *on the hat
lane* and the entire `hihat_split` LLM call**; nothing else. **ADTOF stays in
the pipeline** for kick / snare / toms / ride / crash; those lanes are
unchanged. Full-kit replacement (and the ADTOF-retirement licensing win) is
**deferred**, not pursued here; see §7 and §9.9. The model therefore does *not*
emit non-hat channels; it is a focused hat detector that happens to also call
open vs closed.

Approaches considered and rejected:

- **Onset-conditioned patch classifier** (keep ADTOF, classify a window around
  each onset): lightest, but inherits ADTOF's recall and its phantom-train
  behaviour on the hat lane, and works against the decay-is-the-signal grain.
  Kept only as a conceptual fallback.
- **Hi-hat source separator** (open-track / closed-track audio): open and
  closed hats overlap heavily in time and spectrum, clean separated targets do
  not exist (would have to be synthesized), and a detector is still needed on
  the output. All cost, no benefit for a binary articulation label. Rejected.

Why the chosen approach wins: it solves the real problem at the root in a
single in-domain model, replaces the DSP-features + LLM split with a learned
articulation target (keeping only the cheap `_open_tail_filter` backstop), and
the full drum-stem context (kick/snare placement, whether a ring is followed by
another strike) is exactly what disambiguates open vs closed. Note the model is
fed the **full drum stem** as context (so it can use kick/snare placement etc.)
but only *emits* the hat channels, context-in, hat-out.

---

## 3. Inputs and feature representation

- **Source signal: the BS-Roformer drum stem**, 44.1 kHz mono. *Not* the
  MDX23C isolated hi-hat stem; that is where the open tail gets smeared, and
  the full drum context is what disambiguates articulation. (This matches the
  existing observation that ADTOF works better on the drum stem than the
  isolated hat stem.)
- **Feature: log-mel spectrogram.**
  - `n_fft = 2048` (~46 ms analysis window).
  - `hop = 441` → **100 frames/sec (10 ms hop)**. Matches ADTOF; 10 ms
    onset resolution sits comfortably inside the standard ±50 ms eval
    tolerance. (Frames are 10 ms apart, ~46 ms wide, ~78% overlap → smooth,
    peak-pickable activations.)
  - **192–229 mel bins, fmax = 22050 Hz.** The open/closed cue *is* broadband
    5–15 kHz sizzle energy and its decay, and mel spacing is near-linear/coarse
    up there, so 128 bins devote too few bins above 10 kHz to the exact band
    that matters. Start high. A dedicated **high-band log-energy feature** (or a
    CQT / per-octave HF branch) is a strong v1 option if mel still
    under-resolves the sizzle. This is a representation decision, not a late
    tuning knob: under-resolving the sizzle caps articulation accuracy
    regardless of architecture.
  - Log-compressed, per-corpus normalized.
- **Input channels: start drum-stem-only (1 channel).** Designed-in upgrade
  path: stack the isolated hi-hat and cymbals stems as extra input channels
  *only if* the single channel underperforms, to avoid baking in separation
  artifacts prematurely.

> The 10 ms hop / 46 ms window is only the per-frame snapshot resolution. The
> model's *decision context* for open/closed comes from the recurrent core
> integrating across 20–50 frames (the 200–500 ms tail), not from a single
> frame.

---

## 4. Model architecture

A **CRNN**: convolutional front-end + recurrent temporal core. This is the
proven modern-ADT recipe (Vogl et al.; ADTOF's own Frame_RNN) and is the right
shape for a decay-driven decision.

### Relationship to ADTOF: same approach, fresh weights. NOT fine-tuning

Two clarifications that are easy to lose and both matter for licensing and for
understanding what this model is:

- **This model IS the onset detector; there is no separate detector "in
  front" of it.** It is *not* a classifier that waits for ADTOF (or anything
  else) to find onsets and then labels a window around each. It is a **frame-wise
  detector**: it consumes the whole drum-stem spectrogram and emits, for every
  10 ms frame and every class, a 0–1 activation ("does an onset of this class
  start here?"). Onsets come from **peak-picking those curves** (§7); detection
  and articulation fall out of one forward pass. This is exactly how ADTOF
  itself works (`adtof_onsets.py`: "we take ADTOF's dense per-frame activations
  and run our OWN deterministic peak-pick"), so our model is a drop-in at the
  same activations→peak-pick boundary. The detect-then-classify alternative was
  considered and **rejected** (§2) precisely because it can never remove ADTOF.
- **We replicate ADTOF's *architecture family*, not its *weights*.** We do
  **not** fine-tune ADTOF's checkpoint. ADTOF's weights are CC-BY-NC-SA;
  continuing training from them would make ours a derivative and re-import the
  ShareAlike taint we are trying to escape (§9.9). Instead we **train new
  weights from scratch** on the same *kind* of network, pretraining on
  permissively-licensed data (StemGMD / STAR / E-GMD) and fine-tuning on the
  6289 corpus. ADTOF contributes the *idea* (a frame-wise CRNN), not a single
  weight or training sample. Consequence: the model must **earn** ADTOF-level
  quality on each lane; it inherits none of ADTOF's competence; which is why
  rollout is gated per-lane on the benchmark (§7, §8).

### Components

- **Conv front-end:** stacked 3×3 convolutions with pooling in the **frequency
  axis only** (preserve time resolution) → a compact per-frame feature vector.
- **Temporal core: 2-layer BiGRU (~128 hidden).** The recurrence sees the
  entire post-onset tail, so the model decides open-vs-closed from the
  decay/sizzle that unfolds over the next 200–500 ms, and learns that one long
  ring = one open onset rather than a phantom train.
- **Upgrade path (not v1):** swap the BiGRU core for a Conformer /
  self-attention encoder once the CRNN baseline is trustworthy. 6289 songs can
  support it; CRNN first because it is robust, fast to train, and proven.

### Outputs: per-frame hat-onset activations

- Two **sigmoid activation curves per frame** (multi-label, not softmax;
  `hh_closed` and `hh_open` can in principle both fire). **These two channels
  are the entire output**; no kick/snare/tom/cymbal channels, because ADTOF
  still owns those lanes (§2 scope). That separation *is* the deliverable.
- The "ternary" hat outcome (closed / open / neither) **emerges from the two
  independent binary channels**; there is no 3-way softmax, and peak-picking is
  uniform across both.
- **`hh_pedal` is deferred out of v1.** Pedal hi-hat (GM 44) is rare, may be
  thin or absent in the labels, has no eval signal (the benchmark folds it into
  HH), and `h:pedal` is **not a routable pitch** anywhere in the pipeline today
  (`onsets_midi.py` has no mapping for it, so it would be silently dropped). v1
  folds any pedal hits into `hh_closed`. Add a dedicated channel later only if a
  verified pedal-labelled source and a routing target both exist.
- **No `hh_any` auxiliary channel in v1.** It existed only to let
  no-articulation data (ADTOF) supervise hat-onset detection via masked loss.
  Since the corpus is pre-filtered to songs with clean open/closed labels and
  the public pretraining sets (StemGMD/STAR/E-GMD) all carry articulation, every
  training source has full hat labels; so there is nothing to mask and no need
  for `hh_any`. (Re-introduce it only if a no-articulation hat source is ever
  added to training.)

---

## 5. Datasets

Ordered by proximity to the target domain (open/closed hat onsets on a drum
stem separated from a real mix).

| Dataset | Size | Real / synth | Open/closed labels | Role |
|---|---|---|---|---|
| **Our 6289 songs** | very large | real mix → our separation | ✅ reliable, meticulous | **Gold. Primary fine-tune + held-out test.** |
| **STAR Drums** (2025) | ~124 h | synth drums mixed into **real** melodic/vocal recordings | ✅ distinct (18-class) | Closest public analog to our domain; pretrain realism + articulation. CC BY 4.0 (Zenodo). |
| **StemGMD** (2023) | ~136 h mixtures / ~1224 h total stem audio | synth (10 Logic sample libs), **isolated stems** | ✅ separate open-HH & closed-HH stems | Articulation/decay prior at scale; isolated stems = a built-in augmentation engine with perfect labels. *(Verify exact hours against the Zenodo record before weighting the mix.)* |
| **ADTOF** (2021, exp. 359 h) | 359 h | **real commercial music** (rhythm-game charts) | ❌ single hi-hat class | **Not used for training.** No articulation labels (zero value to a hat-articulation model) *and* CC-BY-NC-SA (would taint the weights). Excluded. |
| **E-GMD** (2020) | ~444 h | re-recorded e-kits | ✅ 42/46/44 | One more pretraining source. |
| **ENST / MDB / IDMT** | small | **real acoustic** | ENST ✅ | External real-acoustic **test** + small real fine-tuning. |

### Masked / partial-label loss; not needed in v1

The earlier design needed per-dataset label masking to fold in no-articulation
real-music data (ADTOF). With ADTOF dropped from training (above) and the corpus
pre-filtered to clean open/closed labels (§6), **every training source carries
full hat articulation**; our corpus, STAR, StemGMD, E-GMD. There is nothing to
mask, so the masking mechanism (and the `hh_any` channel) is omitted. Keep it in
mind only if a no-articulation hat source is ever reintroduced.

### StemGMD augmentation engine

Because StemGMD provides open-HH and closed-HH as separate audio, we can
synthesize unlimited drum-stem training clips by summing stems at varied
relative levels (and applying light bus processing), each with perfect
open/closed labels; directly targeting the articulation decision.

### Licensing (decides what may enter the training pool)

**ADTOF is *not* retired by this project (decided): it keeps running for the
non-hat lanes.** So the pipeline as a whole remains under ADTOF's CC-BY-NC-SA
runtime dependency (`MODELS.md:52,83-98`) for now; that blocker is tracked
separately in `MODELS.md` and is out of scope here.

Even so, **keep this hat model's training data clean**, for cheap future
optionality: if ADTOF is eventually replaced, we don't want this model to have
become a *second* NC-SA blocker.

- **Do not train on the ADTOF dataset.** A model trained on CC-BY-NC-SA data is
  a derivative; ShareAlike would force NC-SA onto the weights. ADTOF is excluded
  from training anyway (no articulation labels; §5 table), so this costs us
  nothing here.
- **Permissive sources only:** STAR Drums is **CC-BY-4.0**; E-GMD is
  **CC-BY-4.0**; **verify StemGMD's Zenodo license** before relying on it.
  Confirm each is attribution-only (commercial OK) so the pool stays clean.
- This keeps the hat model itself commercially clean **regardless** of the
  ADTOF runtime situation, so it never has to be retrained on licensing grounds.
- **Separator dependency:** the cached training stems are produced by
  BS-Roformer + MDX23C, both **license-unknown** (`MODELS.md:48-49,65-77`) and
  flagged for possible replacement (Demucs). If the separator is swapped after
  training, the cached drum-stem domain shifts and the model's train==inference
  guarantee breaks. Pin the separator as a hard dependency of this model, or
  budget to re-separate + retrain on a swap (see §9.5).

---

## 6. Training recipe

1. **Pretrain** on the permissive synth/semi-synth pool (**StemGMD + STAR Drums
   + E-GMD**) to learn the open/closed timbre + decay prior with abundant
   reliable articulation labels.
2. **Fine-tune** on our **pre-filtered corpus** of separated drum stems
   (in-domain gold; the subset of the 6289 that cleanly encode open/closed; §6 label conversion). This adapts to acoustic kits, genre variety, and our
   separation artifacts, and is the model's only real-music supervision, which
   is why a real corpus is load-bearing.
3. **Test** on a held-out slice of that corpus + **ENST/MDB/IDMT** (independent
   real-acoustic). (ADTOF is not in the training pool, §5.)

Caveat stated plainly: STAR and StemGMD drums are *synthesized* (virtual
instruments), so they are a prior, not a substitute. The 6289 real songs are
what make the model work on real audio. The 2026 ADT_STR method for
*generating* articulation-labelled synthetic data is an optional augmentation
source, not core.

### Data preparation

- **One-time corpus build:** run all 6289 full mixes through our own
  separation pipeline (BS-Roformer → drum stem) and **cache the drum stems and
  their log-mels** so epochs are cheap. This is the expensive step
  (GPU-hours × 6289) but is done once.
- **Label conversion (read the instrument the event hit, not a MIDI note):**
  RLRR `RlrrEvent` is `{name, vel, loc, time}` (`src/rlrr/schema.ts:30-51`).
  There are **three** places open/closed could be encoded, and the codebase
  shows they are genuinely used differently by different authoring paths, so
  the extractor must be **mechanism-agnostic** rather than assume one:
  1. **`event.midi` extension** (GM 46 open / 42 closed / 44 pedal). This is how
     *this* repo's own converters carry it; `drums.ts:93-97` states plainly
     "Paradiddle uses the same class so we can't actually disambiguate at the
     RLRR layer … the MIDI round-trip uses the `event.midi` extension"
     (`schema.ts:44-50`). If the 6289 charts came through Drumjot's own
     MIDI→RLRR path, this is where the signal lives.
  2. **Separate instrument `class`/`name`**; `class` is free-form with no enum
     (`schema.ts:23`, upstream `rlrrschema.json`), so a charter *can* define a
     distinct open-hat instrument and reference it by `name`. Authoring tools
     that do this put the signal here.
  3. **`loc` hit-zone index** (`schema.ts:35`, "always 0 in the reference
     tool"); unlikely, but some kits could encode a zone on one hat instrument.
  The label path reads the raw RLRR (**bypassing `CLASS_TO_DRUM`**, which only
  knows `BP_HiHat_C`→closed and folds 42/46 together; `drums.ts:28,62-64`) and
  resolves articulation from whichever of (1)/(2)/(3) carries it. Channel set
  (hat-only model): {hh_closed, hh_open}.
  - **No corpus audit; clean encoding is assumed (decided).** We do not audit or
    handle ambiguous charts. The data owner **pre-filters the corpus to songs
    that cleanly separate open vs closed** and discards the rest, so every
    retained song has a usable label in one of (1)/(2)/(3). Consequence: the
    effective corpus is **smaller than 6289** (acceptable), and the
    single-hat-masking machinery is unneeded. The extractor still has to read
    the *correct* field, so confirm once which of (1)/(2)/(3) the corpus uses
    before wiring it up, but treat that as an implementation detail, not a
    go/no-go gate.
- **Stem↔label timing (one-time check, then watch the residual):** verify the
  separation models are truly sample-aligned, not merely free of *constant*
  latency; a global-offset correction only catches constant lag, not
  chunk-boundary/frame-edge jitter from the overlap-add transformers. Spot-check
  alignment on a handful of tracks.
- **Target encoding:** for each onset, a peak on that class's channel at the
  onset frame with a small **Gaussian/triangular spread** to ease optimization
  and absorb alignment jitter (standard ADT target smoothing). Use ±1–2 frames
  for the tightly-aligned synthetic sets; **widen to ~±3–4 frames for the real
  charts**, where chart-vs-audio offsets of 20–40 ms are routine. The true peak
  stays on the true frame.

### Splits

- **Split by song, grouped by artist/album** (no siblings straddle
  train/test → prevents kit/production leakage), **stratified by genre** (metal
  vs pop hats differ). Hold out a clean test set up front and never touch it
  during tuning.

### Loss and class imbalance

- **Per-frame binary cross-entropy per channel** (multi-label). Frames are
  overwhelmingly negative → **focal loss or positive-weighting** so the model
  is not trivially "predict silence."
- **Open ≪ closed**, so weight the `hh_open` positive loss up and/or oversample
  songs containing open hats. (Watch a side effect: open-hat sizzle overlaps
  crash/cymbal energy on the merged drum stem, so oversampling open-hat material
  also oversamples crash-heavy material; monitor the crash channel for
  co-adaptation.)
- **Pedal** is folded into `hh_closed` for v1 (§4); revisit only with a verified
  pedal-labelled source and a routing target.

---

## 7. Inference and pipeline integration

- New pipeline stage (working name `learned_onsets`) produces `OnsetCandidate`s
  per channel into `onsets_by_pitch`:
  - `hh_closed` → `h`
  - `hh_open` → `H` (existing synthetic open-hat routing pitch; rendered to
    GM 46 by `onsets_midi.py` and folded back to `h:o` on the frontend)
  - (no pedal lane in v1; see §4)
- Reuse the existing **height + min-distance + prominence peak-pick**, with
  **per-channel thresholds tuned on the validation set** (open's threshold will
  differ from closed's). Refine each peak's time against the audio onset-strength
  envelope as ADTOF does today, if needed.
- **Replaces ADTOF's *hat lane* + the entire `hihat_split` LLM call.** ADTOF
  still runs for kick / snare / toms / cymbals; its hat-lane activations are
  simply ignored in favour of this model's `hh_closed` / `hh_open`. Keep
  `_open_tail_filter` as a near-silent safety net; drop the `discard` LLM logic.
- Downstream is unchanged: `beats.py` beat-relative annotation, MIDI emission,
  `note_provenance`, frontend Jot conversion.
- **Scope is hat-only; full-kit replacement is explicitly out of scope (§2).**
  ADTOF stays in the deployed pipeline, so its CC-BY-NC-SA *runtime* dependency
  remains (tracked in `MODELS.md`, not this project). Expanding this model to
  the full kit to retire ADTOF is a possible *future* effort, not part of v1.

---

## 8. Evaluation: how we know it worked

**Blocker to address first: no articulation eval exists today.** The benchmark
harness scores only 3 folded classes (KD/SD/HH; `benchmarks/core/classes.py`),
and *every* wired loader folds open hat into HH (`MDB_LABEL_TO_CLASS` maps
`OH`→HH; the E-GMD path discards 42/44/46; IDMT has no open label). So before
the success criterion below is even measurable we must build:

1. **A real-audio, open/closed-labelled test set.** A held-out slice of the
   6289 (gold) is the primary one; if feasible also hand-annotate a few hundred
   open/closed hat onsets from real songs *outside* the 6289 as an independent
   check. This is the only thing that can validate the **articulation head**,
   which (per §9.3) leans on synthetic timbre to the extent the real corpus is
   thin on articulation; synth val metrics can look great while real
   articulation silently fails.
2. **New scoring code** for articulation (the harness's 3-class folding cannot
   express it).
3. **New dataset loaders.** `benchmarks/loaders/` has only `egmd`, `idmt_smt`,
   `mdb_drums`; there are **no StemGMD / STAR / ADTOF loaders**, so those sets
   need loader code before they can be trained on or scored, not just new
   scoring.

Then report, against that test set **and head-to-head vs the current ADTOF +
LLM baseline**:

- **Onset F1 per class** (±50 ms tolerance); standard ADT metric.
- **Articulation accuracy** (the metric that actually matters): of
  correctly-detected hat onsets, the fraction labelled with the right
  articulation, plus a **closed/open confusion matrix** and per-class
  precision/recall. Report on the real-audio set specifically, not just synth val.
- **Phantom rate**; spurious hat onsets inside open tails (directly measures
  whether the sizzle-train problem is gone).

### Success criteria (proposed; confirm)

- Beat the current ADTOF + LLM baseline on **hat-lane onset F1**.
- **> 90 % open/closed articulation accuracy on correctly-detected hats** on
  the held-out **real-music** test set (not synth val).
- **Phantom rate ≈ 0** inside open tails.

---

## 9. Risks and open questions

1. **Label encoding assumed clean (audit skipped, decided).** Open/closed lives
   in one of `event.midi` / instrument `class` / `loc` (§6); the data owner
   pre-filters the corpus to songs that cleanly carry it and discards the rest.
   So feasibility is assumed, not audited, and the effective corpus is `<6289`
   (acceptable). Residual implementation step: confirm once *which* field the
   retained corpus uses before wiring the extractor.
2. **[BLOCKER] No articulation eval today.** The harness scores 3 folded
   classes; a real-audio open/closed-labelled test set + new scoring code must
   be built before the success criteria are measurable (§8).
3. **Articulation-head synth→real gap.** Real-music articulation supervision
   comes from the (pre-filtered) corpus itself, which is the point of having a
   real corpus; the synth sets (STAR/StemGMD/E-GMD) are a prior. The real-audio
   articulation eval (§8) is the guard; watch the synth-val vs real-test gap.
4. **Separation compute for 6289 songs**; one-time, cacheable, but budget for
   it.
5. **Separator dependency + cache brittleness.** Cached stems are pinned to
   BS-Roformer + MDX23C, both license-unknown and flagged for possible
   replacement (`MODELS.md:48-49,65-77`). A separator swap shifts the training
   domain and invalidates the cache (train==inference breaks). Pin the separator
   as a hard model dependency, or budget to re-separate + retrain on a swap.
6. **Stem↔label timing**; not just constant latency; check for chunk/frame
   jitter; widen target smoothing on real charts (§6).
7. **Genre imbalance**; mitigated by stratified sampling + per-class weighting.
8. **HF resolution**; 128 mel bins under-resolve the 5–15 kHz sizzle band; start at 192–229 bins or add an HF-band/CQT feature (§3), not a late knob.
9. **Licensing (decided): hat-only, ADTOF stays.** ADTOF is *not* retired here;
   it keeps running for non-hat lanes, so its CC-BY-NC-SA *runtime* dependency
   remains (tracked in `MODELS.md`, out of scope). To preserve future
   optionality we still keep this model's *training data* clean: ADTOF dataset
   excluded (no articulation + NC anyway), pretraining only on STAR (CC-BY-4.0),
   E-GMD (CC-BY-4.0), and StemGMD (**verify** its Zenodo license). See §5.
10. **Corpus provenance.** This spec assumes 6289 real full mixes per the data
    owner; `AGENTS.md` documents a different/older ~2000-song synthesized-MIDI
    corpus plan. The data owner has confirmed real mixes; update `AGENTS.md` to
    match so this stops re-surfacing.

---

## 10. Phasing

Scope decided: **hat-only, ADTOF stays; no corpus audit (owner pre-filters).**

1. **Eval-set + scoring first** (gates the success criteria and de-risks
   everything): carve a real-audio open/closed-labelled held-out set; add
   articulation scoring (confusion matrix + per-class P/R + phantom rate) to the
   harness; record the current ADTOF+LLM baseline on it.
2. **Data pipeline:** confirm which RLRR field carries open/closed, then label
   extraction → `hh_closed` / `hh_open` events; ingest the owner's pre-filtered
   corpus; separation + mel cache; timing check; **new loaders for
   StemGMD/STAR**; extend existing E-GMD/MDB/IDMT loaders. (No ADTOF loader; no
   masking; every source is fully articulated.)
3. **Model + training harness:** CRNN (two hat output channels), multi-label
   BCE/focal loss, pretrain (clean synth pool) → fine-tune on the corpus.
4. **Pipeline integration:** `learned_onsets` (hat lane) replaces ADTOF's hat
   lane + `hihat_split`; ADTOF retained for other lanes; `_open_tail_filter`
   kept as backstop.
5. **Future (out of v1):** full-kit expansion to retire ADTOF entirely; optional
   Conformer core; optional extra input channels; optional ADT_STR augmentation;
   revisit pedal.

---

## References

- STAR Drums (TISMIR 2025): https://transactions.ismir.net/articles/10.5334/tismir.244
- StemGMD / LarsNet (2023): https://arxiv.org/abs/2312.09663 · Zenodo: https://zenodo.org/records/7860223
- ADTOF (ISMIR 2021): https://arxiv.org/abs/2111.11737 · datasets: https://zenodo.org/doi/10.5281/zenodo.10084510
- Synthetic-to-real gap in ADT (2024): https://arxiv.org/pdf/2407.19823
- Realistic synthetic data for ADT (2026): https://arxiv.org/pdf/2601.09520 · code: https://github.com/pier-maker92/ADT_STR
