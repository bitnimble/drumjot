# Models research

> Living reference for every ML model the transcriber uses or has evaluated,
> with a focus on **commercial licensing** and **lyrics alignment quality**.
> Compiled 2026-05-27 from a long investigation into lyrics-alignment cascade
> failures + a full-stack license audit. Update as models change.

**Headline takeaways:**
- We are migrating off CC-BY-NC models because they block any commercial /
  cost-recovery deployment (see [Licensing notes](#licensing-notes-cc-by-nc)).
- **Three confirmed CC-BY-NC blockers** in the stack today: MMS-300m
  (lyrics), madmom pretrained weights (beats), ADTOF Frame_RNN weights
  (drum onsets). Two more models have **no declared license** (jarredou
  separators).
- English lyrics alignment is already migrated to a clean Apache-2.0 model.
- Hardware ceiling: dev box is a **GTX 1660 Super, 6 GB VRAM, Turing
  (no Tensor Cores, no bf16)**. ~5 GB free during the alignment stage.
  This rules out 1B models at fp32 and makes fp16 mandatory for anything
  large.

---

## Complete list of models investigated

| Stage | Model | Params | License | Commercial? | Status |
|---|---|---|---|---|---|
| Lyrics align (en) | `facebook/wav2vec2-large-robust-ft-libri-960h`[^w2v-robust] | 317M | Apache 2.0[^apache] | ✅ | **In use** |
| Lyrics align (non-en) | `MahmoudAshraf/mms-300m-1130-forced-aligner`[^mms300m] | 315M | CC-BY-NC 4.0[^cc-by-nc] | ❌ | In use, **blocker** |
| Lyrics align (en alt) | `facebook/wav2vec2-large-960h-lv60-self`[^w2v-lv60] | 317M | Apache 2.0[^apache] | ✅ | Evaluated, not chosen |
| Lyrics align (en alt) | `facebook/wav2vec2-base-960h`[^w2v-base] | 95M | Apache 2.0[^apache] | ✅ | Evaluated; too weak |
| Lyrics align (en alt) | `facebook/hubert-large-ls960-ft`[^hubert] | 316M | Apache 2.0[^apache] | ✅ | Evaluated; no singing edge |
| Lyrics align (multiling) | `espnet/owsm_ctc_v4_1B`[^owsm-v4] | 1.0B | CC-BY-4.0[^cc-by] | ✅ | **Preferred non-en replacement** (deferred) |
| Lyrics align (multiling) | `espnet/owsm_ctc_v3.2_ft_1B`[^owsm-v32] / `v3.1_1B`[^owsm-v31] | 1.0B | CC-BY-4.0[^cc-by] | ✅ | Older OWSM-CTC |
| Lyrics align (multiling) | `FunAudioLLM/Fun-ASR-MLT-Nano-2512`[^funasr-nano] | 800M | Apache 2.0 (verify) | ✅? | Architecture unverified (likely Paraformer/non-CTC) |
| Lyrics align (multiling) | `FunAudioLLM/SenseVoiceSmall`[^sensevoice] | ~234M | FunASR License v1.1 (non-OSI) | ⚠️ | Commercial-with-attribution, legal review |
| Lyrics align (ja) | `nvidia/parakeet-tdt_ctc-0.6b-ja`[^parakeet-ja] | 600M | CC-BY-4.0[^cc-by] | ✅ | Best ja in isolation; NeMo loader |
| Lyrics align (ko) | `kresnik/wav2vec2-large-xlsr-korean`[^kresnik-ko] | 317M | Apache 2.0[^apache] | ✅ | **Cheapest experiment** (native HF) |
| Lyrics align (zh) | `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`[^grosman-zh] | 317M | Apache 2.0[^apache] | ✅ | Only clean Apache zh; mediocre |
| Lyrics align (en, patchwork) | `jonatasgrosman/wav2vec2-large-xlsr-53-english`[^grosman-en] | 317M | Apache 2.0[^apache] | ✅ | Patchwork-path en option |
| Lyrics align backbone | `facebook/wav2vec2-large-xlsr-53`[^xlsr53] | 317M | Apache 2.0[^apache] | ✅ | Pretrained-only, **no CTC head** |
| Lyrics align backbone | `facebook/wav2vec2-xls-r-300m/1b/2b`[^xlsr] | 0.3-2B | Apache 2.0[^apache] | ✅ | Pretrained-only, no multiling CTC head |
| Lyrics align backbone | `facebook/w2v-bert-2.0`[^w2vbert] | 580M | Apache 2.0[^apache] | ✅ | Pretrained-only, no CTC head |
| Lyrics align (rejected) | `facebook/mms-1b-all`[^mms1b] | 1B | CC-BY-NC 4.0[^cc-by-nc] | ❌ | Better than 300m but NC |
| Lyrics align (rejected) | `facebook/seamless-m4t-v2-large`[^seamless] | 2.3B | CC-BY-NC 4.0[^cc-by-nc] | ❌ | seq2seq, not CTC, + NC |
| Lyrics align (lead) | `Fhrozen/owsm_ctc_v3.2_aligner`[^fhrozen] | unknown | Apache 2.0 | ✅? | Empty README; inspect repo files |
| Lang detect (audio) | OpenAI Whisper[^whisper] via faster-whisper[^faster-whisper] | varies | MIT | ✅ | In use |
| Vocals isolation (lyrics) | `UVR-MDX-NET-Voc_FT.onnx`[^uvr] |; | MIT (UVR) | ✅ attrib | In use |
| Drum stems S1 | jarredou `BS-ROFO-SW-Fixed.ckpt`[^bsrofo] |; | **unknown** | ⚠️ | In use, **no license** |
| Drum stems S2 | jarredou MDX23C DrumSep 5-stem[^drumsep] |; | **none declared** | ⚠️ | In use, **no license** |
| Beats (default) | madmom pretrained `.pkl`[^madmom] |; | CC-BY-NC-SA 4.0[^cc-by-nc-sa] | ❌ | In use, **blocker** |
| Beats (alt) | Beat Transformer[^beat-transformer] |; | MIT | ✅ | Opt-in; clean replacement for madmom |
| Drum onsets | adtof_pytorch Frame_RNN weights[^adtof-pt] |; | CC-BY-NC-SA 4.0[^cc-by-nc-sa] | ❌ | In use, **blocker** |
| LLM filter | Anthropic Claude (API)[^anthropic] |; | API | ✅ | No local weights |

---

## Stem separation models

All separation runs through `audio-separator`[^audio-separator] (the library
code is **MIT**; only the model weights have separate licenses).

- **`UVR-MDX-NET-Voc_FT.onnx`**[^uvr]; vocals isolation for the lyrics path.
  **MIT** via Anjok07's UVR project. Commercial OK with attribution. Used in
  `config.py` / `provision.py`.
- **`BS-ROFO-SW-Fixed.ckpt`**[^bsrofo] (jarredou). Stage-1 drum extraction.
  **License literally "unknown"** in the HF front-matter. No grant = all
  rights reserved by default. **Follow-up:** contact jarredou for a written
  grant, or replace with Demucs v4[^demucs] (`htdemucs_ft`, MIT) for
  first-stage drum extraction.
- **MDX23C DrumSep 5-stem**[^drumsep] (`jarredou/models` GH release
  `drumsep_5stems_mdx23c_jarredou.ckpt`). Stage-2 5-stem drum split.
  **No license declared** anywhere in the repo/release. Same risk as above.
  Harder to replace; no obvious permissively-licensed 5-stem drum splitter
  exists.

**Verdict:** UVR clean; both jarredou separators are license-unknown and need
either author contact or replacement before commercial launch.

---

## Onset models

- **adtof_pytorch Frame_RNN**[^adtof-pt] (xavriley/ADTOF-pytorch, weights
  bundled in the wheel at
  `adtof_pytorch/data/adtof_frame_rnn_pytorch_weights.pth`); the **sole**
  drum-onset detector across every stem. Loaded in
  `app/pipeline/adtof_onsets.py`.
  - **License: CC-BY-NC-SA 4.0**[^cc-by-nc-sa], inherited from upstream
    MZehren/ADTOF[^adtof]. The pytorch port ships no LICENSE of its own; the
    weights are converted from MZehren's Keras checkpoints, so they're
    derivative works of CC-BY-NC-SA material.
  - **Commercial: NO; confirmed blocker.**
  - **Mitigation (hard):** (a) retrain Frame_RNN on a permissively-licensed
    drum dataset, (b) license MZehren weights commercially via the author,
    or (c) swap to an MIT/Apache drum-onset model. No off-the-shelf
    permissive drop-in exists. This is the **largest licensing lift** in the
    stack and the one that can't be indefinitely deferred if drum
    transcription is the product differentiator.

---

## Correction, quantisation and jitter models

- **Beat / downbeat tracking** has two backends:
  - **madmom 0.17.dev0 (default)**[^madmom]; source code is BSD-3, but the
    pretrained model files (`beats_lstm_*`, `downbeats_blstm_*`,
    `downbeats_bgru_*.pkl`) are **CC-BY-NC-SA 4.0**[^cc-by-nc-sa] (verified in
    the installed `madmom-*.dist-info/LICENSE`: "If you want to include any
    of these files… in a commercial product, please contact Gerhard
    Widmer."). The `RNNDownBeatProcessor` front-end loads these. **Confirmed
    blocker.** Note the `DBNDownBeatTrackingProcessor` post-processor is BSD
    *code* with no model file; only the RNN front-end weights are encumbered.
  - **Beat Transformer (alt, opt-in via `beat_tracker=beat_transformer`)**[^beat-transformer]. Zhao et al. 2022, **MIT**, checkpoints in the same MIT repo. Vendored
    in `app/vendor/beat_transformer/`.
  - **Mitigation (cheap):** flip the `settings.beat_tracker` default to
    `beat_transformer`. ~5-minute change; eliminates the madmom-weights
    blocker entirely. **The single easiest licensing win in the audit.**

---

## Lyrics models

### Current state (post-2026-05-27 changes)

Per-language CTC dispatch lives in `app/pipeline/lyrics_align.py`
(`_pick_alignment_model`):
- **English** → `facebook/wav2vec2-large-robust-ft-libri-960h`[^w2v-robust]
  (Apache 2.0, ~317M, multi-domain robust pretraining + LS-960 fine-tune).
- **Everything else** → MMS-300m[^mms300m] (the package default, CC-BY-NC,
  blocker).

Both share the package's runtime-appended `<star>` wildcard column and the
`preprocess_text(romanize=True, language=iso3)` tokenization. The aligner
package is `MahmoudAshraf97/ctc-forced-aligner`[^cfa] (BSD code).

### How ctc_forced_align works (mechanism)

1. `generate_emissions` runs the CTC model over the full audio (internal 30 s
   chunking + posterior stitching) → per-frame log-prob matrix `(T, V)`.
2. A `<star>` wildcard column is appended at runtime (`(T, V+1)`),
   model-agnostic; this is why any wav2vec2-family CTC checkpoint is a
   drop-in. `<star>` is meant to absorb non-lyric / OOV frames.
3. `preprocess_text` tokenizes the caller's lyric text (romanized Latin for
   most languages; char-level for jpn/chi) and inserts `<star>` between
   every word.
4. `get_alignments` runs Viterbi to find the single globally-optimal
   monotonic path through the trellis aligning tokens to frames.
5. `get_spans` + `postprocess_results` convert the path to per-word
   `{start, end, score}` where **`score` = mean log-prob along the path for
   that word's frames**. Sharply negative = Viterbi forced through
   low-probability frames.

### The cascade-failure investigation (key finding)

**Symptom:** ~95% correct alignment, but some words drift 1-2 s and cascade
forward before re-syncing. Worst case observed: the word "heart" stretched
across a ~4.7 s sustained vowel (`t=[55.68, 60.42]s`, `score=-18.1`,
`max_phoneme_prob=0.010`), pushing the next words 3 bars late.

**Root cause (confirmed, not a backbone problem):** CTC training produces
**peaky posteriors**, proven in Zeyer et al. "Why does CTC result in peaky
behavior?"[^zeyer]. On a sustained vowel the model emits ~1% phoneme
probability across the whole hold. Viterbi has nowhere to put those frames
(the `<star>` column gets near-zero posterior at speech-active frames in any
real wav2vec2 model, so the absorber path is non-viable), so it stretches the
most recent lyric word to cover the hold.

**Things we tried that did NOT work:**
- **Per-word `<star>` insertion**; no-op. The package *already* inserts a
  `<star>` between every word (`['<star>', 'No,', '<star>', 'sir', …]`,
  confirmed via `_log_token_sequence`). Our change just replicated it.
  Scores were identical to the digit.
- **Local re-alignment of low-score words** (`_repair_low_score_words`); no-op by Bellman's principle of optimality.[^bellman] A subpath of a
  globally optimal Viterbi path is itself optimal for the sub-trellis, so
  re-running Viterbi on the same emissions + tokens between the same
  endpoints returns the identical path. Every rerun came back with the same
  score to the digit (`-6.73 → -6.73`, etc.). **This code is dead weight;
  candidate for removal** unless repurposed (see below).

**Levers that WOULD work (in order of leverage):**
1. **Duration clamp (heuristic, zero compute, no new deps)**; when a word's
   duration exceeds N× median AND score < threshold AND max_phoneme_prob is
   tiny, clamp `end_sec` to `start + 1.5× median_word_duration`. Don't try to
   re-align the held audio; just refuse to claim the word covers it. UI shows
   the word briefly then a gap during the hold; which is what the audio
   actually contains. **The right next step for the user-visible problem.**
   (This is what `_repair_low_score_words` should be rewritten into.)
2. **Re-run `generate_emissions` on a fresh audio slice** (not an emissions
   slice); the model's LayerNorm/positional-encoding/chunking give genuinely
   different posteriors on a short clip vs. a slice of the long-form output.
   Might help; might be worse without context. Untested.
3. **Swap the aligner model**. English-specialized model concentrates its
   param budget instead of spreading across 1100+ languages. Done for
   English. Does NOT fix the held-vowel cascade (it's objective-level), but
   improves median score generally.
4. **Singing-specific aligner**; `qiuqiao/SOFA`[^sofa] (MIT) or
   `wolfgitpr/HubertFA`[^hubertfa] (Apache, HuBERT-based, "designed for
   singing voice"). No published benchmarks; would need an A/B on
   held-note-heavy songs. The principled fix per the research (Huang et al.
   "Less Peaky CTC by Label Priors"[^huang]) is retraining with a
   label-prior penalty; out of scope, but SOFA/HubertFA may have done
   something equivalent.

**Secondary finding:** `That's` recurred as a worst-scorer 5×. Likely an
**apostrophe-stripping tokenization artifact** (`preprocess_text` drops the
`'`, so the glottal `'s` the model hears has no token to align to). This is
a text-layer issue, independent of model choice; won't change with a
backbone swap.

### Diagnostics added (in `lyrics_align.py`)

All log under the `lyrics:` prefix at INFO:
- `_log_audio_stats`; duration, RMS, near-silent fraction, nan/inf.
- `_log_emissions_stats`; global nan/inf, mean/std, top-3 argmax classes,
  `<star>` margin (distinguishes "genuinely OOV" from "corrupt model load").
- `_log_emissions_windowed`; per-5s-window `max_phoneme_prob`, blank_frac,
  star_frac, top argmax classes. **Time axis derived from
  `audio_seconds / total_frames`, NOT the package's `stride`** (whose unit is
  version-dependent; ms in our version, samples in others; trusting it gave
  timestamps off by 1000×).
- `_log_word_score_diagnostics`; score distribution (min/p10/median/p90/max,
  count below threshold) + worst-10 words each annotated with
  max_phoneme_prob over its frame range.
- `_log_token_sequence`; dumps the token/`<star>` structure fed to Viterbi.

**Triage matrix** (read the two logs together):
- low score + low max_phoneme_prob → dead-audio span (cascade victim /
  instrumental). Fix upstream (separator / VAD) or duration-clamp.
- low score + high max_phoneme_prob → model heard a phoneme but it disagreed
  with the forced text token. Wrong language, missing `<star>`, or LRC
  mismatch.
- sustained high star_frac → instrumental section; words there are cascade
  victims by definition.

### Model comparison for English (singing vocals)

Research found **no published benchmark** comparing HuBERT-large[^hubert] vs
wav2vec2-large-robust[^w2v-robust] on singing (DALI/DSing/Jamendo). Key facts:
- The held-vowel cascade is **CTC-objective-level, not backbone-level**
  (Zeyer 2021[^zeyer]); no checkpoint swap fixes it.
- wav2vec2-large-robust's "robust" pretraining = Libri-Light + CommonVoice +
  Switchboard + Fisher. Switchboard/Fisher are **8 kHz telephone**; robust
  to narrowband noise, **not** to vocal sustain/reverb. Don't expect it to
  help singing specifically; chosen because it's the safe Apache-2.0 English
  default and concentrates capacity on English.
- HuBERT and wav2vec2 are behaviorally more similar than commonly believed
  (arXiv:2508.08110[^hubert-iter]); differences trace to HuBERT's iterative
  pseudo-labeling, not the objective. Don't expect HuBERT to fix peakiness.
- Singing-ALT context: Ou et al. (ISMIR 2022)[^ou] is the canonical wav2vec2
  lyrics-transcription adaptation paper (code: ALT_SpeechBrain[^ou-repo]) but
  contains no held-vowel boundary analysis.

### fp16 note (hardware-specific)

We reverted fp16 (commit `37ba1ae`) due to instability on MMS-300m
(LayerNorm/softmax over fp16 activations → nan/inf → degenerate Viterbi).
On the **GTX 1660 Super (Turing, no Tensor Cores, no bf16)**[^turing], fp16
runs but without the throughput win and **bf16 (the usual stability
workaround) is unavailable**. Any future large model on this box must run
fp16 and prove fp16-stable via the `_log_emissions_stats` nan/inf check.
Currently English runs fp16 on CUDA, fp32 on CPU (`_load_ctc_aligner`).

---

## Regarding multi-lingual lyric alignment

### The core constraint

Mixed-language handling is a **tokenization-and-vocab problem, not an
alignment-algorithm problem**. You need either (a) one CTC model whose vocab
spans every language in a song, or (b) compose multiple language-specific
models into one Viterbi pass; and (b) is research-grade hard. (a) is a model
swap.

There is **no Apache-2.0 multilingual CTC model** covering en/ja/ko/zh. MMS
(CC-BY-NC)[^mms300m], Canary (CC-BY-NC)[^canary], Seamless (CC-BY-NC)[^seamless]
are all blocked. The only permissive multilingual CTC is
**OWSM-CTC (CC-BY-4.0)**[^owsm-v4] (OWSM v3.1 paper[^owsm-paper]), from
ESPnet/CMU; not Meta. XLS-R[^xlsr] /
XLSR-53[^xlsr53] / w2v-bert[^w2vbert] backbones are Apache but ship **no
multilingual CTC head** (Meta only released English-only and
pretrained-backbone weights under Apache; the multilingual+fine-tuned combo
is always CC-BY-NC).

### Why OWSM-CTC v4 was deferred (NOT rejected)

It's the **preferred non-English replacement**, tracked in the
`lyrics_align.py` module docstring. Deferred because:
- Not native HF `transformers`; loads via `espnet`[^espnet] +
  `espnet_model_zoo`, so `load_alignment_model` can't take it directly. Needs
  a ~50-line adapter to expose `(emissions, stride)`.
- Needs a tokenizer swap; uses multilingual BPE, not the romanized/char
  output of `preprocess_text`. Word-count partitioning needs BPE→word mapping.
- The English fix was the urgent, clean piece; OWSM was the larger deferred
  one.
- **No smaller OWSM-CTC variant exists**; every CTC checkpoint
  (v3.1[^owsm-v31]/v3.2[^owsm-v32]/v4[^owsm-v4]) is 1B. The smaller v4
  variants (`owsm_v4_small_370M`[^owsm-small], `owsm_v4_base_102M`[^owsm-base])
  are **encoder-decoder seq2seq, not CTC**; unusable for our pipeline.

### OWSM-CTC v4 1B on the 6 GB box (the binding constraint)

| Mode | Weights | Peak | 6 GB verdict |
|---|---|---|---|
| fp32 batch=4 | ~4 GB | 6.5-8 GB | ❌ OOM |
| fp32 batch=2 | ~4 GB | 5.5-6.5 GB | ❌ likely OOM (display takes 0.5-1 GB) |
| fp32 batch=1 | ~4 GB | 5-6 GB | ⚠️ edge |
| **fp16 batch=1-2** | ~2 GB | 3.5-5 GB | ✅ only viable config |

fp16 is **mandatory** on this hardware, and fp16-stability for OWSM's
E-Branchformer encoder is **unproven** (could differ from MMS's transformer
stack; test with the nan/inf check first). Reported quality (UNVERIFIED; these per-language CER figures are NOT on the model card[^owsm-v4]; they
appear to come from the OWSM-CTC paper[^owsm-ctc-paper] but I could not
confirm the exact values, treat as ballpark): ja CER ~7.9 / ko CER ~16.7 /
zh CER ~6.4; roughly MMS-equivalent.

**Recommended experiment sequencing on the 1660 Super:** throwaway script
that loads OWSM at fp16, runs emissions on a 30 s clip, checks nan/inf +
`torch.cuda.max_memory_allocated()`. ~1 hour to know if it's viable before
building the full integration.

### Mixed-language scenario support

Three scenarios in increasing difficulty, and which models can handle them:

1. **Sprinkle** (English song, few foreign words); works today via uroman
   romanization (approximate); OWSM best (native tokens). No architecture
   change.
2. **Line/verse-level** (kpop: whole EN verse, whole KO verse); multilingual
   single-pass model (MMS/OWSM/Fun-ASR) handles in 1 pass, no change.
   Patchwork per-language needs a two-pass coarse-then-refine + seam-stitching
   layer (re-introduces the per-segment windowing problem we left whisperx to
   escape; ~1-2 weeks).
3. **Word-level code-switching** (KO rap with EN loanwords mid-line); **only
   multilingual-native-vocab models work** (OWSM, Fun-ASR-if-CTC). Patchwork
   can't reach this (per-word model switching within one Viterbi path is
   research-grade). MMS does it approximately via romanization.

| Model | Sprinkle | Line-level | Word-level |
|---|---|---|---|
| wav2vec2-large-robust (en)[^w2v-robust] | 🟡 | ❌ | ❌ |
| MMS-300m (CC-BY-NC)[^mms300m] | ✅ | ✅ | 🟡 |
| **OWSM-CTC v4**[^owsm-v4] | ✅ | ✅ | ✅ |
| Patchwork per-language | 🟡 | 🟡 (2-pass) | ❌ |
| Fun-ASR-Nano (if CTC)[^funasr-nano] | ✅ | ✅ | ✅ |

**Key implication:** the OWSM integration work (BPE tokenizer + adapter) is
the **same regardless of which scenario you target**; it's prerequisite for
all three. The lever is the model swap, not the architecture. If
mixed-language is on the roadmap at all, OWSM-CTC v4 is the bet that scales
monotonically.

### Cheapest experiments to run now

1. `kresnik/wav2vec2-large-xlsr-korean`[^kresnik-ko] (Apache, native HF,
   ~1.2 GB fp32, fits 6 GB with margin); one-line swap in
   `_pick_alignment_model` for `ko`. Validates the patchwork path. CER 1.8 on
   Zeroth-KR (beats MMS).
2. `Fhrozen/owsm_ctc_v3.2_aligner`[^fhrozen]; inspect repo files; could be a
   smaller distilled CTC aligner or just a re-export. 10-min check.
3. `FunAudioLLM/Fun-ASR-MLT-Nano-2512`[^funasr-nano]; check `config.json`
   `architectures` field; if it exposes a CTC head it's the cheapest
   multilingual path (Apache, native HF, 800M). If Paraformer/CIF → skip.
   10-min check.

### Language detection (current)

`_detect_language_from_text` uses **majority-by-character-count** routing
(changed 2026-05-27 from priority-order). Each alphabetic char votes; most
votes wins; ties → `en`. Ambiguous CJK (kanji/hanzi) resolves to ja/zh via a
pre-pass: any kana → ja; else any simplified-Chinese marker → zh; else ja
(J-pop bias). This handles the Scenario-1 sprinkle correctly (a few kana in
an English song no longer flip the whole song to MMS). Caller-pinned
`language` still overrides. Audio-based fallback uses faster-whisper[^faster-whisper].

---

## Licensing notes (CC-BY-NC)

**CC-BY-NC 4.0 "NonCommercial" = "not primarily intended for or directed
towards commercial advantage or monetary compensation."**[^cc-by-nc] The "or
monetary compensation" clause is the trap. (Not legal advice; consult an IP
lawyer. CC's own NonCommercial interpretation guidance is the starting
point.[^cc-nc-interp])

- **Cost-recovery ("charge only to cover running costs")**. NOT safe on the
  conservative reading. CC 4.0's "or monetary compensation" wording reads
  strict on its face. **Case law is sparse and mixed, not clearly strict**; don't cite it as settled:
  - *Great Minds v. FedEx Office* (2d Cir. 2018)[^great-minds] actually cuts
    the *other* way: it held a commercial copy shop (FedEx) acting as the
    **agent** of NC licensees (school districts) did **not** breach the NC
    term. It's permissive toward commercial agents of NC users, not a
    strict-context ruling. Useful to know precisely because it's often
    mis-cited as strict.
  - The German *Deutschlandradio* matter (LG Köln Az. 28 O 232/13,
    2014)[^deutschlandradio] initially read NC as "purely private use only,"
    but the OLG Köln on appeal (Nov 2014) narrowed the holding to
    unauthorized image **alteration**, not commercial use; so it is weak
    authority for a strict NC reading.
  - Net: the *text* is strict; the *case law* doesn't reliably back a
    cost-recovery carve-out either way. Treat cost-recovery as unsafe by
    default and get a lawyer's read.
- **Cross-subsidy ("free product A subsidized by paid product B")**. NOT
  safe. The subsidy itself is the smoking gun: A exists *because of* B's
  commercial activity → A is "directed towards commercial advantage." Mozilla
  (free Firefox + paid services, non-profit foundation)[^mozilla] **could
  not** legally do this with CC-BY-NC content; which is why their stack is
  MPL/Apache/MIT. Labels/corporate structure alone don't create the
  separation; genuine arm's-length entities with no shared
  infra/branding/funding might, but a *deliberate* subsidy relationship can't.
- **Donations / voluntary tipping**; generally OK.
- **Meta specifically** hasn't published its MMS NC interpretation; given the
  Llama license disputes, assume strict.
- **Conclusion:** treat CC-BY-NC as "no commercial deployment under any
  structure." Get one hour with an IP lawyer before any business-model
  decision. Replacing the 3 NC deps is bounded work; an after-launch ruling
  is unbounded risk.

### Attribution-required (clean licenses, still need credit page)

UVR/Anjok07 (MIT)[^uvr], Beat Transformer/Zhao (MIT)[^beat-transformer],
whisperX[^whisperx] / faster-whisper[^faster-whisper] / OpenAI Whisper
(MIT/BSD)[^whisper], `facebook/wav2vec2-large-robust-ft-libri-960h`
(Apache 2.0)[^w2v-robust].

### Migration priority

1. **Beat tracker default → `beat_transformer`**[^beat-transformer] (~5 min,
   clears madmom NC).
2. **OWSM-CTC v4 1B for non-en lyrics**[^owsm-v4] (~1-2 days, clears MMS NC;
   fp16 only on 6 GB box).
3. **Contact jarredou** re: separator licenses[^bsrofo] (async; saves a
   Demucs migration if friendly).
4. **ADTOF drum-onset replacement**[^adtof] (weeks; the hard one; retrain or
   re-architect; no off-the-shelf permissive drop-in).

---

## References

### Models (HuggingFace)

[^w2v-robust]: facebook/wav2vec2-large-robust-ft-libri-960h; <https://huggingface.co/facebook/wav2vec2-large-robust-ft-libri-960h>
[^mms300m]: MahmoudAshraf/mms-300m-1130-forced-aligner (CC-BY-NC, inherits Meta MMS); <https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner>
[^w2v-lv60]: facebook/wav2vec2-large-960h-lv60-self; <https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self>
[^w2v-base]: facebook/wav2vec2-base-960h; <https://huggingface.co/facebook/wav2vec2-base-960h>
[^hubert]: facebook/hubert-large-ls960-ft; <https://huggingface.co/facebook/hubert-large-ls960-ft>
[^owsm-v4]: espnet/owsm_ctc_v4_1B; <https://huggingface.co/espnet/owsm_ctc_v4_1B>
[^owsm-v32]: espnet/owsm_ctc_v3.2_ft_1B; <https://huggingface.co/espnet/owsm_ctc_v3.2_ft_1B>
[^owsm-v31]: espnet/owsm_ctc_v3.1_1B; <https://huggingface.co/espnet/owsm_ctc_v3.1_1B>
[^owsm-small]: espnet/owsm_v4_small_370M (seq2seq, not CTC); <https://huggingface.co/espnet/owsm_v4_small_370M>
[^owsm-base]: espnet/owsm_v4_base_102M (seq2seq, not CTC); <https://huggingface.co/espnet/owsm_v4_base_102M>
[^funasr-nano]: FunAudioLLM/Fun-ASR-MLT-Nano-2512; <https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512>
[^sensevoice]: FunAudioLLM/SenseVoiceSmall; <https://huggingface.co/FunAudioLLM/SenseVoiceSmall>
[^parakeet-ja]: nvidia/parakeet-tdt_ctc-0.6b-ja; <https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja>
[^kresnik-ko]: kresnik/wav2vec2-large-xlsr-korean; <https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean>
[^grosman-zh]: jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn; <https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn>
[^grosman-en]: jonatasgrosman/wav2vec2-large-xlsr-53-english; <https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english>
[^xlsr53]: facebook/wav2vec2-large-xlsr-53 (pretrained backbone, no CTC head); <https://huggingface.co/facebook/wav2vec2-large-xlsr-53>
[^xlsr]: facebook/wav2vec2-xls-r-300m (also -1b, -2b); <https://huggingface.co/facebook/wav2vec2-xls-r-300m>
[^w2vbert]: facebook/w2v-bert-2.0; <https://huggingface.co/facebook/w2v-bert-2.0>
[^mms1b]: facebook/mms-1b-all (CC-BY-NC); <https://huggingface.co/facebook/mms-1b-all>
[^seamless]: facebook/seamless-m4t-v2-large (CC-BY-NC, seq2seq); <https://huggingface.co/facebook/seamless-m4t-v2-large>
[^canary]: nvidia/canary-1b-v2 (CC-BY-NC); <https://huggingface.co/nvidia/canary-1b-v2>
[^fhrozen]: Fhrozen/owsm_ctc_v3.2_aligner (empty README, needs inspection); <https://huggingface.co/Fhrozen/owsm_ctc_v3.2_aligner>
[^bsrofo]: jarredou/BS-ROFO-SW-Fixed (license "unknown"); <https://huggingface.co/jarredou/BS-ROFO-SW-Fixed>

### Projects / repos

[^cfa]: MahmoudAshraf97/ctc-forced-aligner; <https://github.com/MahmoudAshraf97/ctc-forced-aligner>
[^uvr]: Anjok07/ultimatevocalremovergui (UVR, MIT); <https://github.com/Anjok07/ultimatevocalremovergui>
[^drumsep]: jarredou/models DrumSep release (no LICENSE); <https://github.com/jarredou/models/releases/tag/DrumSep>
[^adtof]: MZehren/ADTOF LICENSE (CC-BY-NC-SA 4.0); <https://github.com/MZehren/ADTOF/blob/master/LICENSE>
[^adtof-pt]: xavriley/ADTOF-pytorch; <https://github.com/xavriley/ADTOF-pytorch>
[^madmom]: CPJKU/madmom LICENSE (BSD code; CC-BY-NC-SA model files); <https://github.com/CPJKU/madmom/blob/main/LICENSE>
[^beat-transformer]: zhaojw1998/Beat-Transformer (MIT); <https://github.com/zhaojw1998/Beat-Transformer>
[^sofa]: qiuqiao/SOFA singing-oriented forced aligner (MIT); <https://github.com/qiuqiao/SOFA>
[^hubertfa]: wolfgitpr/HubertFA singing forced aligner (Apache); <https://github.com/wolfgitpr/HubertFA>
[^ou-repo]: guxm2021/ALT_SpeechBrain (Apache); <https://github.com/guxm2021/ALT_SpeechBrain>
[^faster-whisper]: SYSTRAN/faster-whisper (MIT); <https://github.com/SYSTRAN/faster-whisper>
[^whisper]: openai/whisper (MIT); <https://github.com/openai/whisper>
[^whisperx]: m-bain/whisperX (BSD); <https://github.com/m-bain/whisperX>
[^audio-separator]: nomadkaraoke/python-audio-separator (MIT); <https://github.com/nomadkaraoke/python-audio-separator>
[^demucs]: facebookresearch/demucs (MIT); <https://github.com/facebookresearch/demucs>
[^espnet]: espnet/espnet (OWSM toolkit); <https://github.com/espnet/espnet>
[^anthropic]: Anthropic API (Claude); <https://docs.anthropic.com/>

### Papers

[^zeyer]: Zeyer, Schlüter, Ney, "Why does CTC result in peaky behavior?" (2021), arXiv:2105.14849; <https://arxiv.org/abs/2105.14849>
[^huang]: Huang et al., "Less Peaky and More Accurate CTC Forced Alignment by Label Priors" (ICASSP 2024), arXiv:2406.02560; <https://arxiv.org/abs/2406.02560>
[^ou]: Ou, Gu, Wang, "Transfer Learning of wav2vec 2.0 for Automatic Lyric Transcription" (ISMIR 2022), arXiv:2207.09747; <https://arxiv.org/abs/2207.09747>
[^hubert-iter]: "Iterative refinement, not training objective, makes HuBERT behave differently from wav2vec 2.0" (2025), arXiv:2508.08110; <https://arxiv.org/abs/2508.08110>
[^owsm-paper]: Peng et al., "OWSM v3.1: Better and Faster Open Whisper-Style Speech Models" (2024), arXiv:2401.16658; <https://arxiv.org/abs/2401.16658>
[^owsm-ctc-paper]: Peng et al., "OWSM-CTC: An Open Encoder-Only Speech Foundation Model for Speech Recognition, Translation, and Language Identification" (ACL 2024), arXiv:2402.12654 (likely source of the per-language CER figures; exact values not independently confirmed); <https://arxiv.org/abs/2402.12654>

### Licenses & legal

[^apache]: Apache License 2.0; <https://www.apache.org/licenses/LICENSE-2.0>
[^cc-by]: CC BY 4.0 legal code; <https://creativecommons.org/licenses/by/4.0/legalcode>
[^cc-by-nc]: CC BY-NC 4.0 legal code (NonCommercial definition); <https://creativecommons.org/licenses/by-nc/4.0/legalcode>
[^cc-by-nc-sa]: CC BY-NC-SA 4.0 legal code; <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>
[^cc-nc-interp]: Creative Commons, "NonCommercial interpretation" guidance; <https://wiki.creativecommons.org/wiki/NonCommercial_interpretation>
[^great-minds]: Great Minds v. FedEx Office & Print Servs., Inc., 886 F.3d 91 (2d Cir. 2018). NOTE: held FedEx (commercial agent of NC licensees) did NOT breach the NC term; permissive, not strict; often mis-cited. Justia: <https://law.justia.com/cases/federal/appellate-courts/ca2/17-808/17-808-2018-03-21.html>
[^deutschlandradio]: Deutschlandradio CC-BY-NC-SA matter, LG Köln Az. 28 O 232/13 (2014). NOTE: LG Köln read NC as "purely private use," but OLG Köln on appeal (Nov 2014) narrowed the holding to unauthorized image alteration, not commercial use; weak authority for a strict NC reading. English summary (Council of Europe / Merlin): <https://merlin.obs.coe.int/article/6880>
[^mozilla]: Mozilla licensing policy (MPL/Apache/MIT); <https://www.mozilla.org/en-US/MPL/>

### Hardware

[^bellman]: Bellman's principle of optimality (dynamic programming; why a subpath of an optimal Viterbi path is itself optimal); <https://en.wikipedia.org/wiki/Bellman_equation#Principle_of_optimality>
[^turing]: NVIDIA Turing (TU116, GTX 1660 Super), no Tensor Cores, no bf16, <https://en.wikipedia.org/wiki/GeForce_16_series>
