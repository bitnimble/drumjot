# ParaDB map alignment scorer (spec)

Status: design, approved 2026-05-30. Not yet implemented.

This is the **v1 implementation spec** for the scorer designed in
`research/midi-audio-alignment-score.md`. That research doc remains the
authoritative description of the scoring *algorithm* (§5) and the full
correction *vision* (§8). This spec records the decisions that pin it to
a concrete, shippable v1: the deployment shape (batch + web), the input
formats (ParaDB packs first), and the scope cuts (global align only).
Where the two disagree, this spec wins for v1; the research doc has been
annotated with the same decisions.

## 1. Purpose

Produce a scalar **0-100 quality score** for a drum chart, measuring how
faithfully its notated onsets line up with the actual drum audio. The
driving use case is **corpus filtering**: score the full ParaDB map
corpus (~6k songs / ~15k difficulty tracks) so low-quality charts can be
excluded from the training set of a future drum-transcription model.

The web endpoint (§9) exists only as a manual test harness during
development; the real consumer is a headless batch job that imports the
core function. Everything is therefore designed batch-first: the scoring
core is a plain importable Python function, pure where it can be, with
the web layer a thin wrapper.

The chart is **external** (a third-party ParaDB map, or an uploaded
MIDI), never our own `prediction.mid`. That independence is what makes
reusing the pipeline's own ADTOF onset detector as the audio reference
legitimate.

## 2. The metric you filter on

The headline is **`score_corrected`**: the rigid soft-F1 score
(research §5) re-evaluated *after* a global offset + tempo alignment
(research §8 tiers 0-1). Filtering on the corrected score measures
**notation faithfulness independent of a fixable global drift** -- a
chart authored 2 s late, or at 99.4% tempo, is not wrongly rejected for
an error that a single affine transform removes.

The raw (pre-correction) `score` is also reported; a large
`score_corrected - score` gap is itself a signal ("this chart has gross
global drift"). The recovered `offset_sec` / `tempo_ratio` are red-flag
diagnostics AND the exact transform a batch run applies to emit cleaned
`(audio, chart)` training pairs (`corrected_onsets_by_lane`).

## 3. Scope

### In scope (v1)

- Rigid soft-F1 score (research §5): per-lane soft-Gaussian
  monotonic-injective DP match → soft P/R/F1 → weighted roll-up.
- Global correction, tiers 0-1 only (research §8.0-8.1): cross-correlation
  offset, then bounded affine tempo (Huber least-squares on matched
  pairs). Reports `(a, b)` and the corrected chart times.
- ParaDB `.zip` map packs as the primary input; MIDI + audio upload as a
  secondary test input.
- A streaming `POST /score` endpoint and a minimal frontend upload.

### Non-goals (v1)

- **No per-note ICP nudge** (research §8.2) and **no cleaned-`.mid`
  export.** Global align is enough to make the score fair and to clean
  training pairs; per-note warping is a later step and is the bulk of the
  correction effort.
- **No CLI.** Manual testing goes through the web UI first; a batch
  driver script comes later and imports `score_map.py` directly.
- **No new heavy dependencies.** `mido`, `librosa`, `scipy`, `numpy` are
  already in `pyproject.toml`; zip handling uses the stdlib `zipfile`.
- **No symbolic/musical judgement** (no "is this the right groove"),
  purely onset timing + presence.
- **No reuse of `geometric_snap.py`'s DP.** It is a one-sided
  onset→slot assignment; this scorer is a two-sided Needleman-Wunsch
  match. They share only the monotonic-injective *idea* and the
  pure-module style contract; forcing a shared core loop would hurt both.

## 4. Architecture: new `transcriber/app/scoring/` package

Pure algorithm separated from I/O, mirroring `geometric_snap.py` /
`benchmarks/core/score.py`'s purity contract so the cores test against
synthetic lists and never touch audio.

| Module | Responsibility | Purity |
|---|---|---|
| `lanes.py` | GM-note→pitch (port of `src/midi/gm.ts::GM_PERCUSSION`) + Paradiddle-class→pitch (port of `src/rlrr/drums.ts::CLASS_TO_DRUM`) + pitch→5-lane fold | pure |
| `rlrr_read.py` | parse `.rlrr` JSON (encoding sniff, port of `paradb.ts::decodeRlrrText`) → per-lane onset seconds from `event.time` (`eventTimeSeconds`, number-or-string); resolve each event's instrument-instance `name` to a class via the `instruments[]` array; expose `audioFileData` track refs | pure-ish |
| `paradb_read.py` | `.zip` pack reader via stdlib `zipfile`: pick best-difficulty `.rlrr`, return chart onsets + song/drum audio bytes | pure-ish |
| `midi_read.py` | `mido` whole-file read → tempo-aware raw onset seconds, ch-9 + all-channel fallback, GM fold (research §4) | pure-ish |
| `audio_onsets.py` | drum stem → ADTOF reference onsets, all 5 lanes from one inference | I/O, heavy |
| `alignment.py` | the scorer: soft-Gaussian monotonic-injective DP + per-lane soft P/R/F1 + roll-up (research §5) | pure |
| `correction.py` | tier-0 cross-correlation offset + tier-1 bounded affine tempo → `(a, b)` + corrected times (research §8.0-8.1) | pure |
| `score_map.py` | orchestrator / batch entry point | I/O |
| `models.py` | `AlignmentResult` dataclass | pure |

This deviates from research §9, which placed the modules in
`benchmarks/core/` for an offline CLI. Since v1 is web + batch *inside*
the app and depends on `app/pipeline` stages (`separate`,
`adtof_onsets`), `app/scoring/` is the cohesive home.

**Reuse vs port.** Reused as-is: `separate.Separator.run_stems_all`
(`app/pipeline/separate.py:241`), `adtof_onsets` internals (refactored,
below), `OnsetCandidate` (`app/models.py`), the streaming-endpoint +
`_gpu_lock` / `gpu_park.park_for_transcribe` machinery
(`app/main.py`). Ported from TypeScript (single source of truth stays
the `.ts`, guarded by drift tests, §11): the `gm.ts` GM table, the
`drums.ts` class table, `decodeRlrrText`, and the difficulty-selection
logic in `paradb.ts` (re-expressed on stdlib `zipfile`).

## 5. Inputs and parsing

### 5.1 Lane vocabulary

Five lanes, matching ADTOF's output classes
(`adtof_onsets.py:79` `_LANE_FOR_PITCH`): `k` kick, `s` snare, `t` toms,
`h` hi-hat, `cy` cymbals (ride + crash **merged**. ADTOF has no separate
ride/crash class). Both the GM table and the Paradiddle class table fold
DSL pitches into these five: `f`(floor tom)→`t`, `c`(crash)+`d`(ride)→`cy`,
hi-hat variants→`h`. DSL pitches with no ADTOF lane (`p` clap, `b`
perc) are dropped and counted in `unmapped_notes`. The GM source of
truth is `gm.ts`, **not** the stale 3-class
`benchmarks/core/classes.py::GM_PITCH_TO_CLASS`.

### 5.2 ParaDB pack (`paradb_read.py`, primary input)

A `.zip` containing one or more `.rlrr` charts (one per difficulty) plus
the audio they reference. Mirror `paradb.ts::loadParadbZip` on stdlib
`zipfile`:

1. List entries; pick the `.rlrr` with the highest
   `recordingMetadata.complexity`, ties broken by difficulty word in the
   filename (Expert>Hard>Medium>Easy). (One score per pack = the most
   complete chart. Per-difficulty scoring is a trivial loop later.)
2. Decode the chosen `.rlrr` text (UTF-8 / BOM / UTF-16 LE+BE sniff) and
   `json.loads`.
3. Extract chart onsets via `rlrr_read`.
4. Resolve `audioFileData.songTracks` (full mix) and `drumTracks`
   (drums-only) to zip entries by basename (case-insensitive); extract
   their bytes. The drum track, when present, is the preferred audio
   reference (§6).

### 5.3 RLRR chart onsets (`rlrr_read.py`)

`.rlrr` `events` are `{name, vel, loc, time}` where **`time` is the
absolute recording seconds** (number or 4-decimal string; `loc` is a
hit-zone index, always 0, NOT a timestamp). `time` shares the clock of
the audio tracks in the pack, so no tempo math is needed (contrast
MIDI). Per event: resolve `name` (an instrument *instance* name) to its
class via the `instruments[]` array (`{name, class}`), falling back to
the `BP_<Class>_C_<idx>` regex (`drums.ts::instanceNameToClass`); class
→ pitch → lane. Emit per-lane ascending onset-second lists. Unknown
classes counted, not fatal. **Use `event.time`; never the quantized
grid** `rlrr_to_jot` builds.

### 5.4 MIDI (`midi_read.py`, secondary test input)

Iterate **`for msg in mido.MidiFile(path)`** (merged, tempo-aware,
`msg.time` already in seconds, never iterate `mid.tracks`, the
multi-track-tempo bug from research §4). Take note-ons (velocity > 0),
GM-fold pitch → lane. Channel policy: prefer channel 9; if no channel-9
note-ons, fall back to all channels; if still no drum-mapped notes, raise
`"no drum channel found"`.

## 6. Audio reference (`audio_onsets.py`)

Goal: per-lane reference onset seconds from the real drum audio.

1. **Obtain a drum stem.** If the pack shipped a `drumTracks` entry, use
   it directly and **skip separation**, it is already an
   in-distribution full-drum mix, the cheap path that matters at 15k-track
   scale. Otherwise run `Separator.run_stems_all(audio, work_dir)` on the
   song track (or the MIDI-path uploaded audio) and take `.drum_stem`.
2. **Detect all 5 lanes from one ADTOF inference.** Add
   `detect_all_lanes_adtof(drum_stem) -> dict[lane, list[float]]`,
   refactored out of `detect_onsets_adtof` (`adtof_onsets.py:423`): run
   the CRNN once, peak-pick each of the 5 activation lanes, refine times
   against the audio envelope (existing logic). This avoids the
   pipeline's per-instrument approach, which needs the second separation
   `run_stems_per`, unnecessary for a timing reference, since ADTOF was
   trained on full drum mixes and resolves all 5 lanes from one.

**Reference set: raw ADTOF, no LLM prune.** The pipeline tunes onsets
high-recall ("detect hot, the LLM prunes"), so raw detections carry
phantom hits a clean chart will "miss" → a recall drag (research §4
reference-set bias). We accept it: the LLM prune is far too expensive
across 15k tracks, and because the *same detector* biases every chart
identically, `score_corrected` stays a valid **relative** ranking for
thresholding. Documented, with precision-weighting retained so the score
still discriminates.

## 7. Scoring (`alignment.py`)

Exactly research §5; restated for self-containment. Per lane, chart
onset times `M` and audio onset times `A`:

```
reward(i, j) = exp(-(m_i - a_j)^2 / (2 σ²))   if |m_i - a_j| ≤ B
             = disallowed                      otherwise

dp[i][j] = max(dp[i-1][j],                    # chart i unmatched (insertion)
               dp[i][j-1],                     # audio j unmatched (deletion)
               dp[i-1][j-1] + reward(i,j))     # match  (only if within band B)
TPQ = dp[I][J]
soft_precision = TPQ / I ; soft_recall = TPQ / J ; soft_f1 = 2PR/(P+R)
```

Monotonic + injective (free gaps both sides), restricted to the band for
`O(N·band)` cost. Defaults `B = 50 ms` (correspondence gate),
`σ = 25 ms` (credit kernel), separate knobs on purpose (§3 of research).
Edge cases: empty-both lane skipped; one-sided lane → `f1 = 0`;
`TPQ = 0` → `f1 = 0` (guard 0/0). Roll-up: `f1_macro` (mean of lanes),
`f1_weighted` (by audio-onset count `J`); `score = round(100·f1_weighted)`.

> **Calibrate `σ` to the detector, not blindly to `B`.** ADTOF on stems
> smears transients by tens of ms even after envelope refine; a perfect
> chart can plateau below 100 for reasons that are the reference's error.
> Tune `σ` against the detector's empirical timing variance (research §6).

## 8. Global correction (`correction.py`)

Tiers 0-1 of research §8, no ICP loop (correspondence is solved once per
tier against the frozen DP match):

- **Tier 0, offset.** Cross-correlate summed per-lane onset impulse
  trains (chart vs audio), rasterized to ≤10 ms bins, over a ±2-bar bound;
  argmax lag = `b`. Threshold-free; pulls the chart inside the ±B band so
  the soft score becomes a usable objective. **NB** for drum-dense onsets
  the cross-correlation argmax is noisy and overshoots (see the empirical
  note in research §8.0); prefer / cross-check against the **median
  nearest-peak offset** there, as implemented in `training/scripts/eval_paradb.py`.
- **Tier 1, affine tempo.** With the post-offset DP correspondence, a
  robust (Huber) least-squares fit `t' = a·t + b` on matched pairs.
  **Require ≥3 matched pairs and bound `a ∈ [0.5, 2.0]`**; otherwise
  treat as no tempo correction (a 1-pair or unbounded fit can collapse
  the chart onto coincidences and inflate the corrected score). A large
  `|a − 1|` is a red-flag diagnostic, not a free pass.

`score_corrected` = the §7 score re-evaluated at the corrected times.
Because each tier fits on a *frozen* correspondence and the score is
measured at the resulting positions (not optimized per-note), the
corrected number cannot be inflated by per-note freedom.

## 9. Outputs (`models.py`)

```
AlignmentResult:
  score: int                       # round(100·f1_weighted), pre-correction
  score_corrected: int             # HEADLINE filter metric, post global-align
  f1_macro: float                  # corrected
  f1_weighted: float               # corrected (basis of score_corrected)
  f1_weighted_raw: float
  per_lane: {lane: {soft_f1, soft_precision, soft_recall, n_chart, n_audio}}  # corrected
  offset_sec: float                # tier-0 b
  tempo_ratio: float               # tier-1 a (1.0 = none)
  matched_pairs: int               # pairs the affine fit used (trust signal)
  corrected_onsets_by_lane: {lane: [float]}   # chart times after t' = a·t + b
  unmapped_notes: int
  audio_reference: "drum_track" | "separated"
  separation_skipped: bool
```

## 10. Web endpoint + frontend

`POST /score` (`app/main.py`), reusing the existing
`StreamingResponse` NDJSON pattern (`stage`/`substage`/`result`/`error`
envelopes) and the `_gpu_lock` + `gpu_park.park_for_transcribe`
serialization (it uses the drum models). Accepts either a `.zip` pack, or
a `.mid` plus an audio `UploadFile`. Frontend: one "Score a map" item in
`toolbar.tsx` (sibling to the existing Load-ParaDB / Load-MIDI items)
that POSTs via a `transcriber.ts` client method and renders the
`AlignmentResult` (headline + per-lane table + diagnostics). Deliberately
bare, it is test scaffolding, not a product surface.

## 11. Error handling

- Pack with no `drumTracks` → separate the song track; no song track
  either → `"no audio in pack"`.
- MIDI with no channel-9 drums → all-channel fallback; still nothing
  drum-mapped → `"no drum channel found"`.
- Unmapped GM notes / unknown Paradiddle classes → counted in
  `unmapped_notes`, never fatal.
- Lane empty on one side → `f1 = 0`; empty on both → skipped.
- ADTOF / torch failure → surfaced as an endpoint error (matches the
  pipeline's no-fallback behavior).

## 12. Testing

`transcriber/tests/`:

- `test_alignment_score.py` (pure, synthetic onset lists): perfect → 100;
  uniform 20 ms shift → high but < 100; extra/missing notes → P/R drop;
  saturated wrong-rhythm lane → coverage term bites; crossed-pair case →
  monotonic DP diverges from bipartite as documented.
- `test_alignment_correction.py` (pure): injected offset → tier-0
  recovers to ~0 residual; injected tempo `a` → tier-1 recovers within
  `[0.5, 2.0]`; `<3` pairs → no correction; `score_corrected ≥ score`.
- `test_scoring_lanes.py`, **drift-guard**: fixture GM notes +
  Paradiddle classes → expected lane, so a `gm.ts` / `drums.ts` change
  not mirrored in the Python ports fails CI.
- `test_rlrr_read.py` / `test_paradb_read.py`: fixture `.rlrr` + fixture
  `.zip` → expected per-lane onsets, difficulty selection, UTF-16-BOM
  decode, audio-track extraction.
- `test_midi_read.py`: multi-track fixture with a tempo change →
  tempo-aware seconds (guards the `mid.tracks` multi-track bug).
- ADTOF kept out of unit tests (mock `detect_all_lanes_adtof`); one
  optional `@pytest.mark.integration` end-to-end over a tiny fixture pack.

## 13. Build order

1. `lanes.py` + drift-guard test (ports + 5-lane fold). Foundation for
   every reader.
2. `alignment.py` + `test_alignment_score.py` (pure DP + P/R/F1 +
   roll-up). The scoring core, testable with zero audio.
3. `correction.py` + `test_alignment_correction.py` (tiers 0-1).
4. `rlrr_read.py` + `paradb_read.py` + tests (chart side, audio-free).
5. `midi_read.py` + test (secondary input).
6. `audio_onsets.py`: refactor `detect_all_lanes_adtof` out of
   `detect_onsets_adtof`; drum-track-vs-separate decision.
7. `score_map.py` orchestrator + `models.py`; wire `POST /score` and the
   frontend upload.

Gate after 2-3: synthetic inputs produce expected scores and recover
injected warps. Gate after 7: a real ParaDB pack scores end-to-end and
the corrected score exceeds the raw score on a deliberately offset chart.

## 14. Risks / open questions

- **`σ` vs detector jitter** (§7): a perfect chart may plateau < 100. Fine
  for *relative* corpus filtering; calibrate before reading absolute
  numbers.
- **`loc` ↔ audio clock**: assumes ParaDB `event.loc` shares t=0 with the
  pack's audio. Player/recording latency shows up as a tier-0 offset and
  is corrected; verify on a sample of real packs early.
- **Drum-track quality**: some packs' `drumTracks` are poorly isolated
  (the `paradb.ts` comment notes authors with bad stem separation). When
  a drum track scores implausibly low, separating the song track may be
  the better reference; consider a fallback heuristic after batch
  evidence.
- **Per-difficulty vs per-pack**: v1 scores the highest-complexity chart
  per pack. The ~15k figure (all difficulties) needs a loop over
  `candidates`; trivial, deferred until the per-pack path is validated.
- **Drift-guard maintenance**: the `gm.ts`/`drums.ts` ports must track
  their TS sources; the drift-guard tests fail loudly if they don't, but
  someone still has to update the port. Acceptable given Approach 2
  (Python-owns-everything) was chosen for batch viability.
