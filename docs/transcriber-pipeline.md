# Transcriber pipeline

The Python/FastAPI backend that turns audio into a predicted-onsets MIDI
file. Pure Python, no bun, no TypeScript. See
[architecture.md](architecture.md) for why.

Audio → MIDI flow: BS-Roformer SW separation → MDX23C DrumSep →
madmom beat tracking → ADTOF Frame_RNN per-stem onset detection →
Claude filter LLM (rejects artifact onsets per instrument) → kept onsets
render straight to MIDI with original un-quantized times.
`src/midi/from_midi.ts` on the frontend converts that MIDI to a Jot.

## Layout

```
transcriber/
├── README.md                   Service-level docs (formats, API, perf).
├── pyproject.toml              fastapi, audio-separator[gpu], adtof_pytorch,
│                               madmom (git), mir_eval, anthropic.
├── .env.example                ANTHROPIC_API_KEY, LLM_MODEL,
│                               INSTRUMENT_CONCURRENCY, DEBUG_DIR.
├── docs/ai-midi-to-jot-notes.md   Captured techniques from the deleted DSL
│                               pathway; reference for future AI-assisted
│                               MIDI → Jot work.
├── prompts/                    Markdown prompt templates with placeholders
│                               (filter_onsets.md, split_cymbals.md, split_hihat.md).
└── app/
    ├── main.py                 FastAPI: GET /health, POST /transcribe,
    │                           POST /transcribe/resume.
    ├── config.py               Pydantic-settings (env-driven).
    ├── debug.py                DebugSink + serializers; request-scoped ContextVar.
    ├── models.py               TranscribeResponse, OnsetCandidate (with
    │                           bar/beat_in_bar), BarSummary, TranscriptionSummary.
    └── pipeline/
        ├── runner.py           Stage enum + PipelineContext + run_pipeline().
        ├── resume.py           hydrate_context_from_resume(folder, start_stage).
        ├── separate.py         Two-stage separator (run_stems_all / run_stems_per).
        │                       BS-Roformer SW + Jarredou MDX23C 5-stem DrumSep.
        ├── beats.py            madmom RNN+DBN downbeat tracker (default) OR Beat
        │                       Transformer activations into the shared DBN.
        ├── adtof_onsets.py     ADTOF Frame_RNN per-stem onset detection.
        ├── cymbal_split.py     Splits merged cymbals -> ride (d) / crash (c).
        ├── hihat_split.py      Splits hi-hat -> closed (h) / open (H).
        ├── filter_llm.py       Per-instrument Claude artifact-rejection filter.
        ├── onsets_midi.py      Render kept onsets to prediction.mid.
        ├── note_provenance.py  Per-note debug sidecar (kept + rejected onsets).
        ├── quantise.py / lyrics_align.py / beat_transformer.py / provision.py
        └── llm_util.py         Refusal/content-filter retry + code-fence strip.
```

## Named-stage runner

Both endpoints dispatch through `pipeline/runner.run_pipeline()`. Stages
and their data dependencies:

```
stems_all  -> drum_stem               (consumed by stems_per)
stems_per  -> per_instrument_stems    (consumed by onsets)
beats      -> structure               (consumed by onsets, filter, transcribe)
onsets     -> onsets_by_pitch         (consumed by filter, transcribe)
filter     -> kept_by_pitch           (consumed by transcribe)
transcribe -> predicted_midi          (kept-onsets MIDI deliverable)
```

`filter` runs the per-instrument filter LLM
(`filter_onsets_all_instruments`, one parallel Anthropic call per drum
pitch), persisting survivors to `filter/kept_onsets.json`. `transcribe`
is the deterministic render: `onsets_to_midi_bytes` +
`build_note_provenance` against `kept_by_pitch`, writing `prediction.mid`
and `note_provenance.json`. Splitting them lets the operator iterate on
render/provenance code without re-paying for LLM calls.

Each `_do_<stage>(...)` reads inputs from a shared `PipelineContext`,
writes its output back, and persists artifacts via the debug sink.
Failures wrap into `StageError(stage, original)`; the HTTP layer maps
`StageError.stage == FILTER` to 502 (LLM is external), everything else to
500. The runner is the single place the pipeline order lives, adding a
stage means: extend the enum, add a `_do_*`, add a `_run_stage` case, add
a `resume.py` hydration entry.

There is **no refine stage**; the legacy F1-gated multi-level refinement
loop and the whole DSL-output pathway were removed. See
`transcriber/docs/ai-midi-to-jot-notes.md` for the captured techniques
(per-instrument prompting, deterministic recompose, F1-gated refine,
critic triage, best-of-K, pattern-aware suppression).

## `/transcribe/resume`

Re-runs the pipeline from a chosen stage onward, hydrating earlier
stages' artifacts from a previous run's debug folder. Form params:

- `resume_folder`, absolute path or bare folder name under `DEBUG_DIR`
  (default `/debug`). Path-traversal guard rejects anything outside base.
- `resume_stage`, one of `stems_all`, `stems_per`, `beats`, `onsets`,
  `transcribe`. Required.
- `beat_input`, `include_candidates`, same as `/transcribe`.

Required upstream artifacts by stage: `stems_per` needs
`stems_all/drum_stem.<ext>`; `beats` needs `stems_per/*.<ext>`; `onsets`
needs `stems_per/*.<ext>` + `beats.json`; `transcribe` needs
`beats.json` + `onsets.json` + `stems_per/*.<ext>` (re-hydrates
`kept_by_pitch` from `filter/kept_onsets.json`, identity re-threaded
against `onsets.json` so `build_note_provenance`'s `id(c)` match works).
Missing artifacts → HTTP 400 naming which `resume_stage` regenerates
them. Output overwrites the chosen stage and everything after; upstream
is preserved so re-resuming is idempotent.

## Beat tracking, load-bearing invariants

`analyze_beats` order is **load-bearing**:
`tracker → align_beats_to_onsets → _finalize_bar_tempos → _pad_trailing_bars`.

- **`align_beats_to_onsets`** shifts the **whole grid** by one median
  offset (not per-beat snap). Neural trackers report each beat at its
  activation peak, ~30–50 ms after the transient; for each beat it
  computes the delta to the *strongest* drum onset within ±50 ms, takes
  the **median**, and shifts the entire grid by that single offset. A
  uniform shift removes systematic lag while leaving inter-beat gaps
  (hence per-bar tempo and genuine accelerando) untouched. Per-beat
  snapping folded the drummer's micro-timing into the grid and made
  per-bar tempo wobble 5–10 BPM on steady songs. Gated by
  `MIN_ALIGN_COVERAGE` (.30): below 30 % beats with a nearby onset, the
  grid is left as the tracker produced it. Must run **after** the tracker
  and **before** padding (so synthetic fadeout bars don't pull the
  median). Drum-stem onsets are detected once on `ctx.drum_stem` inside
  `_do_beats` and passed in.
- **`_finalize_bar_tempos`** median-filters per-bar tempos
  (`TEMPO_SMOOTHING_WINDOW`=5) only to *decide* tempo motion: if the
  smoothed reference-bar span is under `SUSTAINED_TEMPO_CHANGE_BPM` (8.0),
  every bar is pinned to one global tempo (median, bar 0 excluded as
  likely anacrusis) and `has_tempo_changes=False`; otherwise the smoothed
  contour is kept. Must run **before** padding (pads inherit the
  pinned/smoothed tempo) and **before** `has_tempo_changes` is consumed
  (it overwrites the naive flag). Don't re-run `_rebuild_bar_fields`
  after it, that reintroduces raw wobble.
- **Anacrusis**: `_summarize` excludes bar 0 when ≥2 bars present, so a
  3-beat pickup doesn't mislabel the whole song as 3/4. Bar 0's own
  `time_signature` is left as-is.

**Onset detector windows are tuned tight** (`pre_max`=`post_max`=3,
`wait`=3, `pre_avg`=`post_avg`=50) on the assumption of **per-instrument
stems** (isolated transients), not the full drum mix. If a refactor ever
runs onset detection on the full mix, widen the windows back to ~20 or
you'll get spurious double-detections.

## Debug folder layout

Written when `DEBUG_DIR` is set or `debug=true` on a request:

```
/debug/<timestamp>_<id>_<slug>/
├── input.<ext>           raw upload
├── stems_all/            drum_stem.wav, no_drums.wav (drumless mix),
│                         bass/other/vocals
├── stems_per/            k,s,h,d,c,t .wav + residual.wav (aux percussion the
│                         5-class model has no lane for; diagnostic-only)
├── beats.json            BeatStructure dump
├── onsets.json           per-pitch candidates
├── onsets_only.mid       one MIDI hit per detected onset (no LLM/quantize), │                         drop into a DAW to hear exactly what the detector heard
├── prediction.mid        kept-onset MIDI (the score)
├── note_provenance.json  per-note kept + rejected onsets
├── filter/kept_onsets.json
├── llm/NN_<purpose>.txt              full hydrated prompt for one LLM call
├── llm/NN_<purpose>.response.json    parsed Anthropic response (same NN):
│                                     content blocks incl. tool_use.input,
│                                     stop_reason, usage. (stop_reason=="max_tokens"
│                                     + empty input == forced-tool truncation.)
└── request.json          filename + options + timings summary
```

## Outputs folder layout

Always populated regardless of the `debug` flag. Each FLAC is written the
instant its producing stage finishes (not deferred), so deliverables are
downloadable while slow stages still run. The request dir is created
lazily on first successful write, so an `/outputs/<...>/` that exists at
all is guaranteed non-empty.

In-stage writes only fire when those stages run. A resume from `beats`+
skips `stems_all`/`stems_per`, so `app.outputs.materialize_pending` runs
once after the pipeline (both endpoints, before temp work_dir teardown)
and backfills missing deliverables from the hydrated context, or for
`no_drums`/`residual` (which no stage carries on the context) by
scavenging `<resume_dir>/stems_all/no_drums.<ext>` /
`<resume_dir>/stems_per/residual.<ext>`. No recomputation.

```
/outputs/<timestamp>_<id>_<slug>/
├── drum_stem.flac        isolated drum mix -> TranscribeResponse.drum_stem_url
├── no_drums.flac         music-minus-drums backing track -> no_drums_url
├── stem_<k|s|h|d|c|t>.flac  one per recovered per-instrument stem
├── residual.flac         diagnostic-only; not surfaced as a URL
├── prediction.mid        kept-onset MIDI -> prediction_midi_url
└── debug.zip             full debug bundle (loaded back via "Load debug bundle")
```

FastAPI serves this via `StaticFiles` at `/outputs/...`; URLs (leading
`/`) compose against `TRANSCRIBER_BASE` (`/api` in dev → Vite proxy;
absolute in prod). Client helper: `src/transcriber.ts::stemUrl`.

## Build / run

```
cd transcriber && cp .env.example .env   # fill in ANTHROPIC_API_KEY
docker compose up --build                # first build downloads ~3 GB of weights
docker compose logs -f transcriber       # ready at "Startup complete… service is ready"
curl http://localhost:8001/health        # 200 only after eager-load completes
```

Dev box has a local uv-managed venv at `transcriber/.venv` (torch cu128,
madmom, audio-separator[gpu], adtof_pytorch), this is the **primary dev
loop**; Docker is only for clean/reproducible builds. `scripts/check-py`
activates it. Invoke Python as `python3` on the bare system (plain
`python` only inside an activated venv). **Do NOT install/upgrade deps
unprompted**, the install graph is ordering-sensitive (numpy/cython
before madmom; torch from cu128 index first); flag dep changes and let
the user run `uv pip install`.

## Accuracy: the two paths

The user picked **Path A** as the implementation. **Path B** is
researched-only.

### Path A, off-the-shelf + LLM (IMPLEMENTED)

Shipped in `transcriber/`. Expected real-world accuracy **F1 0.85–0.92**,
dominated by the upstream-model ceiling (ADTOF caps ~F1 0.87 on clean
drum stems; separation bleed costs more). Per-song cost (3-min track,
serverless GPU): separation ~$0.05 + LLM ~$0.15 = **~$0.20/song**.

### Path B, train own Noise-to-Notes (RESEARCHED ONLY)

The N2N paper (arXiv 2509.21739, Sept 2025) is current academic SOTA
(0.897 F1 E-GMD, 0.879 MDB, 0.949 IDMT): diffusion-based generative model
on EDGE transformer + MERT-330M features. No public code/weights yet
(~ICASSP 2026 / ISMIR 2026). User has confirmed they want to train their
own. Full plan, architecture components, the open training-data set,
the user's 6289-song corpus + Drumgizmo rendering, augmentation suite,
TPU config (~$420/run on spot v6e-1, ~$5–6.5K for the ablation program),
expected outcomes, and the engineering plan; is preserved in
[research/MODELS.md](../research/MODELS.md). Build the `mir_eval` 50 ms
F1 eval harness early and run ADTOF locally as a proxy baseline.

## Things to test next (real-audio, in priority order)

1. 4/4 rock at known tempo, verify F1 ≥ 0.80.
2. Song with a tempo change, verify inline `{{ bpm: ... }}` at the right
   bars + `has_tempo_changes: true`.
3. Triplets / swing (jazz waltz, shuffle blues), verify `feel=triplet` /
   `feel=shuffle` and `(...)_N` groups.
4. 7/8 or 3/4, verify time-sig detection + `{{ time: "..." }}` changes.
5. opus/aac/flac/wav files end-to-end.
6. Drop `.mid` files in `src/midi/__tests__/fixtures/` to activate the
   fixture suite (round-trip on real material).
7. Email the N2N authors for weights/sample outputs while building the
   eval harness (Path B).

## References

- 2509.24853, "Enhanced ADT via Drum Stem Source Separation" (Path A blueprint).
- 2509.21739, N2N paper (Path B target).
- ADTOF: github.com/MZehren/ADTOF, runnable proxy baseline.
- E-GMD: magenta.tensorflow.org/datasets/e-gmd.
- madmom: github.com/CPJKU/madmom, beat/downbeat tracking.
- audio-separator: pypi.org/project/audio-separator/.
- Jarredou MDX23C DrumSep: github.com/jarredou/models/releases.
- Drumgizmo + free kits: drumgizmo.org/wiki (Path B rendering).
- ParadiddleUtilities: github.com/emretanirgan/ParadiddleUtilities (RLRR origin).

## Open questions for the user

1. **N2N timeline**, committed to starting Path B, or still
   research-complete/execution-pending? (Currently the latter.)
2. **The 6289-song corpus**, real full-mix songs with hand-aligned,
   articulation-accurate charts; user has recordings and separates stems
   directly. Agent hasn't seen the files; ask for folder structure before
   starting a rendering/separation pipeline.
3. **Audio-native LLM step** was explicitly excluded, the next lever if
   accuracy plateaus below target.
