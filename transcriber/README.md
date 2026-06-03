# Drumjot Transcriber

Off-the-shelf separation + LLM drum transcription service for Drumjot.
Accepts audio and runs a five-stage pipeline:

1. **`stems_all` — BS-Roformer SW** (jarredou's BS-ROFO-SW-Fixed) extracts
   a drum stem from the full mix. It's a 6-stem Band-Split RoPE
   Transformer chosen over `htdemucs_ft` for its substantially cleaner
   drum stem (drums SDR ~14 vs ~10), especially its preservation of
   high-frequency cymbal / hi-hat transients that Stage 2 then has to
   split. The bass+guitar+piano+other+vocals sum is also exported as a
   "music minus drums" deliverable.
2. **`stems_per` — jarredou 5-stem MDX23C DrumSep** splits the drum stem
   into kick / snare / toms / hi-hat / **cymbals**. Note: this model
   merges ride + crash into a single `cymbals` stem.
3. **`beats` — madmom RNN + DBN downbeat tracker** (or the vendored Beat
   Transformer; the DBN postprocessor is shared, only the activation
   network changes). Each detected beat is snapped to the strongest drum
   onset within ±50 ms to undo the tracker's ~50 ms activation-peak lag.
   Per-bar tempos are finalized — pinned to one global value for
   constant-tempo material, or kept as a smoothed contour when a
   sustained change is present — and we recover the tempo, the per-bar
   time signature, and (via the intra-beat fraction distribution) the
   bar-level "feel": `straight16` / `straight8` / `triplet` / `shuffle` /
   `sparse` / `mixed`. Onsets are mapped to `(bar, beat_in_bar)` rather
   than a fixed 1/16 grid — so triplets, tempo changes and
   time-signature changes are all handled natively.
4. **`onsets`, ADTOF Frame_RNN per stem.** The hi-hat lane is the
   exception: ADTOF runs on its isolated stem with looser gates, an
   audio-domain onset supplement, and an energy floor (the ~14 kHz
   band-limit starves ADTOF of hat sizzle). The merged `cymbals` lane is
   split into ride (`d`) / crash (`c`) via deterministic features plus a
   small LLM pass; the hi-hat lane into closed (`h`) / open (`H`) /
   discard via ring-envelope features + an LLM pass + a deterministic
   envelope guardrail and discard-rescue (see the pipeline doc's "Hi-hat
   lane").
5. **`transcribe` — Claude filter (per instrument).** One small LLM call
   per drum pitch (run in parallel) rejects artifact onsets; the kept
   onsets render straight to a MIDI file (`prediction.mid`) with their
   original un-quantized times. A per-note debug provenance JSON sidecar
   lists every detected onset (kept and rejected) so the UI can annotate
   notes + render rejected onsets as ghost overlays.

The frontend's deterministic `src/midi/from_midi.ts` converts the MIDI
into a Drumjot Jot. (See `docs/ai-midi-to-jot-notes.md` for the techniques
captured from the deleted DSL-output pathway, which a future AI-assisted
MIDI → Jot pass can pull from.)

The response carries the prediction MIDI URL, drum-only / drumless stem
URLs, beat-tracker metadata for the UI status pill, and a debug-zip URL
that bundles the MIDI + provenance + per-stem audio for offline replay.

The service is a stateless FastAPI app. It runs locally during development
and is structured to drop into any IaaS that supports Docker + NVIDIA GPUs.

## Requirements

- **Docker** + **docker compose v2**
- **NVIDIA Container Toolkit** if you want GPU acceleration (strongly
  recommended — on CPU, separation alone takes several minutes per song)
- **`ANTHROPIC_API_KEY`** — see `.env.example`
- ~10 GB free disk for the model cache after first inference

## Local quick start

```bash
cd transcriber
cp .env.example .env
# edit .env to add your ANTHROPIC_API_KEY
docker compose up --build
```

## Linting and tests

```bash
scripts/check-py        # post-change: ruff --fix + pytest (uses transcriber/.venv)
scripts/check-py tests/test_hihat_split.py    # scoped to one test file
# Direct invocation (assumes activated transcriber/.venv):
ruff check .            # lint (config in pyproject.toml [tool.ruff])
pytest                  # pure-Python tests under tests/
```

`scripts/check-py` is the canonical post-change verification path — it
activates the venv, autofixes lint, then runs pytest, exiting on the
first failure. `ruff check .` should always exit clean before pushing;
CI will treat any violation as a failure. The `pytest` suite
intentionally only covers the modules that don't need madmom /
audio-separator / a GPU — it's a sanity net for the parts of the
pipeline that change most often.

Wait for `INFO: Uvicorn running on http://0.0.0.0:8001` then:

```bash
curl http://localhost:8001/health
# -> {"status":"ok","gpu_available":true,"gpu_name":"NVIDIA GeForce GTX 1660 SUPER"}
```

The Drumjot Vite app proxies `/api/transcribe` to this service in dev mode
(see `vite.config.ts`), so the "Transcribe audio" toolbar button works out
of the box once the service is up.

## API

### `GET /health`

Returns service status and GPU availability. Useful for readiness probes.

### `POST /transcribe`

Multipart form upload.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | audio file | required | see "Supported audio formats" below (limit 200 MB) |
| `include_candidates` | bool | false | include raw onset candidates in the response (debug) |
| `beat_input` | `full_mix` \| `drum_stem` | `BEAT_INPUT_DEFAULT` (`full_mix`) | which audio to feed the beat tracker |
| `debug` | bool | false | persist intermediate artifacts to disk; see "Debug artifacts" below |

Time signature, grid, tempo and feel are not form parameters; all are
detected automatically from the audio via the beat tracker.

### `POST /transcribe/resume`

Re-run the pipeline from a chosen stage, hydrating earlier stages'
artifacts from a previous debug folder. Useful for iterating on the
LLM stages without paying for separation every time.

| Field | Type | Default | Description |
|---|---|---|---|
| `resume_folder` | string | required | absolute path or bare folder name under the debug base |
| `resume_stage` | stage | required | one of `stems_all`, `stems_per`, `beats`, `onsets`, `transcribe` |
| (plus all `/transcribe` option fields) | | | `beat_input`, `include_candidates` |

Required artifacts depend on which stages will be skipped (e.g. resuming
at `transcribe` needs `beats.json` + `onsets.json` + `stems_per/*`).
Anything missing comes back as a 400 with a stage-specific message.
Stages from `resume_stage` onward run fresh and overwrite the artifacts
they produce; upstream artifacts are left intact so subsequent resumes
from the same folder stay idempotent.

### Debug artifacts

The pipeline normally runs in a per-request `tempfile.TemporaryDirectory`
that is deleted as soon as the response is sent. To debug a transcription
(listen to the separated stems, inspect the beat tracker output, replay
the LLM input/output, etc.), persist every intermediate file by either:

- Setting `DEBUG_DIR=/debug` in the transcriber's `.env`, which persists
  **every** request, or
- Passing `debug=true` as a form field on a single `/transcribe` call
  (which falls back to `/debug` as the base when no env var is set).

Both modes write into a per-request subdir of the form
`<DEBUG_DIR>/<timestamp>_<short-id>_<filename-slug>/`. With the default
`docker/docker-compose.*.yml` volume mount, this maps 1:1 to the host's
debug directory so files are immediately playable.

Layout of a persisted request:

```
debug/20260517-004530_a1b2c3d4_my-song/
├── input.mp3                # raw upload (original codec)
├── stems_all/               # stage `stems_all`: BS-Roformer SW output
│   ├── drum_stem.wav
│   ├── no_drums.wav         # bass+guitar+piano+other+vocals sum
│   └── <other sub-stems>.wav
├── stems_per/               # stage `stems_per`: MDX23C drum-piece split
│   ├── k.wav                # kick
│   ├── s.wav                # snare
│   ├── h.wav                # hi-hat
│   ├── c.wav                # cymbals (ride+crash merged)
│   └── t.wav                # toms
├── beats.json               # stage `beats`: BeatStructure (beats, bars, feel, tempo)
├── onsets.json              # stage `onsets`: per-stem candidates with (bar, beat_in_bar)
├── onsets_only.mid          # "what the detector heard", no LLM filtering (diagnostic)
├── prediction.mid           # stage `transcribe`: filter-kept onsets as MIDI (the score)
├── note_provenance.json     # stage `transcribe`: per-note keep/reject log for the UI
└── request.json             # filename, options, timings summary
```

Each top-level folder/file is named after the pipeline stage that produces
it, which is also the value to pass as `resume_stage` on
`/transcribe/resume`. Stage ordering:
`stems_all` → `stems_per` → `beats` → `onsets` → `transcribe`.

The response JSON includes a `debug_dir` field with the container path so
the caller can pick the right subdir. The Vite app's "Save debug files"
checkbox sets `debug=true` and shows the path in the success status pill.

The host-side debug directory is gitignored.

### Stem deliverables (`/outputs`)

Separately from debug artifacts, **every** request writes a small fixed
set of FLAC deliverables into `<OUTPUTS_DIR>/<id>/` and serves them under
`/outputs/...` via a `StaticFiles` mount. Each is written the instant its
producing stage finishes (so they're downloadable while the slow LLM
stages are still running) and surfaced as URL paths on the response:

- `drum_stem_url` — the isolated drum mix.
- `no_drums_url` — the bass+guitar+piano+other+vocals "music minus drums" mix.
- `prediction_midi_url` — the kept onsets as a MIDI file (the score).

These are URL paths with no host; compose them against the caller's
transcriber base URL (`/api` in dev, `https://...` in prod). Any can be
`None` if the corresponding artifact couldn't be produced.

### Supported audio formats

The pipeline reads audio through `librosa` → `soundfile`/`audioread`, and
`audio-separator` falls back to **ffmpeg** for anything not natively
supported by libsndfile. Both libraries are installed in the Docker image.
In practice every common audio container works without special handling:

| Format | Extensions | Notes |
|---|---|---|
| WAV (PCM) | `.wav` | Lossless. Fastest decode. |
| FLAC | `.flac` | Lossless. Decoded by libsndfile directly. |
| MP3 | `.mp3` | Lossy. Decoded via libsndfile / ffmpeg. |
| AAC | `.aac`, `.m4a`, `.mp4` | Lossy. Decoded via ffmpeg. |
| Opus | `.opus`, `.ogg` (Opus-in-Ogg), `.oga` | Lossy. Decoded via ffmpeg. |
| OGG Vorbis | `.ogg`, `.oga` | Lossy. Decoded by libsndfile / ffmpeg. |
| WebM audio | `.webm` | Decoded via ffmpeg (best-effort — works if it carries Opus or Vorbis). |

The Vite app's "Transcribe audio" button accepts all of the above by
default (plus the `audio/*` MIME wildcard, so anything the browser labels
as audio passes the picker).

Files that won't work: MIDI (`.mid` — already symbolic, no audio to
transcribe; use `src/midi/fromMidi` directly), proprietary DRM-locked
streams, and exotic containers ffmpeg doesn't ship a demuxer for.

Response (`application/json`):

```json
{
  "metadata": {
    "initial_tempo": 120.0,
    "initial_time_signature": [4, 4],
    "duration_seconds": 184.3,
    "stems_used": ["c", "h", "k", "s", "t"],
    "bars": [
      {"bar": 0, "time_signature": "4/4", "tempo_bpm": 120.0,
       "feel": "straight16", "start_time": 0.0}
    ],
    "has_tempo_changes": false,
    "has_time_sig_changes": false
  },
  "candidates": {},
  "debug_dir": null,
  "drum_stem_url": "/outputs/20260517-004530_a1b2c3d4_my-song/drum_stem.flac",
  "no_drums_url": "/outputs/20260517-004530_a1b2c3d4_my-song/no_drums.flac",
  "prediction_midi_url": "/outputs/20260517-004530_a1b2c3d4_my-song/prediction.mid",
  "debug_zip_url": "/outputs/20260517-004530_a1b2c3d4_my-song/debug.zip"
}
```

## Project layout

All Docker config (Dockerfile, .dockerignore, docker-compose.*.yml,
Caddyfile*, entrypoint.sh) lives in the repo-root `docker/` folder; the
build context is the repo root so it can see both `src/` (frontend) and
`transcriber/` (this dir).

```
transcriber/
├── pyproject.toml          # Python package + deps
├── uv.lock                 # pinned dependency lockfile (uv)
├── .env.example            # ANTHROPIC_API_KEY etc.
├── checkpoints/            # Beat Transformer weights (baked into the image)
├── docs/                   # design notes (e.g. ai-midi-to-jot-notes.md)
├── prompts/                # LLM prompt templates
│   ├── filter_onsets.md         # filter-stage artifact rejection
│   ├── split_cymbals.md         # ride/crash classification
│   └── split_hihat.md           # closed/open hi-hat classification
├── benchmarks/             # accuracy benchmark harness + dataset loaders
└── app/
    ├── main.py             # FastAPI entrypoint (/health, /transcribe, /transcribe/resume)
    ├── config.py           # pydantic-settings
    ├── models.py           # request/response schemas
    ├── debug.py            # opt-in DebugSink for persisting intermediates
    ├── outputs.py          # always-on OutputSink for FLAC/MIDI deliverables
    └── pipeline/
        ├── runner.py        # five-stage dispatcher (shared by both endpoints)
        ├── resume.py        # resume-folder artifact hydration
        ├── separate.py      # BS-Roformer SW + MDX23C DrumSep
        ├── provision.py     # injects + downloads the custom separator weights
        ├── adtof_onsets.py  # ADTOF Frame_RNN onset detector (per stem)
        ├── cymbal_split.py  # split cymbals lane -> ride / crash
        ├── hihat_split.py   # split hi-hat lane -> closed / open
        ├── beats.py         # beat/downbeat/feel tracking
        ├── beat_transformer.py # vendored Beat Transformer activation net
        ├── filter_llm.py    # filter-stage onset rejection (the only LLM call)
        ├── onsets_midi.py   # onsets -> MIDI rendering
        ├── note_provenance.py # per-note debug provenance sidecar
        └── llm_util.py      # shared Anthropic client helpers
```

## Deployment notes

This service is intentionally stateless and 12-factor-friendly:

- All configuration via environment variables (see `app/config.py` and
  `.env.example`).
- No on-disk session state — temp files live in per-request tempdirs by
  default. (Set `DEBUG_DIR` or send `debug=true` to opt a request into
  debug-artifact persistence. The `/outputs` FLAC + MIDI deliverables are
  always written, but to a volume the operator can prune freely.)
- Model weights live on a Docker volume so they persist across restarts.
  The Beat Transformer checkpoint is baked into the image at build time
  from `transcriber/checkpoints/`.
- **Both separation models are loaded eagerly at container startup**
  (FastAPI lifespan). Neither ships in `audio-separator`'s registry, so
  `pipeline/provision.py` injects them and downloads their weights first.
  Watch the logs for `Loading stems_all separator ...` /
  `stems_all ready in N.NNs` / `Loading stems_per separator ...` lines.
  Until you see `Startup complete in N.NNs — service is ready`, the
  `/health` endpoint will not respond — this is the readiness signal
  orchestrators (Cloud Run, Kubernetes, docker-compose health checks)
  use to gate traffic. First-time startup is slower because the model
  weights are downloaded into the cache volume; subsequent restarts take
  ~30 seconds once the volume is warm.
- `/health` is a cheap readiness probe that becomes available the instant
  startup finishes.

To deploy on **GCP Cloud Run** (with GPU): use `gcloud run deploy
--image=drumjot-transcriber --gpu=1 --concurrency=1`. To deploy on
**Modal / Replicate / RunPod / Banana**: this image is a drop-in. Add an
ingress / load balancer in front for production traffic.

## Performance

Approximate timings on a single **NVIDIA GTX 1660 Super** (6 GB VRAM,
Turing, no tensor cores) for a 3-minute track. The transformer-based
separators are the dominant cost on this card — it's roughly an order of
magnitude slower than a current data-center GPU, and 6 GB is tight, so
`audio-separator` chunks the input rather than holding the whole track in
VRAM:

| Step | Time |
|---|---|
| First-ever container start (downloads models) | ~3–5 min |
| Subsequent container starts (models cached) | ~30s |
| `stems_all` — BS-Roformer SW separation (3-min track) | ~70–110s |
| `stems_per` — MDX23C DrumSep | ~30–50s |
| `beats` — beat/downbeat tracking (mostly CPU) | ~8–15s |
| `onsets` — onset detection + cymbal/hi-hat split (all stems) | ~5–10s |
| `transcribe` — per-instrument filter LLM calls (parallel) | ~10–20s |
| **Total per request (after startup)** | **~2–3 min** |

VRAM note: BS-Roformer SW is the heaviest model and the most likely to
OOM on a 6 GB card if the chunk/segment size is raised. The defaults are
sized to fit; if you see CUDA out-of-memory in `stems_all`, leave them
alone (or lower them) rather than chasing speed.

On CPU, request time stretches to ~10+ minutes — separation alone is
several minutes. Don't run this without a GPU unless you're patient.

## Troubleshooting

**`stems_all produced no drum stem`**: the BS-Roformer model file failed
to download or `audio-separator` wrote elsewhere. Check
`docker logs drumjot-transcriber` for the `provision`/audio-separator
download error. Common cause: rate-limited Hugging Face, retry later.

**CUDA out of memory in `stems_all` / `stems_per`**: the separators don't
fit alongside everything else in 6 GB. Make sure nothing else is using
the GPU, and don't raise `audio-separator`'s chunk/segment sizes above
the defaults.

**`ANTHROPIC_API_KEY is not set`**: the `.env` file isn't being picked up.
Make sure it's in `transcriber/` (not the repo root) and `docker compose`
is being run from that directory.

**Filter LLM keeps rejecting real hits**: set `include_candidates=true`
and re-run to see what the LLM was given; usually the ADTOF thresholds in
`app/config.py` need tuning. For faster iteration, persist a debug folder
and replay just the filter stage with `POST /transcribe/resume`
(`resume_stage=transcribe`).
