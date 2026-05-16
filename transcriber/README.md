# Drumjot Transcriber

Off-the-shelf + LLM drum transcription service for Drumjot. Accepts audio,
runs:

1. **Demucs v4 (`htdemucs_ft`)** - drum stem from the full mix
2. **MDX23C 6-stem DrumSep** (aufr33 / jarredou) - drum stem into kick /
   snare / hi-hat / ride / crash / toms
3. **librosa** per-stem high-recall onset detection
4. **madmom RNN + DBN downbeat tracker** - per-beat anchors with
   downbeat classification. From this we recover the tempo curve, the
   per-bar time signature, and (via the intra-beat fraction
   distribution) the bar-level "feel": `straight16` / `straight8` /
   `triplet` / `shuffle` / `sparse` / `mixed`. Onsets are mapped to
   `(bar, beat_in_bar)` rather than a fixed 1/16 grid - so triplets,
   tempo changes and time-signature changes are all handled natively.
5. **Claude (Opus 4.7)** translates the per-bar candidate listing into
   Drumjot DSL with inline `{{ bpm: ... }}` / `{{ time: ... }}` blocks
   where tempo/time signature change, and `(...)_N` triplet groups in
   bars where the feel is triplet/shuffle. Optionally with
   **self-consistency** (K candidates, pick best by F1).
6. **Optional multi-level refinement loop** (off by default per request,
   on by default in compose):
   - *Macro* pass - tempo / time-signature fixes
   - *Structure* pass - factor repeating bars into `[Name=(...)]`
   - *Onsets* pass - missing / extra hits (up to 3 iterations)
   - *Velocity* pass - dynamics matching
   Each iteration uses a deterministic diff against the per-stem onsets
   (`mir_eval` F1) to gate revisions: only revisions that *improve* the
   score are kept. A cheap critic LLM (Haiku) triages issues before the
   expensive generator (Opus) revises.

…and returns a DSL string the Drumjot frontend parses and renders, plus
a structured refinement log that the UI surfaces (initial F1, final F1,
per-iteration accept/reject decisions).

The service is a stateless FastAPI app. It runs locally during development
and is structured to drop into any IaaS that supports Docker + NVIDIA GPUs.

## Requirements

- **Docker** + **docker compose v2**
- **NVIDIA Container Toolkit** if you want GPU acceleration (strongly
  recommended - on CPU, separation alone takes ~5 minutes per song)
- **`ANTHROPIC_API_KEY`** - see `.env.example`
- ~10 GB free disk for the model cache after first inference

## Local quick start

```bash
cd transcriber
cp .env.example .env
# edit .env to add your ANTHROPIC_API_KEY
docker compose up --build
```

Wait for `INFO: Uvicorn running on http://0.0.0.0:8001` then:

```bash
curl http://localhost:8001/health
# -> {"status":"ok","gpu_available":true,"gpu_name":"NVIDIA RTX ..."}
```

The Drumjot Vite app proxies `/api/transcribe` to this service in dev mode
(see `vite.config.ts`), so the "Transcribe audio" toolbar button works
out of the box once the service is up.

## API

### `GET /health`

Returns service status and GPU availability. Useful for readiness probes.

### `POST /transcribe`

Multipart form upload.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | audio file | required | see "Supported audio formats" below |
| `include_candidates` | bool | false | include onset candidates in response (debug) |
| `refine` | bool | `REFINE_BY_DEFAULT` | run the multi-level convergence loop |
| `self_consistency_samples` | int | `SELF_CONSISTENCY_SAMPLES_DEFAULT` (1) | generate K initial candidates and pick the best by F1 against source stems |
| `debug` | bool | false | persist intermediate artifacts to disk; see "Debug artifacts" below |

Time signature and grid are no longer form parameters; both are
detected automatically from the audio via madmom downbeat tracking.

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
`docker-compose.yml` volume mount `./debug:/debug`, this maps 1:1 to
`transcriber/debug/...` on the host so files are immediately playable.

Layout of a persisted request:

```
transcriber/debug/20260517-004530_a1b2c3d4_my-song/
├── input.mp3                # raw upload (original codec)
├── stage1/
│   └── drum_stem.wav        # demucs htdemucs_ft output
├── stage2/
│   ├── k.wav                # MDX23C kick stem
│   ├── s.wav                # snare
│   ├── h.wav                # hi-hat
│   ├── d.wav                # ride
│   ├── c.wav                # crash
│   └── t.wav                # toms
├── beats.json               # full BeatStructure (beats, bars, feel, tempo)
├── onsets.json              # per-stem onset candidates with (bar, beat_in_bar)
├── initial.jot              # first-pass Drumjot DSL (pre-refinement)
├── final.jot                # final DSL (== initial.jot if refinement off)
├── self_consistency.json    # K candidate scores + chosen index (if used)
├── refinement.json          # per-iteration accept/reject log
└── request.json             # filename, options, scores, timings summary
```

The response JSON includes a `debug_dir` field with the container path
so the caller can pick the right subdir. The Vite app's "Save debug
files" checkbox sets `debug=true` and shows the path in the success
status pill.

The host-side `transcriber/debug/` directory is gitignored.

### Supported audio formats

The pipeline reads audio through `librosa` -> `soundfile`/`audioread`,
and `audio-separator` falls back to **ffmpeg** for anything not
natively supported by libsndfile. Both libraries are installed in the
Docker image. In practice this means every common audio container
works without special handling:

| Format | Extensions | Notes |
|---|---|---|
| WAV (PCM) | `.wav` | Lossless. Fastest decode. |
| FLAC | `.flac` | Lossless. Decoded by libsndfile directly. |
| MP3 | `.mp3` | Lossy. Decoded via libsndfile / ffmpeg. |
| AAC | `.aac`, `.m4a`, `.mp4` | Lossy. Decoded via ffmpeg. |
| Opus | `.opus`, `.ogg` (Opus-in-Ogg), `.oga` | Lossy. Decoded via ffmpeg. |
| OGG Vorbis | `.ogg`, `.oga` | Lossy. Decoded by libsndfile / ffmpeg. |
| WebM audio | `.webm` | Decoded via ffmpeg (best-effort - if the WebM contains Opus or Vorbis audio it works). |

The Vite app's "Transcribe audio" button accepts all of the above by
default (plus the `audio/*` MIME wildcard, so anything the browser
labels as audio passes the picker).

Files that won't work: MIDI (`.mid` - already symbolic, no audio to
transcribe; use `src/midi/fromMidi` directly), proprietary DRM-locked
streams, and exotic containers ffmpeg doesn't ship a demuxer for.

Response (`application/json`):

```json
{
  "jot_dsl": "{{ bpm: 120, time: \"4/4\", ... }} | k . s . | ...",
  "metadata": {
    "initial_tempo": 120.0,
    "initial_time_signature": [4, 4],
    "duration_seconds": 184.3,
    "stems_used": ["c", "h", "k", "s", "t"],
    "bars": [
      {"bar": 0, "time_signature": "4/4", "tempo_bpm": 120.0,
       "feel": "straight16", "start_time": 0.0},
      ...
    ],
    "has_tempo_changes": false,
    "has_time_sig_changes": false
  },
  "refinement": {
    "initial_score": 0.81,
    "final_score": 0.89,
    "elapsed_seconds": 42.7,
    "iterations": [
      { "level": "macro", "iteration": 0, "issues_detected": 1,
        "issues_sent_to_llm": 1, "score_before": 0.81, "score_after": 0.83,
        "accepted": true, "note": "8.2s" },
      ...
    ]
  },
  "self_consistency": null,
  "candidates": {}
}
```

## Project layout

```
transcriber/
├── Dockerfile              # NVIDIA CUDA base + Python deps + bun + app
├── docker-compose.yml      # local dev orchestration; build context = repo root
├── pyproject.toml          # Python package + deps
├── .env.example            # ANTHROPIC_API_KEY etc.
├── prompts/                # LLM prompt templates
│   ├── transcribe.md       # main first-pass prompt
│   ├── examples.md         # few-shot DSL examples
│   ├── critic.md           # cheap-LLM triage prompt
│   ├── refine_macro.md     # tempo/time-sig revision
│   ├── refine_structure.md # pattern factoring
│   ├── refine_onsets.md    # missing/extra hit corrections
│   └── refine_velocity.md  # dynamics adjustments
├── tools/                  # Bun TS bridge for the canonical parser
│   ├── jot_to_onsets.ts    # DSL -> per-pitch onset list (used by refine)
│   ├── tsconfig.json
│   └── package.json
└── app/
    ├── main.py             # FastAPI entrypoint
    ├── config.py           # pydantic-settings
    ├── models.py           # request/response schemas
    ├── debug.py            # opt-in DebugSink for persisting intermediates
    └── pipeline/
        ├── separate.py     # Demucs + DrumSep
        ├── onsets.py       # librosa per-stem detection
        ├── beats.py        # madmom beat/downbeat/feel tracking
        ├── llm.py          # Claude initial transcription + self-consistency
        ├── jot_extract.py  # bun subprocess: DSL -> structured onsets
        ├── diff.py         # onset / velocity / tempo issue detectors
        ├── score.py        # per-stem F1 via mir_eval
        ├── critic.py       # Haiku triage of issue lists
        └── refine.py       # multi-level convergence loop
```

## Deployment notes

This service is intentionally stateless and 12-factor-friendly:

- All configuration via environment variables (see `app/config.py`).
- No on-disk session state - temp files live in per-request tempdirs by
  default. (Set `DEBUG_DIR` or send `debug=true` to opt a request into
  artifact persistence; see "Debug artifacts" above.)
- Model weights live on a Docker volume so they persist across restarts.
- **Both separation models are loaded eagerly at container startup**
  (FastAPI lifespan). Watch the logs for `Loading stage 1 separator...`
  / `Stage 1 ready in N.NNs` / `Loading stage 2 separator...` lines.
  Until you see `Startup complete in N.NNs - service is ready`, the
  `/health` endpoint will not respond - this is the readiness signal
  orchestrators (Cloud Run, Kubernetes, docker-compose health checks)
  use to gate traffic. First-time startup is slower because the model
  weights are downloaded into the cache volume; subsequent restarts
  take ~30 seconds once the volume is warm.
- `/health` is a cheap readiness probe that becomes available the
  instant startup finishes.

To deploy on **GCP Cloud Run** (with GPU): use `gcloud run deploy
--image=drumjot-transcriber --gpu=1 --concurrency=1`. To deploy on
**Modal / Replicate / RunPod / Banana**: this image is a drop-in. Add an
ingress / load balancer in front for production traffic.

## Performance

On a single NVIDIA RTX 4090:

| Step | Time |
|---|---|
| First-ever container start (downloads models) | ~3–5 min |
| Subsequent container starts (models cached) | ~30s |
| Demucs separation (3-min track) | ~12s |
| DrumSep separation | ~8s |
| Onset + tempo (all stems) | ~3s |
| LLM call | ~5–10s |
| **Total per request (after startup)** | **~30s** |

On CPU, request time stretches to ~5–10 minutes. Don't run this without a
GPU unless you're patient.

## Troubleshooting

**`Stage 1 separator produced no drum stem`**: the model file failed to
download. Check `docker logs drumjot-transcriber` for the audio-separator
download error. Common cause: rate-limited Hugging Face, retry later.

**`ANTHROPIC_API_KEY is not set`**: the `.env` file isn't being picked up.
Make sure it's in `transcriber/` (not the repo root) and `docker compose`
is being run from that directory.

**LLM emits invalid DSL**: the Vite frontend's parser will raise. Set
`include_candidates=true` and re-run to see what the LLM was given; usually
fixing the onset thresholds in `app/config.py` resolves it.
