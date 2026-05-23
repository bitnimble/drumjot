# Drumjot

A drum notation tool that takes audio (or MIDI, or Paradiddle `.rlrr`)
and produces a compact, human-readable DSL representation you can
visualise in the browser. Round-trip converters export back to MIDI or
RLRR; the optional transcription service turns arbitrary audio into a
Jot via stem separation + beat tracking + an LLM.

The DSL is documented in [SPEC.md](SPEC.md); examples sit in
[src/fakes.ts](src/fakes.ts). The full architectural overview is in
[AGENTS.md](AGENTS.md).

---

## Web app

Browser-based DSL editor + renderer. Vite + React + MobX. Uses Bun as
the package manager and test runner.

### Prerequisites

- **[Bun](https://bun.sh/)** v1.3 or newer (`curl -fsSL https://bun.sh/install | bash`).
- A modern browser (any Chromium / Firefox / Safari release).

### Setup

```bash
bun install
```

### Run the dev server

```bash
bun run dev
```

Then open <http://localhost:5173>.

The toolbar gives you:

- An **example selector** (loads `rockJot` / `tripletJot` from
  `src/fakes.ts`).
- A **Transcribe audio** button that uploads audio and renders the
  resulting Jot. Needs the transcriber service running locally (see
  the next section).
- A **Refine accuracy** checkbox that toggles the LLM convergence
  loop on the transcribed output.
- A **Samples** dropdown that controls how many best-of-K
  candidates the transcriber generates.

### Other useful commands

```bash
bun test               # full test suite (65 tests, ~150 ms)
bunx tsc --noEmit      # typecheck only
bun run build          # production build (tsc + Vite)
bun run preview        # serve the production build
```

### Loading custom Drumjot DSL from the console

Once the page is open, the global `Drumjot` instance lets you load
arbitrary DSL:

```js
drumjot.loadDsl(`
  {{ bpm: 120, time: "4/4",
     instrumentMapping: { k:{name:"Kick"}, s:{name:"Snare"}, h:{name:"HiHat"} } }}
  | h:c h:c h:c h:c h:c h:c h:c h:c |
  ||
  | k . s . k . s . |
`);
```

---

## Transcriber service

A separate Python backend (`transcriber/`) that converts audio into
Drumjot DSL via Demucs + Jarredou (separation) + librosa (per-stem
onset detection) + madmom (beat / downbeat / feel tracking) + Claude
Opus 4.7 (LLM emission + multi-level refinement). Runs in Docker with
optional NVIDIA GPU passthrough; the Vite dev server proxies `/api/*`
to it.

### Prerequisites

- **[Docker](https://www.docker.com/)** and **docker compose v2**.
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)**
  if you want GPU acceleration. Without a GPU the pipeline still
  works but each request takes ~5–10 minutes instead of ~30–60 s.
- An **Anthropic API key** (https://console.anthropic.com/).
- ~10 GB of free disk for the separation model cache (downloaded on
  first run into a Docker volume).

### Setup

```bash
cd transcriber
cp .env.example .env
# Edit .env, set ANTHROPIC_API_KEY.
```

Other env vars you may want to tune (defaults are sensible):

| Var | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `claude-opus-4-7` | Model used by the filter stage (rejects artifact onsets per instrument). |
| `INSTRUMENT_CONCURRENCY` | `4` | Max concurrent per-instrument filter LLM calls. Calls are small; mainly guards API rate limits. |
| `DEVICE` | `auto` | `auto` lets audio-separator pick CUDA/MPS/CPU. Force with `cuda` / `cpu`. |
| `DEBUG_DIR` | _(unset)_ | When set (recommend `/debug`), every request persists drum stems + beat tracking + LLM input/output to `<DEBUG_DIR>/<timestamp>_<id>_<filename>/` so you can listen back and debug. The default compose mount `./debug:/debug` already exposes this on the host. |

### Build and run

```bash
docker compose up --build
```

First-ever build downloads ~3 GB of separation model weights into a
named Docker volume; subsequent restarts skip that. Models load
eagerly at container startup, so the service is ready when the logs
show:

```
Startup complete in N.NNs - service is ready to accept requests.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

### Verify the service

```bash
curl http://localhost:8001/health
# -> {"status":"ok","gpu_available":true,"gpu_name":"NVIDIA RTX ..."}
```

The `/health` endpoint only returns 200 after eager startup completes,
so it doubles as a real readiness probe for any orchestrator
(Kubernetes, Cloud Run, docker-compose health checks).

### Using it from the web app

With the service running and `bun run dev` up, click **Transcribe
audio** in the Drumjot toolbar, pick an audio file (wav / flac / mp3 /
aac / m4a / opus / ogg), wait ~30–90 s, and the rendered Jot
replaces whatever was on screen. The score is delivered as a MIDI
file and converted client-side via `src/midi/from_midi.ts`; the
success pill shows the detected tempo, bar count, and whether tempo or
time-signature changes were detected.

### Using it directly

```bash
curl -X POST http://localhost:8001/transcribe \
  -F file=@your_song.mp3 \
  -F debug=true
```

Response is a JSON `TranscribeResponse` with `metadata` (tempo, time
signature, per-bar info), `prediction_midi_url` (the predicted-onsets
MIDI file), and the per-stem audio deliverable URLs.

When `debug=true` (or `DEBUG_DIR` is set), the response also includes a
`debug_dir` field pointing at a per-request folder under
`transcriber/debug/` containing the original upload, separated drum
stems, per-instrument stems (kick/snare/hat/...), the beat-tracker
output, the detected onsets, the predicted MIDI, and per-note
provenance. The Vite app's "Save debug files" checkbox toggles this.

Full API docs: see [transcriber/README.md](transcriber/README.md).

### Stopping / cleaning up

```bash
docker compose down               # stop the service
docker compose down -v            # also delete the models-cache volume
                                  # (forces re-download next run)
```

---

## Project structure (brief)

```
drumjot/
├── SPEC.md                Drum-DSL grammar (the source of truth).
├── AGENTS.md              Comprehensive context for AI / human contributors.
├── src/                   Frontend + library code.
│   ├── dsl.ts             Core types.
│   ├── jot.ts             RenderedJot + layout pipeline.
│   ├── jot_view.tsx       React renderer.
│   ├── fakes.ts           Example jots.
│   ├── parser/            DSL parser (TS).
│   ├── midi/              MIDI <-> Jot.
│   ├── rlrr/              Paradiddle <-> {Jot, MIDI}.
│   └── transcriber.ts     HTTP client for the transcriber service.
├── transcriber/           Python backend (FastAPI + Docker).
└── package.json           Bun-managed.
```

For a deeper dive (architectural decisions, the long-term Path A vs
Path B accuracy plan, known limitations, things to test next), read
[AGENTS.md](AGENTS.md).

---

## License

MIT.
