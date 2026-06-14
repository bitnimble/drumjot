# Drumjot

A drum notation tool that takes audio (or MIDI, or Paradiddle `.rlrr`)
and produces a compact, human-readable DSL representation you can
visualise in the browser. Round-trip converters export back to MIDI or
RLRR; the optional transcription service turns arbitrary audio into a
Jot via stem separation + beat tracking + an LLM.

The DSL is documented in [SPEC.md](SPEC.md); examples sit in
[src/fakes/fakes.ts](src/fakes/fakes.ts). The full architectural overview
is in [AGENTS.md](AGENTS.md).

---

## Web app

Browser-based DSL editor + renderer. Vite + React + MobX. Bun is the
package manager and unit-test runner; Playwright drives the e2e suite and
Storybook hosts the component/library sandboxes.

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

The toolbar (left → right) gives you:

- **File** — load a `.jot`, a Standard MIDI File, a Paradiddle `.rlrr`
  pack, a transcriber **debug bundle** (`.zip`), or audio track(s) as
  backing; re-open a **Recent** transcription; pick a built-in
  **Example** (`rockJot` / `tripletJot` from `src/fakes/fakes.ts`); or
  load synced **Lyrics** (LRCLIB search, `.lrc` file, or pasted text).
- **Transcribe** — upload audio (or **Resume** a previous run from a
  chosen pipeline stage) to render a Jot. Options: beat input
  (full-mix / drum-stem), drum separator (MDX23C / LarsNet), LLM model,
  and an optional quantise stage with an LLM-adjustment pass. Needs the
  transcriber service (see the next section).
- **View** — horizontal zoom, reference grid lines (beats / 16ths /
  triplets / 48ths), filtered-onset and uniform-waveform overlays, and
  the light/dark/system theme.
- **Playback** — drum kit (SoundFont preset), playback speed, audio
  latency trim, and auto-follow-on-play.

Below the toolbar: the **mixer** (one row per audio track and drum
instrument — mute / solo / volume / colour, drag to reorder), the score
itself, a content-aware **minimap** scrubber, and the playback transport.

### Other useful commands

```bash
scripts/check          # post-change: both sides (lint --fix + typecheck + tests)
scripts/check-ts       # post-change: frontend (stylelint --fix + tsc + bun test)
                       #   the canonical wrapper; runs tsc / bun test with the
                       #   right cwd + flags. Pass a path to scope bun test.
bun run build          # production build (lint:design + tsc + Vite) — compile check
bun run preview        # serve the production build
bun run e2e            # Playwright suite (auto-spawns a dev server)
bun run e2e:perf       # the 120fps perf specs only, in isolation
bun run storybook      # Storybook dev server (see below)
bun run build-storybook # static Storybook build (compile check for stories)
```

Tests live in a `test/` subfolder per feature (`src/<feature>/test/`):
unit `*.test.ts` (Bun) and Playwright `*.e2e.ts` side by side. The e2e
suite is split into a parallel `functional` project and a serial `perf`
project that runs after it, so a single `bun run e2e` gives a clean pass
without the perf medians fighting the functional workers for CPU.

### Storybook

Component and library sandboxes (Storybook 9, `@storybook/react-vite`):

```bash
bun run storybook         # dev server on http://localhost:6006
bun run build-storybook   # static build into storybook-static/
```

`storybook dev` binds to **localhost:6006** (no `--host`), so on a remote
dev box port-forward 6006 to view it. Stories live next to the feature
they cover, in `src/<feature>/stories/*.stories.tsx`, and cover three
things:

- **Primitives** (`components/stories/`) — IconButton, Checkbox,
  NumberStepper, Tabs, Logo, the colour picker, with handlers wired to the
  Actions panel.
- **Feature components** — the PlaybackBar and a mixer InstrumentRow
  driven by real Document / Mixer / Viewport stores, plus the toolbar busy
  pills and theme picker.
- **Library sandbox** (`src/stories/jot_loader.stories.tsx`) — pick a
  `.jot` / `.mid` (or a built-in example), parse it, and view the Jot text
  alongside a live render.

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
provenance. The web app requests `debug=true` for every transcription so
the run's bundle is always saved (and auto-loaded back into the score on
completion); the **Load → debug bundle** menu item re-opens a saved one.

Full API docs: see [transcriber/README.md](transcriber/README.md).

### Stopping / cleaning up

```bash
docker compose down               # stop the service
docker compose down -v            # also delete the models-cache volume
                                  # (forces re-download next run)
```

---

## Project structure (brief)

Each domain is a folder; there are no barrel `index.ts` files (import from
the file that defines a symbol), and tests live in a per-feature `test/`
subfolder.

```
drumjot/
├── SPEC.md                Drum-DSL grammar (the source of truth).
├── AGENTS.md              Comprehensive context for AI / human contributors.
├── src/
│   ├── index.tsx          App entry (the only loose file besides global CSS).
│   ├── dsl/               Core DSL types (dsl.ts) + the Jot formatter.
│   ├── jot/               RenderedJot + layout pipeline (view_config,
│   │                      resolved_jot, pattern_expansion).
│   ├── parser/            DSL parser (TS).
│   ├── midi/              MIDI <-> Jot.
│   ├── rlrr/              Paradiddle <-> {Jot, MIDI}.
│   ├── lyrics/            Synced-lyrics parsing + alignment.
│   ├── linter/            Jot linter + rules.
│   ├── tempo/ tracks/ instruments/ grid/ dynamics/ selection/ fakes/
│   │                      Small shared domains (one file each, for now).
│   ├── utils/             General helpers (geom, zip, download, perf_probe).
│   └── jot_view/          The React app: jot_view.tsx + per-feature folders
│                          (mixer, playback, lyrics, score, toolbar, minimap,
│                          viewport, transcribe, provenance, document,
│                          settings, components, toasts). The transcriber
│                          HTTP client is jot_view/transcribe/transcriber.ts.
├── transcriber/           Python backend (FastAPI + Docker).
└── package.json           Bun-managed.
```

For a deeper dive (architectural decisions, the long-term Path A vs
Path B accuracy plan, known limitations, things to test next), read
[AGENTS.md](AGENTS.md).

---

## License

MIT.
