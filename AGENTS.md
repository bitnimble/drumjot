# Drumjot, agent guide

Browser-based drum notation tool: a compact **DSL** for drum patterns, a
**React/MobX renderer**, and a Python **transcriber** that turns audio
into a predicted-onsets MIDI file. The DSL is the lingua franca all three
share. This file is the index, read the critical rules below for every
request, and pull in the linked docs when a task touches that area.

> `CLAUDE.md` is a symlink to this file. Edit `AGENTS.md`.

## Critical rules (apply to every request)

- **Use `bun`, never npm/yarn. Use `bunx`, never npx.**
- **After any code change, run `scripts/check*`**, never invoke
  pytest / bun test / tsc / stylelint / playwright directly. Those direct
  invocations are **denied** by the permission config; the scripts get
  venv activation, cwd, and autofix flags right. They autofix where
  possible and exit non-zero on first failure.
  - `scripts/check`, both sides (py then ts); default after a
    cross-cutting change.
  - `scripts/check-py [pytest args]`, ruff `--fix` + pytest. Needs
    `transcriber/.venv`.
  - `scripts/test-py [pytest args]`, pytest only (no ruff); for
    iterating on a failing test.
  - `scripts/check-ts [bun test args]`, stylelint `--fix` + tsc
    `--noEmit` + bun test.
- **Don't run `bun run dev`**, it's a human-only long-running watch. For
  a "does it compile?" smoke test use **`bun run build`** (lint:design +
  tsc + Vite), which is agent-friendly.
- **Any front-end change reruns the e2e suite.** After touching anything
  under `src/**` (`.ts`/`.tsx`/`.css`) or `tests/**` (shared fixtures),
  run **`bun run e2e`** (or a scoped `bun run e2e <spec>` while iterating,
  but a full pass before you claim done) and confirm it's green,
  `scripts/check-ts` (tsc + unit + lint) does **not** exercise real
  browser behaviour, so it can't catch a broken interaction, selector, or
  render. E2E specs live next to the feature they cover, at
  **`src/<feature>/tests/*.e2e.ts`** (shared fixtures in `tests/fixtures/`).
  The specs are coupled to the UI (toolbar menu structure, `data-testid`s,
  the ⋯ overflow): if your change moves or renames those, **update the
  affected specs in the same change** and rerun, don't leave them red.
  Needs the one-time Playwright setup (`bunx playwright install chromium`
  + `sudo bunx playwright install-deps chromium`).
- **No naked color literals in CSS modules**, `bun run lint:design`
  fails on hex / `rgb()`/`rgba()`/`hsl()`/`hsla()` in `src/**/*.css`
  outside `src/design_tokens.css`. Typography goes through `composes:`
  from `src/typography.module.css`; shared UI primitives live under
  `src/jot_view/components/`. See [docs/design-system.md](docs/design-system.md).
- **No DOM layout reads in hot paths**, never read `scrollLeft` /
  `clientWidth` / `getBoundingClientRect` / `getComputedStyle` (for
  layout) in any render, effect, MobX reaction, or per-frame/scroll/zoom
  path; read from `JotViewStore` observables instead. See
  [docs/architecture.md](docs/architecture.md#frontend-performance-model).
- **Frame budget is 120 fps / 8.3 ms**, not 60 fps (165 Hz monitor).
- **Browsers: evergreen, last 2 years** (`package.json` `browserslist`).
  Use modern web APIs without polyfills or feature-detection.
- **The transcriber is pure Python**, no bun, no TS in its runtime. It
  emits MIDI; `src/midi/from_midi.ts` converts MIDI→Jot on the frontend.
- **Python**: `transcriber/.venv` (uv) is the primary dev loop; invoke
  `python3`. **Don't install/upgrade deps unprompted**, install ordering
  is fragile; flag dep changes and let the user run them.
- **Don't read skill files with Read**, use the `Skill` tool.

### Workflow

The `superpowers:writing-plans` skill is **off by default** for this
project. Once `superpowers:brainstorming` converges on requirements, go
straight to implementation, no separate spec/plan doc. Reach for
`writing-plans` only when work genuinely needs review checkpoints
(multi-day, multi-agent, or risky cross-cutting); ask the user if unsure.

## Build / test / run

Frontend (`bun`, repo root):

| Command | What it does |
|---|---|
| `bun install` | Install deps. |
| `bun run build` | lint:design + tsc `--noEmit` + Vite build. Agent compile-check. |
| `bun run e2e` | Playwright suite (auto-spawns dev server). See below. |

`bun test` is scoped to `src/` via `bunfig.toml` and matches `*.test.ts`
unit files; Playwright covers the co-located `src/**/tests/*.e2e.ts`
specs (the distinct `.e2e.ts` suffix keeps the two runners from
overlapping, see `bunfig.toml`). Go through `scripts/check-ts` for the
post-change loop.

Transcriber (Docker or the local venv): see
[docs/transcriber-pipeline.md](docs/transcriber-pipeline.md#build--run).

**Sandbox** (`sandbox/Dockerfile`): a throwaway CUDA + Python container
carrying the *same* dep stack as the transcriber (uv-managed venv, torch
cu128, audio-separator, madmom, …) plus `bun`, but running nothing of
Drumjot. Use it to run experimental scripts against the real stack and
GPU, and to inspect prior transcribe output under the mounted
`/codebox-workspace`. Deps stay in lockstep automatically: the image
does `uv pip install -e .` against `transcriber/pyproject.toml`.

- `scripts/sandbox-py '<code>' [argv…]`, runs `python3 -c` in the
  container (extra args → `sys.argv[1:]`; reads stdin if no arg).
- `scripts/sandbox-bun '<code>' [argv…]`, runs `bun -e` (extra args →
  `Bun.argv[1:]`, like `node -e`; reads stdin if no arg).
- `scripts/sandbox-run <cmd…>`, exec any command (e.g. `nvidia-smi`).

**Pass non-trivial code as a file, not inline.** Only a simple one-liner
goes inline as `scripts/sandbox-py '<code>'`. For anything multi-line,
`Write()` it to a temp file in this repo under a **random/unique name**
(e.g. `tmp_a1b2c3.py`, not a fixed `tmp.py`; other agents may be running
concurrently and would collide) and run
`scripts/sandbox-py /abs/path/in/repo/tmp_a1b2c3.py`; the repo is
volume-mounted at the **same path** inside the sandbox, so it resolves
identically.
Multi-line inline code trips the harness into a permission confirmation
even though `scripts/sandbox-*` is allowlisted; a file argument doesn't.
Same for `scripts/sandbox-bun` (write a `.ts`/`.js` temp file) and shell
via `scripts/sandbox-run`. Delete the temp file when done.

The scripts auto-start the `drumjot-sandbox` container if stopped and
fall back to `sudo docker`; they print the build/run recipe if it
doesn't exist yet. Build context is the repo root (needs
`transcriber/pyproject.toml`): `docker build -f sandbox/Dockerfile -t
drumjot-sandbox .`.

**Code intelligence (LSP)**: the `LSP` tool is wired up (`tsgo` +
`pyright-langserver`). **Prefer it over Grep for symbol-level questions**
(definition, references, hover, call hierarchy), typed, no false
positives. Use Grep only for genuinely text-shaped queries (CSS class
names, log strings, cross-language DSL letters, vendored code).

**e2e (Playwright)**: headless Chromium against the Vite dev server, on a
headless box. Setup: `bunx playwright install chromium` + (one-time)
`sudo bunx playwright install-deps chromium` for system libs. Debug via
trace viewer (`bun run e2e:report` + port-forward 9323), no display for
`--headed`. Probe JS state through `window.drumjot` / `window.jotPlayer`
and `data-testid`s; the backend is stubbed per-spec via Playwright
`page.route` (e.g. the LRCLIB mock in `src/lyrics/tests/lyrics.e2e.ts`).
Audio *scheduling* is assertable; audio *fidelity* still needs a human
ear-check. Put **`E2E_DEBUG_BUNDLE`**=`/path/to/bundle.zip` in `.env` to
enable the opt-in "complete viewer" smoke test
(`src/playback/tests/debug_bundle.e2e.ts`, via
`src/playback/tests/debug_bundle.helper.ts::loadDebugBundle`); skipped
when unset, so
the bundle stays machine-local and uncommitted. The `e2e` script loads
`.env` via `bun --env-file` (bun's *implicit* load doesn't reach the node
Playwright subprocess), so plain `bun run e2e` picks it up.

## Detailed docs (pull in when relevant)

- [docs/architecture.md](docs/architecture.md), what Drumjot is, full
  repo layout, architectural decisions (DSL lingua franca, pure-Python
  transcriber, beat-relative coords, LLM-in-the-loop, perf model, browser
  targets).
- [docs/transcriber-pipeline.md](docs/transcriber-pipeline.md), the
  Python audio→MIDI pipeline, named-stage runner, `/transcribe/resume`,
  beat-grid invariants, debug/outputs folder layouts, accuracy Path A/B,
  things to test, open questions.
- [docs/design-system.md](docs/design-system.md), design tokens,
  typography classes, shared UI primitives, the stylelint rules.
- [docs/dsl-reference.md](docs/dsl-reference.md), DSL quick reference
  (the grammar is [SPEC.md](SPEC.md); examples are the parser tests).
- [docs/gotchas.md](docs/gotchas.md), known limitations and load-bearing
  invariants (renderer, playback, transcriber) + a break-glass checklist.
- [docs/score-stacking.md](docs/score-stacking.md), the score's z-index
  ladder + portal/clipping contract. **Read before adding any
  overlay/popover/badge that escapes its bar or row.**
- `transcriber/docs/ai-midi-to-jot-notes.md`, techniques captured from
  the deleted DSL-output pathway, for future AI-assisted MIDI→Jot work.
- `research/`, HIHAT, MODELS (Path B training plan), MIDI↔audio
  alignment, lyrics alignment.
