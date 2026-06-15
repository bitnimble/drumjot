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
  - `scripts/check-py [pytest args]`, ruff `--fix` + pytest across **all
    first-party Python** (transcriber + `training/` + `dsp/`), all from
    `transcriber/.venv` (`training`/`dsp` go on `PYTHONPATH`). A bare run
    does the whole suite; pytest args target the transcriber tests only
    (fast single-test iteration) and skip training/dsp. **Torch caveat:**
    the training tests (and transcriber ADTOF tests) need a working CUDA
    torch in that venv; on a host without one, run them in the sandbox
    (`scripts/sandbox-run env PYTHONPATH=dsp:training python3 -m pytest
    training/tests`).
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
  **`src/<feature>/test/*.e2e.ts`** (shared fixtures in repo-root `tests/fixtures/`).
  The specs are coupled to the UI (toolbar menu structure, `data-testid`s,
  the ⋯ overflow): if your change moves or renames those, **update the
  affected specs in the same change** and rerun, don't leave them red.
  Needs the one-time Playwright setup (`bunx playwright install chromium`
  + `sudo bunx playwright install-deps chromium`).
  - **Two Playwright projects** (`playwright.config.ts`): `functional`
    (everything, fully parallel) and `perf` (the 120fps `perf.e2e.ts`).
    `perf` `dependencies: ['functional']` so it runs **after** functional
    finishes with the worker pool free, its per-frame medians are
    contention-sensitive, so it must not race the parallel functional
    workers. So `bun run e2e` gives a clean 26/26 in one shot. Iterate on
    perf alone with **`bun run e2e:perf`** (`--project=perf --no-deps`).
    Caveat: a functional failure skips the dependent `perf` project; fix
    functional first (or use `e2e:perf`).
- **No naked color literals in CSS modules**, `bun run lint:design`
  fails on hex / `rgb()`/`rgba()`/`hsl()`/`hsla()` in `src/**/*.css`
  outside `src/design_tokens.css`. Typography goes through `composes:`
  from `src/typography.module.css`; shared UI primitives live under
  `src/ui/<component>/` (each in its own folder, e.g.
  `src/ui/dropdown/dropdown.tsx` + `dropdown.module.css`; CSS-only
  primitives like `button`/`modal`/`spinner`/`form` too). See
  [docs/design-system.md](docs/design-system.md).
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
- **ALWAYS use the `LSP` tool to find symbols, never grep/text search.**
  For any symbol-level question, where is this defined, who references
  it, what's its type/signature, find every call site before a
  rename/refactor, the `LSP` tool (`goToDefinition`, `findReferences`,
  `hover`, `documentSymbol`, `workspaceSymbol`, `goToImplementation`,
  call hierarchy) is the **mandatory** first move. It's dramatically
  faster, more accurate, and far cheaper in tokens than sweeping the repo
  with `git grep`/Grep, and it won't miss call sites or trip over
  substring collisions (`parse` vs `JSON.parse`). Reserve text search for
  genuinely text-shaped queries only (CSS class names, log strings,
  cross-language DSL letters, vendored code). If you catch yourself
  grepping for a function/class/variable/type name, stop and use `LSP`.
- **Prefer built-in tools over ad-hoc bash.** Reach for `Read` / `Edit` /
  `Write` for files, the `LSP` tool for symbol-level questions
  (definition / references / hover), and the `Skill` tool for skills, before shelling out. Avoid ad-hoc shell scripting (`echo`/`printf`/`cat`/
  `sed`/`grep`/`find`/`for`-loops); the permission hook denies many of
  them outright and trips a confirmation on multi-line inline code. When
  you genuinely need a text search, a single clean `git grep` is fine, but
  default to the dedicated tools and the project's `scripts/*` wrappers.

### Frontend store / presenter / component architecture

The frontend follows a strict three-layer split: per-domain data stores +
presenters + components. Code is grouped by **feature folder** under
`src/jot_view/<feature>/`, each folder holds that domain's
`<feature>_store.ts` + `<feature>_presenter.ts` + its view `.tsx`/`.css`
(e.g. `mixer/`, `playback/`, `lyrics/`, `viewport/`, `transcribe/`,
`structure/`, …). The loaded jot's composition root sits at the
`jot_view/` root: `jot_view_store.ts` (`JotViewStore`, the data store +
the `buildJotModel` peer constructor), `jot_view_presenter.ts`
(`JotViewPresenter`, load orchestration), `jot_view_contexts.ts`. Shared
UI primitives live in top-level `src/ui/<component>/` (one folder per
component; stories in `src/ui/stories/`); the DSL layer (dsl / parser /
tempo / element-metrics) lives under `src/schema/dsl/` and RLRR under
`src/schema/rlrr/`; `settings/`, `toolbar/`, `ui/` are top-level peers of
`jot_view/`.

- **No barrel files.** Import every symbol straight from the module that
  defines it; never add or import a re-export barrel (`index.ts` /
  `store.ts`). Barrels hide the real dependency graph, invite import
  cycles, and bloat what each consumer pulls in.

When adding or moving frontend state/logic, follow it:

- **Stores = data only.** A store holds MobX `observable`s and
  `computed`s and nothing else: no actions, no setters/toggles, no
  clamping, no reactions, no `AbortController`s, no orchestration. Simple
  read accessors that just reshape store data are fine (a memoised lazy
  cache like `getInstrumentTrack` is a deliberate exception). Red/green
  flag: **stores have only observables + computeds; presenters may have
  reactions, autoruns, computeds, local observables, and actions.**
- **Presenters mutate stores.** Every mutation lives on a presenter, down to trivial `setX`/`toggleX`/clamp, plus all `reaction`/`autorun`,
  cross-store orchestration, and non-view bookkeeping (the in-flight
  `AbortController`s, etc.). Presenters are the only writers.
- **Components bind presenter methods to UI callbacks and derive store
  state into JSX.** They read stores (via per-store React contexts or
  props) and call `presenter.X` for actions.
- **No single top-level store.** Construct the peer stores where the view
  is created and pass each down independently (`viewportStore.foo`,
  `mixerStore.foo`), never through one aggregate `store`.
- **Acyclic dependencies.** A store may take a one-way reference to a
  peer it reads (e.g. most stores read `DocumentStore.currentJot`); the
  presenter depends on all stores; no store depends on a presenter. If two
  stores would form a cycle, extract the shared state into a third store
  both depend on.
- **Why:** this lets business logic be unit-tested against mocked stores,
  and components be rendered (tests / Storybook) with mocked stores +
  presenter, each concern swappable in isolation.
- **Per-domain presenters.** Each presenter owns one store + the
  cross-cutting orchestration for its domain (`settings`, `viewport`,
  `mixer`, `playback`, `provenance`, `lyrics`, `document`, `transcribe`).
  Where an action spans domains, the owning presenter calls a sibling
  presenter rather than writing the sibling's store directly, keeping the
  single-writer rule intact. The dependency graph stays acyclic
  (leaf presenters → `DocumentPresenter` → `TranscribePresenter`).

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
unit files; Playwright covers the co-located `src/**/test/*.e2e.ts`
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
- [training/README.md](training/README.md), the learned drum-onset model
  (frozen MERT encoder + per-lane heads): training loop, datasets
  (E-GMD/STAR), the Docker trainer, the ParaDB `.rlrr` test harness +
  per-instrument scoring, and current findings. Pure Python; not part of
  the transcriber runtime (the spike that wires a checkpoint into the
  pipeline is `transcriber/app/pipeline/learned_onsets.py`).
- `research/`, HIHAT, MODELS (Path B training plan), MIDI↔audio
  alignment, lyrics alignment.
