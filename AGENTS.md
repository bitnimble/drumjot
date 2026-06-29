# Drumjot, agent guide

Browser-based drum notation tool: a compact **DSL** for drum patterns, a
**React/MobX renderer**, and a Python **transcriber** that turns audio
into a predicted-onsets MIDI file. The DSL is the lingua franca all three
share. This file is the index, read the critical rules below for every
request, and pull in the linked docs when a task touches that area.

> `CLAUDE.md` is a symlink to this file. Edit `AGENTS.md`.
>
> Cross-project conventions (close-the-loop, naming, run-the-checks,
> LSP-first, built-in-tools-over-bash, one-statement-per-Bash-call, Linear
> tickets, the store/presenter/component pattern + the DOM-layout-read ban,
> …) live in the user-level `~/.claude/CLAUDE.md` and are **not** repeated
> here; this file is the Drumjot-specific delta. (Note: agent tools that
> don't load `~/.claude/CLAUDE.md` won't see those rules.)

## Critical rules (apply to every request)

- **Use `bun`, never npm/yarn. Use `bunx`, never npx.**
- **After any code change, run the post-change checks.** Never invoke
  pytest / ruff / playwright directly, those are **denied** by the
  permission config. **Python** keeps wrapper scripts (venv activation,
  cwd, autofix); **TypeScript** has no wrapper, run the `package.json`
  scripts directly (see `.claude/settings.json` for the allow-list; raw
  `tsc`/`bunx stylelint`/`bunx vite` are redirected to these).
  - **TypeScript** (`bun run …`, repo root): `typecheck` (tsc `--noEmit`),
    `test` (bun unit tests; trailing args pass through, e.g.
    `bun run test src/editing/store.test.ts`), `lint:design` (stylelint
    `--fix`, autofixes the CSS). `bun run build` runs lint:design + tsc + Vite
    in one shot, the agent compile-check. There's no combined
    typecheck+test+lint command, run the ones the change touches (and
    always `bun run e2e` for `src/**`, below).
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
- **Don't run `bun run dev`**, it's a human-only long-running watch. For
  a "does it compile?" smoke test use **`bun run build`** (lint:design +
  tsc + Vite), which is agent-friendly.
- **Any front-end change reruns the e2e suite.** After touching anything
  under `src/**` (`.ts`/`.tsx`/`.css`) or `tests/**` (shared fixtures),
  run **`bun run e2e`** (or a scoped `bun run e2e <spec>` while iterating,
  but a full pass before you claim done) and confirm it's green,
  `bun run typecheck`/`test`/`lint:design` do **not** exercise real
  browser behaviour, so they can't catch a broken interaction, selector,
  or render. E2E specs live next to the feature they cover, at
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
  - **Flaky perf under load.** The per-frame medians measure main-thread
    busy time, so a loaded box (other builds, agents, GPU jobs) inflates
    them. If a `perf` spec fails but **all three** hold: (1) it's only
    *slightly* over (the failing metric is within ~2×, not an order of
    magnitude, off, and the median is still near budget), (2) you're
    confident your change touched **nothing** on the viewport / zoom /
    scroll / per-frame render path, and (3) the functional suite is green, then treat it as system load, not a regression: skip it and say so.
    Don't skip when your change *does* touch that path (re-measure on an
    unloaded box or against baseline) or when it's badly over budget.
- **No naked color literals in CSS modules**, `bun run lint:design`
  fails on hex / `rgb()`/`rgba()`/`hsl()`/`hsla()` in `src/**/*.css`
  outside `src/design_tokens.css`. Typography goes through `composes:`
  from `src/typography.module.css`; shared UI primitives live under
  `src/ui/<component>/` (each in its own folder, e.g.
  `src/ui/dropdown/dropdown.tsx` + `dropdown.module.css`; CSS-only
  primitives like `button`/`modal`/`spinner`/`form` too). See
  [docs/design-system.md](docs/design-system.md).
- **Frame budget is 120 fps / 8.3 ms**, not 60 fps (165 Hz monitor).
- **Browsers: evergreen, last 2 years** (`package.json` `browserslist`).
  Use modern web APIs without polyfills or feature-detection.
- **The transcriber is pure Python**, no bun, no TS in its runtime. It
  emits MIDI; `src/midi/from_midi.ts` converts MIDI→Jot on the frontend.
- **Python**: `transcriber/.venv` (uv) is the primary dev loop; invoke
  `python3`.
- **Long GPU runs need monitoring set up AT LAUNCH** (the general
  long-running-job rule lives in `~/.claude/CLAUDE.md`). Drumjot
  specifics: applies to ANY multi-hour GPU job, training sweeps AND data
  separation / dataset generation; the dangerous failure is a **silent
  CUDA hang** (process alive, `State R`, `nvidia-smi` ~99% util, GPU mem
  frozen, only the **log mtime going stale** reveals it). Full pattern +
  the kill-then-verify-GPU-freed-before-relaunch sequence live in the
  `long-run-monitoring` memory.
- **Record every GPU run in `training/RESULTS.md`, reproducibly.** After
  any proper GPU run (train/eval/sweep/probe) completes, append a dated
  entry proactively (no need to ask). Each entry MUST be re-runnable from
  itself: the **full verbatim command + all env vars** (`MODELS_DIR`,
  `DRUMJOT_STAR/ENST/EGMD/PARADB`, `PYTHONPATH`, …) **and a complete
  parameter list including defaults that aren't on the command line**
  (`--label-min-support`, `--lr`, `--batch-size`, `--layer`, `--high-band`,
  loss, sib weights, seed, es-* , encoder/layer/fps, box, bf16/TF32). When
  you **change a default param value**, retroactively annotate every past
  entry that used the old default with its explicit value, so historical
  numbers stay interpretable. See the `persist-gpu-results-in-results-md`
  memory.
- **Bash tool/search specifics.** `cat` / `sed` / `awk` / `find` stay
  redirected to `Read` / `Edit` / `Find`; their command-executing flags
  (`rg --pre` / `--pre-glob` / `--hostname-bin`, `git grep -O` /
  `--open-files-in-pager`) are **denied** in `.claude/settings.json` (they
  run an arbitrary command, so they're not read-only). Multi-line inline
  code trips a confirmation; put non-trivial scripts in a `tmp/*` file.

### Frontend store / presenter / component architecture

The frontend follows a strict three-layer split: per-domain data stores +
presenters + components. Code is grouped by **feature folder** under
`src/editing/<feature>/`, each folder holds that domain's
`<feature>_store.ts` + `<feature>_presenter.ts` + its view `.tsx`/`.css`
(e.g. `mixer/`, `playback/`, `lyrics/`, `viewport/`, `transcribe/`,
`structure/`, …). The loaded jot's composition root sits at the
`editing/` root: `jot_editor_store.ts` (`JotEditorStore`, the data store +
the `buildJotModel` peer constructor), `jot_editor_presenter.ts`
(`JotEditorPresenter`, load orchestration), `jot_editor_contexts.ts`. Shared
UI primitives live in top-level `src/ui/<component>/` (one folder per
component; stories in `src/ui/stories/`); the DSL layer (dsl / parser /
tempo / element-metrics) lives under `src/schema/dsl/` and RLRR under
`src/schema/rlrr/`; `settings/`, `toolbar/`, `ui/` are top-level peers of
`editing/`.

The cross-project store / presenter / component rules (three-layer split,
no-barrel-files, stores-data-only, presenters-mutate, acyclic deps,
per-domain presenters) live in `~/.claude/CLAUDE.md`; the layout above is
Drumjot's concrete realisation of them. Drumjot-specific exceptions worth
remembering: a memoised lazy cache like `getInstrumentTrack` is an allowed
read accessor on a store; most stores read `DocumentStore.currentJot`; the
presenter dependency graph is `leaf presenters → DocumentPresenter →
TranscribePresenter`.

## Build / test / run

Frontend (`bun`, repo root):

| Command | What it does |
|---|---|
| `bun install` | Install deps. |
| `bun run typecheck` | tsc `--noEmit`. |
| `bun run test` | bun unit tests (trailing args → single-file iteration). |
| `bun run lint:design` | stylelint `--fix` over `src/**/*.css` (autofixes). |
| `bun run build` | lint:design + tsc `--noEmit` + Vite build. Agent compile-check. |
| `bun run e2e` | Playwright suite (auto-spawns dev server). See below. |

`bun test` is scoped to `src/` via `bunfig.toml` and matches `*.test.ts`
unit files; Playwright covers the co-located `src/**/test/*.e2e.ts`
specs (the distinct `.e2e.ts` suffix keeps the two runners from
overlapping, see `bunfig.toml`). The frontend post-change loop is just
these `bun run` scripts directly (no wrapper script); run `bun run e2e`
for any `src/**` change.

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
`Write()` it into the gitignored **`tmp/` scratch folder** under a
**random/unique name** (e.g. `tmp/a1b2c3.py`, not a fixed `tmp/x.py`; other
agents may be running concurrently and would collide) and run
`scripts/sandbox-py /abs/path/in/repo/tmp/a1b2c3.py`; the repo is
volume-mounted at the **same path** inside the sandbox, so it resolves
identically.
- **All scratch goes in `tmp/`, NOT a `tmp_*`-prefixed file.** `Write()` /
  `Edit()` to any `tmp_*` path (repo root or nested) is **DENIED** by
  `.claude/settings.json` (`deny: Write/Edit(tmp_*)`), so a `tmp_foo.py` just
  gets blocked. Use `tmp/foo.py` (plain name, no `tmp_` prefix). Both `/tmp/`
  and `/tmp_*` are gitignored, so neither surfaces as untracked, but only
  `tmp/` is writable.
Multi-line inline code trips the harness into a permission confirmation
even though `scripts/sandbox-*` is allowlisted; a file argument doesn't.
Same for `scripts/sandbox-bun` (write a `tmp/*.ts`/`.js` file) and shell
via `scripts/sandbox-run`. Delete the scratch file when done (or just leave
it in `tmp/`, which is gitignored).

The scripts auto-start the `drumjot-sandbox` container if stopped and
fall back to `sudo docker`; they print the build/run recipe if it
doesn't exist yet. Build context is the repo root (needs
`transcriber/pyproject.toml`): `docker build -f sandbox/Dockerfile -t
drumjot-sandbox .`.

**Code intelligence (LSP)**: the `LSP` tool is wired up here with `tsgo` +
`pyright-langserver` (the global LSP-first rule applies, see
`~/.claude/CLAUDE.md`).

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
