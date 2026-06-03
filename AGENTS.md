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

`bun test` is scoped to `src/` via `bunfig.toml`; Playwright owns `e2e/`
(separate runner, they never overlap). Go through `scripts/check-ts` for
the post-change loop.

Transcriber (Docker or the local venv): see
[docs/transcriber-pipeline.md](docs/transcriber-pipeline.md#build--run).

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
and `data-testid`s; the backend is stubbed via
`e2e/helpers/transcriber-mock.ts`. Audio *scheduling* is assertable;
audio *fidelity* still needs a human ear-check.

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
