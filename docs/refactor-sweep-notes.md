# Refactor sweep, working notes & findings

Autonomous session against TODO.md "refactor sweep" (bullets 20–26) +
unit-test cleanup + dedup audit + Storybook. Branch
`refactor/frontend-editing-prep`. This file accumulates:

- **Flagged** items: things I judged behaviour-/quality-risky to change
  autonomously, left for human review.
- **Dedup audit** findings (bullet 26).
- Per-feature extraction progress.

Each completed slice is a green-gated commit (`bun run build` +
`scripts/check-ts` + `bun run e2e`).

---

## Flagged for review (NOT changed autonomously)

_(none yet)_

---

## Dedup audit (bullet 26)

_(in progress)_

---

## Per-feature extraction progress

| feature | status | notes |
|---|---|---|
| playback | pending | |
| viewport | pending | |
| transcribe | pending | |
| lyrics | pending | |
| minimap | pending | |
| mixer | pending | |
| toolbar | pending | incl. DebugPanel → provenance |
| score | pending | big; split sub-components |
| contexts.ts split | pending | per-feature contexts |

---

## Unit-test cleanup

_(pending)_

---

## Storybook

_(pending)_
