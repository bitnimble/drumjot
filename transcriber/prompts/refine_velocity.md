# Refinement pass: velocity (dynamics)

You previously transcribed a drum recording into the Drumjot DSL.
A per-onset RMS comparison against the source stems has flagged
specific hits whose velocity in your transcription doesn't match the
audio.

{PARSE_ERROR_HINT}

## Current Jot

```
{CURRENT_JOT}
```

## Detected velocity issues

Each issue tells you a time (seconds) and a `(bar, beat)` reference
in the `notes` string. The current velocity is what the Jot encodes
today via modifiers + vol metadata; the expected velocity is what RMS
in the source audio suggests. A "gap" larger than ~20 is musically
audible; the LLM should adjust.

```json
{ISSUES_JSON}
```

## Encoding velocity in Drumjot DSL

Drumjot doesn't have explicit per-note numeric velocity. Encode
dynamics by:

- Adding `:a` (accent, +24 velocity above baseline) to notes whose
  expected velocity is significantly above ~85.
- Adding `:g` (ghost, -32 velocity below baseline) to notes whose
  expected velocity is significantly below ~55.
- Setting the global `vol:` metadata (`pp`, `p`, `mp`, `mf`, `f`,
  `ff`) for whole sections that play loud or soft.
- Removing `:a` from notes whose expected velocity is actually around
  baseline, and removing `:g` from notes that are actually louder.

## Your task

Adjust the modifier annotations on the flagged notes to make the
encoded velocity match the expected. Don't add or remove notes -
this pass is only about dynamics.

**Hard constraints**:

1. Don't change which slots have hits.
2. Don't change the global metadata block (except potentially adding
   a `vol:` if a whole section needs it).
3. Don't change pattern definitions - if a pattern reference would
   need a different per-occurrence dynamic, leave the pattern and add
   an inline `vol:` metadata on the reference instead.

Output **only** the revised Drumjot DSL. No commentary, no fences.
