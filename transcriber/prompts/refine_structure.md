# Refinement pass: structure (pattern factoring)

You previously transcribed a drum recording into the Drumjot DSL.
This pass is purely **representational** - the same notes must be
played at the same times after this pass; only the DSL structure
changes.

{PARSE_ERROR_HINT}

## Current Jot

```
{CURRENT_JOT}
```

## Refactoring hint

```json
{ISSUES_JSON}
```

## Your task

Look for bar-level repetition in the current Jot:

- Bars that are byte-identical across the song -> factor into a
  single `[Name=(...)]` definition and reference with `[Name]`.
- Bars that are identical except for the last 1-2 slots (typical
  "fill on bar 4" structure) -> define the base pattern and reference
  it with a position substitution: `[Name#15-16=(...)]`.
- Two-bar phrases that repeat at the verse/chorus level -> consider
  whether the larger phrase deserves its own pattern.

**Hard constraints**:

1. The notes that would be played by the new Jot must be exactly
   the same as the notes that would be played by the current Jot.
   Don't drop or add hits during factoring.
2. If the Jot doesn't contain genuine repetition (i.e. every bar is
   substantially different), output the Jot unchanged.
3. Pattern names follow the identifier rules: at least 2 characters,
   start with a letter, no whitespace.

Output **only** the revised Drumjot DSL. No commentary, no fences.
