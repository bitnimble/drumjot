# Refinement pass: onsets (missing / extra / wrong hits)

You previously transcribed a drum recording into the Drumjot DSL. A
deterministic comparison between your transcription and the source
drum stems has flagged specific hits that look wrong - either present
in the source but missing from the Jot, or in the Jot but not in the
source.

{PARSE_ERROR_HINT}

## Current Jot

```
{CURRENT_JOT}
```

## Detected onset issues

Each issue lists a pitch and the absolute `time` (seconds), plus a
`notes` string that gives the `(bar, beat)` reference - that's the
coordinate you actually emit DSL in. Higher confidence = stronger
evidence the source disagrees with the Jot.

```json
{ISSUES_JSON}
```

## Your task

Revise the Jot so that:

- `missing_onset` issues are addressed by adding a hit at the
  referenced bar+beat on the listed pitch. The integer part of the
  beat number is the beat (1-indexed); the fractional part tells you
  where inside the beat the hit lands (0.000 = on the beat, 0.333 =
  triplet middle, 0.500 = the "and", 0.667 = triplet last / shuffle,
  0.750 = "a" of a 1/16 grid).
- `extra_onset` issues are addressed by removing the offending hit
  at the reported bar+beat.

**Hard constraints**:

1. Preserve the global metadata block exactly as-is.
2. Preserve pattern definitions and their references where they're
   still applicable. If addressing an issue would require breaking a
   pattern reference, prefer to add the change as an inline override
   via position substitution (`[Name#3=(...)]`) over inlining the
   entire pattern.
3. Don't introduce new pitches or instruments - only use pitches that
   already appear in the Jot.
4. If a flagged issue conflicts with a musically-obvious pattern
   (e.g. removing a hit would break a `(k.s.kks.)`-style groove
   identically in every bar), prefer to keep the original Jot and
   silently ignore that issue.
5. Address every issue in the input where doing so is musically
   plausible. Don't selectively fix only a subset.

Output **only** the revised Drumjot DSL. No commentary, no fences.
