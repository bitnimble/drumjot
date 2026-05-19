# Refinement pass: onsets (missing / extra / wrong hits)

You previously transcribed a drum recording into the Drumjot DSL. A
deterministic comparison between your transcription and the source
drum stems has flagged hits that *may* be wrong - either present in
the source but missing from the Jot, or in the Jot but not in the
source.

These flags are **evidence, not instructions**. The comparison is
purely numeric: it matches detected onsets and cannot tell a real
strike from cymbal/hi-hat decay re-triggering the detector, bleed
from another drum, or a doubled detection of one hit. You have the
musical context it doesn't. Treat each issue as a hypothesis to
evaluate, and apply your judgement about what was actually played.

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

For each issue, decide whether it reflects a real transcription error
or a detector artifact, then:

- For a `missing_onset` you judge **real**, add a hit at the
  referenced bar+beat on the listed pitch. The integer part of the
  beat number is the beat (1-indexed); the fractional part tells you
  where inside the beat the hit lands (0.000 = on the beat, 0.333 =
  triplet middle, 0.500 = the "and", 0.667 = triplet last / shuffle,
  0.750 = "a" of a 1/16 grid).
- For an `extra_onset` you judge **real**, remove the offending hit
  at the reported bar+beat.
- For an issue you judge to be a **detector artifact or musically
  wrong**, leave the Jot unchanged for that hit. This is expected and
  correct — do not add a note you don't believe was played just
  because it was flagged.

Use `confidence` as a prior, not a verdict: higher confidence means
stronger numeric evidence, but a confident flag on a hi-hat that would
turn a clean 1/8 pulse into 1/16 spam is still an artifact you should
reject. The most common bad flag is a weak `missing_onset` that sits
*between* the hits of an already-regular hi-hat/ride pulse — adding it
back is exactly the over-transcription to avoid.

Do **not** mechanically re-apply a change a previous pass already
considered and left out. If the Jot already represents a deliberate
musical reading and the issues would only push it back toward a
literal onset-for-onset dump, prefer keeping the Jot as-is. It is fine
to address none of the issues in a bar if none of them are real.

**Hard constraints**:

1. Preserve the global metadata block exactly as-is.
2. Preserve pattern definitions and their references where they're
   still applicable. If addressing an issue would require breaking a
   pattern reference, prefer to add the change as an inline override
   via position substitution (`[Name#3=(...)]`) over inlining the
   entire pattern.
3. Don't introduce new pitches or instruments - only use pitches that
   already appear in the Jot.

Output **only** the revised Drumjot DSL. No commentary, no fences.
