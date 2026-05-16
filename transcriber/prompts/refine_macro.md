# Refinement pass: macro (tempo / time signature)

You previously transcribed a drum recording into the Drumjot DSL. The
beat tracker on the source audio disagrees with the global metadata
you emitted - typically because the tempo is off, or you've mistaken a
half/double-tempo ambiguity.

{PARSE_ERROR_HINT}

## Current Jot

```
{CURRENT_JOT}
```

## Detected macro issues

```json
{ISSUES_JSON}
```

## Your task

Adjust the global metadata block (`{{ ... }}`) so the bpm and time
signature match the source. Leave the bar contents alone - if changing
the bpm requires re-bracing the note positions in source-aligned
seconds, the downstream onset pass will handle that.

If the issue looks like a half/double-tempo ambiguity, prefer the
faster of the two options when the original audio's snare backbeats
land on beats 2 and 4 (most popular music). Pick the slower option
when the backbeats would be on the second and fourth eighth of the
bar at the faster tempo.

Output **only** the revised Drumjot DSL. No commentary, no fences.
