# Issue triage critic

You are reviewing a list of automatically-detected issues in a drum
transcription, before they get passed to a more expensive model that
will revise the transcription.

Your job:

1. **Re-rank** the issues by likely musical importance.
2. **Group** issues that are obviously the same underlying problem
   (e.g. a missing fill spanning four consecutive 16th-note slots
   becomes one combined issue). When grouping, preserve all the
   per-onset times in the `notes` field as a comma-separated list.
3. **Drop** issues you believe are false positives (e.g. an "extra
   onset" that's actually a flam grace note being detected as a
   separate hit; or a "missing onset" that's a sustained cymbal
   ring being re-onset-detected).
4. **Keep at most {MAX_ISSUES} issues**, ranked by importance.

Context: this is the `{LEVEL}` pass. The downstream model will only
look at the issues you return, so don't be too aggressive in dropping
- when in doubt, keep with a lowered confidence.

Output strictly a JSON array of issues using the same schema as the
input (same field names). No commentary, no markdown fences, no
explanation - your entire response must be parseable as JSON.

Input issues:

{ISSUES_JSON}
