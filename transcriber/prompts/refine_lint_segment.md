# Refine: lint pass (segment patch)

You're patching a small region of a larger Drumjot transcription. A
deterministic linter found the issues listed below in this region. Your
job is to repair each issue by consulting the audio onset evidence — not
by inferring purely from the diagnostic text.

A lint error means this region contains something physically impossible
or musically nonsensical. The right fix is usually NOT "blindly delete
the offending modifier"; it's "figure out what the source audio actually
contains in that bar and re-encode it correctly". Some examples:

- `:o` (open) on a kick is invalid. This could mean any of several
  things: the LLM accidentally added the `:o` onto a real kick hit, the
  note was actually meant to be a hi-hat `h:o` and the pitch got
  misclassified, or it's something else entirely. Look at the onset
  evidence in that bar to decide which.
- A 3-hand simultaneity (`s+d+c`) could mean one of the three notes was
  actually a foot stroke that got classified as a hand (e.g. a hi-hat
  with `:f`), or that one of the three is spurious and should be
  dropped, or something else. Use the onset evidence to decide.
- A roll spanning two hand instruments could mean the roll boundary is
  in the wrong place, or that the roll should only cover one of the
  instruments, or something else. Consult the audio.

Use the **Audio onsets** block below as ground truth for what each bar
contains. Each onset is `(beat_in_bar, strength)`: a snare hit at
beat 2.0 with strength 11.0 is a more confident match than one at 2.05
with strength 3.0. If a bar shows onsets you didn't encode, prefer
adding them; if the bar shows fewer onsets than your Jot claims, prefer
removing the extras.

Severity meaning:
- `ERROR`: the chart is physically impossible or semantically broken.
  You MUST fix every error.
- `WARNING`: the chart is technically playable but suspicious. Fix when
  the audio context confirms it's wrong; otherwise leave as-is.

## Output contract

You will see a segment of the DSL below — typically one bar containing
the diagnostic plus one bar of read-only context on each side. The
segment starts at a `|` separator and ends just before the next `|`
(or at end-of-file if this segment ends the chart).

- Return **only** the corrected segment. Same number of bars, same
  opening structure, same trailing position. Do not add or remove `|`
  separators.
- Do not touch any bar that has no diagnostic targeting it — those bars
  are context, not edit targets. Their text should appear in your output
  unchanged.
- Preserve any inline `{{...}}` metadata blocks that fall inside the
  segment exactly as written.
- Do not output any explanation, code fences, or other commentary —
  just the corrected DSL fragment.{PARSE_ERROR_HINT}

## Segment to repair

```
{SEGMENT}
```

## Diagnostics in this segment

{LINT_DIAGNOSTICS}

## Audio onsets in the affected bars

{ONSET_CONTEXT}

Now output the corrected segment.
