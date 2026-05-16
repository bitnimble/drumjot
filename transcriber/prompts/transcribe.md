# Drumjot transcription (beat-aware)

You convert per-bar candidate drum onsets into a valid Drumjot DSL
string. Beat tracking has already run on the source audio - every
candidate is labelled with the **bar** it belongs to and its position
within that bar as a floating-point **beat number** (1.000 = on the
downbeat, 1.500 = the off-beat eighth between beats 1 and 2, 2.333 =
one-third of the way into beat 2, i.e. an 1/8 triplet position).

## Rules

1. The output must be **only the Drumjot DSL text** - no markdown
   fences, no commentary, no explanation. The first character must
   start the DSL.
2. Discard onset candidates whose `strength` is < 30 % of the median
   strength for their instrument (these are usually bleed or analysis
   noise).
3. Each bar in your output must correspond to a bar in the input list,
   in order. Do **not** add or remove bars.
4. Match each bar's content to the time signature and feel given in
   its header.
5. Use `+` between simultaneously-played notes (same beat position,
   different instruments). Example: `k+s` plays a kick and a snare
   together.
6. Detect repeating bar-level patterns. When two or more bars are
   byte-identical, emit them once as `[Name=(...)]` and reference
   with `[Name]` afterwards.
7. Add `:a` to notes whose strength is noticeably above the median
   for that instrument (>30 % above); add `:g` to notes noticeably
   below median (>30 % below). Don't tag every note.
8. Emit a global metadata block `{{ bpm: N, time: "X/Y",
   instrumentMapping: { ... } }}` at the top. Use the initial tempo
   and initial time signature from the input. The mapping covers the
   pitches you actually used (`k`=Kick, `s`=Snare, `h`=HiHat,
   `c`=Crash, `d`=Ride, `t`=Tom, `f`=FloorTom).

### Tempo and time-signature changes

When the input reports `tempo_changes: yes`, watch for bars whose
`tempo_bpm` differs from the previous bar by more than 2 BPM. Emit
an **inline** `{{ bpm: N }}` block **between** those two bars in your
output, just before the new bar's `|`. The new tempo stays in effect
for all subsequent bars until you emit another `{{ bpm: ... }}`.

Same for time signatures when `time_sig_changes: yes` - emit
`{{ time: "N/M" }}` between bars where the time signature changes.

### Choosing how to write each bar

Each bar's header includes a `feel` hint. Use it to decide how to
write that bar:

- **straight16**: write 16 elements (or fewer with rests) on the
  obvious 1/16 grid. Each beat is 4 elements; the n-th 1/16 of beat k
  is at position `(k-1)*4 + n`.
- **straight8**: 8 elements per bar (in 4/4), one per 1/8 note.
- **triplet**: prefer a `(...)_N` group for 1/8-triplet positions.
  Three triplet hits filling a half-bar (beats 1.000, 1.333, 1.667
  AND 2.000, 2.333, 2.667) become `(h h h h h h)_4` if all-hi-hat or
  `(k+s k+s k+s k+s k+s k+s)_4` for kick+snare unisons. A half-bar
  triplet fill (three hits across two beats) is `(... ... ...)_4`.
- **shuffle**: the off-beat lands at ~0.667 of the beat. Emit
  `(a . . . . a . a)*N` style swung 16ths, or use group weights to
  bias positions.
- **sparse**: not enough onsets to detect feel - just write the
  obvious hits with rests around them.
- **mixed**: the bar's onsets don't fit any single feel; write the
  hits at their beat positions using sub-bar groups as needed.

### Beat positions and grid math

If the candidate sits at beat `b.f` where `f` is the fraction:

- `f` close to 0.000 -> on the beat
- `f` close to 0.250 -> on the 2nd 1/16 of the beat ("e")
- `f` close to 0.333 -> 1/8 triplet middle ("trip-let middle")
- `f` close to 0.500 -> on the 1/8 ("and")
- `f` close to 0.667 -> 1/8 triplet last position OR shuffle off-beat
- `f` close to 0.750 -> on the 4th 1/16 of the beat ("a")

When in doubt, pick the closest grid position and round consistently
across the whole bar.

## Drumjot DSL grammar

{SPEC}

## Few-shot examples

{EXAMPLES}

## Input

Initial tempo: {INITIAL_TEMPO} BPM
Initial time signature: {INITIAL_TIME_SIG}
Tempo changes detected: {TEMPO_CHANGES}
Time-signature changes detected: {TIME_SIG_CHANGES}
Total bars: {BAR_COUNT}

Per-bar onset candidates. Each candidate is `(beat_in_bar, strength)`
where `beat_in_bar` is a 1-indexed float (1.000 = downbeat).

{BARS}

## Output

Now produce the Drumjot DSL.
