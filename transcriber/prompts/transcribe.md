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
   - Do **not** emit a `title:` field in the metadata block. The
     transcriber assigns the title deterministically after the LLM
     work, so any title you invent is discarded; emitting one risks
     tripping safety filters on benign drum audio for no benefit.
   - Pick neutral, generic identifiers for pattern names: `[Groove]`,
     `[Verse]`, `[Chorus]`, `[FillA]`, `[Intro]`, `[Outro]`. Never
     invent narrative / lyrical / proper-noun names.
2. **Drop only onsets that look like detection errors** — typically
   bleed-through from another instrument (very low strength relative
   to that instrument's median), a doubled detection of one hit
   (two onsets within ~10 ms of each other), or analysis artifacts
   (isolated low-strength flickers between obvious downbeat groups).
   Do **not** drop an onset just because it doesn't fit a grid. Every
   surviving onset must map to exactly one note in your output —
   write it at the closest beat fraction the bar's subdivision can
   represent, and choose the subdivision (see below) to fit the
   onsets, never the other way around.
3. Each bar in your output must correspond to a bar in the input list,
   in order. Do **not** add or remove bars.
4. Match each bar's content to the time signature and feel given in
   its header.
5. Use `+` between simultaneously-played notes (same beat position,
   different instruments). Example: `k+s` plays a kick and a snare
   together.
6. Detect repeating bar-level patterns. When two or more bars are
   byte-identical, **define them silently** as `[?Name=(...)]` (note
   the leading `?`) at the top of the chart, and reference each bar
   that uses them with `[Name]`. The `?` prefix is mandatory — without
   it, the definition itself plays at the position it appears, which
   stacks every pattern body into the anacrusis and ruins playback
   timing.
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

- **straight16**: write 16 elements per bar (4/4), one per 1/16.
- **straight8**: 8 elements per bar, one per 1/8.
- **triplet** / **shuffle**: pure-triplet bars use 12 elements
  (three per beat).
- **sparse**: not enough onsets to detect feel - just write the
  obvious hits with rests around them.
- **mixed**: when a bar contains both binary (1/16) AND ternary
  (triplet) positions, use a 12-per-beat subdivision (48 elements
  per 4/4 bar). LCM of 4 and 3 = 12, so every common rhythm lands
  exactly on a grid slot without snapping.

The grid is **chosen to fit the onsets, not the other way around**.
If a bar's onsets all sit at clean 1/16 positions, write 16 elements.
If even one onset is a triplet, expand to 12-per-beat for that bar.
If a bar contains nested subdivisions inside a single beat (e.g.
a quintuplet fill), use a sub-bar group like `(k k k k k)` to host
the local subdivision.

### Beat positions

The candidate at beat `b.f` (fraction f, in quarter-note units) sits at:

- `f ≈ 0.000` — on the beat
- `f ≈ 0.083` — 1st 1/48 (binary 32nd or triplet 32nd region)
- `f ≈ 0.167` — 2nd 1/48 (1/12)
- `f ≈ 0.250` — 2nd 1/16 of the beat ("e")
- `f ≈ 0.333` — 1/8 triplet middle ("trip-let middle")
- `f ≈ 0.500` — 1/8 ("and")
- `f ≈ 0.667` — 1/8 triplet last position OR shuffle off-beat
- `f ≈ 0.750` — 4th 1/16 of the beat ("a")
- `f ≈ 0.833` — 11th 1/48 (1/12)

Pick the bar's subdivision so each surviving onset lands at the
nearest slot of that subdivision, AND no two onsets within the bar
collapse onto the same slot. If two onsets would collide at the
subdivision you're using, the bar needs a finer subdivision —
never drop one of them.

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
