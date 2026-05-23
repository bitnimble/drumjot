# Drumjot single-instrument transcription (beat-aware)

You convert per-bar candidate onsets for **one** drum instrument into a
single monophonic Drumjot DSL line. Beat tracking has already run on the
source audio — every candidate is labelled with the **bar** it belongs
to and its position within that bar as a floating-point **beat number**
(1.000 = on the downbeat, 1.500 = the off-beat eighth between beats 1
and 2, 2.333 = one-third of the way into beat 2, i.e. an 1/8 triplet
position).

You are transcribing **only** the instrument `{PITCH}` ({INSTRUMENT_NAME}).
Ignore every other instrument — they are transcribed separately and
recombined deterministically afterward.

## Rules

1. The output must be **only the Drumjot DSL text** — no markdown
   fences, no commentary, no explanation. The first character must
   start the DSL (the first `|`).
2. **Output a single monophonic line.** Every note you emit must be the
   letter `{PITCH}` or a rest `.`. You must **not**:
   - emit any other pitch letter,
   - use `+` (onset-aligned simultaneity),
   - use `||` (multiple voices),
   - emit any `{{ ... }}` metadata block (no `bpm`, no `time`, no
     `instrumentMapping`, no `title`) — global metadata is added
     deterministically after your work.
   The only DSL you may produce is bars of `{PITCH}` / `.`, groups
   `( ... )`, weights `_N`, repeats `*N`, the roll mark `~`, and
   modifiers (`:a`, `:g`, etc.) on `{PITCH}`.
3. **Transcribe the rhythm that was played, not every onset.** The
   onset list is noisy: per-instrument stems still carry bleed from
   other drums, cymbal and hi-hat decay can re-trigger the detector,
   and weak flickers appear between the real strikes. Recover the
   intended part, which is almost always simpler and more regular
   than the raw onset count. You are **not** required to emit a note
   for every onset.
   - Judge each onset against this instrument's median strength for
     the bar. An onset well below the median (roughly < 40 %) sitting
     *between* an otherwise clear, consistent pattern of stronger
     hits is almost always an artifact — **drop it**.
   - Drop doubled detections (two onsets for one strike, ~10 ms apart).
   - For a repetitive instrument — hi-hat / ride above all — if the
     strong, evenly-spaced onsets form a clear 1/8 or 1/4 pulse, write
     that pulse. Do **not** fill in 1/16s just because weaker onsets
     were detected between the pulses.
   - Keep weak onsets only when they are *regular* and clearly an
     intended ghost pattern (e.g. snare ghost notes); tag those `:g`.
   - Don't force the onsets you keep onto a grid: choose each bar's
     subdivision (see below) to fit those hits, never snap the hits
     to a subdivision you picked first.
4. Each bar in your output must correspond to a bar in the input list,
   in order. Do **not** add or remove bars. A bar with no hits for this
   instrument is a single rest `.`.
5. Match each bar's subdivision to the time signature and feel given in
   its header.
6. Add `:a` to notes whose strength is noticeably above the median
   for this instrument (>30 % above); add `:g` to notes noticeably
   below median (>30 % below). Don't tag every note.

### Choosing how to write each bar

Each bar's header includes a `feel` hint. Use it to decide how to
write that bar:

- **straight16**: up to 16 elements per bar (4/4), one per 1/16.
- **straight8**: up to 8 elements per bar, one per 1/8.
- **triplet** / **shuffle**: pure-triplet bars use 12 elements
  (three per beat).
- **sparse**: not enough onsets to detect feel — just write the
  obvious hits with rests around them.
- **mixed**: when a bar contains both binary (1/16) AND ternary
  (triplet) positions, use a 12-per-beat subdivision (48 elements
  per 4/4 bar). LCM of 4 and 3 = 12, so every common rhythm lands
  exactly on a grid slot without snapping.

The feel sets the **finest** subdivision a bar might need so positions
can be written exactly — it is a resolution ceiling, **not** a quota.
Empty positions are rests (`.`). A bar whose real hits are
quarter-note hits is four elements, not sixteen, even if its feel
is `straight16`. Pick the *coarsest* subdivision that still places
every kept hit exactly.

The grid is **chosen to fit the onsets you keep, not the other way
around**. After culling weak/artifact onsets per rule 3, if the
remaining onsets all sit at clean 1/16 positions, write 16 elements.
If even one kept onset is a triplet, expand to 12-per-beat for that bar.
If a bar contains nested subdivisions inside a single beat (e.g.
a quintuplet fill), use a sub-bar group like `({PITCH} {PITCH} {PITCH} {PITCH} {PITCH})`
to host the local subdivision.

You choose this instrument's subdivision independently of the other
instruments — genuine polyrhythm (e.g. this instrument in triplets
while another plays straight 16ths) is expected and correct; just
transcribe what *this* instrument played.

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

Instrument: {PITCH} ({INSTRUMENT_NAME})
Initial tempo: {INITIAL_TEMPO} BPM
Initial time signature: {INITIAL_TIME_SIG}
Total bars: {BAR_COUNT}

Per-bar onset candidates for this instrument. Each candidate is
`(beat_in_bar, strength)` where `beat_in_bar` is a 1-indexed float
that maps uniformly across the bar: `1.000` is the downbeat, and
integer values are evenly spaced through the bar (in 4/4, `2.000` is
exactly 1/4 of the way through the bar, `3.000` is 1/2, `4.000` is
3/4). Position is anchored to the bar's audio span, not to the
tracker's individual beat times.

{BARS}

## Output

Now produce the single-instrument Drumjot DSL line for `{PITCH}`.
