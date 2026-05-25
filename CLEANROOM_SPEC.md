# Drumjot, Cleanroom Specification

This document is the input for a cleanroom reimplementation of Drumjot. It
describes the application as a **black-box of user-visible behaviour and
data contracts**, deliberately avoiding any directive about how the
existing code organises itself. A reader of this document who has never
seen the codebase should be able to build a faithful working
implementation; a later reviewer can then diff that cleanroom build
against the existing one to surface bugs, dead code, and drift.

The companion file [SPEC.md](SPEC.md) is the authoritative DSL grammar
and should be treated as a normative reference. This document covers
everything else: the data model, the UI, playback, the transcriber
backend, the conversions in and out, and the invariants worth keeping.

---

## 1. What Drumjot is

Drumjot is a browser-based drum-notation tool composed of three layers:

1. **A textual DSL** for compact representation of drum patterns.
2. **A web app** that parses the DSL, renders it as a per-instrument-lane
   score, plays it back through a sampled drum kit, and mixes it against
   optional backing-audio tracks.
3. **A separate transcriber service** (HTTP API) that takes an arbitrary
   audio file and produces a predicted MIDI score, which the web app
   loads back as a Jot.

The DSL is the lingua franca: every conversion (audio, MIDI, RLRR
chart format) ultimately targets or originates from the DSL's in-memory
form, the **Jot**. The web app, the transcriber, and the conversion
utilities all agree on this shape.

---

## 2. The Jot data model

A **Jot** is the canonical in-memory representation of a drum score.
It is the result of parsing a DSL string, the input of every renderer
and exporter, and the output of every importer.

### 2.1 Top-level shape

```
Jot {
  title: string                       // Empty if not specified
  globalMetadata: Metadata            // Applies to the whole score
  patterns?: Record<string, Pattern>  // Named pattern definitions
  voices: Voice[]                     // Parallel tracks (one per `||` side)
}
```

A Jot has **at least one voice**. Voices play in parallel; the
playback length is the longest voice's length. Voices may carry a
display name (e.g. "Hands", "Feet").

### 2.2 Voices and bars

```
Voice {
  name?: string
  anacrusis?: Element[]    // Pickup elements before bar 1 (free length)
  bars: Bar[]              // Sequence of fixed-length bars
}

Bar {
  elements: Element[]      // Notes, rests, groups, etc.
  metadata?: Metadata      // Per-bar overrides (bpm, time)
}
```

The **sum of element weights** within a bar must equal the bar's
length in quarter-note beats, derived from the active time signature
(see §2.6).

### 2.3 Elements

Every element is one of five kinds. Notes and rests are leaves;
groups, simultaneities, and pattern references are recursive.

```
Note         { kind: 'note', pitch, modifiers?, sticking?, roll?, weight?, repeat?, metadata? }
Rest         { kind: 'rest', weight?, repeat? }
Simultaneity { kind: 'simul', elements: Element[] }       // Onset-aligned cluster
Group        { kind: 'group', elements: Element[], weight?, repeat?, roll?, modifiers?, metadata?, patternSource? }
PatternRef   { kind: 'patternRef', name, substitutions?, weight?, repeat? }
```

- `weight` is the relative duration of an element within its container.
  In a group of N elements with no weights, each occupies 1/N of the
  group's span. Explicit weights override this evenly distribution.
- `repeat` (the `*N` suffix) duplicates the element in place N times.
- A `Simultaneity` is what `a+b` produces: all inner elements share a
  single onset time.
- A `Group` (`(...)`) holds a sequence of elements that sub-divide a
  span. When a group's internal subdivision is non-dyadic (e.g. three
  notes in two beats) it is rendered as a tuplet (see §6.4).
- A `PatternRef` (`[Name]`) plays the named pattern. Substitutions
  (`[Name#3=(x)]`) replace positions in the pattern body before play.

### 2.4 Pitches, modifiers, sticking

A **pitch** is a single lowercase letter `a`–`z`. Pitches are
intentionally untyped; the same letter can mean different
instruments depending on `globalMetadata.instrumentMapping`. There is
no fixed pitch→instrument mapping at the DSL layer.

A **modifier** is one of a fixed set of single- or two-letter codes
attached to a note with `:`. Modifiers play three roles:

| Group | Codes | Effect |
|---|---|---|
| Velocity | `a` accent, `g` ghost | Loud or quiet relative to default |
| Hi-hat state | `c` closed, `h` half-open, `o` open, `f` foot, `s` splash | Selects a variant of the same letter |
| Articulation | `r` rim, `x` cross-stick, `z` buzz, `k` choke, `m` mute, `l` let-ring | Playing technique |
| Grace strokes | `fl` flam, `dr` drag, `rf` ruff | Adds a grace note before the main hit |

Modifiers chain: `s:a:r` is an accented rim-shot snare.

A **sticking** is `@r`, `@l`, `@rf`, `@lf` (right hand, left hand,
right foot, left foot). Sticking is metadata only; it does not
affect playback or MIDI export.

A trailing `~` marks a **roll/buzz**.

### 2.5 Metadata

Metadata can appear at four scopes with this precedence (most
specific wins): **note > group > per-bar > global**.

```
Metadata {
  // Timing
  bpm?: number | BpmTransition
  time?: TimeSignature            // { count, unit }, e.g. { count: 4, unit: 4 }

  // Volume
  vol?: Volume | VolTransition    // 'pp' | 'p' | 'mp' | 'mf' | 'f' | 'ff'

  // Instrument resolution
  instrumentMapping?: Record<string, Instrument>

  // Audio-timeline anchors (seconds from audio file t=0)
  drumsT0Sec?: number             // Time of first drum onset; bar 1 sits here
  signalT0Sec?: number            // First non-silent sample (informational)
  leadBars?: number               // Pre-drum bar count

  // Provenance / display
  title?: string                  // Lifted onto Jot.title if present
  comment?: string

  // Format-specific
  midi?: { note: number, velocity?: number, tick?: number }
  rlrr?: object

  // User extensions
  [string]: unknown
}
```

The two transition shapes (`BpmTransition`, `VolTransition`) describe
linear interpolation over a number of bars and exist so the DSL can
express "ritardando from 120 to 100 over 4 bars" inline.

### 2.6 Time signatures and bar length

```
TimeSignature { count: number, unit: number }   // numerator / denominator
barBeats = count * 4 / unit                     // length in quarter-note beats
```

So `4/4` → 4 beats, `7/8` → 3.5 beats, `5/4` → 5 beats. The bar's
element weights must sum to `barBeats`.

### 2.7 Instruments

A pitch letter resolves to an **Instrument** via
`globalMetadata.instrumentMapping`. The Instrument record describes
what to play for that pitch:

```
Instrument {
  kind: 'kick' | 'snare' | 'hihat' | 'ride' | 'crash' | 'tom' | 'custom'
  name?: string                   // Display label
  limb?: 'lh' | 'rh' | 'lf' | 'rf'
  midi?: { note: number, vol?: Volume }
}
```

When `instrumentMapping` is omitted the renderer falls back to
sensible defaults for the conventional letters (see §10 for the
canonical pitch alphabet).

### 2.8 Patterns

A named pattern is declared with `[Name=(...)]` at the top level and
referenced with `[Name]`. Definitions **do not play at their
position**; only references do. The older "definition implicitly
plays" semantic and the `?`-prefixed silent form have been removed
from the spec; to play a pattern at the point of declaration write
`[Name=(...)][Name]`.

Pattern substitutions replace individual element positions in the
referenced body without re-declaring it: `[Riff#3=(x)]` plays Riff
with element 3 replaced by `(x)`. Ranges are inclusive:
`[Riff#5-8=...]`.

### 2.9 Macros vs patterns

`[$name=...]` is a **textual preprocessor macro**: the definition is
removed and every `[$name]` is substituted as raw text before
parsing. Macros and patterns share the bracket syntax but not the
namespace.

---

## 3. Rendering a Jot

The Jot, once parsed, is converted to a layout suitable for display
in two passes. Both passes are pure functions of their inputs.

### 3.1 Structural pass (zoom-invariant)

Inputs: the Jot.
Outputs: per-bar, per-voice resolved beat-coordinates.

The structural pass:

- Inlines pattern references (each `[Name]` is expanded into a Group
  tagged with `patternSource: { name }`).
- Unrolls every `repeat` so the resulting tree has no `*N` left.
- Walks each bar's element tree and places each note at a `beat`
  position (in quarter-note beats from the bar's start), with a
  `duration` derived from its share of the bar.
- Detects **tuplet spans**: any non-pattern group whose internal
  weight fractions are non-dyadic becomes a `TupletSpan` covering
  the slot range, labelled with the count.
- Detects **pattern spans**: each expanded pattern reference becomes
  a `PatternSpan` covering the slot range, labelled with the pattern
  name. Identical names get a stable colour across voices and bars.
- Assigns each pitch a colour from an 8-slot palette in
  first-seen / mapping-order.
- Computes an **onset density** for the score (max onsets per beat
  across voices) and clamps it into a `densityFactor` in
  [0.4 .. 1.6] (1.0 ≈ 2 onsets/beat, typical rock).

### 3.2 Pixel pass (depends on zoom)

Inputs: the structural output + view config (bar width, palette,
typography).
Outputs: pixel positions for every note, bar, and span.

```
pxPerBeat = (barWidth * densityFactor) / 4
```

So sparse scores compress and busy ones expand, and the user's zoom
slider scales everything by adjusting `barWidth`. The pixel pass is
the only thing that needs to re-run on a zoom change.

### 3.3 Drum offset (user correction)

`RenderedJot` exposes a mutable `drumOffsetBeats` value the user can
adjust live via the toolbar's beat-offset spinner. The renderer
shifts every note by this many beats and rebuckets it into possibly
different bars. Notes pushed before bar 0 or past the last bar are
dropped. Pattern and tuplet spans are cleared while a non-zero
offset is active (a uniform shift can straddle barlines and
invalidate the original groupings). A baseline offset
(`drumOffsetBeatsBaseline`) carried by a debug bundle (see §7)
defines the zero point so the spinner shows 0 at the transcriber's
chosen alignment.

### 3.4 Lead-in bars

When `globalMetadata.drumsT0Sec > 0`, the score reserves pre-bar-1
space sized at the **bars' exact pixels per second** so a loaded
backing track's waveform lines up with the notation. Pre-bars may
come from a quantised MIDI lead-in (in which case they are real
bars with negative indices) or be synthesised as one hatched
spacer derived from `drumsT0Sec` at the global BPM.

The playhead is allowed to travel into the lead-in region; click-to-
seek clamps to the lead-in start.

---

## 4. The DSL syntax (quick reference)

[SPEC.md](SPEC.md) is normative. This is the cheat-sheet:

```
{{ globalKey: value, ... }}            global metadata block
[$name=...]                            macro definition (preprocessor)
[$name]                                macro reference (preprocessor)
[Pattern=(...)]                        pattern definition (silent)
[Pattern]                              pattern reference (plays)
[Pattern#3=(...)]                      pattern reference with substitution

| bar1 | bar2 | bar3 |                 bar separators
|| voice2 bar1 | voice2 bar2 |         second voice (parallel)

a                                      note on pitch a
.                                      rest
a + b                                  simultaneity (onsets align)
(a b c)                                group (3 notes share parent's slot)
(a_2 b)                                group, a takes 2/3 of the slot
a*3                                    repeat a three times
a~                                     roll/buzz
a:a                                    accented a
a:g:r                                  ghost rim a (modifier chain)
a@r                                    a on the right hand (sticking)
a{...}                                 per-note metadata
```

Whitespace is insignificant outside string literals. Modifier and
sticking suffixes attach to the immediately-preceding primary
element.

---

## 5. Format conversions

Every conversion goes through Jot. The web app supports four
import sources and three export sinks, plus a debug bundle import.

### 5.1 DSL text → Jot (user loads `.jot`)

The file is parsed; errors surface as a status pill carrying the
line/column and message. Macros are expanded before parsing. Pattern
definitions populate `jot.patterns`. Per-bar metadata snapshots
carry the active `time` and `bpm` onto each bar's metadata so
downstream consumers don't need to walk the bar tree to know what's
in force.

### 5.2 MIDI bytes → Jot (`fromMidi`)

Triggered by the "Load midi" toolbar action, by the transcriber
response (`prediction_midi_url` is fetched), or by a debug-bundle
load. Assumptions made by the converter:

- Only the drum channel (10 / index 9) is read; other channels are
  ignored.
- Notes are quantised to a 48-per-whole grid (the smallest division
  needed to represent triplets and 16ths in the same grid).
- Note durations are discarded; drums are one-shot strikes.
- The first `setTempo` becomes `globalMetadata.bpm`; later tempo
  changes become per-bar `bpm` overrides.
- `timeSignature` meta events trigger bar boundaries and per-bar
  `time` overrides.
- Velocity is preserved exactly at `note.metadata.midi.velocity`
  for lossless round-trip; it is also mapped to modifiers
  heuristically (≥100 → `:a`, <40 → `:g`).
- The exact original MIDI note number is preserved at
  `note.metadata.midi.note`; the pre-quantisation tick at
  `note.metadata.midi.tick` so the renderer can match notes against
  a separate provenance JSON.
- Leading all-rest bars are counted; the count surfaces as
  `globalMetadata.leadBars` and the time before the first drum onset
  becomes `globalMetadata.drumsT0Sec`.
- An `instrumentMapping` is synthesised from the GM percussion table
  for whichever MIDI notes appear in the file.

### 5.3 Jot → MIDI bytes (`toMidi`)

Outputs a Standard MIDI File, format 1, 480 PPQN, single drum track
on channel 10.

- One `setTempo` at tick 0 from `globalMetadata.bpm`; one extra
  `setTempo` per bar that carries a `bpm` override.
- One `timeSignature` at tick 0; one extra per bar that changes
  signature.
- For each note: `noteOn` at the note's tick with velocity derived
  from accent/ghost modifiers and the active `vol`, then `noteOff`
  at tick+1 (drums are one-shot).
- `note.metadata.midi.note` wins over the pitch's mapped note when
  present.
- `note.metadata.midi.velocity` wins over the derived velocity when
  present.
- The live `drumOffsetBeats` is **not** applied during export; the
  offset is a UI correction only.
- Sticking, modifier-only annotations that don't map to MIDI, and
  comments are not exported.

### 5.4 RLRR ↔ Jot, RLRR ↔ MIDI

The web app reads and writes the **Paradiddle RLRR** chart format
(a JSON song chart from the ParadiddleUtilities project). RLRR
files carry timed events in seconds with per-event velocity, plus a
BPM timeline.

Conversion contracts:

- `events[].time` (seconds) → beat positions within bars using the
  BPM timeline plus the time signature.
- `events[].name`/`class` (e.g. `BP_Snare_C`) → pitch letter via a
  fixed mapping table.
- `events[].vel` → velocity and accent/ghost modifiers (same
  heuristic as MIDI).
- Round-trip fidelity: Jot → RLRR → Jot is lossless for pitch,
  velocity, timing, and modifiers (Paradiddle-specific kit geometry
  is preserved as `metadata.rlrr`).

### 5.5 Audio file → transcriber → Jot

The full flow:

1. User uploads an audio file via the "Transcribe" toolbar
   dropdown.
2. The browser POSTs the file (plus the chosen `beat_input` and the
   `debug` flag) to the transcriber's `/transcribe` endpoint.
3. The response is a stream of NDJSON progress events; the toolbar
   shows the current stage / substage in a busy pill.
4. The final `result` envelope carries a `prediction_midi_url`, two
   audio stem URLs (`drum_stem_url`, `no_drums_url`), and; if
   `debug=true`; a `debug_zip_url`.
5. The browser fetches `prediction_midi_url` and converts the bytes
   to a Jot via §5.2.
6. The browser fetches `debug_zip_url` (if present) and unpacks it
   to seed per-stem audio tracks and the per-note provenance overlay.
7. The Jot renders; the score and waveforms align via `drumsT0Sec`;
   the user can play back immediately.

See §9 for the full transcriber pipeline.

### 5.6 Debug bundle → Jot + audio tracks + provenance

A debug bundle (`debug.zip`) packages a transcription's deliverables
plus inspection data:

```
prediction.mid                # The score MIDI
note_provenance.json          # Per-note keep/reject history
no_drums.mp3, stem_*.mp3      # Per-instrument backing audio
residual.mp3                  # Diagnostic (drum_stem − sum(stems))
debug.json                    # Manifest, stage timings, captured logs
```

The web app's "Load debug bundle" action:

- Reads `debug.json` for the manifest and the pitch → filename map.
- Parses `prediction.mid` into a Jot.
- Parses `note_provenance.json` into the renderer so each kept note
  shows its detected onset details, and rejected onsets render as
  dashed ghost overlays when the user enables "Show filtered".
- Materialises each unique mp3 as a `File` and wires it up as an
  audio track in the mixer, paired with its instrument lane via
  the `groupId` so each pitch's audio row sits next to its
  notation row.

---

## 6. The web app; frontend specification

The frontend is a single-page application with the following
top-level regions:

```
+--------------------------------------------------------+
| Toolbar                                                |
+--------------------------------------------------------+
| Score area (scrollable)                                |
|   Title / subtitle                                     |
|   Legend                                               |
|   Sticky gutter: timeline header + mixer rows          |
|   Bars row (per voice, per pitch, per audio track)     |
+--------------------------------------------------------+
| Debug panel (collapsible, only when bundle loaded)     |
+--------------------------------------------------------+
| Playback transport bar                                 |
+--------------------------------------------------------+
```

### 6.1 Toolbar

The toolbar is a horizontal strip at the top of the page. It
contains:

#### Load dropdown
Opens a panel with:
- **Examples submenu**; built-in example jots (the current
  selection is highlighted; choosing one loads it).
- **Load .jot file**; file picker for `.jot` / `.txt` files.
- **Load midi**; file picker for `.mid` / `.midi`.
- **Load ParaDB map (.zip)**; picker for a Paradiddle song
  package; loads its chart plus audio tracks for play-along.
- **Load debug bundle (.zip)**; picker for a transcriber debug
  bundle.
- **Load audio track(s)**; multi-select picker for any audio
  format. Each file becomes its own mixer row.

A semi-transparent modal overlay ("Loading X…") blocks clicks while
files decode. File pickers reset after selection so re-picking the
same file fires again.

#### Transcribe dropdown
Opens a panel with:
- **Beat input** selector: `full_mix` (default) or `drum_stem`.
- **Select file** picker; uploading sends to `/transcribe` and
  updates the status pill with the live stage/substage detail.
- **Resume previous run** controls (visible when prior runs
  exist):
  - A "previous runs" dropdown populated from the backend's
    `GET /transcribe/list` and refreshed on open.
  - A "stage" dropdown listing the six pipeline stages
    (`stems_all`, `stems_per`, `beats`, `onsets`, `filter`,
    `transcribe`); options for which the prior run lacks the
    required artifacts are disabled.
  - A **Resume** button, enabled only when both selections are
    present.
- A **Stop transcription** button while an upload is in flight.

#### Status pill (right of toolbar)
Hidden when idle. While uploading: filename + spinner + stage +
substage detail (e.g. "filtering 3/5 instruments (latest: snare)").
On success: green pill with filename, tempo, bar count, and any
"has tempo/time changes" notice, plus a `[debug.zip]` download
link if a bundle was produced. On error: red pill with up to 60
characters of the message. Click the pill to dismiss.

#### Zoom slider
Range 0.3× – 3.0×. Wheel scrolling (or Ctrl/Cmd+wheel) anywhere on
the score also zooms. The slider drives `barWidth`; the structural
pass does not re-run.

#### "Show filtered" checkbox
Visible only when a debug bundle with note-provenance is loaded.
When checked, dashed ghost overlays appear at each rejected
onset's position; clicking one opens a tooltip with the rejection
reason.

#### Drum loading indicator
Visible only the first time playback is started in a session
(while the drum SoundFont downloads). Shows a progress bar with
phases: connecting → downloading → decoding.

### 6.2 Score area

#### Title / subtitle / legend
Centred at the top of the score. Title defaults to "Untitled jot".
Subtitle shows tempo, time signature, and default volume marking.
The legend lists every pitch present in the score with its colour
swatch and instrument name.

#### Sticky gutter
A left-pinned column shared across the timeline header and every
mixer row. Its width is user-adjustable by dragging a vertical
resize handle on its right edge (clamped to roughly 128–480 px).

#### Timeline header
Top of the score, sticky to the top during vertical scroll. Shows
each bar's number (1-indexed) and the playback time at the bar's
left edge in `mm:ss` format. Clicking the timeline seeks (subject
to the data-noseek rule, §6.5).

#### Mixer (stacked rows)

Two **master rows** always sit at the top:
- **Audio**; master fader for all loaded audio tracks.
- **Drums**; master fader for all drum samples.

Below them, in user-configurable order, sit two row types:

**Audio track row**:
- Gutter: drag handle, filename (truncated with tooltip), volume
  slider, clear button (×), mute (M), solo (S).
- Bars area: a Canvas waveform of the loaded audio, drawn aligned
  to the bar grid; muted/excluded rows render dimmed.

**Pitch (drum) row**:
- Gutter: drag handle, pitch letter (and instrument name when
  mapped), volume slider, mute, solo. The position of the clear
  button is taken by an invisible spacer so the columns line up.
- Bars area: the per-pitch notes (see §6.3), pattern brackets,
  tuplet brackets, and bar lines.

The drag handles allow reordering rows by drag-and-drop. A
secondary keyboard affordance (Alt+Up/Alt+Down) exists for
accessibility but is not surfaced in the UI text.

Each row carries a `groupId`; consecutive rows with the same group
id render flush (no inter-row gap), so debug bundles cluster each
pitch's audio row directly next to its notation row.

### 6.3 Notes and brackets in the bars area

Each note renders as a coloured glyph at its `(bar, beat)` position
within the row. Variants:

| Modifier | Appearance |
|---|---|
| `a` (accent) | Stronger fill / outline |
| `g` (ghost) | Lower-contrast fill |
| `x` (cross) | Hollow circle |
| `fl` / `dr` / `rf` (grace) | A smaller grace glyph just before the main note |
| `~` (roll) | Extra roll decoration |
| Off-grid (non-dyadic beat) | Dotted outline, except when the note is already inside a tuplet bracket |

Sticking renders as a small badge (`R`, `L`, `RF`, `LF`) next to
the note.

**Tuplet brackets**: a thin numbered bracket spanning the slot
range of any non-dyadic group. The label is the bare slot count
(3 for a triplet, 5 for a quintuplet, …). Tooltip explains
"N-tuplet (not a straight subdivision)". Brackets are visual only;
they do not affect timing or playback.

**Pattern brackets**: a coloured box spanning the slot range of
every pattern reference. Multiple usages of the same pattern share
a stable colour. Clicking the pattern's label highlights every
other usage of the same pattern across the score (toggle).

### 6.4 Note interactions

- **Click a note**: selects it; a popover appears with the pitch,
  instrument, modifiers, sticking, and; when a debug bundle is
  loaded with provenance; a collapsible "debug details" block
  showing the detected onset's time, strength, kept/rejected
  status, beat-alignment offset, and final quantised position.
- **Hover a note**: shows the same tooltip without selecting.
- **Click-drag in empty score area**: draws a marquee rectangle
  that selects every note it overlaps. Clicking on a note clears
  the marquee.
- **Click anywhere with `data-noseek`** (notes, pattern labels,
  the playhead itself, the debug-bundle download button, ghost
  overlays): no seek occurs. Other clicks in the bars area seek.

### 6.5 Scrolling and panning

- Standard horizontal/vertical scroll via the usual controls.
- **Middle-mouse drag** pans both axes (cursor becomes "grabbing").
- During playback, the playhead is kept centred in the viewport; the score auto-scrolls horizontally to track it. The user can
  still scroll manually to break out of the auto-track.

### 6.6 Debug panel

Visible only after a debug bundle is loaded. Collapsed it is a
thin clickable strip at the bottom; expanded it shows two
columns side by side: **Stage timings** (each stage with its
elapsed seconds) and **Logs** (each entry prefixed with elapsed
seconds, a level badge, and the logger name). A top-edge drag
handle resizes the panel height.

### 6.7 Playback transport bar

Pinned to the bottom of the viewport. Contents from left to
right:

- **Play/pause/resume button**; single toggle. Glyphs:
  - Idle → ▶ (Play)
  - Playing → ⏸ (Pause)
  - Paused → ▶ (Resume)
  - Loading → ⏳ (disabled)
  - Error → ⚠ (disabled, hover for message)
  Disabled until a jot is loaded.
- **Stop button** ■; enabled only while playing/paused; clears
  playback and hides the playhead.
- **Master volume fader** (page-wide).
- **Drum kit dropdown** (after the first play, once the SoundFont
  exposes preset names: Standard, Room, Power, Electronic, TR-808,
  Jazz, Brush, Orchestra, …).
- **Speed dropdown**; fixed options 0.25, 0.5, 0.75, 1.0, 1.25.
- **Beat-offset spinner** (when a jot is loaded); units of
  quarter beats, no enforced range; commits on every change.
- **Audio-offset spinner** (when at least one audio track is
  loaded); units of seconds, min 0; live-edits `drumsT0Sec`.
- **Error indicator** (right side, only when something went wrong); red pill showing the truncated error message.

#### Global spacebar shortcut

Pressing space anywhere on the page (except in text inputs,
selects, or contentEditables) toggles play/pause/resume.

---

## 7. Playback subsystem

Playback presents four user-visible states:

| State | Trigger | Visuals | Allowed actions |
|---|---|---|---|
| **Idle** | Default; or after Stop | No playhead unless cued | Play (from start or cue), seek |
| **Cued** | Click to seek while idle | Playhead parked at click position | Play (from cue), seek |
| **Loading** | First Play of a session | ⏳ on the transport button + toolbar progress bar | None (button disabled) |
| **Playing** | Play from idle/cued/paused | Playhead animates, score auto-scrolls | Pause, Stop, Seek, M/S, faders, speed |
| **Paused** | Pause while playing | Playhead static | Resume, Stop, Seek, M/S, faders, speed |

### 7.1 Transport semantics

- **Play** from idle starts at the cued position if one exists,
  otherwise at `-drumsT0Sec` (so the playhead enters bar 1 exactly
  when the recording's drums would start) or at 0 if no lead-in.
- **Pause** freezes the playhead and stops audio elements explicitly
  (they have an independent media clock; suspending the AudioContext
  alone is not enough).
- **Resume** continues from the frozen position with no audible gap.
- **Stop** halts playback unconditionally and hides the playhead.
  Crucially, this must cancel both already-sounding notes *and*
  every future-scheduled drum onset (the underlying sampler's
  built-in stop typically only halts notes already sounding).
- **Click-to-seek** during play re-anchors the playback clock and
  reschedules all pending drum events; audio tracks seek to the
  matching media time. During pause, the same happens but the audio
  graph stays suspended; resuming continues from the new position.

### 7.2 Master / per-row mixing

There are three master gains (page, drums master, audio master)
plus per-row faders for every audio track and every pitch row. All
of them are live: adjustments mid-playback take effect with no
glitch.

Per-pitch default gains compensate for SoundFont imbalance: hi-hat
is ducked, kick is boosted, others at unity. These trims are baked
into velocity calculation so accent/ghost dynamics remain intact
when the user adjusts the row fader.

### 7.3 Mute / solo

Mute is per-row (silences immediately). Solo is **global**; soloing any row (drum or audio) puts the whole mixer into solo
mode; all non-soloed rows go silent. Explicit mute beats solo: a
muted-and-soloed row stays silent. Toggling either takes effect
mid-flight by cancelling and rescheduling.

### 7.4 Playback speed

Allowed speeds: 0.25, 0.5, 0.75, 1.0, 1.25 (no faster than 1.25
today). Pitch is preserved on both drums and audio:

- Drum samples play at native rate; the spacing between scheduled
  onsets is divided by the speed.
- Audio elements use `playbackRate` with `preservesPitch = true`.

Changing speed mid-playback re-anchors the playback clock so the
playhead does not jump.

### 7.5 Lead-in and click-to-seek interactions

The playhead is allowed to ride into the pre-bar lead-in region
(rendering at negative `currentTime` values). Click-to-seek
clamps to the lead-in start as the leftmost target; the user
cannot scrub past the audio file's t=0.

### 7.6 Audio track sync and drift correction

Audio tracks play via `MediaElementAudioSourceNode` so their
output is routed through the same Web Audio graph as the drums.
A periodic correction loop (every ~0.5 s) compares the element's
`currentTime` against the expected jot-time-derived position and:

- Within tolerance (<~40 ms): trim `playbackRate` by ≤1% to drift
  back into alignment without an audible step.
- Out of tolerance (>~500 ms; e.g. a backgrounded tab): hard-seek
  to the correct position.

### 7.7 Drum sounds and kit

The default drum kit is the **GeneralUser GS SoundFont**'s
percussion bank. The first play in a session downloads it
(~30 MB) with a visible progress indicator; subsequent plays use
the browser cache. After load, the kit dropdown exposes the
SoundFont's presets and the user can switch kits mid-playback.

The DSL pitch letters map to General MIDI percussion as follows:

| Pitch | Default GM note | Variants via modifier |
|---|---|---|
| `k` (kick) | 36 |; |
| `s` (snare) | 38 | `:x` → 37 (side stick) |
| `h` (hi-hat) | 42 (closed) | `:o` → 46 (open), `:f` → 44 (pedal) |
| `t` (tom) | 50 (high tom) |; |
| `f` (floor tom) | 41 |; |
| `c` (crash) | 49 |; |
| `d` (ride) | 51 |; |
| `p` (clap) | 39 |; |
| `b` (bell/cowbell) | 56 |; |

Grace strokes (`:fl`, `:dr`, `:rf`) emit a quieter additional note
~50 ms before the main hit.

---

## 8. The transcriber backend

The transcriber is an HTTP service (FastAPI in the reference
implementation, served over docker-compose during development).
The web app talks to it through the Vite dev proxy under `/api`.

### 8.1 HTTP API surface

#### `GET /health`
Readiness probe. Returns 200 once both separator models have
loaded.

```
{ "status": "ok", "gpu_available": bool, "gpu_name": string | null }
```

#### `POST /transcribe`

Multipart form:

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | audio | required | ≤200 MB, any format ffmpeg accepts |
| `beat_input` | enum | `full_mix` | `full_mix` or `drum_stem` |
| `include_candidates` | bool | `false` | Debug: include pre-LLM onset candidates in the response |
| `debug` | bool | `false` | Persist intermediate artifacts to disk + build the debug zip |

Response: **newline-delimited JSON stream**. Each line is one
envelope:

```
{"type":"stage","stage":"stems_all","phase":"start"}
{"type":"stage","stage":"stems_all","phase":"end","elapsed_seconds":75.2}
{"type":"substage","stage":"filter","detail":"filtering 3/5 instruments (latest: snare)"}
{"type":"result","data":<TranscribeResponse>}
{"type":"error","status_code":502,"stage":"filter","message":"..."}
```

The final result envelope shape:

```
TranscribeResponse {
  metadata: {
    initial_tempo: number
    initial_time_signature: [count, unit]
    duration_seconds: number
    stems_used: string[]                       // e.g. ["c","h","k","s","t"]
    bars: Array<{ bar, time_signature, tempo_bpm, feel, start_time }>
    has_tempo_changes: bool
    has_time_sig_changes: bool
  }
  candidates: Record<pitch, OnsetCandidate[]>  // empty unless include_candidates
  debug_dir: string | null
  drum_stem_url: string                         // "/outputs/<id>/drum_stem.flac"
  no_drums_url: string                          // "/outputs/<id>/no_drums.flac"
  prediction_midi_url: string                   // "/outputs/<id>/prediction.mid"
  debug_zip_url: string | null                  // "/outputs/<id>/debug.zip" if debug=true
}
```

All URLs are leading-slash paths designed to be appended to the
service's base URL.

Error status codes:

| Code | When |
|---|---|
| 400 | Missing required form field; resume artifact not found |
| 413 | Upload exceeds 200 MB |
| 500 | Local-compute stage failed (`stems_all`, `stems_per`, `beats`, `onsets`, `transcribe`) |
| 502 | LLM stage failed (`filter`); external dependency |
| 503 | Service misconfigured |

#### `POST /transcribe/resume`

Same form fields as `/transcribe`, minus `file`, plus:

| Field | Type | Required |
|---|---|---|
| `resume_folder` | string | yes; absolute path or bare folder name under the debug base |
| `resume_stage` | enum | yes; one of the six stage names |

Re-runs the pipeline starting at `resume_stage`, hydrating earlier
stages' outputs from the chosen folder. Stage-by-stage required
artifacts:

| Resume at | Required upstream artifacts |
|---|---|
| `stems_all` | input audio only |
| `stems_per` | `stems_all/drum_stem.<ext>` |
| `beats` | `stems_per/*.<ext>` (and the drum stem if `beat_input=drum_stem`) |
| `onsets` | `stems_per/*.<ext>` + `beats.json` |
| `filter` | `onsets.json` + `beats.json` + `stems_per/*.<ext>` |
| `transcribe` | `filter/kept_onsets.json` + `beats.json` |

Missing artifacts produce HTTP 400 with a message naming which
stage would regenerate them. Outputs of `resume_stage` and later
are overwritten; earlier outputs are preserved; re-resuming the
same folder is idempotent.

#### `GET /transcribe/list`

Returns recent transcriptions for the UI's resume picker:

```
[
  {
    "folder": "20260517-004530_a1b2c3d4_my-song",
    "original_filename": "my-song.mp3",
    "requested_at": "2026-05-17T00:45:30Z",
    "last_run_at": "2026-05-17T00:47:15Z",
    "last_resume_stage": "transcribe",
    "resumable_stages": ["stems_all", "stems_per", "beats", "onsets", "filter", "transcribe"]
  },
  ...
]
```

`resumable_stages` lists only the stages whose required upstream
artifacts still exist on disk.

### 8.2 Pipeline stages

Six stages in fixed order: each reads its inputs from prior
stages' outputs and writes new artifacts. A single stage failure
fails the whole pipeline (no fallback).

#### 1. `stems_all`; full-mix separation
Runs BS-Roformer SW (a 6-stem source separator) on the input
audio. Produces `drum_stem.wav` (isolated drums; fed to next
stage) and `no_drums.wav` (sum of bass + guitar + piano + other +
vocals; user deliverable as the "music minus drums" backing
track).

#### 2. `stems_per`; drum decomposition
Runs MDX23C 5-stem DrumSep on the drum stem. Produces five
per-instrument stems (`k`, `s`, `h`, `c`, `t`) plus a `residual.wav`
diagnostic (the drum stem minus the sum of the five; captures
auxiliary percussion and the separator's reconstruction error).
The cymbals stem (`c`) carries both ride and crash; the hi-hat
stem (`h`) carries both open and closed.

#### 3. `beats`; beat / downbeat / feel
Runs madmom (RNN + DBN) on whichever audio `beat_input` selects.
Detects beats, downbeats, per-bar tempo, per-bar time signature,
and per-bar "feel" (`straight16`, `straight8`, `triplet`,
`shuffle`, `sparse`, `mixed`). Snaps the whole grid by one
median offset to undo the activation-peak lag. Outputs a
`BeatStructure` saved to `beats.json` with bars, tempos, feels,
initial tempo, initial time signature. Also detects the audio
duration.

#### 4. `onsets`; onset detection + cymbal/hi-hat split
Runs ADTOF Frame_RNN per stem to detect onsets, then:

- **Cymbal split**: classifies each `c` onset as ride (`d`),
  crash (`c`), or discard, using deterministic features
  (decay, spectral flatness/centroid, gaps) plus an LLM
  classification pass.
- **Hi-hat split**: classifies each `h` onset as closed (`h`),
  open (`H`), or discard, using deterministic features plus an
  LLM classification pass.

Maps every surviving onset to `(bar, beat_in_bar)` using the
grid from stage 3. Outputs `onsets.json` (per-pitch kept onsets)
and `onsets_only.mid` (a diagnostic MIDI rendering of every
detected onset, no LLM filtering, so the operator can hear what
the detector heard).

#### 5. `filter`; per-instrument artifact rejection
Runs one Claude call per drum pitch in parallel (concurrency
gated by `INSTRUMENT_CONCURRENCY`). Each call receives the
instrument's onset list plus shared context (beat structure,
other instruments' onset positions for each bar) and returns
indices to reject as artifacts. The default is to keep; false
positives in rejection are worse than false negatives.

Failure modes (key missing, rate limit, timeout, tool-call
refusal) surface as HTTP 502; there is no fallback to "keep
everything".

Outputs the surviving onsets to `filter/kept_onsets.json`.

#### 6. `transcribe`. MIDI rendering + provenance
Pure deterministic render. Builds the MIDI from the kept onsets
using the per-bar tempo map, writes `prediction.mid`, and writes
`note_provenance.json` recording every onset (kept and rejected)
with its detected time, strength, beat position, MIDI tick (or
null for rejected), and rejection reason.

Per-pitch velocity is percentile-normalised (p10 → 64, p90 → 104,
clamped 1–127).

### 8.3 Disk layout

```
/outputs/<id>/                 # always populated, served by FastAPI StaticFiles
├── drum_stem.flac             # from stems_all
├── no_drums.flac              # from stems_all
├── stem_<k|s|h|c|d|t|H>.flac  # from stems_per (post-split)
├── residual.flac              # diagnostic
├── prediction.mid             # from transcribe
└── debug.zip                  # built post-pipeline when debug=true

/debug/<id>/                   # populated only when debug=true or DEBUG_DIR is set
├── input.<ext>                # raw upload
├── stems_all/{drum_stem,no_drums}.wav
├── stems_per/{k,s,h,c,d,t,H,residual}.wav
├── beats.json
├── onsets.json
├── onsets_only.mid
├── filter/kept_onsets.json
├── hihat_split/decision.json
├── cymbal_split/decision.json
├── prediction.mid
├── note_provenance.json
└── request.json               # filename, options, timings summary
```

Folder names are timestamp + short id + slug of the filename and
are stable across resume requests.

### 8.4 Configuration

The service reads configuration from environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Claude API key for filter + splits |
| `LLM_MODEL` | claude-opus-4-7 | Model name |
| `INSTRUMENT_CONCURRENCY` | 4 | Max parallel filter LLM calls |
| `BEAT_TRACKER` | `madmom` | `madmom` or `beat_transformer` |
| `BEAT_INPUT_DEFAULT` | `full_mix` | `full_mix` or `drum_stem` |
| `DEBUG_DIR` | unset | If set, every request persists debug artifacts even without `debug=true` |
| `OUTPUTS_DIR` | `/outputs` | Static-served deliverables |
| `MODELS_DIR` | `/models` | Cached separator weights (Docker volume) |
| `DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `WORKER_ROLE` | `pipeline` | `pipeline` (loads models, serves `/transcribe`) or `api` (lightweight) |
| `CORS_ORIGINS` | localhost:5173 | Allowed origins |

ADTOF tuning knobs (`ADTOF_*`) expose detector parameters; the
defaults are tuned for per-instrument stems (tight windows, low
threshold + LLM filtering does the cleanup).

### 8.5 Pitch alphabet produced by the transcriber

| Pitch | Source | Notes |
|---|---|---|
| `k` | stems_per kick | |
| `s` | stems_per snare | |
| `h` | hihat_split → closed | |
| `H` | hihat_split → open | Synthetic; frontend folds back to `h:o` |
| `t` | stems_per toms | |
| `c` | cymbal_split → crash | |
| `d` | cymbal_split → ride | |

The frontend's MIDI converter receives the post-split pitches and
should render `H` notes as `h:o` (open hi-hat).

### 8.6 Guarantees / non-guarantees

The transcriber **promises**:
- Stages always fire in fixed order.
- Same input + same config produces the same MIDI (modulo
  timestamp metadata in the bundle).
- Streaming progress events fire in real time.
- Resume from any stage is idempotent.
- Per-note audit trail in `note_provenance.json` covers every
  detected onset (kept and rejected).
- Stage names and required artifacts appear verbatim in error
  messages.

It does **not** promise:
- Transactional cleanup on client disconnect.
- Deterministic LLM filter decisions across invocations (Claude
  is non-deterministic; in practice decisions are stable for
  obvious cases).

---

## 9. Round-trip and lossiness summary

| Conversion | Lossless on | Lossy on |
|---|---|---|
| DSL → Jot | element tree, metadata snapshots | comments, whitespace, source structure (macros / patterns / repeats are expanded) |
| Jot → DSL | not provided by the spec | n/a |
| Jot → MIDI → Jot | pitch, onset beat, velocity, tempo, time sig (via `note.metadata.midi`) | sticking, modifier-only annotations, comments, the live `drumOffsetBeats` |
| MIDI → Jot | exact MIDI note + velocity + tick (preserved on metadata) | sub-grid timing (quantised to 1/48 of whole), note durations |
| Jot → RLRR → Jot | pitch, velocity, timing, modifiers | Paradiddle 3D kit geometry survives only as `metadata.rlrr` blob |
| RLRR → MIDI | tempo map, per-instrument velocity |; |

---

## 10. Canonical pitch alphabet

The DSL itself does not fix a pitch alphabet, but the rest of the
system (MIDI converter, transcriber, default kit) agrees on these
conventional letters:

| Letter | Instrument | Default GM note | Modifier variants |
|---|---|---|---|
| `k` | Kick | 36 |; |
| `s` | Snare | 38 | `:x` → 37 (side stick), `:r` → 38 (rim shot) |
| `h` | Hi-hat (closed) | 42 | `:o` → 46 (open), `:f` → 44 (pedal), `:h` (half) |
| `H` | Hi-hat (open) | 46 | Synthetic from the transcriber; folded back to `h:o` |
| `t` | High tom | 50 |; |
| `f` | Floor tom | 41 |; |
| `c` | Crash | 49 |; |
| `d` | Ride | 51 |; |
| `p` | Clap | 39 |; |
| `b` | Cowbell | 56 |; |

---

## 11. Invariants worth preserving

These are behaviours that the existing system gets right and that
a cleanroom implementation should also satisfy. They are the
load-bearing details a reviewer will check the cleanroom against.

1. **Bar/element weights sum to bar length.** After macro and
   pattern expansion and repeat unrolling, the weights in each
   bar sum to the bar's beat count (derived from its active time
   signature). The parser does not enforce this; the renderer
   should rely on it.

2. **Pattern definitions never play at their position.** Only
   references play. The older "definition is also a reference"
   semantic and the `?`-prefixed silent form are gone.

3. **Per-bar metadata snapshots.** The parser walks the bar tree
   carrying a running snapshot of the active `time` and `bpm`,
   and stamps each bar's metadata so downstream consumers don't
   have to re-walk to know what's in force.

4. **MIDI round-trip preserves exact note + velocity.** Achieved
   by stashing the raw values on `note.metadata.midi`; both
   numbers win over derived values during export.

5. **Beat-offset is a UI correction, not a stored mutation.** The
   spinner adjusts the renderer's beat offset only; export
   functions write from the source jot. The baseline lives on
   `drumOffsetBeatsBaseline` so the spinner reads zero at the
   transcriber's chosen alignment.

6. **Lead-in space is sized at the bars' exact pixels per
   second.** `pxPerBeat = (barWidth * densityFactor) / 4` is the
   single source of truth; the lead-in spacer and any loaded
   waveform must derive from the same formula so they align with
   the bar grid at any zoom.

7. **Click-to-seek opts out via `data-noseek`.** Notes, pattern
   labels, the playhead itself, ghost overlays, and the
   debug-bundle download link all carry this attribute. New
   clickable score chrome that shouldn't seek must opt out the
   same way.

8. **Stop kills both sounding and scheduled notes.** The
   sampler's own stop typically only halts sounding notes. The
   transport must collect per-note stop callbacks from each
   schedule call and invoke them all on Stop.

9. **Solo is global across drums and audio tracks.** Soloing a
   drum row silences audio tracks (unless one is also soloed),
   and vice versa. Mute beats solo (explicit mute always wins).

10. **Audio tracks sync via shared AudioContext clock.** A
    periodic drift-correction loop nudges `playbackRate` for
    small drift and hard-seeks for large drift.

11. **Playback speed preserves pitch on both drums and audio.**
    Drum samples play at native rate with onset spacing
    rescaled; audio uses `preservesPitch = true`.

12. **`/transcribe` and `/transcribe/resume` both stream NDJSON.**
    A single result envelope at the end carries the URLs and
    metadata; the UI reads stage events for live progress.

13. **Resume is idempotent and surfaces missing-artifact errors
    by stage name.** The 400 message must name the stage that
    would regenerate the missing artifact.

14. **The transcriber filter LLM does not silently degrade.** A
    failed Anthropic call surfaces as HTTP 502; there is no
    fallback to "keep everything", because that would silently
    deliver an artifact-heavy MIDI.

15. **Note provenance covers every detected onset, kept or
    rejected.** Tied back to MIDI ticks for the kept onsets so
    the renderer can match notes against the provenance JSON;
    rejected onsets carry the reject reason for the UI tooltip.

16. **Tuplet bracket labels are the bare slot count.** A
    deliberate simplification; the label is `el.elements.length`,
    not a musical ratio. Equal-note triplets / quintuplets label
    correctly (3, 5, …). Weight-expressed swung pairs label as 2
    even though they are musically triplet-based. This is the
    accepted trade-off; cleanroom should match.

17. **The transcriber's `H` pitch is folded to `h:o` on the
    frontend.** Open hi-hat is a synthetic routing pitch from the
    splitter and should not appear in the final rendered Jot as a
    distinct letter.

18. **Pre-bar lead-in supports negative `currentTime`.** The
    playhead is allowed to travel into the pre-bar space; the
    rAF clamp must permit negative time. Click-to-seek clamps to
    the lead-in start as the leftmost target.

19. **Each row carries a `groupId` for visual grouping.** Audio
    track and pitch row pairs from a debug bundle share an id so
    they render flush. Solo rows or rows with different ids get
    inter-row separation.

20. **Per-pitch default gains compensate for SoundFont
    imbalance.** Hi-hat ducked (~0.6×), kick boosted (~1.5×),
    others at unity. Applied at the scheduler level so
    accent/ghost dynamics are preserved.

---

## 12. Out of scope for the cleanroom

The cleanroom rewrite does **not** need to produce:

- The build system (Vite, bun, Docker, ruff/pytest configs).
- The exact ML model wiring (which separator library version,
  which madmom commit); the cleanroom can take any equivalent
  models that produce the same kind of output.
- The exact wire format of the existing in-flight progress JSON
  (the consuming agent should match the streaming-NDJSON shape
  in §8.1; but tiny field additions are fine).
- Performance / cost benchmarking.
- Tests for the existing code's internal helpers; instead,
  re-derive tests from this spec.

The cleanroom **does** need to produce, in priority order:

1. A faithful DSL parser per [SPEC.md](SPEC.md).
2. The Jot ↔ MIDI converters per §5.2, §5.3.
3. The score renderer with the regions and interactions per §6.
4. The playback subsystem per §7.
5. The transcriber HTTP surface per §8 (the pipeline can use any
   stack as long as the API contract and pitch alphabet match).
6. The debug-bundle import path per §5.6 and §6.2.
7. The RLRR converters per §5.4 if time permits.

---

End of cleanroom specification.
