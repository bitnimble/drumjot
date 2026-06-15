# Drumming DSL — Spec v2

A textual DSL for representing drumming notation/tablature. Designed to be unambiguous, compact, and easy to manipulate programmatically (including by LLMs). Works equally as a single string or a multi-line file. Whitespace outside metadata strings is insignificant.

---

## Notes, rests, sequencing

- **Note**: a single lowercase letter `a`–`z`. Resolved via `instrumentMapping`.
- **Rest**: `.` — occupies one position.
- **Sequence**: notes/rests written one after another play in order.
- **Onset-aligned simultaneity**: `a+b` plays `a` and `b` at the same instant. `+` also works on groups, enabling polyrhythms: `(a a a)_4 + (b b b b)_4`.

## Groups

`(...)` groups elements. Groups can be nested. Within a group, positions are distributed evenly across the group's duration unless individual elements override with `_N`. A **top-level group** spans one bar of the current time signature.

## Bars

`|` separates bars. Content between two `|` must sum to one bar of the current time signature. Content **before** the first `|` is treated as an anacrusis/pickup and is not length-checked.

```
| k.s.kks. | k.s.k.s. |
```

## Global simultaneity

`||` plays the parts on each side in parallel for the entire track. Sides may have different lengths; they start together, each runs to its own end, and the track length equals the longer side. Use `||` for hand-vs-foot independence or stem-up/stem-down style voicing.

```
| h:c h:c h:c h:c h:c h:c h:c h:c |
||
| k...s...k...s... |
```

## Duration weight: `_N`

`_N` makes an element occupy N positions instead of 1. Applies to notes, rests, and groups.

```
(k . s . (k+s k+s k+s)_4)
```

The inner triplet occupies 4 of the 8 effective positions = half-bar triplet.

## Repeat: `*N`

Repeats the immediately preceding element N times.

```
(k.s.)*4
```

## Roll / buzz fill: `~`

Marks a roll (rapid multi-bounce fill). Combine with `_N` for explicit duration.

- `a~` — roll over 1 position
- `a~_4` — roll over 4 positions
- `(...)~` — roll across a group

For let-ring/sustain without bouncing, use the `:l` modifier.

## Modifiers: `:`

Modifiers attach to a note or group, separated by `:`. Multiple modifiers chain. Modifiers that do not apply to the target instrument are silently ignored at playback (e.g. `:o` on a snare).

**Single-character (common)**

| Mod | Meaning |
|---|---|
| `:a` | Accent |
| `:g` | Ghost |
| `:c` | Closed (hi-hat) |
| `:h` | Half-open |
| `:o` | Open |
| `:f` | Foot / chick |
| `:s` | Splash (foot open-close) |
| `:r` | Rim shot |
| `:x` | Cross-stick / stick click |
| `:z` | Buzz / press roll |
| `:k` | Choke |
| `:m` | Mute |
| `:l` | Let-ring / sustain |

**Multi-character (less common)**

| Mod | Meaning |
|---|---|
| `:fl` | Flam |
| `:dr` | Drag |
| `:rf` | Ruff |

Chaining: `s:a:r` = accented rim shot. `s:g:x` = ghost cross-stick.

## Sticking: `@`

Suffixes a note. `@r` right hand, `@l` left hand, `@rf` right foot, `@lf` left foot.

```
s:fl@l k@r s@r s@r:a
```

## Metadata

Two scopes, distinguished by brace count:

- **Note / group metadata**: `{ key: value, ... }` immediately follows a note or group.
- **Global metadata**: `{{ key: value, ... }}` applies to the rest of the track until overridden.

Precedence (highest to lowest): **note > group > global > instrumentMapping**. Applies to every key, including `vol`.

Special keys:

```ts
type Volume = 'pp' | 'p' | 'mp' | 'mf' | 'f' | 'ff';

type BpmTransition = {
  start?: number;
  end: number;
  duration: number; // bars
};

type VolTransition = {
  start?: Volume;
  end: Volume;
  duration: number; // bars
};

// One drum-kit instrument (kick, snare, hi-hat, ...).
type Instrument = {
  name?: string;
  limb?: 'lh' | 'rh' | 'lf' | 'rf';
  midi?: { note: number; vol?: Volume };
};

type Metadata = {
  bpm?: number | BpmTransition;
  vol?: Volume | VolTransition;
  time?: string;            // e.g. "4/4", "7/8"
  // Maps each lane letter to an Instrument. Order is the rendered lane order.
  instrumentMapping?: Record<string, Instrument>;
  comment?: string;
  // Three timeline epochs the playback / score / waveform code coordinates
  // around. All seconds, measured forward from t=0 of the loaded audio file
  // (audioT0, the origin by definition — no field). Expected ordering:
  // audioT0 (=0) <= signalT0Sec <= drumsT0Sec.
  //
  // drumsT0Sec — audio time of the first drum onset. Bar 1 of the score
  //   sits exactly here; the player delays its schedule by this much so
  //   rendered drums hit at the same wall-clock offset as in the source.
  //   Replaces the legacy `startOffset` field.
  // signalT0Sec — audio time of the first non-silent sample (e.g. a vocal
  //   or guitar pickup before drums). Informational today; the waveform /
  //   debug overlays may use it to mark "music starts here" distinct from
  //   "drums start here".
  // leadBars — number of pre-drum bars in the rendered score (those that
  //   sit before bar 1; they get negative bar indices). Carried so
  //   consumers don't have to recount the leading rest bars themselves.
  drumsT0Sec?: number;
  signalT0Sec?: number;
  leadBars?: number;
  // user-defined keys allowed
};
```

`time` and `bpm` may be set globally or per-group; they remain in effect until the next override.

## Patterns

`[Name=(...)]` defines a pattern silently — the definition itself does not play at its position. To play it at the same time as defining, follow it with an explicit reference: `[Name=(...)][Name]`.

Reference: `[Name]`.

**Identifier rules**: starts with a letter, allows letters/digits/`_`, minimum 2 characters, no whitespace, no reserved characters. Uppercase allowed.

**Manipulation**: `[Name#N=(...)]` returns a copy with position N replaced. Positions are 1-based, and each note, rest, group, and `+` simultaneity counts as one position. Descend into nested groups by chaining: `[Name#3#2=(...)]`. Range: `[Name#4-8=(...)]`. Multiple at once: `[Name#3=(...),#5=(...)]`.

Manipulation does not mutate the original; reassign with `=` to persist:

```
[Verse=[Verse#3=(k+s)]]
```

## Macros

`[$name=...]` defines a macro. Macros are **preprocessor substitutions**: the definition itself is removed from output, and every `[$name]` reference is replaced verbatim with the raw definition text before the full parse runs. The definition may be any text fragment (not necessarily a complete group).

Macro names follow the same identifier rules. Macros and patterns occupy separate namespaces (differentiated by `$`).

```
[$grv=k.s.kks.]
([$grv])*4
```

Macros are always silent by virtue of preprocessing — the definition itself produces no output, only its references do.

---

## Reserved characters

| Token | Purpose |
|---|---|
| `a`–`z` | Note lanes |
| `A`–`Z` | Allowed in pattern/macro identifiers (not as notes) |
| `0`–`9` | Numeric arguments (`*N`, `_N`, `#N`); allowed inside identifiers |
| `.` | Rest |
| `+` | Onset-aligned simultaneity |
| `\|` | Bar separator |
| `\|\|` | Global simultaneity |
| `(` `)` | Group |
| `*` | Repeat (`*N`) |
| `_` | Duration weight (`_N`) |
| `~` | Roll / buzz fill |
| `:` | Modifier prefix |
| `@` | Sticking prefix |
| `{` `}` | Note/group metadata |
| `{{` `}}` | Global metadata |
| `[` `]` | Pattern/macro define or invoke |
| `=` | Pattern/macro assignment |
| `#` | Pattern position access |
| `-` | Position range in `#N-M` |
| `,` | Separator in metadata and pattern arg lists |
| `$` | Macro identifier prefix |
| `"` | String delimiter inside metadata |

---

## Examples

### 1. Basic groove

```
{{ bpm: 120, time: "4/4",
   instrumentMapping: { k:{name:"Kick"}, s:{name:"Snare"}, h:{name:"HiHat"} } }}
| h:c h:c h:c h:c h:c h:c h:c h:c |
||
| k . s . k . s . |
```

### 2. Half-bar triplet

```
| k . s . (k+s k+s k+s)_4 |
```

### 3. Pattern with manipulation

```
[Groove=(k.s.kks.)]
[Groove]*3
[Groove#5-8=(k+s k+s k+s)_4]
```

### 4. 3:4 polyrhythm

```
| (a a a)_4 + (b b b b)_4 |
```

### 5. Flam, accent, sticking

```
| s:fl@l k@r s@r:a . k@r s@l k@r k@r |
```

### 6. Crescendo across two bars

```
{{ vol: { start: "mp", end: "ff", duration: 2 } }}
| h:c h:c h:c h:c | h:c h:c h:c h:c |
```

### 7. Anacrusis

```
{{ time: "4/4" }}
k k k | s . k . s . k . | s . k . s . k . |
```

### 8. Mixed meter

```
{{ time: "7/8" }}
| k . s . k k s |
{{ time: "4/4" }}
| k . s . k . s . |
```

### 9. Macro + pattern

```
[$std=k.s.kks.]
[Verse=([$std])*4]
[Chorus=([$std])*2 (k+s k+s k+s k+s)_8]
[Verse] [Chorus] [Verse]
```

### 10. Cymbal swell roll then choke

```
| c~_8:o | c:k . . . k . s . |
```
