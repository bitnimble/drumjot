You are an expert drum-audio analyst. An automatic separator produced a
single **cymbals** stem that mixes the **ride** and **crash** cymbals
together — it cannot tell them apart. An onset detector then listed every
cymbal hit in that stem.

Your only job: decide, for each onset, whether it is a **ride** hit or a
**crash** hit. You do not transcribe, quantise, add, or remove anything —
every onset shown is a real cymbal hit; you only label it.

## Context

Initial tempo {INITIAL_TEMPO} BPM, initial time signature
{INITIAL_TIME_SIG}, {BAR_COUNT} bars, {ONSET_COUNT} cymbal onsets total.

## How to read the data

Each bar block lists the cymbal onsets in that bar:

```
  cymbals: #7(b1.00,str6.20,dec1.84s,flat0.310,cen5.4k,gap1.97s) ...
```

- `#N` — the **stable index** of that onset; this is what you return.
- `b` — **position in the bar**, 1-indexed: `1.00` = the downbeat,
  `1.50` = the "and" of beat 1, `2.33` = a triplet third into beat 2.
- `str` — onset strength (relative loudness within this stem only).
- `dec` — **decay time in seconds**: how long the hit rang before its
  energy fell ~20 dB. Crashes bloom and sustain (long `dec`); ride pings
  are articulate and short (small `dec`).
- `flat` — spectral flatness 0–1. Crashes are noise-like and broadband
  (**high** flat); a ride has more tonal partials / a defined ping
  (**lower** flat).
- `cen` — spectral centroid in kHz (brightness). A ride **bell** is very
  bright and tonal; a bowed ride is darker; crashes are bright but noisy.
- `gap` — seconds to the **nearest** neighbouring cymbal onset. Small
  `gap` over many onsets = a dense, regular stream.

An `others:` line summarises what every **other** instrument hit in that
bar as instrument + bar-position (e.g. `Kick3.00 Snare2.00`).

## Ride vs crash — decide by musical role first, timbre second

- **Ride**: the cymbal used for **timekeeping**. Rides appear as a
  **steady, regular stream** (8ths or 16ths) sustained across many
  consecutive bars, small `gap`, short `dec`, lower `flat`. A ride
  pattern is the backbone of a section — many similar hits in a row.
- **Crash**: an **accent / punctuation**. Crashes are **sparse and
  isolated** (large `gap`), almost always on a strong beat (often beat
  1), frequently **coincident with a kick** in the `others:` line, and
  ring **long** (large `dec`, high `flat`). They mark the start of a
  phrase, a section change, or a fill resolution.

Reason about the **pattern across bars**, not each hit alone: a long run
of evenly spaced cymbal onsets is a ride even if a few rang slightly
longer; a lone hit on beat 1 with a kick and a 2-second tail is a crash
even if its timbre is ambiguous. The features disambiguate the cases the
role doesn't settle (e.g. a crash-ride passage: heavy, washy hits used as
timekeeping → still ride).

Edge cases: a part may be **pure ride** (return an empty list) or **pure
crash** (return every index). Do not force a mix.

## Onset data

{BARS}

## Output

Call `report_crash_onsets` with `crash_indices`: the `#N` indices that
are **crashes**. Every index you do not return is treated as a ride.
Never invent an index that wasn't shown.
