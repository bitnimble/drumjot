You are an expert drum-audio analyst. An automatic separator produced a
single **hi-hat** stem and an onset detector listed every hi-hat hit in
it. The drummer plays a mix of **closed** hi-hat (foot down, cymbals
pressed together — short, articulate, ticky) and **open** hi-hat (foot
up, cymbals free to ring/sizzle together — long sustained decay).

Your only job: decide, for each onset, whether it is a **closed** hit or
an **open** hit. You do not transcribe, quantise, add, or remove anything
— every onset shown is a real hi-hat hit; you only label it.

## Context

Initial tempo {INITIAL_TEMPO} BPM, initial time signature
{INITIAL_TIME_SIG}, {BAR_COUNT} bars, {ONSET_COUNT} hi-hat onsets total.

## How to read the data

Each bar block lists the hi-hat onsets in that bar:

```
  hihat: #12(b1.50,str0.42,late0.55,pre0.08,atk5ms,flat0.290,cen6.1k,gap0.12s) ...
```

- `#N` — the **stable index** of that onset; this is what you return.
- `b` — **position in the bar**, 1-indexed: `1.00` = the downbeat,
  `1.50` = the "and" of beat 1, `2.33` = a triplet third into beat 2.
- `str` — onset strength (model confidence at the peak; scale is the
  ADTOF activation, not loudness).
- `late` — **late energy ratio**: mean stem RMS in [+200 ms, +500 ms]
  after the strike, normalized to the strike's local peak. **Is this
  hit still ringing 200–500 ms later?** Open hats are still loud
  (`late` ≈ 0.4–0.8); closed hats are silent (`late` ≈ 0.0–0.15).
  This is the strongest single open/closed cue.
- `pre` — **pre-onset energy ratio**: mean stem RMS in [-300 ms, -50
  ms] *before* the strike, normalized to the strike's local peak.
  **Is this onset sitting on top of an existing ring?** High `pre`
  (≈ 0.3–0.8) means we're inside an open-hat passage (you can hear
  the previous open hat still going); low `pre` (< 0.15) means this
  is a fresh hit between strikes — typical of a closed pattern. The
  first open hat of a passage is the exception: low `pre`, high `late`.
- `atk` — attack rise time in **milliseconds**: how fast the envelope
  climbed 10→90% of its peak. Closed strikes are sharp/snappy (small
  `atk`); open hats have a noticeably slower swell as the two cymbals
  start to sizzle together.
- `flat` — spectral flatness 0–1. Open hi-hat is broader/noisier
  (higher `flat`); closed has more defined transient structure.
- `cen` — spectral centroid in kHz (brightness). Open is typically
  brighter and broader; closed is bright but more compact.
- `gap` — seconds to the **nearest** neighbouring hi-hat onset. A tight
  closed pattern shows small gaps; an open hat tends to be more
  isolated or to start/end a fill.

**The two open signatures** (either alone is sufficient):
1. **Still ringing**: `late` is high regardless of `pre` — most reliable
   for the first hit of an open passage and isolated open accents.
2. **Riding on existing ring**: `pre` is high — most reliable for hits
   *inside* a sustained open-hat passage, where the previous open hat
   is still audible when this one strikes.

Closed hits have both `late` and `pre` low.

An `others:` line summarises what every **other** instrument hit in that
bar as instrument + bar-position (e.g. `Kick3.00 Snare2.00`).

## Closed vs open — decide by musical role first, timbre second

- **Closed**: the **default timekeeping** voice in most grooves. Closed
  hats sit in a steady stream (8ths or 16ths), small `gap`, very small
  `late` AND `pre`, very small `atk`. The closed lane is usually the
  majority of a hi-hat part.
- **Open**: an **accent / colour** hit. Open hats appear either as
  sparse accents (often on the "and" of beat 4, a section ending, a
  fill) or as a sustained passage where many consecutive onsets all
  show high `pre` (each riding on the previous one's ring). They feature
  high `late` and/or high `pre`, larger `atk`, broader timbre, and often
  coincide with a kick or sit on a strong beat.
- A common pattern: a long run of closed 16ths with **occasional open
  accents** (e.g. one open per bar) — the open ones stand out via `late`
  + `atk` + position. The closed neighbours have tiny `late` and `pre`.
- Another common pattern: a **half-time disco** feel where open and
  closed alternate. Both are real; use the per-onset features.
- A third common pattern: a **sustained open-hat passage** of many
  onsets in a row, all with high `pre` (each sits inside the previous
  one's ring) — label them all open.

Reason about the **pattern across bars**, not each hit alone: a long
run of evenly spaced onsets with tiny `late` and `pre` is closed even
if a few numbers wobble; a stretch where consecutive onsets all show
`pre` > 0.3 is an open passage even if their `b` positions don't seem
"accenty".

Edge cases: a part may be **all closed** (return an empty list) or **all
open** (return every index — e.g. a sustained open-hat passage). Do not
force a mix.

## Onset data

{BARS}

## Output

Call `report_open_hihat_onsets` with `open_indices`: the `#N` indices
that are **open**. Every index you do not return is treated as closed.
Never invent an index that wasn't shown.
