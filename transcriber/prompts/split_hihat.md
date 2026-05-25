You are an expert drum-audio analyst. An automatic separator produced a
single **hi-hat** stem and an onset detector listed every detected
hi-hat hit in it. The drummer plays a mix of **closed** hi-hat (foot
down, cymbals pressed together — short, articulate, ticky) and **open**
hi-hat (foot up, cymbals free to ring/sizzle together — long sustained
decay). The detector is imperfect: when an open hi-hat rings, its
sustained sizzle re-triggers the detector into a stream of **phantom
onsets** that don't correspond to anything the drummer played; the stem
also contains some **bleed** from louder instruments.

Your job is to **classify each onset into one of three buckets**:

1. **OPEN** — a real open hi-hat hit.
2. **CLOSED** — a real closed hi-hat hit. *(This is the default; you
   don't return these — every onset you don't list ends up here.)*
3. **DISCARD** — not a real hit. Reject it.

You do not transcribe, quantise, add, or move anything; you only label.

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
  climbed 10→90% of its peak. Real strikes are sharp (small `atk`); a
  sizzle bump *inside* a ring rises slowly (large `atk`) because the
  "peak" the detector picked is barely above the existing sizzle.
- `flat` — spectral flatness 0–1. Open hi-hat is broader/noisier
  (higher `flat`); closed has more defined transient structure.
- `cen` — spectral centroid in kHz (brightness). Open is typically
  brighter and broader; closed is bright but more compact.
- `gap` — seconds to the **nearest** neighbouring hi-hat onset. A tight
  closed pattern shows small gaps; an open hat tends to be more
  isolated or to start/end a fill; a sizzle train shows VERY small
  gaps (often <50 ms) between many high-`pre` onsets in a row.

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

## When to DISCARD an onset

This is where the prior version of this prompt got things wrong: phantom
sizzle inside an open tail looks superficially like a closed hi-hat
pattern (a string of tightly-spaced onsets). Treating them as closed
fills the closed lane with garbage. They are **not** closed hits — they
are detector artifacts. Discard them.

The clearest discard signatures:

- **Sizzle re-trigger inside an open tail.** Hallmark combination: high
  `pre` (we're sitting in a still-ringing open), low `late` *or*
  comparable to the surrounding noise floor, a slow `atk` (no fresh
  attack — the "peak" is just a bump on the existing sizzle), and the
  detected strength is weak relative to a nearby genuine open hit.
  Several of these often appear in a row at very small `gap` between an
  open accent and the next downbeat. If you see a stretch where every
  other feature says "we're inside an open hi-hat ring" *and* there is
  no fresh attack, those interior bumps are sizzle. Discard them.
- **If it looks like a closed hit but is sitting deep in an open
  passage with high `pre` and a slow / barely-there attack, prefer
  discard over closed.** A real fast closed hit between two open
  accents would still show a sharp `atk`; a sizzle bump does not.
- **Bleed.** A weak onset whose `b` position aligns (±a hair) with a
  hit on a *louder* instrument in the `others:` line, and that does
  not fit any plausible hi-hat pattern — typical: a faint "hi-hat"
  exactly under every snare backbeat with no matching `late` or `pre`.
- **Double-trigger.** Two onsets implausibly close together (`gap`
  < ~25–30 ms) where the drummer almost certainly struck once and the
  second is much weaker.

When **unsure between OPEN and DISCARD**, prefer DISCARD for an obvious
sizzle bump (low strength + high `pre` + slow `atk` + tiny `gap` to a
nearby genuine open) and OPEN for a hit with a real fresh attack. When
**unsure between CLOSED and DISCARD**, prefer CLOSED — removing a real
soft closed hit costs recall, and tail-filtering downstream catches the
worst lingering artifacts as a safety net.

Reason about the **pattern across bars**, not each hit alone: a long
run of evenly spaced onsets with tiny `late` and `pre` is closed even
if a few numbers wobble; a stretch where consecutive onsets all show
`pre` > 0.3 is an open passage even if their `b` positions don't seem
"accenty"; a tight cluster of weak high-`pre` low-`late` slow-`atk`
bumps right after an open accent is sizzle to discard.

Edge cases: a part may be **all closed** (return empty arrays for both
open and discard), **all open** (return every index in `open_indices`,
empty `discard_indices`), or contain **no real hits at all** in some
bars (all discard). Do not force a particular distribution.

## Onset data

{BARS}

## Output

Call `report_hihat_classification` with two arrays of `#N` indices:

- `open_indices`: the indices that are **open** hi-hat hits.
- `discard_indices`: the indices that are **artifacts** (sizzle bumps,
  bleed, double-triggers).

Every index in neither array is treated as **closed**. The two arrays
must be disjoint. Never invent an index that wasn't shown.
