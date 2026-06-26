# Tom sub-classification (model post-process)

The learned onset model emits a single merged **tom** lane (`t`). This stage
splits each song's tom onsets into distinct toms (floor / low / mid / high) by
**per-song pitch clustering**, so the transcriber distinguishes a floor-tom
fill from a rack-tom fill instead of dumping everything as one tom.

It is part of **the model's post-processing**, not a separate transcriber
pipeline stage: it runs inside the learned-onset handoff
(`learned_onsets.detect_all_pitches_learned`), reusing the per-instrument tom
and kick stems the model already consumes. The transcriber gets sub-classified
toms "for free" via the model's output, no new stage, no new inputs.

## Why clustering, not a trained head

Tom pitch is **absolute per kit**; one kit's "high tom" is another's "low
tom", so a globally-trained per-tom classifier does not transfer. The split
has to be **relative, per song**: find how many distinct tom pitches this kit
used and order them. That is a clustering problem on per-onset fundamental
pitch, run independently per song. (A learned *embedding* could do better on
the hardest cases; see Limitations. We deliberately do not train one here.)

## Algorithm

Per song, given the model's tom-lane onset times + the isolated tom stem +
the kick onset times + the isolated kick stem:

1. **Kick-aware low-cut.** Measure the kick fundamental `kf` (pYIN over kick
   onsets on the kick stem). Set `fmin = max(40, kf * 1.2)` and high-pass the
   tom stem there. This removes kick bleed that otherwise contaminates the
   low end of tom hits (a high-tuned kick sits *inside* the tom range), while
   staying below any real tom (toms are tuned above their own kick).

2. **Per-onset pitch** (monophonic). For each tom onset, run pYIN over the
   decay window `[t+4ms, t+180ms]` (`fmin`..400 Hz) and take the median of the
   voiced frames as the fundamental, in semitones (`12*log2(Hz)`). Drop
   unvoiced onsets from the fit (they are re-attached at assignment).

3. **Valley-depth clustering** (auto-k). Recursively split the 1-D pitch
   population at the deepest density valley, accepting a split **only** when
   the valley is real: with a fixed-bandwidth (0.5 st) KDE, split at a valley
   `v` between peaks `lp,rp` iff `density(v) < 0.5 * min(density(lp),
   density(rp))`, the peaks are `>= 2.0 st` apart, and each side has `>= 3`
   onsets. Recurse on each side. This is the crux: it splits a genuine gap
   (e.g. a floor tom an octave below the rack toms) but refuses a tail-induced
   pseudo-mode (closely-tuned toms that only *look* bimodal), so it neither
   over-splits a single drum nor under-splits well-separated ones.

4. **Order + assign.** Order clusters low→high pitch; assign every onset
   (including the dropped unvoiced ones) to the nearest cluster centre.

5. **Map to GM tom notes** by rank (low→high), so the MIDI carries the tier:
   `1 tom → [50]`, `2 → [41,50]`, `3 → [41,45,50]`, `4 → [41,45,47,50]`.
   These flow through `onsets_midi.PITCH_TO_MIDI` → `gm.ts` (41/43 → floor
   `f`, 45–50 → rack `t`), so today's frontend renders floor-vs-rack; finer
   per-tom Jot lanes are a future frontend change the MIDI is already ready for.

Locked parameters: `CUT_MULT=1.2`, pYIN `fmax=400`, window `4–180 ms`,
KDE bandwidth `0.5 st`, valley ratio `0.5`, min peak separation `2.0 st`,
min cluster count `3`, analysis SR `22050`.

## Validation

Measured on the isolated tom stems (the real inference domain), scored against
ear-verified acoustic ground truth (degenerate kit-position labels merged):

- **Independent data (E-GMD, distinct-tom drum-module renders):** 35/40
  kit-count recovery, **0.72 ARI**. MDB 0.74, ENST 0.64.
- **Control set (12 ear-verified clips): 9/12 exact tom-count**, including the
  cases that defeated every simpler method simultaneously: recovers a sparse
  floor tom (Snowmine), keeps an octave-apart tom (FusionJazz), and refuses to
  split genuinely-degenerate tiers (Easton/Faces/Punk/FreeJazz).

The large gap between raw-label and acoustic-truth scores is **ground-truth
degeneracy** (datasets label kit position, not pitch), confirmed by ear, not
clustering error.

## Limitations (documented, by design)

- **Monophonic.** Two toms struck simultaneously read as one pitch; the second
  is missed. A tom's strong inharmonic overtone is indistinguishable from a
  concurrent second tom by peak magnitude or harmonic ratio, so simple
  multi-peak extraction does not work, true polyphony needs multi-F0 / NMF
  templates or a learned model. Out of scope.
- **Closely-tuned adjacent toms** (within the ~1–2 st per-onset pitch-noise
  floor) collapse together rather than split. This is the correct, safe
  behaviour: it never over-splits one drum into phantoms.
- **Harmonic-artifact phantoms** (rare kits where some hits read a strong
  non-octave partial) can add a spurious tier; resisted a per-onset octave
  fix.

The stage degrades gracefully: too few onsets, all-unvoiced, or any failure
falls back to the single merged `t` lane (current behaviour).
