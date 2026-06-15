# Known limitations & gotchas

Things that will bite if you don't know them. Present state of the
project, not history.

## Renderer / frontend

- **The TS renderer doesn't apply per-bar tempo to its pixel layout.**
  It only uses `barBeats` for bar widths (time signature, not bpm), so a
  Jot with tempo changes renders all bars at the same visual width even
  if real durations differ. Fine for notation; would need work for
  "playback-accurate" rendering. Two song-global scalars sit on top:
  `densityFactor` scales every bar by onset density, and `leadInPx`
  reserves pre-roll before bar 1 from `globalMetadata.songLeadIn`. Both
  derive at the bars' exact px/second
  (`(barWidth·densityFactor/4)·(bpm/60)`), that equivalence with
  `buildTimeline` is load-bearing for waveform/playhead alignment; change
  one side and you must change the other.

- **Lead-in / seek coordinate invariants.** Click-to-seek and the lead-in
  both assume `bar.x` is the single source of truth for the bars-row
  pixel space (origin after the gutter). The bars row needs the `leadIn`
  spacer as its first flex child or the flex-positioned bars desync from
  `bar.x` / the absolutely-placed playhead. `seekFromClick` bails on
  `[data-noseek]` ancestors (notes, pattern label), new clickable score
  chrome that shouldn't seek must opt out the same way. `xToTime` clamps
  the lead-in region to time 0; `timeToX` *does* map negative time into
  it for the playhead.

- **Tuplet bracket labels are the bare slot count** (`el.elements.length`).
  Detection is robust (any non-dyadic weight fraction → tuplet), so
  equal-note triplets/quintuplets label correctly. But a weight-expressed
  swung pair `(a_2 a)` is flagged and labeled **2** when musically it's
  triplet-based. `(a_3 a)` (3:1 dotted, dyadic) is correctly not flagged.

## Playback

- **`JotPlayer.currentTime` is in JOT time, not real time.** With
  `playbackSpeed < 1.0`, real elapsed seconds ≠ reported `currentTime`.
  `setPlaybackSpeed` mid-flight re-anchors so the playhead doesn't jump.
  Anything timing wall-clock events against it must divide by
  `playbackSpeed` first. **It can also go negative during the lead-in**, tolerate that.

- **Stop semantics need both halves.** smplr's `drums.stop()` only halts
  notes already sounding; future-scheduled notes (via
  `drums.start({ time })`) keep firing. `JotPlayer.stop()` collects the
  per-note stopFns returned by each `start` and invokes them all. Anyone
  adding a new scheduling path must push their stop fns onto
  `this.scheduledStops` for Stop to remain truthful.

- **The playback module assumes one global BPM.** The timeline anchors to
  `globalMetadata.bpm` because `toMidi` only emits one `setTempo` at tick
  0, per-bar `{{ bpm }}` overrides aren't carried into the MIDI bytes
  that drive playback. If MIDI export ever emits multiple tempo events,
  `playback/events.ts` and `playback/timeline.ts` need updating in
  lockstep.

- **smplr's TR-808 group names are non-obvious**: `hihat-close` (no
  trailing `d`), `mid-tom` (not `tom-mid`), no separate `ride` group
  (`drums.ts` falls back to `cymbal` for ride). Swapping kits
  (`Casio-RZ1`, `LM-2`, …) means re-verifying these against the new kit's
  `getGroupNames()`.

## Transcriber

- **madmom installs from a git main branch**, not PyPI (last PyPI release
  predates modern Python). The Dockerfile pins it. On ImportError,
  `beats.py` falls back to librosa beat tracking (no downbeat detection),
  degrading per-bar feel to default `straight16`. Second backend, Beat
  Transformer, is available via `settings.beat_tracker =
  "beat_transformer"` (feeds the same DBN).

- **Per-instrument stems are the input to onset detection, not the full
  drum mix.** Detector windows are tuned tight on this assumption, see
  [transcriber-pipeline.md](transcriber-pipeline.md#beat-tracking-load-bearing-invariants).

- **The hi-hat lane is a special case**, don't assume it behaves like the
  other ADTOF lanes. It runs on the isolated stem with its own looser
  `adtof_hihat_*` gates, an audio-domain onset supplement, and an energy
  floor (the ~14 kHz band-limit starves ADTOF of hat sizzle); open/closed
  is decided by a deterministic envelope guardrail that *overrides* the
  LLM, plus a discard-rescue. Full contract in
  [transcriber-pipeline.md](transcriber-pipeline.md#hi-hat-lane-load-bearing-specifics).

- **Beat-grid finalization order is load-bearing.** Preserve
  `tracker → align_beats_to_onsets → _finalize_bar_tempos →
  _pad_trailing_bars`. Full rationale in
  [transcriber-pipeline.md](transcriber-pipeline.md#beat-tracking-load-bearing-invariants).

- **Mid-bar tempo changes aren't handled.** The DSL spec allows it, but
  the parser snapshots metadata only at bar boundaries. Sub-bar
  `{{bpm:...}}` blocks parse but don't affect timing inside their bar.

- **Refinement-style assumptions about tempo**: if tempo is wrong
  (half/double confusion), onset reasoning sees hundreds of bogus
  positions. Watch for this on songs at unusual tempos.

- **`audio-separator[gpu]` install size** is ~3 GB of PyTorch + CUDA
  wheels, first Docker build is slow.

## Misc

- **`<input type="file">` accept is best-effort** across browsers (Safari
  especially is loose about MIME types for non-Apple formats). The
  `audio/*` wildcard catches most cases; the backend doesn't enforce
  format and uses ffmpeg for anything libsndfile rejects.

- **Long songs and token budget.** A 5-minute song's per-bar prompt fits
  Opus's context budget; a 10-minute song could push it. Batch by
  song-section if you hit limits.

- **F1 is noisy.** LLM output sample variation can mask small real gains;
  run multiple seeds when measuring whether a pipeline change helped.

## Sanity checklist if something breaks

- `bun run build` fails → usually a TS error from a type-rename ripple;
  `scripts/check-ts` gives the full list.
- A `src/parser/__tests__/` test fails → the parser is the most
  load-bearing TS piece; revert the last parser change and re-run.
- Transcriber Docker build fails on madmom → `beats.py` falls back to
  librosa automatically on ImportError; madmom can be removed from
  `pyproject.toml` temporarily.
- `bun test` shows 0 tests → check the `test` script and `*.test.ts` glob.
- Frontend can't reach the transcriber → confirm `docker compose up`
  finished startup and `curl http://localhost:8001/health` returns 200.
  Proxy target is in `vite.config.ts`.
