/**
 * Time-to-pixel mapping for the playback playhead.
 *
 * Tempo follows `jot.tempoEvents` (sticky tempo changes anchored at
 * `(barIndex, beat)`, mid-bar precision) on top of the initial
 * `globalMetadata.bpm`. Browser playback does NOT go through the MIDI
 * bytes; the scheduler in `events.ts` walks the same per-bar tempo
 * segments so the playhead, the audio-track waveform and the scheduled
 * drums all share one clock.
 *
 * Jot-time anchor: bar 1 (= the first drum bar, `voice.bars[leadBars]`)
 * sits at jot time 0 by construction. That coincides with the audio time
 * stored in `globalMetadata.drumsT0Sec` so the player's
 * `media = jot + drumsT0Sec` identity lines the synth drums up with the
 * recorded audio drums. Pre-drum lead-in bars (if any) sit at negative
 * startSec; their notes (rare, but possible if an upstream generator
 * stamps drums into a pre-drum bar) fire at negative jot time, which
 * maps back to positive media time during playback. Anacrusis, when
 * present, is itself drum content; under the drums-t0 convention it
 * sits at jot 0 alongside bar 1's downbeat.
 *
 * Pixel offsets are looked up against the LIVE `RenderedJot` on every call
 * to {@link timeToX}, not cached at build time, so the playhead stays in
 * sync with the score even when the user zooms during playback (the
 * `ViewConfig.barWidth` change reflows the rendered bars reactively and
 * the playhead re-reads their new `x` / `width`).
 */
import { TimeSignature } from 'src/dsl';
import { Pixels, RenderedJot, px } from 'src/jot';
import { buildBarTempos, resolveBpm } from 'src/tempo';

export { resolveBpm };

/**
 * Pick the BPM and time signature the song spends the largest proportion
 * of its audio duration in, excluding pre-drum lead-in bars (negative
 * `index`) whose bpm is artificially scaled to fit the audio pre-roll.
 *
 * Used by the subtitle formatter (cosmetic), and by `store.setDrumOffset`
 * to convert a beat-shift delta to the audio-compensation seconds.
 * `globalMetadata.bpm` is the initial tempo (before any tempoEvent
 * fires), which for transcribed bundles is the back-solved lead-in
 * tempo and can differ markedly from the song's playing tempo.
 */
export function pickDominantBpmAndTime(jot: RenderedJot): {
  dominantBpm: number | undefined;
  dominantTime: TimeSignature | undefined;
} {
  const voice = jot.structure.voices[0];
  if (!voice || voice.bars.length === 0) {
    return { dominantBpm: undefined, dominantTime: undefined };
  }
  const tempos = buildBarTempos(jot.source, voice.bars);
  const bpmDur = new Map<number, number>();
  const timeDur = new Map<string, { time: TimeSignature; duration: number }>();
  for (let i = 0; i < voice.bars.length; i++) {
    const bar = voice.bars[i];
    if (bar.index < 0) continue;
    const barTempos = tempos[i];
    for (const seg of barTempos.segments) {
      const segDuration = (seg.endBeat - seg.startBeat) * (60 / seg.bpm);
      const bpmKey = Math.round(seg.bpm);
      bpmDur.set(bpmKey, (bpmDur.get(bpmKey) ?? 0) + segDuration);
    }
    const timeKey = `${bar.time.count}/${bar.time.unit}`;
    const prev = timeDur.get(timeKey);
    if (prev) prev.duration += barTempos.durationSec;
    else timeDur.set(timeKey, { time: bar.time, duration: barTempos.durationSec });
  }
  let dominantBpm: number | undefined;
  let bestBpmDur = -Infinity;
  for (const [bpm, dur] of bpmDur) {
    if (dur > bestBpmDur) {
      bestBpmDur = dur;
      dominantBpm = bpm;
    }
  }
  let dominantTime: TimeSignature | undefined;
  let bestTimeDur = -Infinity;
  for (const { time, duration } of timeDur.values()) {
    if (duration > bestTimeDur) {
      bestTimeDur = duration;
      dominantTime = time;
    }
  }
  return { dominantBpm, dominantTime };
}

export type BarTiming = {
  startSec: number;
  durationSec: number;
};

export type JotTimeline = {
  totalDurationSec: number;
  bars: BarTiming[];
  /**
   * Reference to the laid-out jot whose bars drive playback. Held by
   * reference so `timeToX` always reads current `bar.x` / `bar.width` —
   * critical for zoom changes during playback.
   *
   * `undefined` only on {@link EMPTY_TIMELINE}.
   */
  rendered: RenderedJot | undefined;
};

export const EMPTY_TIMELINE: JotTimeline = {
  totalDurationSec: 0,
  bars: [],
  rendered: undefined,
};

export function buildTimeline(rendered: RenderedJot): JotTimeline {
  // The audio-time fields we need (`bar.beats`, `bar.index`,
  // `jot.tempoEvents`, `globalMetadata.bpm`) all live on the structural
  // cache or on the source jot, which are zoom-invariant. Reading from
  // `rendered.structure` rather than `rendered.resolved` means observers
  // calling `buildTimeline` don't pick up a spurious dependency on
  // `viewConfig.barWidth`, so the per-bar timings stay stable across
  // wheel ticks.
  const structure = rendered.structure;
  // All voices in a Jot share the same bar grid (they're laid out from the
  // same global metadata), so the first voice's timing is canonical.
  const voice = structure.voices[0];
  if (!voice || voice.bars.length === 0) return EMPTY_TIMELINE;

  // Per-bar tempo segments come from `jot.tempoEvents`. Mid-bar tempo
  // changes are honoured natively: each bar's `durationSec` is the sum
  // of its constant-tempo intra-bar segments.
  const tempos = buildBarTempos(rendered.source, voice.bars);
  const durations: number[] = new Array(voice.bars.length);
  for (let i = 0; i < voice.bars.length; i++) durations[i] = tempos[i].durationSec;

  // Anchor bar 1 (= the first non-lead-in bar) at jot time 0, so the
  // audio scheduler's "media = jot + drumsT0Sec" identity lines up the
  // synth drums with the recorded audio drums. Pre-drum bars (the
  // lead-in, identified by `bar.index < 0`) get negative `startSec`;
  // playback's rAF loop already accepts negative jot times for the
  // pre-roll scrub, and `timeToX` resolves them via the per-bar loop.
  // Lead-in count is read directly from the structure (counting the
  // leading run of negative-indexed bars) rather than from
  // `globalMetadata.leadBars`: `structureForVoice` materialises both
  // the explicit-leadBars and the chrome-only (`drumsT0Sec` without
  // `leadBars`) source shapes into the same negative-indexed-bar form,
  // so reading the count from the bars themselves keeps the timeline
  // path single-source-of-truth.
  let leadBars = 0;
  for (const b of voice.bars) {
    if (b.index >= 0) break;
    leadBars++;
  }
  let leadOffsetSec = 0;
  for (let i = 0; i < leadBars; i++) leadOffsetSec += durations[i];

  const bars: BarTiming[] = new Array(voice.bars.length);
  let cursor = -leadOffsetSec;
  for (let i = 0; i < voice.bars.length; i++) {
    bars[i] = { startSec: cursor, durationSec: durations[i] };
    cursor += durations[i];
  }
  // `cursor` now sits at the end of the last bar in jot time. The total
  // playable duration (used as the playback stop sentinel) covers
  // [-leadOffsetSec, cursor].
  return { totalDurationSec: cursor, bars, rendered };
}

/**
 * Map an absolute playback time (seconds from start) to the playhead's
 * pixel x within a voice's `barsRow`. Reads `bar.x` / `bar.width` from
 * the live `RenderedJot` on each call so zoom changes are picked up
 * automatically. Clamps to the bar grid at both ends so the playhead
 * never escapes the score.
 */
export function timeToX(timeline: JotTimeline, seconds: number): Pixels {
  const rendered = timeline.rendered;
  if (!rendered) return px(0);
  const voice = rendered.resolved.voices[0];
  const renderedBars = voice?.bars ?? [];
  const bars = timeline.bars;
  if (bars.length === 0 || renderedBars.length === 0) return px(0);
  // Notes sit `notePadPx` inside their bar's left edge (engraving
  // inset). The playhead marks when an onset *sounds*, so it has to
  // ride the note grid, not the time-anchored bar box — every branch
  // below adds `pad` so it lands on the note instead of a constant
  // `pad` px to its left.
  const pad = (voice?.notePadPx as number) ?? 0;
  // Lead-in (when present) lives in `voice.bars` as negative-indexed
  // bars with negative `startSec`, so the per-bar loop below resolves
  // negative seek targets that fall inside them without a separate
  // chrome branch.
  const firstStartSec = bars[0]?.startSec ?? 0;
  const lastTiming = bars[bars.length - 1];
  const lastBar = renderedBars[renderedBars.length - 1];
  if (seconds >= lastTiming.startSec + lastTiming.durationSec) {
    return px((lastBar.x as number) + pad + (lastBar.width as number));
  }
  // Clamp `seconds` earlier than the first bar's start (the pre-pre-roll
  // window) to that bar's left edge; keeps the playhead anchored
  // somewhere visible if a seek lands before any bar's range.
  if (seconds < firstStartSec) {
    return px((renderedBars[0].x as number) + pad);
  }

  // Linear scan; jots are typically under ~64 bars so binary search adds
  // complexity without measurable benefit at the rAF rate.
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    if (seconds < bar.startSec + bar.durationSec) {
      const within = bar.durationSec > 0 ? (seconds - bar.startSec) / bar.durationSec : 0;
      const renderedBar = renderedBars[i] ?? lastBar;
      return px((renderedBar.x as number) + pad + within * (renderedBar.width as number));
    }
  }
  return px((lastBar.x as number) + pad + (lastBar.width as number));
}

/**
 * Inverse of {@link timeToX}: map a pixel x within the voice's
 * `barsRow` (same coordinate space `bar.x` is in — origin at the left
 * edge of the bars region, *after* the gutter) back to an absolute
 * playback time in seconds. Used for click-to-seek on the score and
 * the audio-track waveforms. Clamps to the bar grid at both ends so a click
 * in the margins lands on 0 / the final bar boundary.
 */
export function xToTime(timeline: JotTimeline, x: number): number {
  const rendered = timeline.rendered;
  if (!rendered) return 0;
  const voice = rendered.resolved.voices[0];
  const renderedBars = voice?.bars ?? [];
  const bars = timeline.bars;
  if (bars.length === 0 || renderedBars.length === 0) return 0;
  // Inverse of timeToX's note-grid shift: the clickable axis is offset
  // `pad` px right of the bar boxes, so subtract it before mapping x
  // back to time.
  const pad = (voice?.notePadPx as number) ?? 0;
  // Lead-in (when present) lives in `voice.bars` as negative-indexed
  // bars; the per-bar loop below maps clicks inside their pixel range
  // back to their negative `startSec` jot times directly.

  const lastBar = renderedBars[renderedBars.length - 1];
  const lastTiming = bars[bars.length - 1];
  const endX = (lastBar.x as number) + pad + (lastBar.width as number);
  const endSec = lastTiming.startSec + lastTiming.durationSec;
  if (x >= endX) return endSec;

  for (let i = 0; i < renderedBars.length; i++) {
    const rb = renderedBars[i];
    const x0 = (rb.x as number) + pad;
    const w = rb.width as number;
    if (x < x0 + w) {
      const timing = bars[i];
      if (!timing) return endSec;
      const within = w > 0 ? (x - x0) / w : 0;
      return timing.startSec + within * timing.durationSec;
    }
  }
  return endSec;
}
