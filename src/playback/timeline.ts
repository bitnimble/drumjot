/**
 * Time-to-pixel mapping for the playback playhead.
 *
 * Anchored to `globalMetadata.bpm` (a single, constant tempo for the whole
 * jot) because `toMidi` only emits a `setTempo` meta event at tick 0 —
 * per-bar `{{ bpm: ... }}` overrides in the DSL aren't carried through to
 * the MIDI bytes that drive playback, so the playhead would drift relative
 * to the audio if we honoured them here.
 *
 * Pixel offsets are looked up against the LIVE `RenderedJot` on every call
 * to {@link timeToX}, not cached at build time, so the playhead stays in
 * sync with the score even when the user zooms during playback (the
 * `ViewConfig.barWidth` change reflows the rendered bars reactively and
 * the playhead re-reads their new `x` / `width`).
 */
import { Pixels, RenderedJot, px } from 'src/jot';

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
  const resolved = rendered.resolved;
  // All voices in a Jot share the same bar grid (they're laid out from the
  // same global metadata), so the first voice's timing is canonical.
  const voice = resolved.voices[0];
  if (!voice || voice.bars.length === 0) return EMPTY_TIMELINE;

  const bpmField = resolved.globalMetadata.bpm;
  const bpm = typeof bpmField === 'number' && bpmField > 0 ? bpmField : 120;
  const secondsPerBeat = 60 / bpm;

  let timeSec = 0;
  const bars: BarTiming[] = [];
  for (const bar of voice.bars) {
    const durationSec = bar.beats * secondsPerBeat;
    bars.push({ startSec: timeSec, durationSec });
    timeSec += durationSec;
  }
  return { totalDurationSec: timeSec, bars, rendered };
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
  const renderedBars = rendered.resolved.voices[0]?.bars ?? [];
  const bars = timeline.bars;
  if (bars.length === 0 || renderedBars.length === 0) return px(0);
  if (seconds <= 0) return renderedBars[0].x;

  const lastTiming = bars[bars.length - 1];
  const lastBar = renderedBars[renderedBars.length - 1];
  if (seconds >= lastTiming.startSec + lastTiming.durationSec) {
    return px((lastBar.x as number) + (lastBar.width as number));
  }

  // Linear scan; jots are typically under ~64 bars so binary search adds
  // complexity without measurable benefit at the rAF rate.
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    if (seconds < bar.startSec + bar.durationSec) {
      const within = bar.durationSec > 0 ? (seconds - bar.startSec) / bar.durationSec : 0;
      const renderedBar = renderedBars[i] ?? lastBar;
      return px((renderedBar.x as number) + within * (renderedBar.width as number));
    }
  }
  return px((lastBar.x as number) + (lastBar.width as number));
}
