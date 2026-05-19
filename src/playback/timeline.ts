/**
 * Time-to-pixel mapping for the playback playhead.
 *
 * Tempo follows `globalMetadata.bpm` and any per-bar `{{ bpm: ... }}`
 * overrides, accumulated bar by bar. Browser playback does NOT go through
 * the MIDI bytes (which only carry a `setTempo` at tick 0) — the scheduler
 * in `events.ts` walks the rendered jot with the same per-bar tempo logic
 * — so honouring per-bar bpm here is exactly what keeps the playhead and
 * the audio-track waveform locked to the scheduled drums. Ignoring it (the
 * old behaviour) is what made a variable-tempo chart drift against its
 * recording.
 *
 * Pixel offsets are looked up against the LIVE `RenderedJot` on every call
 * to {@link timeToX}, not cached at build time, so the playhead stays in
 * sync with the score even when the user zooms during playback (the
 * `ViewConfig.barWidth` change reflows the rendered bars reactively and
 * the playhead re-reads their new `x` / `width`).
 */
import { BpmTransition } from 'src/dsl';
import { Pixels, RenderedJot, px } from 'src/jot';

/**
 * Resolve a `Metadata.bpm` field (a number, a {@link BpmTransition}, or
 * absent) to a positive BPM, falling back to `fallback` when missing or
 * non-positive. Transitions are not interpolated — we take `start` (else
 * `end`), mirroring the no-interpolation policy `events.ts` uses for
 * volume so the scheduler, playhead and waveform share one definition.
 */
export function resolveBpm(
  field: number | BpmTransition | undefined,
  fallback: number,
): number {
  if (typeof field === 'number') return field > 0 ? field : fallback;
  if (field && typeof field === 'object') {
    const v = field.start ?? field.end;
    return typeof v === 'number' && v > 0 ? v : fallback;
  }
  return fallback;
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
  const resolved = rendered.resolved;
  // All voices in a Jot share the same bar grid (they're laid out from the
  // same global metadata), so the first voice's timing is canonical.
  const voice = resolved.voices[0];
  if (!voice || voice.bars.length === 0) return EMPTY_TIMELINE;

  const globalBpm = resolveBpm(resolved.globalMetadata.bpm, 120);

  let timeSec = 0;
  // Per-bar `{{ bpm }}` overrides are sticky — a change holds until the
  // next one — so carry the effective tempo forward across bars rather
  // than re-reading the global each iteration.
  let currentBpm = globalBpm;
  const bars: BarTiming[] = [];
  for (const bar of voice.bars) {
    const override = bar.source.metadata?.bpm;
    if (override !== undefined) currentBpm = resolveBpm(override, currentBpm);
    const durationSec = bar.beats * (60 / currentBpm);
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
  if (seconds <= 0) {
    // Lead-in: jot time runs from -leadInSec (start of the recording's
    // pre-roll) up to 0 (bar 1 / first beat). Map it linearly across
    // the reserved lead-in pixels so the playhead travels the pre-roll
    // and meets bar 1 exactly when the drums enter — the same instant
    // the audio-track waveform shows them. Without an offset both values are
    // 0 and this collapses to the old "park at bar 1 start".
    const leadInPx = (voice?.leadInPx as number) ?? 0;
    const leadInSec = voice?.leadInSec ?? 0;
    if (leadInSec > 0 && leadInPx > 0) {
      if (seconds <= -leadInSec) return px(pad);
      return px(pad + (leadInPx * (seconds + leadInSec)) / leadInSec);
    }
    return px((renderedBars[0].x as number) + pad);
  }

  const lastTiming = bars[bars.length - 1];
  const lastBar = renderedBars[renderedBars.length - 1];
  if (seconds >= lastTiming.startSec + lastTiming.durationSec) {
    return px((lastBar.x as number) + pad + (lastBar.width as number));
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

  const first = renderedBars[0];
  if (x <= (first.x as number) + pad) {
    // Lead-in region. The reserved pre-roll pixels [0, leadInPx] map
    // back onto negative jot time [-leadInSec, 0] — the exact inverse
    // of timeToX's lead-in branch — so click-to-seek can land inside
    // the recording's drumless intro instead of snapping to bar 1.
    // Without a lead-in, clamp to 0 as before.
    const leadInPx = (voice?.leadInPx as number) ?? 0;
    const leadInSec = voice?.leadInSec ?? 0;
    if (leadInSec > 0 && leadInPx > 0) {
      if (x <= pad) return -leadInSec;
      return ((x - pad - leadInPx) / leadInPx) * leadInSec;
    }
    return 0;
  }

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
