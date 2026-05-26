/**
 * Chunk layout for the tiled mixer waveform.
 *
 * The audio-track waveform used to be one wide canvas spanning the
 * whole score (capped at a 16 384 px backing-store dimension, so very
 * long or very zoomed tracks lost horizontal resolution). It's now a
 * row of fixed-jot-time-width canvases; one per `SECONDS_PER_CHUNK`
 * window; and each chunk picks its own backing-store size, so the
 * effective resolution is unbounded.
 *
 * `buildChunkLayout` walks the structural bars ONCE to produce
 * everything a chunk row needs to lay itself out:
 *
 *  - Per-bar `BarBeat`: cumulative beat position + jot-time anchor +
 *    duration. Beat-space is zoom-invariant (pure structure / tempo),
 *    so the values stay stable across wheel ticks.
 *  - `WaveformChunk[]`: each chunk owns a contiguous beat range and
 *    the matching jot-time window. The chunk's CSS position is driven
 *    by `--chunk-start-beat × --px-per-beat` so live zoom is a pure
 *    CSS recompute (no React work).
 *
 * The lead-in (negative-indexed bars at the front of `voice.bars`) is
 * absorbed transparently: its bars contribute to `startBeat` and
 * `startSec` like any other bar. A chunk that straddles the lead-in /
 * drum-content boundary is fine; the per-pixel mapping inside the
 * worker iterates bars, so each section gets its real audio-time
 * mapping regardless of the bar's role.
 */
import { RenderedJot } from 'src/jot';
import { resolveBpm } from 'src/playback/timeline';

/**
 * Default chunk width in jot-time seconds, used at zoom levels at or
 * below 1×. Chosen to keep each chunk's backing-store dimensions
 * comfortably under the 16 384 px cross-browser canvas cap at zoom=1
 * while keeping the chunk count low (a 5-minute song produces 10
 * chunks). Aligned to seconds; bar boundaries are not consulted; so
 * chunks straddle bar edges; the worker's per-bar pixel-to-time mapping
 * handles that without seams.
 *
 * At higher zoom the consumer passes a smaller `secondsPerChunk` (see
 * the parameter on {@link buildChunkLayout}) so chunks shrink in
 * jot-time and their bitmaps stay within the canvas cap; without that,
 * a 30 s chunk at zoom 2+ would exceed 16 384 bitmap pixels and the
 * browser would silently downsample it into a blurry tile.
 */
export const SECONDS_PER_CHUNK = 30;

/**
 * Zoom-invariant per-bar layout: where the bar sits in the voice's
 * cumulative beat axis (used to position chunks via CSS calc), the
 * bar's own beat count, and the jot-time window the bar covers (used
 * by the worker to map each chunk pixel column back to an audio
 * sample range).
 */
export type BarBeat = {
  /** Sum of `beats` for every bar before this one in the voice. */
  startBeat: number;
  /** This bar's own beat count. */
  beats: number;
  /**
   * Jot time at the bar's left edge. Negative for lead-in bars (they
   * sit before the drum entrance at jot 0).
   */
  startSec: number;
  /** Jot duration this bar occupies. */
  durationSec: number;
};

/**
 * One tile in the row. Each chunk renders its own canvas, sized to
 * `totalBeats × renderedPxPerBeat` (chunk-local pixel width) and
 * positioned at `startBeat × pxPerBeat + notePadPx` (in score-pixel
 * space). The CSS calc reads the LIVE `--px-per-beat`, so zoom moves
 * every chunk reactively without any React re-render.
 */
export type WaveformChunk = {
  /** Stable React key, monotonically rising from 0. */
  key: number;
  /** Beat position of the chunk's left edge in the voice. */
  startBeat: number;
  /** Beat span the chunk covers. */
  totalBeats: number;
  /** Jot time at the chunk's left edge. */
  jotStart: number;
  /** Jot time at the chunk's right edge. */
  jotEnd: number;
};

export type ChunkLayout = {
  /** Per-bar beat + jot-time anchors. */
  bars: BarBeat[];
  /** Sum of `bars[*].beats`. Caller can sanity-check chunk coverage. */
  totalBeats: number;
  /** The chunk list. Empty when the jot has no bars. */
  chunks: WaveformChunk[];
};

const EMPTY_LAYOUT: ChunkLayout = { bars: [], totalBeats: 0, chunks: [] };

/**
 * Walk the structural voice once, accumulating beat and jot-time
 * cursors, then slice the jot-time axis into {@link SECONDS_PER_CHUNK}
 * windows. Per-bar `{{ bpm }}` overrides are honoured the same way
 * `events.ts` / `buildTimeline` do: sticky until the next override.
 *
 * Reads ONLY `rendered.structure` (the zoom-invariant layout), so the
 * returned `bars[*].startBeat / .beats / .startSec / .durationSec` and
 * the derived `chunks[*]` are stable across zoom changes. Callers can
 * memo on `rendered.structure` and reuse the result across every wheel
 * tick.
 */
export function buildChunkLayout(
  rendered: RenderedJot,
  secondsPerChunk: number = SECONDS_PER_CHUNK,
): ChunkLayout {
  const structureVoice = rendered.structure.voices[0];
  if (!structureVoice || structureVoice.bars.length === 0) return EMPTY_LAYOUT;
  const chunkSeconds = secondsPerChunk > 0 ? secondsPerChunk : SECONDS_PER_CHUNK;

  const globalBpm = resolveBpm(rendered.globalMetadata.bpm, 120);

  // Forward pass: per-bar duration in seconds at the in-force tempo.
  let currentBpm = globalBpm;
  const durations: number[] = new Array(structureVoice.bars.length);
  for (let i = 0; i < structureVoice.bars.length; i++) {
    const bar = structureVoice.bars[i];
    const override = bar.source.metadata?.bpm;
    if (override !== undefined) currentBpm = resolveBpm(override, currentBpm);
    durations[i] = bar.beats * (60 / currentBpm);
  }

  // Count of leading lead-in (negative-indexed) bars; same definition
  // `buildTimeline` uses to anchor bar 1 at jot time 0.
  let leadBars = 0;
  for (const b of structureVoice.bars) {
    if (b.index >= 0) break;
    leadBars++;
  }
  let leadOffsetSec = 0;
  for (let i = 0; i < leadBars; i++) leadOffsetSec += durations[i];

  const bars: BarBeat[] = new Array(structureVoice.bars.length);
  let cursorBeat = 0;
  let cursorSec = -leadOffsetSec;
  for (let i = 0; i < structureVoice.bars.length; i++) {
    const sb = structureVoice.bars[i];
    bars[i] = {
      startBeat: cursorBeat,
      beats: sb.beats,
      startSec: cursorSec,
      durationSec: durations[i],
    };
    cursorBeat += sb.beats;
    cursorSec += durations[i];
  }

  const firstJotStart = bars[0].startSec;
  const lastJotEnd = bars[bars.length - 1].startSec + bars[bars.length - 1].durationSec;

  const chunks: WaveformChunk[] = [];
  let key = 0;
  for (let t = firstJotStart; t < lastJotEnd; t += chunkSeconds) {
    const jotStart = t;
    const jotEnd = Math.min(t + chunkSeconds, lastJotEnd);
    const startBeat = jotTimeToBeat(bars, jotStart);
    const endBeat = jotTimeToBeat(bars, jotEnd);
    const totalBeats = endBeat - startBeat;
    if (totalBeats <= 0) continue;
    chunks.push({ key: key++, startBeat, totalBeats, jotStart, jotEnd });
  }

  return { bars, totalBeats: cursorBeat, chunks };
}

/**
 * Linear-interpolation lookup: which (fractional) beat position does
 * the given jot-time second correspond to? Out-of-range inputs clamp
 * to the first / last bar's edges. Mirrors `timeToX`'s per-bar walk
 * inside the timeline module.
 */
function jotTimeToBeat(bars: BarBeat[], jotSec: number): number {
  if (bars.length === 0) return 0;
  const first = bars[0];
  if (jotSec <= first.startSec) return first.startBeat;
  for (const bar of bars) {
    const barEnd = bar.startSec + bar.durationSec;
    if (jotSec < barEnd) {
      const within = bar.durationSec > 0 ? (jotSec - bar.startSec) / bar.durationSec : 0;
      return bar.startBeat + within * bar.beats;
    }
  }
  const last = bars[bars.length - 1];
  return last.startBeat + last.beats;
}
