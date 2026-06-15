/**
 * Chunk layout for the tiled mixer waveform.
 *
 * The audio-track waveform used to be one wide canvas spanning the
 * whole score (capped at a 16 384 px backing-store dimension, so very
 * long or very zoomed tracks lost horizontal resolution). It's now a
 * row of fixed-beat-width canvases; one per `BEATS_PER_CHUNK` window;
 * and each chunk picks its own backing-store size, so the effective
 * resolution is unbounded.
 *
 * Chunk identity is stable for the lifetime of the score: chunks are
 * keyed by `floor(startBeat / BEATS_PER_CHUNK)`, so a zoom change
 * resizes existing chunks (via CSS-driven `left` / `width` recompute)
 * without unmounting / remounting them. This is what makes the high-
 * quality redraw happen on the same frame as the zoom: there's no
 * IntersectionObserver fire latency to wait through (eliminated, see
 * mixer's parent-driven visibility derivation), no bucket transition
 * to churn, no stretched-bitmap holdover gap to mask.
 *
 * `buildChunkLayout` walks the structural bars ONCE to produce
 * everything a chunk row needs to lay itself out:
 *
 *  - Per-bar `BarBeat`: cumulative beat position + jot-time anchor +
 *    duration. Beat-space is zoom-invariant (pure structure / tempo),
 *    so the values stay stable across wheel ticks.
 *  - `WaveformChunk[]`: each chunk owns a contiguous beat range
 *    (`BEATS_PER_CHUNK` beats wide, last chunk possibly shorter). The
 *    chunk's CSS position is driven in JS from its `startBeat` × the
 *    live `pxPerBeat`.
 *
 * The lead-in (negative-indexed bars at the front of `voice.bars`) is
 * absorbed transparently: its bars contribute to `startBeat` and
 * `startSec` like any other bar. A chunk that straddles the lead-in /
 * drum-content boundary is fine; the per-pixel mapping inside the
 * worker iterates bars, so each section gets its real audio-time
 * mapping regardless of the bar's role.
 */
import type { StructuralPresenter } from 'src/jot_view/structure/structural_presenter';
import { buildBarTempos } from 'src/schema/dsl/tempo';

/**
 * Beats per chunk. Chunks are sliced on this beat-aligned grid,
 * independent of bar structure (a 3/4 bar followed by a 4/4 bar
 * doesn't realign the grid; the worker's per-bar mapping handles the
 * tempo / duration boundary inside a chunk without seams).
 *
 * Sized so the worst-case backing bitmap stays comfortably under the
 * cross-browser 16 384 px canvas cap. Worst case is `pxPerBeat = 112
 * × densityFactor 1.6 = 179.2` (see `MAX_DENSITY_FACTOR` in
 * `src/jot.ts`) × `MAX_ZOOM = 4` × `devicePixelRatio = 3` ≈ 2150
 * backing-px/beat. 4 × 2150 ≈ 8 600 backing-px per chunk, well under
 * the cap. At normal zoom (pxPerBeat ≈ 112, DPR 2) a chunk is ~896
 * backing-px, a typical viewport holds 10–30 chunks per track.
 */
export const BEATS_PER_CHUNK = 4;

/**
 * Zoom-invariant per-bar layout: where the bar sits in the voice's
 * cumulative beat axis (used to position chunks via JS), the bar's own
 * beat count, and the jot-time window the bar covers (used by the
 * worker to map each chunk pixel column back to an audio sample range).
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
 * `totalBeats × pxPerBeat` (chunk-local pixel width). `key` is stable
 * across the lifetime of the score (= the chunk's beat-aligned bucket
 * index), so React preserves the canvas DOM element across zoom
 * changes; only `left` / `width` change.
 */
export type WaveformChunk = {
  /** Stable React key: `startBeat / BEATS_PER_CHUNK`. */
  key: number;
  /** Beat position of the chunk's left edge in the voice. */
  startBeat: number;
  /** Beat span the chunk covers (`BEATS_PER_CHUNK`, or smaller for the trailing chunk). */
  totalBeats: number;
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
 * cursors, then slice the beat axis into `BEATS_PER_CHUNK` windows.
 * Per-bar `{{ bpm }}` overrides are honoured the same way `events.ts`
 * / `buildTimeline` do: sticky until the next override.
 *
 * Reads ONLY the zoom-invariant structural voices, so the returned
 * `bars[*].startBeat / .beats / .startSec / .durationSec` and the derived
 * `chunks[*]` are stable across zoom changes. Callers can memo on the
 * `StructuralPresenter` and reuse the result across every wheel tick.
 */
export function buildChunkLayout(structural: StructuralPresenter): ChunkLayout {
  const structureVoice = structural.voices[0];
  if (!structureVoice || structureVoice.bars.length === 0) return EMPTY_LAYOUT;

  const tempos = buildBarTempos(structural.source, structureVoice.bars);
  const durations: number[] = new Array(structureVoice.bars.length);
  for (let i = 0; i < structureVoice.bars.length; i++) {
    durations[i] = tempos[i].durationSec;
  }

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
  const totalBeats = cursorBeat;

  const chunks: WaveformChunk[] = [];
  for (let startBeat = 0; startBeat < totalBeats; startBeat += BEATS_PER_CHUNK) {
    const span = Math.min(BEATS_PER_CHUNK, totalBeats - startBeat);
    if (span <= 0) continue;
    chunks.push({
      key: startBeat / BEATS_PER_CHUNK,
      startBeat,
      totalBeats: span,
    });
  }

  return { bars, totalBeats, chunks };
}
