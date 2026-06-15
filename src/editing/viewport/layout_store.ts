import { computed, makeObservable } from 'mobx';
import type { StructureStore } from 'src/editing/structure/structure_store';

/**
 * Beat→pixel layout derived from the score structure + the active bar
 * width, the pixel layer that used to be `ResolvedJot` (`layoutJot` /
 * `pixelLayer` / density). Belongs to the viewport / coordinate-system
 * domain: it owns `pxPerBeat` (the single multiplier the renderer exposes
 * as `--px-per-beat`), the onset-density width scaling, the engraving
 * inset, and per-bar pixel offsets.
 *
 * Per-note pixel x/width are deliberately NOT materialised here, the
 * renderer computes them in CSS from `--px-per-beat` so a zoom doesn't
 * re-render; consumers that need a number use {@link pxPerBeat} +
 * {@link notePadPx}.
 */

// Onset density (onsets per quarter beat) the static bar width was tuned
// for: a 4/4 bar of straight eighths (8 onsets / 4 beats).
const REFERENCE_ONSETS_PER_BEAT = 2;
const MIN_DENSITY_FACTOR = 0.4;
const MAX_DENSITY_FACTOR = 1.6;

export type LayoutBar = {
  id: string;
  index: number;
  beats: number;
  /** Cumulative beats before this bar. */
  startBeat: number;
  /** Pixel x of the bar's left edge. */
  x: number;
  width: number;
};

export class LayoutStore {
  constructor(
    private readonly structure: StructureStore,
    private readonly getBarWidth: () => number,
    private readonly getNotePadBeats: () => number
  ) {
    makeObservable(this, {
      densityFactor: computed,
      pxPerBeat: computed,
      notePadPx: computed,
      bars: computed,
      contentWidthPx: computed,
    });
  }

  /** Whole-song width scale from onset density (max over layers), clamped.
   *  A "column" is a distinct onset beat within a bar (a simultaneity is
   *  one column), matching the legacy `countOnsets`. */
  get densityFactor(): number {
    let maxRatio = 0;
    for (const layer of this.structure.layers) {
      let onsets = 0;
      let beats = 0;
      for (const bar of layer.bars) {
        const cols = new Set<number>();
        for (const lane of Object.keys(bar.tracks)) {
          for (const note of bar.tracks[lane].notes) cols.add(note.beat);
        }
        onsets += cols.size;
        beats += bar.beats;
      }
      if (beats > 0) maxRatio = Math.max(maxRatio, onsets / beats);
    }
    if (maxRatio <= 0) return 1;
    return Math.max(
      MIN_DENSITY_FACTOR,
      Math.min(MAX_DENSITY_FACTOR, maxRatio / REFERENCE_ONSETS_PER_BEAT)
    );
  }

  /** Quarter-note-beat → pixel multiplier currently in effect. */
  get pxPerBeat(): number {
    return (this.getBarWidth() * this.densityFactor) / 4;
  }

  /** Engraving inset (px) the note grid is shifted right of each bar edge. */
  get notePadPx(): number {
    return this.getNotePadBeats() * this.pxPerBeat;
  }

  /** Per-bar pixel layout for the shared bar grid (all layers tile the
   *  same grid; read the primary layer). */
  get bars(): LayoutBar[] {
    const layer = this.structure.layers[0];
    if (!layer) return [];
    const pxPerBeat = this.pxPerBeat;
    const out: LayoutBar[] = [];
    let startBeat = 0;
    let x = 0;
    for (const bar of layer.bars) {
      const width = bar.beats * pxPerBeat;
      out.push({ id: bar.id, index: bar.index, beats: bar.beats, startBeat, x, width });
      startBeat += bar.beats;
      x += width;
    }
    return out;
  }

  get contentWidthPx(): number {
    const bars = this.bars;
    if (bars.length === 0) return 0;
    const last = bars[bars.length - 1];
    return last.x + last.width;
  }
}
