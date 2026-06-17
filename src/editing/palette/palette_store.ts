import { computed, makeObservable } from 'mobx';
import type { MutableJot } from 'src/schema/schema';
import { INSTRUMENT_FALLBACK_COLOR } from 'src/editing/tracks/tracks';
import type { StructureStore } from 'src/editing/structure/structure_store';

/**
 * Per-lane palette colours + the legend, derived from the score
 * structure's lane order and the active palette. This is the track-view
 * colour concern (`defaultPaletteColorFor` / `assignTrackColors` /
 * `legendLanes`), an independent peer of the structural / tempo domains.
 *
 * Colour is a function of jot-wide lane order: the Nth lane (in
 * mapped-then-first-seen order) gets palette slot N (wrapping). Per-
 * instrument user overrides still live on the track view-models in the
 * mixer; this store only provides the palette default.
 */
export class PaletteStore {
  constructor(
    private readonly structure: StructureStore,
    private readonly getPalette: () => readonly string[],
    private readonly getJot: () => MutableJot | undefined
  ) {
    makeObservable(this, { jotLanes: computed.struct, legend: computed.struct });
  }

  /** Union of every layer's lane order, first-seen, the colour-slot order.
   *  Reads the store's `computed.struct` lane sets (not the note-content
   *  `layers`), and is itself `computed.struct`, so an in-lane note edit
   *  neither recomputes it nor notifies `colorForLane`, no row re-colours /
   *  re-renders on an edit that doesn't change which lanes exist. */
  get jotLanes(): string[] {
    const out: string[] = [];
    for (const layer of this.structure.layerOrder) {
      for (const lane of this.structure.lanesForLayer(layer.id)) {
        if (!out.includes(lane)) out.push(lane);
      }
    }
    return out;
  }

  /** Palette colour for a lane (its slot in {@link jotLanes}, wrapped),
   *  or a neutral grey when the lane is absent or the palette is empty. */
  colorForLane(lane: string): string {
    const palette = this.getPalette();
    if (palette.length === 0) return INSTRUMENT_FALLBACK_COLOR;
    const idx = this.jotLanes.indexOf(lane);
    if (idx < 0) return INSTRUMENT_FALLBACK_COLOR;
    return palette[idx % palette.length];
  }

  /** Palette colour for a lane, or `undefined` when the palette is empty
   *  or the lane isn't in the jot. Unlike {@link colorForLane} this does
   *  NOT substitute the grey fallback, callers (the mixer track-colour
   *  default) use `undefined` to mean "fall through to the instrument's
   *  own colour". */
  paletteColorFor(lane: string): string | undefined {
    const palette = this.getPalette();
    if (palette.length === 0) return undefined;
    const idx = this.jotLanes.indexOf(lane);
    return idx >= 0 ? palette[idx % palette.length] : undefined;
  }

  /**
   * `[lane, { color, name }]` for every lane that has notes, first-seen
   * in the structure's bar/track walk order (the order the score lays its
   * lanes out, which is what the score legend has always shown). Colour is
   * the palette slot; name comes from the instrument mapping.
   */
  get legend(): ReadonlyArray<readonly [string, { color: string; name?: string }]> {
    const jot = this.getJot();
    const seen = new Map<string, { color: string; name?: string }>();
    for (const layer of this.structure.layers) {
      for (const bar of layer.bars) {
        for (const lane of Object.keys(bar.tracks)) {
          if (!seen.has(lane)) {
            seen.set(lane, {
              color: this.colorForLane(lane),
              name: jot?.instruments.get(lane)?.name,
            });
          }
        }
      }
    }
    return Object.freeze(Array.from(seen.entries()));
  }
}
