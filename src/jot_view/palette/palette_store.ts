import { computed, makeObservable } from 'mobx';
import type { Jot } from 'src/schema/schema';
import { INSTRUMENT_FALLBACK_COLOR } from 'src/jot_view/tracks/tracks';
import type { StructureStore } from 'src/jot_view/structure/structure_store';

/**
 * Per-pitch palette colours + the legend, derived from the score
 * structure's lane order and the active palette. This is the track-view
 * colour concern (`defaultPaletteColorFor` / `assignTrackColors` /
 * `legendPitches`), an independent peer of the structural / tempo domains.
 *
 * Colour is a function of jot-wide pitch order: the Nth pitch (in
 * mapped-then-first-seen order) gets palette slot N (wrapping). Per-
 * instrument user overrides still live on the track view-models in the
 * mixer; this store only provides the palette default.
 */
export class PaletteStore {
  constructor(
    private readonly structure: StructureStore,
    private readonly getPalette: () => readonly string[],
    private readonly getJot: () => Jot | undefined
  ) {
    makeObservable(this, { jotPitches: computed, legend: computed });
  }

  /** Union of every voice's lane order, first-seen, the colour-slot order. */
  get jotPitches(): string[] {
    const out: string[] = [];
    for (const voice of this.structure.voices) {
      for (const pitch of voice.pitches) {
        if (!out.includes(pitch)) out.push(pitch);
      }
    }
    return out;
  }

  /** Palette colour for a pitch (its slot in {@link jotPitches}, wrapped),
   *  or a neutral grey when the pitch is absent or the palette is empty. */
  colorForPitch(pitch: string): string {
    const palette = this.getPalette();
    if (palette.length === 0) return INSTRUMENT_FALLBACK_COLOR;
    const idx = this.jotPitches.indexOf(pitch);
    if (idx < 0) return INSTRUMENT_FALLBACK_COLOR;
    return palette[idx % palette.length];
  }

  /** Palette colour for a pitch, or `undefined` when the palette is empty
   *  or the pitch isn't in the jot. Unlike {@link colorForPitch} this does
   *  NOT substitute the grey fallback, callers (the mixer track-colour
   *  default) use `undefined` to mean "fall through to the instrument's
   *  own colour". */
  paletteColorFor(pitch: string): string | undefined {
    const palette = this.getPalette();
    if (palette.length === 0) return undefined;
    const idx = this.jotPitches.indexOf(pitch);
    return idx >= 0 ? palette[idx % palette.length] : undefined;
  }

  /**
   * `[pitch, { color, name }]` for every pitch that has notes, first-seen
   * in the structure's bar/track walk order (the order the score lays its
   * lanes out, which is what the score legend has always shown). Colour is
   * the palette slot; name comes from the instrument mapping.
   */
  get legend(): ReadonlyArray<readonly [string, { color: string; name?: string }]> {
    const jot = this.getJot();
    const seen = new Map<string, { color: string; name?: string }>();
    for (const voice of this.structure.voices) {
      for (const bar of voice.bars) {
        for (const pitch of Object.keys(bar.tracks)) {
          if (!seen.has(pitch)) {
            seen.set(pitch, {
              color: this.colorForPitch(pitch),
              name: jot?.instruments.get(pitch)?.name,
            });
          }
        }
      }
    }
    return Object.freeze(Array.from(seen.entries()));
  }
}
