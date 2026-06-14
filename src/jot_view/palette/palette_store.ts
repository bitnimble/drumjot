import { computed, makeObservable } from 'mobx';
import type { Jot } from 'src/schema/schema';
import { INSTRUMENT_FALLBACK_COLOR } from 'src/tracks/tracks';
import type { StructureStore } from 'src/jot_view/structure/structure_store';

/**
 * Per-pitch palette colours + the legend, derived from the score
 * structure's lane order and the active palette. This is the track-view
 * colour concern that used to live on `RenderedJot`
 * (`defaultPaletteColorFor` / `assignTrackColors` / `legendPitches`).
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

  /** `[pitch, { color, name }]` per lane, in colour-slot order. */
  get legend(): ReadonlyArray<readonly [string, { color: string; name?: string }]> {
    const jot = this.getJot();
    return this.jotPitches.map(
      (pitch) =>
        [pitch, { color: this.colorForPitch(pitch), name: jot?.instruments.get(pitch)?.name }] as const
    );
  }
}
