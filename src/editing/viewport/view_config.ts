import { makeAutoObservable } from 'mobx';
import { PICKER_PALETTE } from 'src/editing/tracks/tracks';

/** Branded pixel scalar to avoid mixing pixel and beat measurements. */
export type Pixels = number & { __pixels: never };
export const px = (n: number) => n as Pixels;

// ---------- Layout config ----------

export class ViewConfig {
  /** Pixel width of one whole bar at default zoom. */
  barWidth = px(448);
  /** Vertical height of one rendered lane track. */
  trackHeight = px(36);
  /** Padding above/below each layer block. */
  layerPadding = px(12);
  /** Note dot diameter. */
  noteDiameter = px(14);
  /**
   * Horizontal offset applied to every note from its bar's left edge,
   * expressed as a fraction of a quarter-note beat (so it scales with
   * zoom). Acts as `P/2`, where `P` is the gap across a barline between
   * the last subbeat of bar N and the first subbeat of bar N+1, set to
   * exactly the inter-subbeat distance at the finest grid (48ths,
   * `pxPerBeat/12`), so a 48th sitting on the last slot of bar N lands
   * at `barline - P/2` and the next bar's downbeat at `barline + P/2`.
   * The barline becomes a true separator rather than something the
   * last 48th overflows past, which both fixes the visual (the last
   * subbeat used to look like it belonged to the next bar) and the hit
   * region (an overflowing note got covered by bar N+1 and clicks
   * landed on the seek handler instead of the note).
   *
   * Default `1/24`: half of `1/12` (the 48ths inter-note distance in
   * beats) so `padLeft = pxPerBeat/24 = P/2`.
   */
  barNotePaddingBeats = 1 / 24;
  /** Palette used when a lane has no explicit colour. Mirrors the
   *  shared {@link PICKER_PALETTE} so a colour the user picks from the
   *  picker's first swatch row matches a lane that landed on the same
   *  palette slot. */
  palette: string[] = [...PICKER_PALETTE];

  constructor() {
    makeAutoObservable(this);
  }
}
