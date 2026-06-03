/**
 * Text-measurement primitive for the lyrics row's word-collision pass.
 *
 * Replaces the previous DOM round-trip (`Range.getBoundingClientRect()`
 * after writing the rendered text, then reading `dataset.beatOffset` back
 * out) with a canvas `measureText` call whose font string mirrors the
 * variable-font axes the CSS clamps against `--px-per-beat`. The output
 * (`shifts: Map<key, px>`) is the source of truth; the renderer writes
 * each value into `--lyric-word-shift` as a render sink.
 *
 * CSS axis formulas (must stay in lockstep with `lyrics_row.module.css`):
 *   letter-spacing: clamp(-0.6, (pxPerBeat - 224) * 0.6/179, 0)   px
 *   --lyric-wdth  : clamp(75,   75 + (pxPerBeat - 45) * 25/179, 100)
 *   --lyric-wght  : clamp(300, 300 + (pxPerBeat - 45) * 100/179, 400)
 *   active word overrides wght to 800 (.lyricWordActive .lyricWordText).
 *   Floor reached at pxPerBeat ≈ 45 (≈40% zoom on a density-1 song);
 *   max spread reached at pxPerBeat ≈ 224 (≈200% zoom on a density-1 song).
 *
 * Canvas font shorthand carries `wdth` via `font-stretch <percent>` and
 * `wght` via the numeric `font-weight`. Variable axes other than these
 * two would need `font-variation-settings`, which canvas 2D doesn't
 * expose, but lyrics_row only animates wdth/wght so the canvas form
 * round-trips faithfully on Chromium / Firefox / Safari ≥ 16.
 *
 * Font loading: `document.fonts.ready` flips `fontReady` true once
 * Bricolage Grotesque has loaded; consumers read it (it's a MobX
 * observable) so the row re-renders with the real widths after the
 * web-font swap. Before that, canvas falls back to the system stack
 * and shifts may be off by a glyph or two; same behaviour as the old
 * DOM walk during the font-load window.
 */
import { makeAutoObservable, runInAction } from 'mobx';

const FONT_FAMILY =
  "'Bricolage Grotesque', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
/** Font size inherited from `.lyricLine`. Keep in lockstep with
 *  `lyrics_row.module.css::.lyricLine { font-size: 18px }`. */
const LYRIC_FONT_SIZE_PX = 18;
const ACTIVE_FONT_WEIGHT = 800;
const MIN_GAP_PX = 4;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function computeLetterSpacingPx(pxPerBeat: number): number {
  return clamp(((pxPerBeat - 224) * 0.7) / 179, -0.7, 0);
}

function computeWdth(pxPerBeat: number): number {
  return clamp(75 + ((pxPerBeat - 80) * 25) / 179, 75, 120);
}

function computeWght(pxPerBeat: number): number {
  return clamp(300 + ((pxPerBeat - 80) * 100) / 179, 300, 600);
}

/** Canvas + font-load state, exposed as a MobX-observable singleton so a
 *  font-load completion triggers consumers to re-derive their shifts. */
class LyricsMeasurer {
  /** True once the Bricolage Grotesque web font has finished loading.
   *  Until then, canvas measurement falls back to the system stack. */
  fontReady: boolean = false;
  private canvas: HTMLCanvasElement | OffscreenCanvas | undefined;
  private ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | undefined;

  constructor() {
    makeAutoObservable(this);
    if (typeof document !== 'undefined' && document.fonts?.ready) {
      void document.fonts.ready.then(() => {
        runInAction(() => {
          this.fontReady = true;
        });
      });
    }
  }

  private ensureCtx(): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | undefined {
    if (this.ctx) return this.ctx;
    if (typeof OffscreenCanvas !== 'undefined') {
      this.canvas = new OffscreenCanvas(1, 1);
      this.ctx = this.canvas.getContext('2d') ?? undefined;
    } else if (typeof document !== 'undefined') {
      this.canvas = document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d') ?? undefined;
    }
    return this.ctx;
  }

  /** Measure `text` as it would render under the row's variable-font
   *  state for the given zoom. `isActive` picks the active-word weight
   *  override (`wght: 800`). Returns 0 when canvas isn't available. */
  measureWordPx(text: string, pxPerBeat: number, isActive: boolean): number {
    const ctx = this.ensureCtx();
    if (!ctx) return 0;
    const wdth = computeWdth(pxPerBeat);
    const wght = isActive ? ACTIVE_FONT_WEIGHT : computeWght(pxPerBeat);
    // CSS font shorthand: <weight> <stretch%> <size>px <family>.
    ctx.font = `${Math.round(wght)} ${wdth.toFixed(2)}% ${LYRIC_FONT_SIZE_PX}px ${FONT_FAMILY}`;
    // `letterSpacing` is the CSS letter-spacing applied during measureText;
    // Chromium / Firefox honour it directly. Safari < 16 ignores; the
    // (sub-)pixel drift sits well inside MIN_GAP_PX.
    type WithSpacing = {
      letterSpacing?: string;
    } & (CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D);
    (ctx as WithSpacing).letterSpacing = `${computeLetterSpacingPx(pxPerBeat).toFixed(3)}px`;
    return ctx.measureText(text).width;
  }
}

export const lyricsMeasurer = new LyricsMeasurer();

/** A word's identity inside the per-row shift map. `lineIdx` is the
 *  positioned-line index (NOT `line.i`; keep stable across the walk),
 *  `sourceIdx` matches `PositionedWord.sourceIdx`. */
export type LyricShiftKey = string;
export function lyricShiftKey(lineIdx: number, sourceIdx: number): LyricShiftKey {
  return `${lineIdx}:${sourceIdx}`;
}

export type LyricWordMeasureInput = {
  /** Stable identity inside the line's source array. */
  sourceIdx: number;
  text: string;
  /** Beats from the line's start. Same units as `--lyric-word-beat-offset`. */
  beatOffset: number;
};

export type LyricLineMeasureInput = {
  /** Index into the positioned-line array. */
  lineIdx: number;
  /** Active line + word indices; either both set (one word in this line
   *  is active) or both absent (line inactive). */
  activeWordSourceIdx: number | undefined;
  words: LyricWordMeasureInput[];
};

/** Walk each line left-to-right and compute the per-word `--lyric-
 *  word-shift` (px) that keeps adjacent words at least {@link MIN_GAP_PX}
 *  apart. Empty result is fine: lines with no collisions return an empty
 *  Map and the renderer leaves `--lyric-word-shift` at 0.
 *
 *  Pure: same inputs → same output. The measurer's `fontReady` is read
 *  by callers (to invalidate before/after the font load); this fn itself
 *  doesn't track it. */
export function computeLyricShifts(
  lines: readonly LyricLineMeasureInput[],
  pxPerBeat: number
): Map<LyricShiftKey, number> {
  const out = new Map<LyricShiftKey, number>();
  if (!Number.isFinite(pxPerBeat) || pxPerBeat <= 0) return out;
  for (const line of lines) {
    let prevRight = -Infinity;
    for (const w of line.words) {
      const natural = w.beatOffset * pxPerBeat;
      const isActive = line.activeWordSourceIdx === w.sourceIdx;
      const textWidth = lyricsMeasurer.measureWordPx(w.text, pxPerBeat, isActive);
      const required = Math.max(natural, prevRight + MIN_GAP_PX);
      const shift = required - natural;
      if (shift > 0) {
        out.set(lyricShiftKey(line.lineIdx, w.sourceIdx), shift);
      }
      prevRight = required + textWidth;
    }
  }
  return out;
}
