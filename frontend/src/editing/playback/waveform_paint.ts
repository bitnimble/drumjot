/**
 * Shared waveform paint: rasterise a `[min, max]`-per-column peaks array into a
 * 2D canvas context as a single `Path2D` fill of 1px column bars. Used by the
 * mixer tile worker ({@link waveform_worker}) and the isolated render perf
 * harness ({@link test/waveform_perf_harness}), so both paint identically; kept
 * out of `waveform_compute` (which stays canvas-free) but reused by both.
 *
 * The caller sizes the canvas (`canvas.width`/`height = backingW`/`backingH`,
 * which also resets context state) before calling; this sets the transform that
 * maps the CSS-pixel column space onto the backing store, clears, and fills.
 */
type Canvas2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

export function paintWaveform(
  ctx: Canvas2D,
  peaks: Float32Array,
  widthPx: number,
  height: number,
  backingW: number,
  backingH: number,
  laneColor: string,
  ampScale: number
): void {
  ctx.imageSmoothingEnabled = false;
  ctx.setTransform(backingW / widthPx, 0, 0, backingH / height, 0, 0);
  ctx.clearRect(0, 0, widthPx, height);
  ctx.fillStyle = laneColor;
  const mid = height / 2;
  const yScale = mid * 0.95 * ampScale;
  // Collect every column's min..max bar into one Path2D and fill once, rather
  // than `widthPx` separate fillRect calls. The bars are 1 px wide at distinct
  // integer x so they never overlap; the single union fill is pixel-identical
  // to per-column fillRects at integer dpr (and on fractional dpr just drops a
  // faint double-blend seam between adjacent columns).
  // No skip-zero shortcut: silent columns still get a 1 px centerline (mn=mx=0
  // collapses to `rect(p, mid, 1, 1)`) so the baseline reads as a continuous
  // line instead of breaking into dashes wherever the audio is quiet.
  const path = new Path2D();
  for (let p = 0; p < widthPx; p++) {
    const mn = peaks[p * 2];
    const mx = peaks[p * 2 + 1];
    const y0 = Math.max(0, mid - mx * yScale);
    const y1 = Math.min(height, mid - mn * yScale);
    path.rect(p, y0, 1, Math.max(1, y1 - y0));
  }
  ctx.fill(path);
}
