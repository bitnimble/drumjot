import { expect, test } from '@playwright/test';
import type { WfPerfResult } from './waveform_perf_harness';

/**
 * Isolated perf measurement of the waveform RENDER path (build pyramid +
 * computeWaveformPeaks + paintWaveform), run on the main thread of a blank
 * harness page (`/waveform_perf.html`) -- no app, no worker. In production this
 * code runs off-thread in the tile worker, where a frame-budget probe can't see
 * it; here it's directly timed.
 *
 * Lives in the `perf` project (serial, after functional, GPU on) so the numbers
 * aren't inflated by parallel-worker contention. The hard gate is the
 * zoom-RATIO of compute time, which cancels machine speed and so survives a
 * loaded box; the absolute ceilings are deliberately ~10x loose, only there to
 * catch a gross regression. The per-zoom numbers are logged either way.
 */
test.describe('waveform render perf (isolated harness)', () => {
  test.describe.configure({ mode: 'serial' });

  test('compute is O(pixels): flat across zoom; build/paint bounded', async ({ page }) => {
    test.setTimeout(60_000);
    await page.goto('/waveform_perf.html');
    await page.waitForFunction(
      () => typeof (window as unknown as { __waveformPerf?: { run?: unknown } }).__waveformPerf?.run === 'function',
    );

    const result = await page.evaluate(
      (opts) =>
        (window as unknown as { __waveformPerf: { run: (o: unknown) => WfPerfResult } }).__waveformPerf.run(opts),
      { trackSeconds: 240, widthPx: 400, height: 64, dpr: 2, iterations: 300 },
    );

    console.log(`[WF-PERF] build ${result.buildMs.toFixed(1)}ms for ${(result.samples / 1e6).toFixed(1)}M samples`);
    for (const r of result.renders) {
      console.log(
        `[WF-PERF] tile ${r.tileSeconds}s (~${r.samplesPerPx} samples/px): ` +
          `compute ${r.computeMs.toFixed(4)}ms  paint ${r.paintMs.toFixed(4)}ms`,
      );
    }

    expect(result.renders.length).toBeGreaterThanOrEqual(2);
    const zin = result.renders[0]; // most zoomed in (fewest samples/px)
    const zout = result.renders[result.renders.length - 1]; // most zoomed out

    // The pyramid makes compute ~flat regardless of zoom (the zoomed-out tile is
    // not meaningfully slower than the zoomed-in one). A regression to raw
    // per-sample scanning would make zout scale with samples/px (~15x more here).
    // Ratio-based so it survives a loaded box; +0.1ms absorbs sub-ms timer noise.
    expect(zout.computeMs).toBeLessThan(zin.computeMs * 4 + 0.1);

    // Gross-regression ceilings (NOT tight budgets): ~10x typical, so a loaded
    // box won't flake them.
    expect(result.buildMs).toBeLessThan(2000);
    for (const r of result.renders) {
      expect(r.computeMs).toBeLessThan(3);
      expect(r.paintMs).toBeLessThan(20);
    }
  });
});
