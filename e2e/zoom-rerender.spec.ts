import { expect, test } from '@playwright/test';

/**
 * Regression guard for the zoom-performance fix.
 *
 * Zooming must NOT re-render `JotView` (and therefore not its score
 * subtree: mixer → every InstrumentRow → BarViews → NoteViews). Two
 * mechanisms keep it off the React render path:
 *   1. positions flow through the `--px-per-beat` CSS variable, not props;
 *   2. `createJotView` hands `JotView` referentially-stable props, and the
 *      live `store.zoom` read is isolated in the toolbar's `ZoomControl`,
 *      so a zoom tick re-renders neither `View` nor `JotView`.
 *
 * The test counts `JotView` renders via the `window.__perf` hook
 * (src/perf_probe.ts). Toggling the drum MASTER mute is used as a
 * POSITIVE CONTROL: it flips a boolean that's a `voiceControls` memo dep,
 * so the bundle's identity changes and `JotView` legitimately re-renders.
 * That proves the counter is wired, so a future change that deletes the
 * probe can't make this test pass vacuously. (A per-pitch mute would NOT
 * re-render JotView; it mutates a Set in place, leaving the memo dep
 * unchanged, and only the affected leaf rows re-render. That's by design.)
 */

const JOT = `{{ bpm: 120, time: "4/4", title: "Zoom Regression",
  instrumentMapping: { h: { name: "HiHat" }, s: { name: "Snare" }, k: { name: "Kick" } } }}
${Array.from({ length: 16 }, () => '(k+h s+h k+h s+h)').join('\n')}
`;

test('zoom does not re-render JotView', async ({ page }) => {
  await page.goto('/');
  await page.evaluate((src) => (window as any).drumjot.loadDsl(src), JOT);
  await page.waitForSelector('[data-testid^="instrument-row-"]');

  const result = await page.evaluate(async () => {
    const w = window as any;
    const store = w.drumjot.store;
    const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
    const settle = async () => {
      await nextFrame();
      await nextFrame();
    };
    await settle();

    // POSITIVE CONTROL: a master-mute toggle changes a voiceControls memo
    // dep, so it must re-render JotView; proving the counter is live.
    w.__perf = {};
    store.toggleDrumMasterMute();
    await settle();
    const onControl = w.__perf.JotView ?? 0;
    store.toggleDrumMasterMute(); // restore
    await settle();

    // REGRESSION ASSERTION: a sweep of zoom levels must NOT re-render it.
    w.__perf = {};
    const zooms = [0.3, 0.5, 0.8, 1.2, 1.7, 2.3, 3.0];
    for (const z of zooms) {
      store.setZoom(z);
      await settle();
    }
    const onZoom = w.__perf.JotView ?? 0;
    const finalZoom = store.zoom;
    delete w.__perf;
    return { onControl, onZoom, zoomTicks: zooms.length, finalZoom };
  });

  // Sanity: the zoom sweep actually took effect.
  expect(result.finalZoom).toBeCloseTo(3.0, 5);
  // Positive control: the probe is wired and JotView reacts to real changes.
  expect(result.onControl).toBeGreaterThan(0);
  // The fix: zooming re-renders JotView zero times.
  expect(result.onZoom).toBe(0);
});
