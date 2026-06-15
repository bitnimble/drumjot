import { expect, test } from '@playwright/test';

/**
 * Regression guard for the zoom-performance fix.
 *
 * Zooming must NOT re-render `JotEditor` (and therefore not its score
 * subtree: mixer → every InstrumentRow → BarViews → NoteViews). Two
 * mechanisms keep it off the React render path:
 *   1. positions flow through the `--px-per-beat` CSS variable, not props;
 *   2. `createJotEditor` hands `JotEditor` referentially-stable props, and the
 *      live `store.zoom` read is isolated in the toolbar's `ZoomControl`,
 *      so a zoom tick re-renders neither `View` nor `JotEditor`.
 *
 * The test counts `JotEditor` renders via the `window.__perf` hook
 * (src/perf_probe.ts). Toggling the drum MASTER mute is used as a
 * POSITIVE CONTROL: it flips a boolean that's a `layerControls` memo dep,
 * so the bundle's identity changes and `JotEditor` legitimately re-renders.
 * That proves the counter is wired, so a future change that deletes the
 * probe can't make this test pass vacuously. (A per-lane mute would NOT
 * re-render JotEditor; it mutates a Set in place, leaving the memo dep
 * unchanged, and only the affected leaf rows re-render. That's by design.)
 */

const JOT = `{{ bpm: 120, time: "4/4", title: "Zoom Regression",
  instrumentMapping: { h: { name: "HiHat" }, s: { name: "Snare" }, k: { name: "Kick" } } }}
${Array.from({ length: 16 }, () => '(k+h s+h k+h s+h)').join('\n')}
`;

test('zoom does not re-render JotEditor', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => typeof (window as any).drumjot?.loadDsl === 'function');
  await page.evaluate((src) => (window as any).drumjot.loadDsl(src), JOT);
  await page.waitForSelector('[data-testid^="instrument-row-"]');

  const result = await page.evaluate(async () => {
    const w = window as any;
    const viewport = w.drumjot.viewport;
    const mixerPresenter = w.drumjot.mixerPresenter;
    const viewportPresenter = w.drumjot.viewportPresenter;
    const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
    const settle = async () => {
      await nextFrame();
      await nextFrame();
    };
    await settle();

    // POSITIVE CONTROL: a master-mute toggle changes a layerControls memo
    // dep, so it must re-render JotEditor; proving the counter is live.
    w.__perf = {};
    mixerPresenter.toggleDrumMasterMute();
    await settle();
    const onControl = w.__perf.JotEditor ?? 0;
    mixerPresenter.toggleDrumMasterMute(); // restore
    await settle();

    // REGRESSION ASSERTION: a sweep of zoom levels must NOT re-render it.
    w.__perf = {};
    const zooms = [0.3, 0.5, 0.8, 1.2, 1.7, 2.3, 3.0];
    for (const z of zooms) {
      viewportPresenter.setZoom(z);
      await settle();
    }
    const onZoom = w.__perf.JotEditor ?? 0;
    const finalZoom = viewport.zoom;
    delete w.__perf;
    return { onControl, onZoom, zoomTicks: zooms.length, finalZoom };
  });

  // Sanity: the zoom sweep actually took effect.
  expect(result.finalZoom).toBeCloseTo(3.0, 5);
  // Positive control: the probe is wired and JotEditor reacts to real changes.
  expect(result.onControl).toBeGreaterThan(0);
  // The fix: zooming re-renders JotEditor zero times.
  expect(result.onZoom).toBe(0);
});

/**
 * Regression guard for the per-note popover cascade.
 *
 * There is one `PopoverPortal` per note (NoteView) and per filtered-onset
 * ghost. At most one is ever shown (the selected/hovered note's label).
 * Before the fix every HIDDEN popover still subscribed to `store.zoom` /
 * `store.scrollX`, so a single zoom tick woke one observer per note and
 * synchronously reconciled thousands of (null-rendering) fibers, ~111ms on
 * a real song. The fix gates the subscribing observer (`PopoverPortalShown`)
 * behind a hookless `show` check so hidden popovers subscribe to nothing.
 *
 * Counts `PopoverPortal` renders via `window.__perf` (src/perf_probe.ts).
 * POSITIVE CONTROL: with one note selected, its open popover must re-render
 * on zoom (so it repositions), proving the counter is wired. REGRESSION:
 * with nothing selected, a zoom sweep must re-render zero popovers.
 */
const DENSE_JOT = `{{ bpm: 120, time: "4/4", title: "Popover Zoom Regression",
  instrumentMapping: { h: { name: "HiHat" }, s: { name: "Snare" }, k: { name: "Kick" },
    r: { name: "Ride" }, c: { name: "Crash" }, t: { name: "Tom" }, f: { name: "Floor" } } }}
${Array.from({ length: 40 }, () => '(k+h+r s+h+r k+h+t s+h+c) (k+f s+h k+h s+h)').join('\n')}
`;

test('zoom does not re-render hidden note popovers', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => typeof (window as any).drumjot?.loadDsl === 'function');
  await page.evaluate((src) => (window as any).drumjot.loadDsl(src), DENSE_JOT);
  await page.waitForSelector('[data-testid^="instrument-row-"]');

  const settle = async () => {
    // Two RAFs to let MobX reactions + React commit flush.
    await page.evaluate(
      () =>
        new Promise<void>((r) =>
          requestAnimationFrame(() => requestAnimationFrame(() => r()))
        )
    );
  };
  await settle();

  // REGRESSION: nothing selected (no popover shown) → a zoom sweep must
  // re-render zero popovers. This is the cascade the fix removes.
  const onZoom = await page.evaluate(async () => {
    const w = window as any;
    const viewportPresenter = w.drumjot.viewportPresenter;
    const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
    w.__perf = {};
    for (const z of [0.4, 0.7, 1.1, 1.6, 2.2, 3.0]) {
      viewportPresenter.setZoom(z);
      await nextFrame();
      await nextFrame();
    }
    const n = w.__perf.PopoverPortal ?? 0;
    delete w.__perf;
    return n;
  });

  // POSITIVE CONTROL: select a note so exactly one popover is shown, then a
  // zoom tick must re-render it (it repositions against the moved anchor).
  await page.locator('[data-noseek="true"]').first().click();
  await settle();
  const onControl = await page.evaluate(async () => {
    const w = window as any;
    const viewportPresenter = w.drumjot.viewportPresenter;
    const viewport = w.drumjot.viewport;
    const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
    const popoverShown = !!document.querySelector('[data-popover="note-label"]');
    w.__perf = {};
    viewportPresenter.setZoom(viewport.zoom * 1.3);
    await nextFrame();
    await nextFrame();
    const n = w.__perf.PopoverPortal ?? 0;
    delete w.__perf;
    return { n, popoverShown };
  });

  // Positive control: the probe is wired and the open popover reacts to zoom.
  expect(onControl.popoverShown).toBe(true);
  expect(onControl.n).toBeGreaterThan(0);
  // The fix: zooming re-renders zero hidden popovers.
  expect(onZoom).toBe(0);
});
