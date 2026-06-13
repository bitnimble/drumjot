import { expect, test } from '@playwright/test';
import { DEBUG_BUNDLE_PATH, loadDebugBundle } from './debug_bundle.helper';

/**
 * Per-frame performance guards. Both run against the full transcriber debug
 * bundle (`E2E_DEBUG_BUNDLE`, see AGENTS.md) - a real, heavy song - because
 * a perf assertion against the tiny 4s rock-loop example proves nothing.
 * Skipped when the var is unset, so the default `bun run e2e` stays green
 * and deterministic.
 *
 * ## Why busy-time, not rAF interval
 *
 * `requestAnimationFrame` deltas are pinned to the compositor's BeginFrame
 * cadence (~16.7ms in headless Chromium even when the page is idle), so a
 * "frame interval < 17ms" assertion would be meaningless - every idle frame
 * already sits at the vsync floor. Instead we measure the *main-thread busy
 * time* per frame: a `MessageChannel` message posted from inside the rAF
 * callback runs as a task immediately after that frame's render steps
 * (style/layout/paint), so `messageTime - frameStart` is the work the main
 * thread did for the frame, independent of the vsync wait. At idle this
 * reads ~0ms, so a 17ms (60fps) budget is a real bar.
 *
 * Note: numbers come from headless Chromium with software rendering, which
 * paints slower than a real GPU; treat a red result here as "investigate",
 * not necessarily "broken on a real machine".
 */

/** Drumjot targets 120fps (see AGENTS.md "Frame budget"): the typical
 *  frame's main-thread work must fit one 120fps frame. */
const FRAME_BUDGET_120_MS = 8.3;
/** One 60fps frame. The slowest frames are allowed up to here: the most
 *  zoomed-OUT frames lay out + paint the most bars at once (a viewport
 *  covers ~2x the beats at 0.5x zoom), and headless software rendering
 *  paints several times slower than the GPU these numbers are really for,
 *  so a hard "every frame < 8.3ms" gate would measure render-backend
 *  noise more than the renderer. The gate below is therefore two-tier. */
const FRAME_BUDGET_60_MS = 16.7;
/** Fraction of frames allowed to exceed the 60fps budget (rare GC pauses
 *  + the deepest-zoom-out frames). */
const MAX_SLOW_FRACTION = 0.1;

type BusyResult = {
  /** Per-frame busy times (ms) after warm-up frames are dropped. */
  steady: number[];
  /** Did the driven activity actually do work (guards against a vacuous
   *  pass where nothing rendered)? */
  active: boolean;
};

/**
 * Run a measured rAF loop in the page and return the per-frame main-thread
 * busy times with the first `warmup` frames dropped.
 *
 * `driver`:
 *  - `'zoom'`: sweep `store.zoom` every frame (a triangle wave), the
 *    same work a continuous wheel-zoom gesture does - the score's hottest
 *    path.
 *  - `'scroll'`: pan `store.scrollX` every frame. Horizontal
 *    virtualization makes this the windowing hot path: each tick re-runs
 *    the visible-range filter and mounts/unmounts the bars crossing the
 *    buffer edge.
 *  - `'playback'`: drive nothing; the player's own rAF advances
 *    `currentTime` and the playhead/scroll observers react. Measures the
 *    cost of a playback frame.
 */
async function measureFrameBusyMs(
  page: import('@playwright/test').Page,
  driver: 'zoom' | 'scroll' | 'playback',
  frames: number,
  warmup: number,
): Promise<BusyResult> {
  return page.evaluate(
    async ({ driver, frames, warmup }) => {
      const viewport = (window as any).drumjot.viewport;
      const presenter = (window as any).drumjot.presenter;
      const player = (window as any).jotPlayer;
      const busy: number[] = [];
      const mc = new MessageChannel();
      let resolveDone: () => void;
      const done = new Promise<void>((r) => (resolveDone = r));
      let frameStart = 0;
      let i = 0;
      const startTime = driver === 'playback' ? player.currentTime : 0;
      const startZoom = driver === 'zoom' ? viewport.zoom : 0;
      const startScroll = driver === 'scroll' ? viewport.scrollX : 0;

      // A message posted during the rAF callback runs as a task once the
      // frame's render steps have completed, so it brackets the frame's
      // main-thread work without waiting for the next vsync.
      mc.port1.onmessage = () => {
        busy.push(performance.now() - frameStart);
        if (i >= frames) return resolveDone();
        requestAnimationFrame(loop);
      };
      const loop = () => {
        // Triangle sweep 0.5 <-> 2.5 in 0.1 steps (zoom in for 20 frames,
        // back out for 20). Models a *sustained* wheel/pinch gesture,
        // where pxPerBeat moves a few percent per frame; the previous
        // sawtooth `i % 20` snapped 2.4 -> 0.5 every 20th frame (a 5x
        // zoom-out in one frame), which mounts every newly-revealed bar
        // at once - a discrete slider jump, not the 120fps gesture this
        // budget is for.
        if (driver === 'zoom') presenter.setZoom(0.5 + Math.abs((i % 40) - 20) * 0.1);
        // Triangle pan 0 <-> ~12000px (200px/frame), a brisk continuous
        // horizontal scroll that repeatedly crosses bar/window boundaries.
        else if (driver === 'scroll') presenter.setScrollX(Math.abs((i % 120) - 60) * 200);
        i++;
        frameStart = performance.now();
        mc.port2.postMessage(null);
      };
      requestAnimationFrame(loop);
      await done;

      const active =
        driver === 'zoom'
          ? viewport.zoom !== startZoom // setZoom is wired and the sweep took effect
          : driver === 'scroll'
            ? viewport.scrollX !== startScroll
            : player.state === 'playing' && player.currentTime > startTime;
      return { steady: busy.slice(warmup), active };
    },
    { driver, frames, warmup },
  );
}

/** p(percentile) of an unsorted sample, for the failure message. */
function pct(values: number[], p: number): number {
  const s = [...values].sort((a, b) => a - b);
  return Math.round((s[Math.floor(s.length * p)] ?? 0) * 100) / 100;
}

/**
 * Two-tier smoothness gate over a steady-state frame sample:
 *   1. The MEDIAN frame holds 120fps (the project target) - i.e. the
 *      sustained gesture is smooth, not just occasionally fast.
 *   2. At most {@link MAX_SLOW_FRACTION} of frames exceed the 60fps
 *      budget - rare GC pauses and the deepest-zoom-out frames (which
 *      lay out the most bars) are tolerated, a sustained run of dropped
 *      frames is not.
 * The single shared message reports the full distribution so a red gate
 * shows where it went (median vs tail).
 */
function expectSmooth(label: string, steady: number[]): void {
  const median = pct(steady, 0.5);
  const slow = steady.filter((v) => v >= FRAME_BUDGET_60_MS).length;
  const slowFraction = slow / steady.length;
  const worst = Math.round(Math.max(...steady) * 100) / 100;
  const msg =
    `${label}: median ${median}ms, p95 ${pct(steady, 0.95)}ms, max ${worst}ms, ` +
    `${slow}/${steady.length} frames over ${FRAME_BUDGET_60_MS}ms`;
  expect(median, `${msg} - median must hold 120fps (<${FRAME_BUDGET_120_MS}ms)`).toBeLessThan(
    FRAME_BUDGET_120_MS,
  );
  expect(
    slowFraction,
    `${msg} - too many frames below 60fps (>${MAX_SLOW_FRACTION * 100}%)`,
  ).toBeLessThanOrEqual(MAX_SLOW_FRACTION);
}

test.describe('per-frame performance', () => {
  // Serial: these measure main-thread busy time, so running them in
  // parallel workers (Playwright's default) makes the heavy bundle loads
  // contend for CPU and inflates every frame ~2x. One at a time keeps the
  // numbers a real signal.
  test.describe.configure({ mode: 'serial' });
  test.skip(!DEBUG_BUNDLE_PATH, 'E2E_DEBUG_BUNDLE not set');

  test('zooming the score holds 120fps', async ({ page }) => {
    test.setTimeout(180_000); // large zip unpack + multi-track audio decode
    await loadDebugBundle(page);

    // 120 frames of continuous zooming; drop the first 20 so a one-off
    // initial layout / style-cache warm-up doesn't count against the
    // steady-state gesture.
    const { steady, active } = await measureFrameBusyMs(page, 'zoom', 120, 20);
    expect(active).toBe(true);
    expect(steady.length).toBeGreaterThan(50);
    expectSmooth('zoom', steady);
  });

  test('scrolling the score holds 120fps', async ({ page }) => {
    test.setTimeout(180_000);
    await loadDebugBundle(page);
    await page.evaluate(() => (window as any).drumjot.presenter.setZoom(1));

    // 120 frames of continuous horizontal panning; drop the first 20 for
    // warm-up. With windowing, each tick re-runs the visible-range filter
    // and mounts the bars entering the buffer.
    const { steady, active } = await measureFrameBusyMs(page, 'scroll', 120, 20);
    expect(active).toBe(true);
    expect(steady.length).toBeGreaterThan(50);
    expectSmooth('scroll', steady);
  });

  test('playback holds 120fps', async ({ page }) => {
    test.setTimeout(180_000); // bundle load + ~30MB SoundFont fetch on first play

    await loadDebugBundle(page);
    await page.evaluate(() => (window as any).drumjot.presenter.setZoom(1));

    // Start playback and wait for it to actually run. Reaching 'playing'
    // downloads the TR-808 SoundFont from the smplr CDN on a cold cache, so
    // give it a generous budget (same signal the audio-track e2e relies on).
    await page.evaluate(() => (window as any).drumjot.store.togglePlayPause());
    await page.waitForFunction(
      () => (window as any).jotPlayer.state === 'playing',
      null,
      { timeout: 60_000 },
    );

    // Measure 150 frames, dropping the first 30: the first frame after
    // playback starts carries one-off startup work (sample scheduling, the
    // initial scroll-follow jump, first playhead paint) that isn't a
    // steady-state playback cost.
    const { steady, active } = await measureFrameBusyMs(page, 'playback', 150, 30);
    expect(active).toBe(true);
    expect(steady.length).toBeGreaterThan(50);
    expectSmooth('playback', steady);

    await page.evaluate(() => (window as any).drumjot.store.stopPlayback());
  });
});
