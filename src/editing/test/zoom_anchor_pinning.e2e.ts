import { expect, test } from '@playwright/test';

/**
 * Regression guard for cursor-anchored wheel-zoom deep in a long song.
 *
 * Two failure modes this pins down:
 *   1. The musical point under the cursor must stay put across a zoom
 *      gesture (the anchor must keep it at `clientX`). `delta ≈
 *      scrollX·(factor−1)` is many screens deep in a long song, so any
 *      drift here surfaces as the viewport jumping to a different section.
 *   2. Scale (`--px-per-beat`) and scroll (`--scroll-x`) must reach the
 *      DOM in lockstep. They used to land via two different React effect
 *      phases (ScoreZoomVar's passive `useEffect` vs ScrollVar's
 *      `useLayoutEffect`), so for ≥1 painted frame the wrapper sat at the
 *      post-zoom scroll offset with the pre-zoom scale, painting an
 *      earlier section (a random "jump"). The flush now writes both vars
 *      in the same tick; we assert they stay mutually consistent.
 *
 * Bars are `|`-separated; ~240 of them => content many screens wide.
 */
const JOT = `{{ bpm: 120, time: "4/4", title: "Zoom Anchor",
  instrumentMapping: { h: { name: "HiHat" }, s: { name: "Snare" }, k: { name: "Kick" } } }}
|${Array.from({ length: 240 }, () => ' k+h s+h k+h s+h ').join('|')}|
`;

test('cursor-anchored zoom keeps the anchor pinned, scale+scroll in lockstep', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => typeof (window as any).drumjot?.loadDsl === 'function');
  await page.evaluate((src) => (window as any).drumjot.loadDsl(src), JOT);
  await page.waitForSelector('[data-testid^="instrument-track-"]');

  const result = await page.evaluate(async () => {
    const w = window as any;
    const viewportStore = w.drumjot.viewport;
    const viewportPresenter = w.drumjot.viewportPresenter;
    const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
    const settle = async () => {
      await nextFrame();
      await nextFrame();
      await nextFrame();
    };
    await settle();

    const scroller = document.querySelector('[data-jot-scroller]') as HTMLElement;
    const barsRow = scroller.querySelector('[data-bars-row]') as HTMLElement;
    const viewport = scroller.querySelector('[data-jot-scroll-content]') as HTMLElement;
    const scrollerRect = scroller.getBoundingClientRect();

    // Scroll most of the way into the song so the per-tick scroll delta is
    // large (this is where a desync paints multiple screens off).
    viewportPresenter.setScrollX(
      Math.max(0, viewportStore._contentWidth - viewportStore._viewportWidth) * 0.8
    );
    await settle();

    const clientX = scrollerRect.left + scrollerRect.width * 0.7;
    const clientY = scrollerRect.top + scrollerRect.height * 0.5;

    // The DOM's live scale/scroll (what the browser actually paints).
    const ppbDom = () =>
      Number(scroller.style.getPropertyValue('--px-per-beat')) || w.drumjot.jotEditorStore.structural.pxPerBeat;
    const scrollDom = () => Number(viewport.style.getPropertyValue('--scroll-x')) || 0;

    // Musical beat under the cursor right now. Bar layout: content-x of
    // beat b = padLeft + b*pxPerBeat, padLeft = pxPerBeat/24.
    const ppb0 = ppbDom();
    const trackedBeat = (clientX - barsRow.getBoundingClientRect().left - ppb0 / 24) / ppb0;

    let maxDrift = 0; // how far the tracked beat slips from the cursor (px)
    let maxDesync = 0; // |store.scrollX − DOM --scroll-x| sampled per tick (px)
    const sample = () => {
      const ppb = ppbDom();
      const trackedScreenX = barsRow.getBoundingClientRect().left + ppb / 24 + trackedBeat * ppb;
      maxDrift = Math.max(maxDrift, Math.abs(trackedScreenX - clientX));
      maxDesync = Math.max(maxDesync, Math.abs(viewportStore.scrollX - scrollDom()));
    };

    const fireWheel = (dir: number) =>
      scroller.dispatchEvent(
        new WheelEvent('wheel', {
          deltaY: dir * 3, // mouse-wheel notch: <0 zoom in, >0 zoom out
          deltaMode: 1,
          clientX,
          clientY,
          bubbles: true,
          cancelable: true,
        })
      );

    // Rapid zoom IN then OUT, 3 coalesced notches per frame, single-frame
    // yields (no settling) to stress the effect-phase race.
    for (let i = 0; i < 10 && viewportStore.zoom < 3.99; i++) {
      fireWheel(-1);
      fireWheel(-1);
      fireWheel(-1);
      await nextFrame();
      sample();
    }
    for (let i = 0; i < 16 && viewportStore.zoom > 0.11; i++) {
      fireWheel(+1);
      fireWheel(+1);
      fireWheel(+1);
      await nextFrame();
      sample();
    }
    await settle();
    sample();

    return { maxDrift: Math.round(maxDrift), maxDesync: Math.round(maxDesync) };
  });

  // The cursor's musical point must stay within a few px of the cursor
  // through the whole gesture (sub-pixel snap + measurement slack).
  expect(result.maxDrift).toBeLessThan(40);
  // Store scrollX and the painted --scroll-x must agree every tick.
  expect(result.maxDesync).toBeLessThan(2);
});
