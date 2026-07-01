import type { Page } from '@playwright/test';

/**
 * Synthetic-touch helpers for the score gesture specs: Playwright's high-level
 * input can't drive multi-finger / multi-frame touch, so these dispatch real
 * `Touch`/`TouchEvent` sequences at the score's own listeners
 * (`[data-jot-scroller]`), one `page.evaluate` per gesture with a rAF yield
 * between frames. Assertions read back from the store / DOM like the mouse specs.
 */

export type Pt = { x: number; y: number };

/** The score's touch listeners live on the scroll container. */
export const SCROLLER = '[data-jot-scroller]';

/** Wide score (60 bars): the 2-bar built-in loop fits the viewport and wouldn't pan. */
const WIDE_JOT = `{{ bpm: 120, time: "4/4", title: "Touch Gestures",
  instrumentMapping: { h: { name: "HiHat" }, s: { name: "Snare" }, k: { name: "Kick" } } }}
|${Array.from({ length: 60 }, () => ' k+h s+h k+h s+h ').join('|')}|
`;

/** Yield a few animation frames so the ResizeObservers have populated the
 *  viewport/content extents the scroll clamp reads (otherwise pan clamps to 0). */
export function settle(page: Page): Promise<void> {
  return page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        let n = 0;
        const step = () => (++n >= 3 ? resolve() : requestAnimationFrame(step));
        requestAnimationFrame(step);
      })
  );
}

/** Load the wide gesture-test score and wait until it's laid out + measured. */
export async function loadWideScore(page: Page): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(
    () => typeof (window as unknown as { drumjot?: { loadDsl?: unknown } }).drumjot?.loadDsl === 'function'
  );
  await page.evaluate(
    (src) => (window as unknown as { drumjot: { loadDsl(s: string): void } }).drumjot.loadDsl(src),
    WIDE_JOT
  );
  await page.waitForSelector('[data-bars-row]');
  await settle(page);
}

/** Interpolate `n` intermediate points between `a` and `b` (inclusive of `b`). */
function lerpPoints(a: Pt, b: Pt, steps: number): Pt[] {
  const pts: Pt[] = [];
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    pts.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
  }
  return pts;
}

/**
 * One-finger press → drag → release, with a `requestAnimationFrame` yield
 * before each move frame. `targetSelector` is the element the touch originates
 * on (it becomes the touch target); default is the scroller.
 */
export async function touchPan(
  page: Page,
  from: Pt,
  to: Pt,
  opts: { steps?: number; targetSelector?: string } = {}
): Promise<void> {
  const steps = opts.steps ?? 8;
  const targetSelector = opts.targetSelector ?? SCROLLER;
  const moves = lerpPoints(from, to, steps);
  await page.evaluate(
    async ({ from, moves, targetSelector }) => {
      const target = document.querySelector(targetSelector);
      if (!target) throw new Error(`touchPan: no element for ${targetSelector}`);
      const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
      const fire = (type: string, pts: { x: number; y: number }[]) => {
        const touches = pts.map(
          (p, i) => new Touch({ identifier: i, target, clientX: p.x, clientY: p.y })
        );
        const active = type === 'touchend' ? [] : touches;
        target.dispatchEvent(
          new TouchEvent(type, {
            cancelable: true,
            bubbles: true,
            touches: active,
            targetTouches: active,
            changedTouches: touches,
          })
        );
      };
      fire('touchstart', [from]);
      for (const m of moves) {
        await nextFrame();
        fire('touchmove', [m]);
      }
      await nextFrame();
      fire('touchend', [moves[moves.length - 1]]);
    },
    { from, moves, targetSelector }
  );
}

/**
 * A single tap: press then release at (nearly) the same point. `jitterPx`
 * nudges the release so a within-budget wobble can be exercised; `holdMs`
 * holds between press and release to exercise the duration budget.
 */
export async function touchTap(
  page: Page,
  at: Pt,
  opts: { targetSelector?: string; jitterPx?: number; holdMs?: number } = {}
): Promise<void> {
  const targetSelector = opts.targetSelector ?? SCROLLER;
  const jitterPx = opts.jitterPx ?? 0;
  const holdMs = opts.holdMs ?? 0;
  await page.evaluate(
    async ({ at, targetSelector, jitterPx, holdMs }) => {
      const target = document.querySelector(targetSelector);
      if (!target) throw new Error(`touchTap: no element for ${targetSelector}`);
      const wait = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
      const fire = (type: string, p: { x: number; y: number }) => {
        const touch = new Touch({ identifier: 0, target, clientX: p.x, clientY: p.y });
        const active = type === 'touchend' ? [] : [touch];
        target.dispatchEvent(
          new TouchEvent(type, {
            cancelable: true,
            bubbles: true,
            touches: active,
            targetTouches: active,
            changedTouches: [touch],
          })
        );
      };
      fire('touchstart', at);
      if (jitterPx) fire('touchmove', { x: at.x + jitterPx, y: at.y });
      if (holdMs) await wait(holdMs);
      const end = jitterPx ? { x: at.x + jitterPx, y: at.y } : at;
      fire('touchend', end);
    },
    { at, targetSelector, jitterPx, holdMs }
  );
}

/**
 * Two-finger pinch centred on `center`: both fingers start `startGap` apart
 * (horizontally) and end `endGap` apart. `endGap > startGap` spreads (zoom in),
 * `endGap < startGap` pinches (zoom out).
 */
export async function touchPinch(
  page: Page,
  center: Pt,
  startGap: number,
  endGap: number,
  opts: { steps?: number; targetSelector?: string } = {}
): Promise<void> {
  const steps = opts.steps ?? 8;
  const targetSelector = opts.targetSelector ?? SCROLLER;
  await page.evaluate(
    async ({ center, startGap, endGap, steps, targetSelector }) => {
      const target = document.querySelector(targetSelector);
      if (!target) throw new Error(`touchPinch: no element for ${targetSelector}`);
      const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
      const pair = (gap: number) => [
        { x: center.x - gap / 2, y: center.y },
        { x: center.x + gap / 2, y: center.y },
      ];
      const fire = (type: string, pts: { x: number; y: number }[]) => {
        const touches = pts.map(
          (p, i) => new Touch({ identifier: i, target, clientX: p.x, clientY: p.y })
        );
        const active = type === 'touchend' ? [] : touches;
        target.dispatchEvent(
          new TouchEvent(type, {
            cancelable: true,
            bubbles: true,
            touches: active,
            targetTouches: active,
            changedTouches: touches,
          })
        );
      };
      fire('touchstart', pair(startGap));
      for (let i = 1; i <= steps; i++) {
        const gap = startGap + (endGap - startGap) * (i / steps);
        await nextFrame();
        fire('touchmove', pair(gap));
      }
      await nextFrame();
      fire('touchend', pair(endGap));
    },
    { center, startGap, endGap, steps, targetSelector }
  );
}

/** Read the live virtual-scroll offsets (store, not DOM) after a gesture. */
export function scrollOffsets(page: Page): Promise<{ x: number; y: number }> {
  return page.evaluate(() => {
    const v = (window as unknown as { drumjot: { viewport: { scrollX: number; scrollY: number } } })
      .drumjot.viewport;
    return { x: v.scrollX, y: v.scrollY };
  });
}

/** Read the current zoom multiplier. */
export function zoom(page: Page): Promise<number> {
  return page.evaluate(
    () => (window as unknown as { drumjot: { viewport: { zoom: number } } }).drumjot.viewport.zoom
  );
}

/** Read whether the playhead auto-follow is engaged. */
export function followPlayhead(page: Page): Promise<boolean> {
  return page.evaluate(
    () =>
      (window as unknown as { drumjot: { playback: { followPlayhead: boolean } } }).drumjot.playback
        .followPlayhead
  );
}

/** Force the auto-follow flag on, so a spec can assert a gesture disengages it. */
export function engageFollow(page: Page): Promise<void> {
  return page.evaluate(() =>
    (
      window as unknown as { drumjot: { playbackPresenter: { setFollowPlayhead(on: boolean): void } } }
    ).drumjot.playbackPresenter.setFollowPlayhead(true)
  );
}

/** Playhead time in seconds (media clock), the same value the mouse specs read. */
export function playheadTime(page: Page): Promise<number> {
  return page.evaluate(
    () => (window as unknown as { jotPlayer: { currentTime: number } }).jotPlayer.currentTime
  );
}

/** Bounding rect of the first bars row, in client (viewport) pixels. */
export function barsRowRect(page: Page): Promise<{ left: number; top: number; width: number; height: number }> {
  return page.evaluate(() => {
    const el = document.querySelector('[data-bars-row]');
    if (!el) throw new Error('no [data-bars-row]');
    const r = el.getBoundingClientRect();
    return { left: r.left, top: r.top, width: r.width, height: r.height };
  });
}

/** Bounding rect of the scroll container, in client (viewport) pixels. */
export function scrollerRect(page: Page): Promise<{ left: number; top: number; width: number; height: number }> {
  return page.evaluate((sel) => {
    const el = document.querySelector(sel);
    if (!el) throw new Error(`no ${sel}`);
    const r = el.getBoundingClientRect();
    return { left: r.left, top: r.top, width: r.width, height: r.height };
  }, SCROLLER);
}
