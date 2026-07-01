import { expect, test } from '@playwright/test';
import { noteGeom } from './editing_e2e_utils';
import {
  barsRowRect,
  engageFollow,
  followPlayhead,
  loadWideScore,
  playheadTime,
  scrollerRect,
  scrollOffsets,
  touchPan,
  touchPinch,
  touchTap,
  zoom,
} from './touch_e2e_utils';

/**
 * Black-box coverage of the score's touch gestures (jot_editor.tsx's native
 * touchstart/move/end handler): one-finger pan, two-finger pinch-zoom,
 * tap-to-seek. Load-bearing invariant: a drag pans WITHOUT moving the playhead,
 * only an isolated tap seeks; gated on a move budget (TAP_MOVE_PX = 8) and a
 * duration budget (TAP_MAX_DURATION_MS = 250), both exercised here.
 */

test.beforeEach(async ({ page }) => {
  await loadWideScore(page);
});

test('one-finger drag pans the score horizontally', async ({ page }) => {
  const before = await scrollOffsets(page);
  expect(before.x).toBe(0);
  const r = await scrollerRect(page);
  const y = r.top + r.height / 2;
  // Finger travels LEFT, so the content scrolls right (scrollX grows), the
  // natural direct-manipulation direction.
  await touchPan(page, { x: r.left + r.width * 0.75, y }, { x: r.left + r.width * 0.3, y });
  const after = await scrollOffsets(page);
  expect(after.x).toBeGreaterThan(100);
  // A horizontal drag shouldn't drift the vertical offset.
  expect(Math.abs(after.y - before.y)).toBeLessThan(2);
});

test('a one-finger pan does NOT move the playhead', async ({ page }) => {
  const before = await playheadTime(page);
  const r = await scrollerRect(page);
  const y = r.top + r.height / 2;
  await touchPan(page, { x: r.left + r.width * 0.75, y }, { x: r.left + r.width * 0.3, y });
  const after = await playheadTime(page);
  // Panning is a "look elsewhere" gesture; it must never seek.
  expect(after).toBeCloseTo(before, 3);
});

test('an isolated tap on empty bars seeks the playhead', async ({ page }) => {
  const before = await playheadTime(page);
  expect(before).toBeCloseTo(0, 3);
  const bars = await barsRowRect(page);
  // Tap well into the bars row (dispatched on the row itself, i.e. empty space,
  // not a note), so the seek target is a clearly-positive time.
  await touchTap(
    page,
    { x: bars.left + 360, y: bars.top + bars.height / 2 },
    { targetSelector: '[data-bars-row]' }
  );
  const after = await playheadTime(page);
  expect(after).toBeGreaterThan(before + 0.05);
  // A tap seeks; it must not pan.
  expect((await scrollOffsets(page)).x).toBe(0);
});

test('a tap that wobbles within the move budget still seeks', async ({ page }) => {
  const before = await playheadTime(page);
  const bars = await barsRowRect(page);
  // 5px of jitter is under TAP_MOVE_PX (8), so this stays a tap.
  await touchTap(
    page,
    { x: bars.left + 360, y: bars.top + bars.height / 2 },
    { targetSelector: '[data-bars-row]', jitterPx: 5 }
  );
  expect(await playheadTime(page)).toBeGreaterThan(before + 0.05);
});

test('a slow press beyond the duration budget does NOT seek', async ({ page }) => {
  const before = await playheadTime(page);
  const bars = await barsRowRect(page);
  // Held past TAP_MAX_DURATION_MS (250ms) without moving: not a tap, so no seek.
  await touchTap(
    page,
    { x: bars.left + 360, y: bars.top + bars.height / 2 },
    { targetSelector: '[data-bars-row]', holdMs: 320 }
  );
  expect(await playheadTime(page)).toBeCloseTo(before, 3);
});

test('tapping a note does not seek the playhead', async ({ page }) => {
  const before = await playheadTime(page);
  const n = await noteGeom(page, 'h', 1);
  // The touch lands on the note glyph (data-noseek), so seekFromClick bails.
  await touchTap(page, { x: n.x, y: n.y }, { targetSelector: `[data-note-id="${n.id}"]` });
  expect(await playheadTime(page)).toBeCloseTo(before, 3);
});

test('an isolated tap keeps playhead auto-follow engaged', async ({ page }) => {
  await engageFollow(page);
  expect(await followPlayhead(page)).toBe(true);
  const bars = await barsRowRect(page);
  await touchTap(
    page,
    { x: bars.left + 360, y: bars.top + bars.height / 2 },
    { targetSelector: '[data-bars-row]' }
  );
  // Seeking via a tap is not a "look elsewhere" gesture, so follow stays on.
  expect(await followPlayhead(page)).toBe(true);
});

test('a pan disengages playhead auto-follow', async ({ page }) => {
  await engageFollow(page);
  expect(await followPlayhead(page)).toBe(true);
  const r = await scrollerRect(page);
  const y = r.top + r.height / 2;
  await touchPan(page, { x: r.left + r.width * 0.75, y }, { x: r.left + r.width * 0.3, y });
  expect(await followPlayhead(page)).toBe(false);
});

test('two-finger pinch-out zooms the score in', async ({ page }) => {
  const before = await zoom(page);
  const r = await scrollerRect(page);
  const center = { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  await touchPinch(page, center, 120, 320);
  expect(await zoom(page)).toBeGreaterThan(before * 1.2);
});

test('two-finger pinch-in zooms the score out', async ({ page }) => {
  const before = await zoom(page);
  const r = await scrollerRect(page);
  const center = { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  await touchPinch(page, center, 320, 120);
  expect(await zoom(page)).toBeLessThan(before * 0.85);
});
