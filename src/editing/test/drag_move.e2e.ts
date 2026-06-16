import { expect, test } from '@playwright/test';
import {
  disableSnapping,
  dragMouse,
  laneCentreY,
  laneNotes,
  loadRockJot,
  noteGeom,
  noteLeft,
  selectedNotes,
  selectionFrame,
} from './editing_e2e_utils';

/**
 * Black-box coverage of drag-moving notes: horizontally within a lane, across
 * to another instrument lane, and as a multi-note group preserving relative
 * spacing. Positions/lane-membership are observed from the DOM after release.
 * Snapping is disabled so these exercise raw free-drag geometry (grid snapping
 * is covered in snapping.e2e.ts).
 */

test.beforeEach(async ({ page }) => {
  await loadRockJot(page);
  await disableSnapping(page);
});

test('dragging a note horizontally moves it within its lane', async ({ page }) => {
  const n = await noteGeom(page, 'h', 1);
  await dragMouse(page, { x: n.x, y: n.y }, { x: n.x + 70, y: n.y });
  const left = (await noteLeft(page, n.id))!;
  expect(left - n.left).toBeGreaterThan(40);
});

test('dragging a note onto another lane re-homes it there', async ({ page }) => {
  const src = await noteGeom(page, 'h', 1);
  const kY = await laneCentreY(page, 'k');
  await dragMouse(page, { x: src.x, y: src.y }, { x: src.x, y: kY });

  // The same note now lives in the kick row and no longer in the hi-hat row.
  await expect(
    page.locator(`[data-testid="instrument-track-k"] [data-note-id="${src.id}"]`)
  ).toHaveCount(1);
  await expect(
    page.locator(`[data-testid="instrument-track-h"] [data-note-id="${src.id}"]`)
  ).toHaveCount(0);
});

test('the note stays vertically put while dragging horizontally (no down-shift)', async ({ page }) => {
  const n = await noteGeom(page, 'h', 1);
  await page.mouse.move(n.x, n.y);
  await page.mouse.down();
  await page.mouse.move(n.x + 35, n.y, { steps: 5 });
  // Mid-drag: the glyph must keep its resting vertical centre, not drop by
  // half its height (the regression where translateX clobbered translateY).
  const midY = await page.evaluate((id) => {
    const r = document.querySelector<HTMLElement>(`[data-note-id="${id}"]`)!.getBoundingClientRect();
    return r.top + r.height / 2;
  }, n.id);
  await page.mouse.up();
  expect(Math.abs(midY - n.y)).toBeLessThan(3);
});

test('dragging across lanes previews on the destination row before release', async ({ page }) => {
  const src = await noteGeom(page, 'h', 1);
  const kY = await laneCentreY(page, 'k');
  // The kick bars-row centre, where the glyph should land mid-drag (landing on
  // any other row would be a whole row-pitch off, catching a wrong-row preview).
  const kRowCentre = await page.evaluate(() => {
    const r = document
      .querySelector('[data-testid="instrument-track-k"] [data-bars-row]')!
      .getBoundingClientRect();
    return r.top + r.height / 2;
  });
  await page.mouse.move(src.x, src.y);
  await page.mouse.down();
  await page.mouse.move(src.x, kY, { steps: 6 });
  const midY = await page.evaluate((id) => {
    const r = document.querySelector<HTMLElement>(`[data-note-id="${id}"]`)!.getBoundingClientRect();
    return r.top + r.height / 2;
  }, src.id);
  await page.mouse.up();
  expect(Math.abs(midY - kRowCentre)).toBeLessThan(10);
});

test('releasing a note drag over empty bars does not move the playhead', async ({ page }) => {
  const playheadTime = () => page.evaluate(() => (window as any).jotPlayer.currentTime as number);
  const before = await playheadTime();
  const n = await noteGeom(page, 'h', 1);
  // Drag right into empty bar space and release there; pre-fix this fell
  // through to click-to-seek and parked the playhead at the release x.
  await dragMouse(page, { x: n.x, y: n.y }, { x: n.x + 45, y: n.y });
  const after = await playheadTime();
  expect(after).toBeCloseTo(before, 3);
});

test('dragging a multi-note selection moves the whole group, preserving spacing', async ({ page }) => {
  const a = await noteGeom(page, 'h', 1);
  const b = await noteGeom(page, 'h', 3);
  const gap = b.left - a.left;

  await laneNotes(page, 'h').nth(1).click();
  await laneNotes(page, 'h').nth(3).click({ modifiers: ['Control'] });
  await expect(selectedNotes(page)).toHaveCount(2);

  // Drag the group by grabbing one of its members.
  await dragMouse(page, { x: a.x, y: a.y }, { x: a.x + 60, y: a.y });

  const aLeft = (await noteLeft(page, a.id))!;
  const bLeft = (await noteLeft(page, b.id))!;
  // Both shifted right...
  expect(aLeft - a.left).toBeGreaterThan(30);
  expect(bLeft - b.left).toBeGreaterThan(30);
  // ...by the same amount, so their spacing is preserved.
  expect(Math.abs(bLeft - aLeft - gap)).toBeLessThan(3);
  // Still a live multi-selection.
  await expect(selectedNotes(page)).toHaveCount(2);
  await expect(selectionFrame(page)).toBeVisible();
});

test('the selection frame tracks the group while dragging', async ({ page }) => {
  const a = await noteGeom(page, 'h', 1);
  await laneNotes(page, 'h').nth(1).click();
  await laneNotes(page, 'h').nth(3).click({ modifiers: ['Control'] });
  await expect(selectedNotes(page)).toHaveCount(2);

  const frameLeft = () =>
    page.evaluate(
      () => document.querySelector('[data-testid="selection-frame"]')!.getBoundingClientRect().left
    );
  const before = await frameLeft();
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(a.x + 50, a.y, { steps: 5 });
  // Mid-drag (still held): the dashed frame must travel with the glyphs, not
  // stay anchored at the pre-drag bounding box.
  const mid = await frameLeft();
  await page.mouse.up();
  expect(mid - before).toBeGreaterThan(30);
});
