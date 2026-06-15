import { expect, test } from '@playwright/test';
import {
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
 */

test.beforeEach(async ({ page }) => {
  await loadRockJot(page);
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
