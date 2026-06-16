import { expect, test } from '@playwright/test';
import { laneNotes, loadRockJot, noteGeom, setMode } from './editing_e2e_utils';

/**
 * The rock loop is authored as two `||` layers (hands: hi-hat + snare; feet:
 * kick). The mixer renders one row per lane, so every lane, including the
 * kick, which lives in the non-first layer, must show its notes, and editing
 * a lane's row must target that lane's layer (so an inserted/moved note lands
 * in the row that was clicked, not whichever layer is first).
 */

test.beforeEach(async ({ page }) => {
  await loadRockJot(page);
});

test('the kick lane (a non-first layer) renders its notes', async ({ page }) => {
  // Two bars: kicks on beats 1 & 5 of bar 1, and 1, 1-and, 5 of bar 2.
  await expect(laneNotes(page, 'k')).toHaveCount(5);
});

test('inserting into the kick row lands in the kick row', async ({ page }) => {
  const before = await laneNotes(page, 'k').count();
  await setMode(page, 'insert');
  // Empty kick spot: ~0.7 beats right of the first kick (beats 1-3 are empty).
  const k0 = await noteGeom(page, 'k', 0);
  const x = k0.x + 80;
  await page.mouse.move(x, k0.y);
  await page.mouse.click(x, k0.y);
  await expect(laneNotes(page, 'k')).toHaveCount(before + 1);
});
