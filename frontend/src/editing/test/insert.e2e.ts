import { expect, test } from '@playwright/test';
import { laneNotes, loadRockJot, noteGeom, setMode } from './editing_e2e_utils';

/**
 * Black-box coverage of inserting notes: switch to insert mode via the
 * floating editing toolbar, move the cursor over an empty spot in a lane, and
 * click, a new note glyph appears in that lane. Uses the snare lane (sparse,
 * with wide empty stretches at beats 0/1/3/4/5).
 */

test.beforeEach(async ({ page }) => {
  await loadRockJot(page);
});

test('insert mode adds a note to the lane that is clicked', async ({ page }) => {
  const before = await laneNotes(page, 's').count();
  await setMode(page, 'insert');

  // The snare's first note is on beat 2; click well to its left (empty beat ~1).
  const s0 = await noteGeom(page, 's', 0);
  const x = s0.left - 60;
  await page.mouse.move(x, s0.y);
  await page.mouse.click(x, s0.y);

  await expect(laneNotes(page, 's')).toHaveCount(before + 1);
});

test('insert mode does not add notes once switched back to select', async ({ page }) => {
  await setMode(page, 'insert');
  const s0 = await noteGeom(page, 's', 0);
  await page.mouse.move(s0.left - 60, s0.y);
  await page.mouse.click(s0.left - 60, s0.y);
  const afterInsert = await laneNotes(page, 's').count();

  await setMode(page, 'select');
  await page.mouse.click(s0.left - 90, s0.y);
  await expect(laneNotes(page, 's')).toHaveCount(afterInsert);
});
