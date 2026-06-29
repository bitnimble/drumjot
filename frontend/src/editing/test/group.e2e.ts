import { expect, test } from '@playwright/test';
import { groupFrames, laneNotes, loadRockJot, selectedNotes, selectionFrame } from './editing_e2e_utils';

/**
 * Black-box coverage of grouping a multi-note selection (Ctrl+G) and
 * ungrouping it (Ctrl+Shift+G). Grouping is in-place: the notes stay put and a
 * group frame appears around them; ungrouping removes the frame and leaves the
 * notes where they were.
 */

test.beforeEach(async ({ page }) => {
  await loadRockJot(page);
});

test('Ctrl+G groups a multi-selection in place: notes stay, a frame appears', async ({ page }) => {
  const before = await laneNotes(page, 'h').count();
  await expect(groupFrames(page)).toHaveCount(0);

  await laneNotes(page, 'h').nth(0).click();
  await laneNotes(page, 'h').nth(1).click({ modifiers: ['Control'] });
  await expect(selectedNotes(page)).toHaveCount(2);

  await page.keyboard.press('Control+g');

  // Notes are untouched (count unchanged) and the group frame is drawn.
  await expect(laneNotes(page, 'h')).toHaveCount(before);
  await expect(groupFrames(page)).toHaveCount(1);
});

test('Ctrl+G is a no-op for a single-note selection (no frame)', async ({ page }) => {
  await laneNotes(page, 'h').nth(0).click();
  await expect(selectedNotes(page)).toHaveCount(1);

  await page.keyboard.press('Control+g');

  await expect(groupFrames(page)).toHaveCount(0);
});

test('Ctrl+Shift+G ungroups: the frame disappears, the notes remain', async ({ page }) => {
  const before = await laneNotes(page, 'h').count();
  await laneNotes(page, 'h').nth(0).click();
  await laneNotes(page, 'h').nth(1).click({ modifiers: ['Control'] });
  await page.keyboard.press('Control+g');
  await expect(groupFrames(page)).toHaveCount(1);

  // The grouped notes are still selected (they keep their ids through grouping),
  // so ungroup acts on them directly.
  await page.keyboard.press('Control+Shift+g');

  await expect(groupFrames(page)).toHaveCount(0);
  await expect(laneNotes(page, 'h')).toHaveCount(before);
  // The restored notes get fresh ids, so the stale selection is cleared.
  await expect(selectedNotes(page)).toHaveCount(0);
  await expect(selectionFrame(page)).toHaveCount(0);
});
