import { expect, test } from '@playwright/test';
import {
  laneNotes,
  loadRockJot,
  selectedNotes,
  selectionFrame,
} from './editing_e2e_utils';

/**
 * Black-box coverage of deleting selected notes via the keyboard. The note
 * glyph disappears from the DOM and the selection clears.
 */

test.beforeEach(async ({ page }) => {
  await loadRockJot(page);
});

test('Delete removes the single selected note', async ({ page }) => {
  const before = await laneNotes(page, 'h').count();
  await laneNotes(page, 'h').nth(0).click();
  await expect(selectedNotes(page)).toHaveCount(1);

  await page.keyboard.press('Delete');

  await expect(laneNotes(page, 'h')).toHaveCount(before - 1);
  await expect(selectedNotes(page)).toHaveCount(0);
});

test('Backspace removes every selected note in a multi-selection', async ({ page }) => {
  const before = await laneNotes(page, 'h').count();
  await laneNotes(page, 'h').nth(0).click();
  await laneNotes(page, 'h').nth(2).click({ modifiers: ['Control'] });
  await expect(selectedNotes(page)).toHaveCount(2);

  await page.keyboard.press('Backspace');

  await expect(laneNotes(page, 'h')).toHaveCount(before - 2);
  await expect(selectedNotes(page)).toHaveCount(0);
  await expect(selectionFrame(page)).toHaveCount(0);
});

test('deleting a marquee selection clears all of it', async ({ page }) => {
  await laneNotes(page, 'h').nth(0).click();
  await laneNotes(page, 'h').nth(1).click({ modifiers: ['Control'] });
  await laneNotes(page, 'h').nth(2).click({ modifiers: ['Control'] });
  await expect(selectedNotes(page)).toHaveCount(3);

  await page.keyboard.press('Delete');
  await expect(selectedNotes(page)).toHaveCount(0);
});
