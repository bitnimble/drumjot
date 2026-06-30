import { expect, test } from '@playwright/test';
import { laneNotes, loadRockJot, selectedNotes } from './editing_e2e_utils';

/**
 * Black-box coverage of undo/redo via the keyboard. Each user gesture is one
 * Loro commit, so it's one undo step; undo/redo replay through the same event
 * path as a live edit, so the score DOM reflects the reverted state. On this
 * headless Linux Chromium, `Control` is the platform `Mod` key.
 */

test.beforeEach(async ({ page }) => {
  await loadRockJot(page);
});

test('Ctrl+Z restores a deleted note and Ctrl+Shift+Z deletes it again', async ({ page }) => {
  const before = await laneNotes(page, 'h').count();
  await laneNotes(page, 'h').nth(0).click();
  await expect(selectedNotes(page)).toHaveCount(1);

  await page.keyboard.press('Delete');
  await expect(laneNotes(page, 'h')).toHaveCount(before - 1);

  // Undo: the note comes back.
  await page.keyboard.press('Control+z');
  await expect(laneNotes(page, 'h')).toHaveCount(before);

  // Redo: gone again.
  await page.keyboard.press('Control+Shift+z');
  await expect(laneNotes(page, 'h')).toHaveCount(before - 1);
});

test('Ctrl+Y also redoes (Windows convention)', async ({ page }) => {
  const before = await laneNotes(page, 'h').count();
  await laneNotes(page, 'h').nth(0).click();
  await page.keyboard.press('Delete');
  await expect(laneNotes(page, 'h')).toHaveCount(before - 1);

  await page.keyboard.press('Control+z');
  await expect(laneNotes(page, 'h')).toHaveCount(before);

  await page.keyboard.press('Control+y');
  await expect(laneNotes(page, 'h')).toHaveCount(before - 1);
});

test('each gesture is its own undo step', async ({ page }) => {
  const before = await laneNotes(page, 'h').count();

  // Two separate deletions = two undo steps.
  await laneNotes(page, 'h').nth(0).click();
  await page.keyboard.press('Delete');
  await laneNotes(page, 'h').nth(0).click();
  await page.keyboard.press('Delete');
  await expect(laneNotes(page, 'h')).toHaveCount(before - 2);

  await page.keyboard.press('Control+z');
  await expect(laneNotes(page, 'h')).toHaveCount(before - 1);
  await page.keyboard.press('Control+z');
  await expect(laneNotes(page, 'h')).toHaveCount(before);
});

test('the Edit menu Undo/Redo items reflect availability, show the shortcut, and act', async ({
  page,
}) => {
  const openEditMenu = () => page.getByRole('button', { name: 'Edit' }).click();
  const undoItem = page.getByTestId('edit-menu-undo');
  const redoItem = page.getByTestId('edit-menu-redo');

  // Nothing to undo/redo yet: both visible but disabled, with their shortcuts
  // shown (Ctrl on this Linux Chromium; the pill text comes from the keymap).
  await openEditMenu();
  await expect(undoItem).toBeVisible();
  await expect(undoItem).toBeDisabled();
  await expect(redoItem).toBeDisabled();
  await expect(page.getByTestId('edit-menu-undo-shortcut')).toHaveText('Ctrl+Z');
  await expect(page.getByTestId('edit-menu-redo-shortcut')).toHaveText('Ctrl+Shift+Z');
  await page.keyboard.press('Escape');

  // Make an edit; Undo becomes available.
  const before = await laneNotes(page, 'h').count();
  await laneNotes(page, 'h').nth(0).click();
  await page.keyboard.press('Delete');
  await expect(laneNotes(page, 'h')).toHaveCount(before - 1);

  await openEditMenu();
  await expect(undoItem).toBeEnabled();
  await undoItem.click(); // also dismisses the menu
  await expect(laneNotes(page, 'h')).toHaveCount(before);

  // Redo is now available from the menu.
  await openEditMenu();
  await expect(redoItem).toBeEnabled();
  await redoItem.click();
  await expect(laneNotes(page, 'h')).toHaveCount(before - 1);
});

test('a fresh edit after undo invalidates the redo stack', async ({ page }) => {
  const before = await laneNotes(page, 'h').count();
  await laneNotes(page, 'h').nth(0).click();
  await page.keyboard.press('Delete');
  await page.keyboard.press('Control+z');
  await expect(laneNotes(page, 'h')).toHaveCount(before);

  // A new deletion supersedes the redo; the earlier redo must not resurrect.
  await laneNotes(page, 'h').nth(0).click();
  await page.keyboard.press('Delete');
  await page.keyboard.press('Control+Shift+z'); // no-op: redo stack was cleared
  await expect(laneNotes(page, 'h')).toHaveCount(before - 1);
});
