import { expect, test, type Locator, type Page } from '@playwright/test';
import { laneNotes, loadRockJot, noteGeom, selectedNotes } from './editing_e2e_utils';

/**
 * Black-box coverage of when the single-note inline label (the popover portaled
 * to `document.body` with `data-popover="note-label"`) is shown. It's a
 * single-note affordance: it appears for the sole selected note (or a hovered
 * note), but NOT for any member of a multi/marquee selection, and it stays
 * hidden while a note is being dragged.
 */

const label = (page: Page): Locator => page.locator('[data-popover="note-label"]');

/** Park the pointer off any note so a stale hover doesn't keep the label up;
 *  the label under test is then purely selection-driven. */
async function parkPointer(page: Page): Promise<void> {
  await page.mouse.move(2, 2);
}

test.beforeEach(async ({ page }) => {
  await loadRockJot(page);
});

test('a sole selected note shows its label', async ({ page }) => {
  await laneNotes(page, 'h').nth(1).click();
  await parkPointer(page);
  await expect(selectedNotes(page)).toHaveCount(1);
  await expect(label(page)).toHaveCount(1);
});

test('a multi-selection suppresses the label', async ({ page }) => {
  await laneNotes(page, 'h').nth(1).click();
  await laneNotes(page, 'h').nth(3).click({ modifiers: ['Control'] });
  await parkPointer(page);
  await expect(selectedNotes(page)).toHaveCount(2);
  await expect(label(page)).toHaveCount(0);
});

test('the label is hidden while dragging a note', async ({ page }) => {
  const n = await noteGeom(page, 'h', 1);
  await laneNotes(page, 'h').nth(1).click();
  await parkPointer(page);
  // Re-hover then confirm the sole-selection label is up before the drag.
  await page.mouse.move(n.x, n.y);
  await expect(label(page)).toHaveCount(1);

  await page.mouse.down();
  await page.mouse.move(n.x + 25, n.y, { steps: 5 }); // past the drag threshold
  await expect(label(page)).toHaveCount(0);
  await page.mouse.up();
});
