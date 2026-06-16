import { expect, test } from '@playwright/test';
import {
  laneNotes,
  loadRockJot,
  noteGeom,
  selectedNotes,
  selectionFrame,
  dragMouse,
} from './editing_e2e_utils';

/**
 * Black-box coverage of note selection: plain click, ctrl-click (toggle),
 * shift-click (range), and marquee. Selection is observed via the
 * `data-selected` attribute the score paints onto each selected glyph, and the
 * multi-note "selection frame" overlay, never by reading store state.
 */

test.beforeEach(async ({ page }) => {
  await loadRockJot(page);
});

test('plain click selects exactly one note', async ({ page }) => {
  await laneNotes(page, 'h').nth(0).click();
  await expect(selectedNotes(page)).toHaveCount(1);
  await expect(laneNotes(page, 'h').nth(0)).toHaveAttribute('data-selected', 'true');
  // No frame for a single-note selection.
  await expect(selectionFrame(page)).toHaveCount(0);
});

test('clicking another note moves the selection', async ({ page }) => {
  await laneNotes(page, 'h').nth(0).click();
  await laneNotes(page, 'h').nth(3).click();
  await expect(selectedNotes(page)).toHaveCount(1);
  await expect(laneNotes(page, 'h').nth(3)).toHaveAttribute('data-selected', 'true');
  await expect(laneNotes(page, 'h').nth(0)).not.toHaveAttribute('data-selected', 'true');
});

test('ctrl-click adds and removes individual notes', async ({ page }) => {
  await laneNotes(page, 'h').nth(0).click();
  await laneNotes(page, 'h').nth(3).click({ modifiers: ['Control'] });
  await expect(selectedNotes(page)).toHaveCount(2);
  // The frame appears once two notes are selected.
  await expect(selectionFrame(page)).toBeVisible();

  // Ctrl-clicking the second note again removes just it.
  await laneNotes(page, 'h').nth(3).click({ modifiers: ['Control'] });
  await expect(selectedNotes(page)).toHaveCount(1);
  await expect(laneNotes(page, 'h').nth(0)).toHaveAttribute('data-selected', 'true');
  await expect(selectionFrame(page)).toHaveCount(0);
});

test('shift-click selects a contiguous range from the anchor', async ({ page }) => {
  await laneNotes(page, 'h').nth(0).click();
  await laneNotes(page, 'h').nth(3).click({ modifiers: ['Shift'] });
  // A range (anchor..target inclusive, in document order), more than the two
  // clicked endpoints, both of which are selected.
  const count = await selectedNotes(page).count();
  expect(count).toBeGreaterThanOrEqual(3);
  await expect(laneNotes(page, 'h').nth(0)).toHaveAttribute('data-selected', 'true');
  await expect(laneNotes(page, 'h').nth(3)).toHaveAttribute('data-selected', 'true');
  await expect(selectionFrame(page)).toBeVisible();
});

test('re-shift-click recomputes the range from the same anchor', async ({ page }) => {
  await laneNotes(page, 'h').nth(0).click();
  await laneNotes(page, 'h').nth(5).click({ modifiers: ['Shift'] });
  const wide = await selectedNotes(page).count();
  await laneNotes(page, 'h').nth(2).click({ modifiers: ['Shift'] });
  const narrow = await selectedNotes(page).count();
  expect(narrow).toBeLessThan(wide);
});

test('marquee drag selects every enclosed note and shows the frame', async ({ page }) => {
  // A thin horizontal box across the start of the hi-hat row, beginning in the
  // empty space just left of the first note so the press starts a marquee
  // rather than selecting a note.
  const first = await noteGeom(page, 'h', 0);
  const third = await noteGeom(page, 'h', 2);
  await dragMouse(
    page,
    { x: first.left - 10, y: first.y - 8 },
    { x: third.right + 8, y: first.y + 8 }
  );
  await expect(selectedNotes(page)).toHaveCount(3);
  await expect(selectionFrame(page)).toBeVisible();
});

test('a marquee that encloses nothing clears the selection', async ({ page }) => {
  await laneNotes(page, 'h').nth(0).click();
  await expect(selectedNotes(page)).toHaveCount(1);
  // Drag a small box in empty space well right of the last bar's content.
  const first = await noteGeom(page, 'h', 0);
  await dragMouse(
    page,
    { x: first.x, y: first.y - 40 },
    { x: first.x + 30, y: first.y - 20 }
  );
  await expect(selectedNotes(page)).toHaveCount(0);
});
