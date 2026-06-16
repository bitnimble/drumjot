import { expect, type Locator, type Page } from '@playwright/test';

/**
 * Shared black-box helpers for the editing e2e specs. Everything here drives
 * the app the way a user would (clicks, drags, keys, real DOM) and reads back
 * from the DOM, no reaching into `window.drumjot` for assertions.
 */

/** Load the built-in "Simple rock loop" example from the empty-state picker
 *  (h/s/k lanes, two 4/4 bars) and wait for the score to render. */
export async function loadRockJot(page: Page): Promise<void> {
  await page.goto('/');
  await page.getByRole('button', { name: 'Simple rock loop' }).click();
  await expect(page.locator('h2')).toContainText('Simple rock loop');
  await page.waitForSelector('[data-note-id]');
}

/** All note glyphs in one instrument lane row, in document order. */
export function laneNotes(page: Page, lane: string): Locator {
  return page.locator(`[data-testid="instrument-track-${lane}"] [data-note-id]`);
}

/** Every currently-selected note glyph (across all lanes). */
export function selectedNotes(page: Page): Locator {
  return page.locator('[data-note-id][data-selected]');
}

export function selectionFrame(page: Page): Locator {
  return page.getByTestId('selection-frame');
}

/** Switch the floating editing toolbar between select / insert mode. */
export async function setMode(page: Page, mode: 'select' | 'insert'): Promise<void> {
  const btn = page.getByTestId(`mode-${mode}`);
  await btn.click();
  await expect(btn).toHaveAttribute('aria-pressed', 'true');
}

type NoteGeom = { x: number; y: number; left: number; right: number; id: string };

/** Viewport geometry + id of the nth note glyph in a lane. */
export async function noteGeom(page: Page, lane: string, nth = 0): Promise<NoteGeom> {
  return page.evaluate(
    ({ lane, nth }) => {
      const row = document.querySelector(`[data-testid="instrument-track-${lane}"]`)!;
      const note = row.querySelectorAll('[data-note-id]')[nth] as HTMLElement;
      const r = note.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2, left: r.left, right: r.right, id: note.dataset.noteId! };
    },
    { lane, nth }
  );
}

/** Current viewport `left` of a specific note id (or null if it's gone). */
export async function noteLeft(page: Page, id: string): Promise<number | null> {
  return page.evaluate((id) => {
    const el = document.querySelector(`[data-note-id="${id}"]`) as HTMLElement | null;
    return el ? el.getBoundingClientRect().left : null;
  }, id);
}

/** Vertical centre of a lane row, for dropping a dragged note onto it. */
export async function laneCentreY(page: Page, lane: string): Promise<number> {
  return page.evaluate((lane) => {
    const r = document.querySelector(`[data-testid="instrument-track-${lane}"]`)!.getBoundingClientRect();
    return r.top + r.height / 2;
  }, lane);
}

/** Turn on grid snapping via the Edit toolbar menu (leaves the menu state as
 *  the user would: opens, toggles on, asserts the checkmark). */
export async function enableSnapping(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Edit' }).click();
  const item = page.getByTestId('edit-menu-snapping');
  await item.click();
  await expect(item).toHaveAttribute('aria-checked', 'true');
  // Dismiss the menu so it doesn't overlay the score for subsequent clicks.
  await page.keyboard.press('Escape');
}

/** Ids of every note glyph currently in a lane, in DOM order. */
export async function laneNoteIds(page: Page, lane: string): Promise<string[]> {
  return page.evaluate((lane) => {
    const row = document.querySelector(`[data-testid="instrument-track-${lane}"]`)!;
    return [...row.querySelectorAll('[data-note-id]')].map((el) => (el as HTMLElement).dataset.noteId!);
  }, lane);
}

/** Press-drag-release with intermediate steps so a real pointer drag (past the
 *  threshold) is emulated rather than a teleport. */
export async function dragMouse(
  page: Page,
  from: { x: number; y: number },
  to: { x: number; y: number }
): Promise<void> {
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  const midX = (from.x + to.x) / 2;
  const midY = (from.y + to.y) / 2;
  await page.mouse.move(midX, midY, { steps: 6 });
  await page.mouse.move(to.x, to.y, { steps: 6 });
  await page.mouse.up();
}
