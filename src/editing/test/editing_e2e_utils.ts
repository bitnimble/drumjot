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

/** Drag-move preview glyphs (the placeholders shown while dragging) in a lane. */
export function dragPreviews(page: Page, lane: string): Locator {
  return page.locator(`[data-testid="instrument-track-${lane}"] [data-testid="drag-preview-note"]`);
}

/** Viewport geometry of the (first) drag-move preview glyph on a lane, or null
 *  if there's none there. Read mid-drag while the pointer is held. */
export async function previewGeom(
  page: Page,
  lane: string
): Promise<{ x: number; y: number; left: number } | null> {
  return page.evaluate((lane) => {
    const el = document.querySelector(
      `[data-testid="instrument-track-${lane}"] [data-testid="drag-preview-note"]`
    ) as HTMLElement | null;
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2, left: r.left };
  }, lane);
}

/** Viewport `top + height/2` of a lane's bars row, for asserting a preview is
 *  vertically centred on the row it lands on. */
export async function laneRowCentreY(page: Page, lane: string): Promise<number> {
  return page.evaluate((lane) => {
    const r = document
      .querySelector(`[data-testid="instrument-track-${lane}"] [data-bars-row]`)!
      .getBoundingClientRect();
    return r.top + r.height / 2;
  }, lane);
}

export function selectionFrame(page: Page): Locator {
  return page.getByTestId('selection-frame');
}

/** Every group-frame slice currently drawn (one per bar+row a group spans). */
export function groupFrames(page: Page): Locator {
  return page.getByTestId('group-frame');
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

/** Set grid snapping to `on` via the Edit toolbar menu, idempotently (snapping
 *  is on by default, so callers can't assume a fixed starting state). Opens the
 *  menu, toggles only if needed, asserts the resulting checkmark, then dismisses
 *  the menu so it doesn't overlay the score for subsequent clicks. */
async function setSnapping(page: Page, on: boolean): Promise<void> {
  await page.getByRole('button', { name: 'Edit' }).click();
  const item = page.getByTestId('edit-menu-snapping');
  const checked = (await item.getAttribute('aria-checked')) === 'true';
  if (checked !== on) await item.click();
  await expect(item).toHaveAttribute('aria-checked', String(on));
  await page.keyboard.press('Escape');
}

/** Turn grid snapping on (idempotent). */
export const enableSnapping = (page: Page): Promise<void> => setSnapping(page, true);

/** Turn grid snapping off (idempotent); snapping is on by default. */
export const disableSnapping = (page: Page): Promise<void> => setSnapping(page, false);

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
