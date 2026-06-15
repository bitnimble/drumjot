import { expect, test, type Page } from '@playwright/test';

/**
 * End-to-end coverage for the editing interactions: multi-select (ctrl /
 * marquee), the selection frame, delete, grid-snapping toggle, and drag-move
 * (within a lane and across lanes). Drives the app through `window.drumjot`
 * (stores/presenters exposed for e2e) and the DOM (`[data-note-id]`,
 * `[data-testid="instrument-track-<lane>"]`).
 */

type Win = {
  drumjot: {
    loadTestJot(): void;
    selection: { effectiveIds: ReadonlySet<string> };
    editingStore: { snappingEnabled: boolean };
  };
};

async function loadRockJot(page: Page): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(
    () => typeof (window as unknown as { drumjot?: { loadTestJot?: unknown } }).drumjot?.loadTestJot === 'function'
  );
  await page.evaluate(() => (window as unknown as Win).drumjot.loadTestJot());
  await expect(page.locator('h2')).toContainText('Simple rock loop');
  await page.waitForSelector('[data-note-id]');
}

const selectionSize = (page: Page) =>
  page.evaluate(() => (window as unknown as Win).drumjot.selection.effectiveIds.size);

/** Centre of the nth note glyph in a given instrument row. */
async function noteCentre(page: Page, lane: string, nth = 0) {
  return page.evaluate(
    ({ lane, nth }) => {
      const row = document.querySelector(`[data-testid="instrument-track-${lane}"]`)!;
      const note = row.querySelectorAll('[data-note-id]')[nth] as HTMLElement;
      const r = note.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2, id: note.dataset.noteId! };
    },
    { lane, nth }
  );
}

test('ctrl-click builds a multi-note selection and shows the selection frame', async ({ page }) => {
  await loadRockJot(page);
  const a = await noteCentre(page, 'h', 0);
  const b = await noteCentre(page, 'h', 1);

  await page.mouse.click(a.x, a.y);
  expect(await selectionSize(page)).toBe(1);
  await expect(page.locator('[data-testid="selection-frame"]')).toHaveCount(0);

  await page.keyboard.down('Control');
  await page.mouse.click(b.x, b.y);
  await page.keyboard.up('Control');

  expect(await selectionSize(page)).toBe(2);
  await expect(page.locator('[data-testid="selection-frame"]')).toBeVisible();

  // A plain click collapses back to one note and hides the frame.
  await page.mouse.click(a.x, a.y);
  expect(await selectionSize(page)).toBe(1);
  await expect(page.locator('[data-testid="selection-frame"]')).toHaveCount(0);
});

test('Delete removes the selected notes', async ({ page }) => {
  await loadRockJot(page);
  const before = await page.locator('[data-testid="instrument-track-h"] [data-note-id]').count();
  const a = await noteCentre(page, 'h', 0);

  await page.mouse.click(a.x, a.y);
  expect(await selectionSize(page)).toBe(1);
  await page.keyboard.press('Delete');

  expect(await selectionSize(page)).toBe(0);
  await expect(page.locator('[data-testid="instrument-track-h"] [data-note-id]')).toHaveCount(
    before - 1
  );
});

test('Edit menu toggles grid snapping', async ({ page }) => {
  await loadRockJot(page);
  expect(await page.evaluate(() => (window as unknown as Win).drumjot.editingStore.snappingEnabled)).toBe(
    false
  );
  await page.getByRole('button', { name: 'Edit' }).click();
  await page.getByTestId('edit-menu-snapping').click();
  expect(await page.evaluate(() => (window as unknown as Win).drumjot.editingStore.snappingEnabled)).toBe(
    true
  );
});

test('drag-move shifts a note horizontally within its lane', async ({ page }) => {
  await loadRockJot(page);
  const a = await noteCentre(page, 'h', 1);

  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  // Several steps past the drag threshold so it promotes to a drag.
  await page.mouse.move(a.x + 30, a.y, { steps: 6 });
  await page.mouse.move(a.x + 60, a.y, { steps: 6 });
  await page.mouse.up();

  // Same note (id stable) now renders at a different x.
  const after = await page.evaluate((id) => {
    const el = document.querySelector(`[data-note-id="${id}"]`) as HTMLElement | null;
    return el ? el.getBoundingClientRect().left : null;
  }, a.id);
  expect(after).not.toBeNull();
  expect(Math.abs((after as number) - (a.x - 0)) > 10).toBe(true);
});

test('drag-move across lanes re-homes the note to the target instrument row', async ({ page }) => {
  await loadRockJot(page);
  const src = await noteCentre(page, 'h', 1);
  const dstRow = await page.evaluate(() => {
    const r = document
      .querySelector('[data-testid="instrument-track-k"]')!
      .getBoundingClientRect();
    return { y: r.top + r.height / 2 };
  });

  await page.mouse.move(src.x, src.y);
  await page.mouse.down();
  await page.mouse.move(src.x + 5, dstRow.y, { steps: 8 });
  await page.mouse.move(src.x, dstRow.y, { steps: 8 });
  await page.mouse.up();

  // The same note id now lives under the kick row.
  await expect(
    page.locator(`[data-testid="instrument-track-k"] [data-note-id="${src.id}"]`)
  ).toHaveCount(1);
});
