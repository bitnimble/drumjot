import { expect, test } from '@playwright/test';
import {
  disableSnapping,
  dragPreviews,
  laneNotes,
  loadRockJot,
  noteGeom,
  selectedNotes,
} from './editing_e2e_utils';

/**
 * Black-box coverage of cut / copy / paste. Copy/cut/paste ride the DOM
 * clipboard events (Ctrl+C/X/V on this headless Linux Chromium). Paste does NOT
 * write immediately: it loads the copied notes as a preview cluster that
 * follows the cursor, then commits on click (Esc cancels). Snapping is disabled
 * so the placement geometry is predictable.
 */

test.beforeEach(async ({ page }) => {
  await loadRockJot(page);
  await disableSnapping(page);
});

test('copy then paste shows a following preview that commits on click', async ({ page }) => {
  const before = await laneNotes(page, 'h').count();
  const n = await noteGeom(page, 'h', 1);
  await laneNotes(page, 'h').nth(1).click();
  await expect(selectedNotes(page)).toHaveCount(1);

  await page.keyboard.press('Control+c');
  await page.keyboard.press('Control+v');

  // Nothing is written yet; the cluster appears under the cursor once it moves.
  await expect(laneNotes(page, 'h')).toHaveCount(before);
  const targetX = n.x + 90;
  await page.mouse.move(targetX, n.y);
  await expect(dragPreviews(page, 'h')).toHaveCount(1);

  // Click commits one new note and ends the placement; the paste is selected.
  await page.mouse.click(targetX, n.y);
  await expect(laneNotes(page, 'h')).toHaveCount(before + 1);
  await expect(dragPreviews(page, 'h')).toHaveCount(0);
  await expect(selectedNotes(page)).toHaveCount(1);
});

test('Escape cancels a paste placement without writing', async ({ page }) => {
  const before = await laneNotes(page, 'h').count();
  const n = await noteGeom(page, 'h', 1);
  await laneNotes(page, 'h').nth(1).click();

  await page.keyboard.press('Control+c');
  await page.keyboard.press('Control+v');
  await page.mouse.move(n.x + 90, n.y);
  await expect(dragPreviews(page, 'h')).toHaveCount(1);

  await page.keyboard.press('Escape');
  await expect(dragPreviews(page, 'h')).toHaveCount(0);
  await expect(laneNotes(page, 'h')).toHaveCount(before);
});

test('cut removes the selection; paste places it back', async ({ page }) => {
  const before = await laneNotes(page, 'h').count();
  const n = await noteGeom(page, 'h', 1);
  await laneNotes(page, 'h').nth(1).click();

  await page.keyboard.press('Control+x');
  await expect(laneNotes(page, 'h')).toHaveCount(before - 1);

  await page.keyboard.press('Control+v');
  await page.mouse.move(n.x, n.y);
  await expect(dragPreviews(page, 'h')).toHaveCount(1);
  await page.mouse.click(n.x, n.y);
  await expect(laneNotes(page, 'h')).toHaveCount(before);
});

test('pasting a cluster spanning lanes preserves each note lane', async ({ page }) => {
  const beforeH = await laneNotes(page, 'h').count();
  const beforeK = await laneNotes(page, 'k').count();
  const h = await noteGeom(page, 'h', 1);

  await laneNotes(page, 'h').nth(1).click();
  await laneNotes(page, 'k').nth(0).click({ modifiers: ['Control'] });
  await expect(selectedNotes(page)).toHaveCount(2);

  await page.keyboard.press('Control+c');
  await page.keyboard.press('Control+v');

  // Cursor on the hi-hat row, but the cluster spans h+k, so previews land on
  // BOTH lanes (lanes preserved, the same span rule as a multi-lane drag).
  const targetX = h.x + 100;
  await page.mouse.move(targetX, h.y);
  await expect(dragPreviews(page, 'h')).toHaveCount(1);
  await expect(dragPreviews(page, 'k')).toHaveCount(1);

  await page.mouse.click(targetX, h.y);
  await expect(laneNotes(page, 'h')).toHaveCount(beforeH + 1);
  await expect(laneNotes(page, 'k')).toHaveCount(beforeK + 1);
});

test('loading a new song cancels an in-flight paste placement', async ({ page }) => {
  // Begin a paste, then load a different song before committing. The placement
  // captured the OLD song's bar layout, so it must be cancelled on the swap
  // rather than surviving into the new song (where a click would commit
  // misplaced notes or the editor would stay stuck in paste mode).
  const n = await noteGeom(page, 'h', 1);
  await laneNotes(page, 'h').nth(1).click();
  await page.keyboard.press('Control+c');
  await page.keyboard.press('Control+v');
  await page.mouse.move(n.x + 90, n.y);
  await expect(page.locator('[data-testid="drag-preview-note"]')).toHaveCount(1);

  // Swap the song wholesale.
  await page.evaluate(
    (s) => (window as unknown as { drumjot: { loadDsl(x: string): void } }).drumjot.loadDsl(s),
    '{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"}, s:{name:"Snare"}, k:{name:"Kick"} } }} | k h s h |'
  );
  await page.waitForSelector('[data-note-id]');

  // No preview survives, and a click on the new score commits nothing (paste is
  // inactive, the new bar keeps exactly its two hi-hats).
  await expect(page.locator('[data-testid="drag-preview-note"]')).toHaveCount(0);
  const after = await laneNotes(page, 'h').count();
  const t = await noteGeom(page, 'h', 0);
  await page.mouse.click(t.x + 50, t.y);
  await expect(laneNotes(page, 'h')).toHaveCount(after);
});

test('a committed paste is a single undo step', async ({ page }) => {
  const before = await laneNotes(page, 'h').count();
  const n = await noteGeom(page, 'h', 1);
  await laneNotes(page, 'h').nth(1).click();

  await page.keyboard.press('Control+c');
  await page.keyboard.press('Control+v');
  await page.mouse.move(n.x + 90, n.y);
  await page.mouse.click(n.x + 90, n.y);
  await expect(laneNotes(page, 'h')).toHaveCount(before + 1);

  await page.keyboard.press('Control+z');
  await expect(laneNotes(page, 'h')).toHaveCount(before);
});

test('copy writes the custom MIME only, never text/plain', async ({ page }) => {
  await laneNotes(page, 'h').nth(1).click();
  await expect(selectedNotes(page)).toHaveCount(1);

  // Observe the clipboard event AFTER the app handler has populated it (the
  // app's document listener is registered at mount, so it runs first). The
  // custom format must be present and text/plain must be empty (no clobbering
  // the user's text clipboard).
  const seen = page.evaluate(
    () =>
      new Promise<{ custom: string; text: string }>((resolve) => {
        const onCopy = (e: ClipboardEvent) => {
          document.removeEventListener('copy', onCopy);
          resolve({
            custom: e.clipboardData?.getData('application/x-drumjot-notes+json') ?? '',
            text: e.clipboardData?.getData('text/plain') ?? '',
          });
        };
        document.addEventListener('copy', onCopy);
      })
  );
  await page.keyboard.press('Control+c');
  const { custom, text } = await seen;
  expect(custom.length).toBeGreaterThan(0);
  expect(text).toBe('');
});
