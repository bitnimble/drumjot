import { expect, test, type Page } from '@playwright/test';

/**
 * Black-box coverage of tuplet-bracket placement: a tuplet whose notes are all
 * in one lane draws above THAT lane's row (not always the topmost row). Loads a
 * jot via the console hook (setup only), a hi-hat row on top, a kick triplet
 * on its own layer below, and checks the bracket sits above the kick row.
 */

async function loadDsl(page: Page, src: string): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(
    () => typeof (window as unknown as { drumjot?: { loadDsl?: unknown } }).drumjot?.loadDsl === 'function'
  );
  await page.evaluate((s) => (window as unknown as { drumjot: { loadDsl(x: string): void } }).drumjot.loadDsl(s), src);
  await page.waitForSelector('[data-note-id]');
}

async function rowBox(page: Page, lane: string) {
  return (await page.locator(`[data-testid="instrument-track-${lane}"]`).boundingBox())!;
}

test('a single-lane tuplet bracket renders above its own lane, not the top row', async ({ page }) => {
  await loadDsl(
    page,
    '{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"}, k:{name:"Kick"} } }} ' +
      '| h h h h | || | (k k k) . . . |'
  );

  const bracket = (await page.getByTestId('tuplet-bracket').first().boundingBox())!;
  const hRow = await rowBox(page, 'h'); // top row
  const kRow = await rowBox(page, 'k'); // bottom row, owns the triplet

  expect(kRow.y).toBeGreaterThan(hRow.y); // sanity: kick row is below the hi-hat row
  // The bracket sits in the band just above the kick row, i.e. below the
  // hi-hat row's top (the OLD always-top-row behaviour would put it above
  // hRow.y) and not below the kick notes.
  expect(bracket.y).toBeGreaterThan(hRow.y + 5);
  expect(bracket.y).toBeLessThan(kRow.y);
});
