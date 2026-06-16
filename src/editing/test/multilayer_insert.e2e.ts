import { expect, test, type Page } from '@playwright/test';
import { setMode } from './editing_e2e_utils';

/**
 * With the same lane in two layers (a snare in layer 1 AND layer 2), the score
 * shows two independent rows. Inserting on a specific layer's row must land in
 * THAT layer, not whichever layer happens to be first, the placeholder carries
 * the clicked row's layerId.
 */

const META = '{{ time: "4/4", instrumentMapping: { s:{name:"Snare"} } }}';

async function load(page: Page, dsl: string): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(
    () => typeof (window as unknown as { drumjot?: { loadDsl?: unknown } }).drumjot?.loadDsl === 'function'
  );
  await page.evaluate(
    (s) => (window as unknown as { drumjot: { loadDsl(x: string): void } }).drumjot.loadDsl(s),
    dsl
  );
  await page.waitForSelector('[data-note-id]');
}

function snareNotes(page: Page, layerId: string) {
  return page.locator(
    `[data-testid="instrument-track-s"][data-layer-id="${layerId}"] [data-note-id]`
  );
}

test("inserting on a non-first layer's row lands in that layer", async ({ page }) => {
  await load(page, `${META} | s . . . | || | s . . . |`);
  await expect(snareNotes(page, 'v0')).toHaveCount(1);
  await expect(snareNotes(page, 'v1')).toHaveCount(1);

  await setMode(page, 'insert');
  // Click an empty beat on layer 2's snare row.
  const row = page.locator('[data-testid="instrument-track-s"][data-layer-id="v1"] [data-bars-row]');
  const box = (await row.boundingBox())!;
  const x = box.x + box.width * 0.55;
  const y = box.y + box.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.click(x, y);

  // The new note landed in layer 2, not layer 1.
  await expect(snareNotes(page, 'v1')).toHaveCount(2);
  await expect(snareNotes(page, 'v0')).toHaveCount(1);
});
