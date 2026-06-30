import { expect, test, type Page } from '@playwright/test';

/**
 * Dragging a trackhead's ≡ handle in the score gutter reorders rows directly,
 * writing through `LayersPresenter` (the same `ordering` the Layers panel uses).
 * Regression guard: this wiring was briefly a no-op after the first-class-layers
 * refactor.
 */

const META =
  '{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"}, s:{name:"Snare"}, k:{name:"Kick"} } }}';

async function load(page: Page, dsl: string): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(
    () => typeof (window as unknown as { drumjot?: { loadDsl?: unknown } }).drumjot?.loadDsl === 'function'
  );
  await page.evaluate(
    (s) => (window as unknown as { drumjot: { loadDsl(x: string): void } }).drumjot.loadDsl(s),
    dsl
  );
  await page.waitForSelector('[data-testid="instrument-track-k"]');
}

/** The lane order of the instrument rows in the score, top to bottom. */
function laneOrder(page: Page): Promise<string[]> {
  return page
    .locator('[data-testid^="instrument-track-"]')
    .evaluateAll((els) =>
      els.map((e) => e.getAttribute('data-testid')?.replace('instrument-track-', '') ?? '')
    );
}

test('dragging a trackhead handle reorders the row', async ({ page }) => {
  await load(page, `${META} | h s k |`);
  // Canonical lane order for h/s/k.
  expect(await laneOrder(page)).toEqual(['h', 's', 'k']);

  // Drag the Kick handle up onto the HiHat row's top edge -> Kick goes first.
  const kickHandle = page
    .locator('[data-testid="instrument-track-k"]')
    .getByRole('button', { name: /Reorder/ });
  await kickHandle.dragTo(page.locator('[data-testid="instrument-track-h"]'), {
    targetPosition: { x: 40, y: 2 },
  });

  await expect.poll(() => laneOrder(page)).toEqual(['k', 'h', 's']);
});

test('gutter reorder and the Layers panel stay in sync', async ({ page }) => {
  await load(page, `${META} | h s k |`);
  // Move Kick to the top via the gutter handle.
  await page
    .locator('[data-testid="instrument-track-k"]')
    .getByRole('button', { name: /Reorder/ })
    .dragTo(page.locator('[data-testid="instrument-track-h"]'), { targetPosition: { x: 40, y: 2 } });
  await expect.poll(() => laneOrder(page)).toEqual(['k', 'h', 's']);

  // The Layers panel (reads the same `ordering`) shows the same order.
  await page.getByTestId('sidebar-item-layers').click();
  await expect(
    page.locator('[data-testid="layers-tree"] [data-testid="layers-track"]')
  ).toHaveText([/Kick/, /HiHat/, /Snare/]);
});
