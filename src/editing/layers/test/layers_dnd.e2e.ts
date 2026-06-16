import { expect, test, type Page } from '@playwright/test';

/**
 * Drag-and-drop in the Layers panel: dragging a track onto a track in another
 * layer moves it there (joining before the drop target), and dragging a layer
 * header reorders whole layers. Writes go through LayersPresenter, the same
 * `ordering` the score reads.
 */

const META =
  '{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"}, s:{name:"Snare"}, k:{name:"Kick"} } }}';

async function openPanel(page: Page, dsl: string): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(
    () => typeof (window as unknown as { drumjot?: { loadDsl?: unknown } }).drumjot?.loadDsl === 'function'
  );
  await page.evaluate(
    (s) => (window as unknown as { drumjot: { loadDsl(x: string): void } }).drumjot.loadDsl(s),
    dsl
  );
  await page.waitForSelector('[data-testid="instrument-track-k"]');
  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('layers-tree')).toBeVisible();
}

/** Track rows (by visible label) inside a given layer band in the panel. */
function trackInLayer(page: Page, layerId: string, label: string) {
  return page
    .locator(`[data-testid="layers-layer"][data-layer-id="${layerId}"]`)
    .locator('[data-testid="layers-track"]', { hasText: label });
}

test('dragging a track onto another layer moves it there', async ({ page }) => {
  // v0 = {HiHat, Snare}; v1 = {Kick}.
  await openPanel(page, `${META} | h s | || | k |`);
  await expect(trackInLayer(page, 'v0', 'Snare')).toHaveCount(1);
  await expect(trackInLayer(page, 'v1', 'Snare')).toHaveCount(0);

  await trackInLayer(page, 'v0', 'Snare').dragTo(trackInLayer(page, 'v1', 'Kick'));

  // Snare moved out of v0 into v1, in the panel...
  await expect(trackInLayer(page, 'v0', 'Snare')).toHaveCount(0);
  await expect(trackInLayer(page, 'v1', 'Snare')).toHaveCount(1);
  await expect(trackInLayer(page, 'v0', 'HiHat')).toHaveCount(1);
  // ...and in the score (both read the same `ordering`), so the panel and
  // score never disagree.
  await expect(
    page.getByTestId('score-layer-v1').locator('[data-testid="instrument-track-s"]')
  ).toHaveCount(1);
  await expect(
    page.getByTestId('score-layer-v0').locator('[data-testid="instrument-track-s"]')
  ).toHaveCount(0);
});

test('create a group from a track, then ungroup it', async ({ page }) => {
  await openPanel(page, `${META} | h s | || | k |`);
  await expect(page.getByTestId('layers-group')).toHaveCount(0);

  // "New group" on the Snare row wraps it in a group.
  await trackInLayer(page, 'v0', 'Snare').getByTestId('layers-new-group').click();
  await expect(page.getByTestId('layers-group')).toHaveCount(1);
  await expect(page.getByTestId('layers-group')).toContainText('Group 1');
  // Snare is now inside the group.
  await expect(
    page.getByTestId('layers-group').locator('[data-testid="layers-track"]', { hasText: 'Snare' })
  ).toHaveCount(1);

  // Ungroup via the group ⋯ menu.
  await page.getByTitle('Options for Group 1').click();
  await page.getByTestId('group-ungroup').click();
  await expect(page.getByTestId('layers-group')).toHaveCount(0);
  await expect(trackInLayer(page, 'v0', 'Snare')).toHaveCount(1); // track survives
});

test('dragging a group header moves the whole group', async ({ page }) => {
  await openPanel(page, `${META} | h s k |`); // v0 = HiHat, Snare, Kick (loose)
  await trackInLayer(page, 'v0', 'Snare').getByTestId('layers-new-group').click();
  await expect(page.getByTestId('layers-group')).toHaveCount(1);
  // Loose h, k precede the new group, so the first row is HiHat.
  await expect(page.getByTestId('layers-track').first()).toContainText('HiHat');

  // Drag the group header onto the HiHat row -> the group moves before it.
  await page
    .locator('[data-testid="layers-group"] [class*="groupHeader"]')
    .dragTo(trackInLayer(page, 'v0', 'HiHat'));
  await expect(page.getByTestId('layers-track').first()).toContainText('Snare');
});

test('dragging a layer header reorders whole layers', async ({ page }) => {
  await openPanel(page, `${META} | h | || | k |`);
  const order = () =>
    page.locator('[data-testid="layers-layer"]').evaluateAll((els) =>
      els.map((e) => e.getAttribute('data-layer-id'))
    );
  expect(await order()).toEqual(['v0', 'v1']);
  // Drag layer 2's header up onto layer 1's header.
  await page
    .locator('[data-layer-id="v1"] [class*="layerHeader"]')
    .dragTo(page.locator('[data-layer-id="v0"] [class*="layerHeader"]'));
  expect(await order()).toEqual(['v1', 'v0']);
});
