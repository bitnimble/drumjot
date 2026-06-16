import { expect, test, type Page } from '@playwright/test';

/**
 * Black-box coverage of the collapsible right sidebar: the rail is always
 * visible and full-height, clicking the Layers item opens its (stub) panel, the
 * collapse/open button toggles it, and, crucially, opening the panel narrows
 * the score's measured width so its scroll virtualization stops rendering the
 * bars that would fall under the sidebar.
 */

/** A long single-lane song so the score is wide enough that virtualization
 *  actually windows (only the visible slice of bars is in the DOM). */
const LONG_SONG =
  '{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"} } }} | ' +
  Array(60).fill('h h h h').join(' | ') +
  ' |';

async function loadLongSong(page: Page): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(
    () => typeof (window as unknown as { drumjot?: { loadDsl?: unknown } }).drumjot?.loadDsl === 'function'
  );
  await page.evaluate((s) => (window as unknown as { drumjot: { loadDsl(x: string): void } }).drumjot.loadDsl(s), LONG_SONG);
  await page.waitForSelector('[data-note-id]');
}

const scoreWidth = async (page: Page) =>
  (await page.locator('[data-jot-scroller]').boundingBox())!.width;

test.beforeEach(async ({ page }) => {
  await loadLongSong(page);
});

test('the rail is visible and full-height, the panel starts collapsed', async ({ page }) => {
  const sidebar = page.getByTestId('sidebar');
  await expect(sidebar).toBeVisible();
  await expect(page.getByTestId('sidebar-panel')).toHaveCount(0);

  const box = (await sidebar.boundingBox())!;
  const viewportH = page.viewportSize()!.height;
  expect(box.height).toBeGreaterThan(viewportH - 2); // spans the full height
  expect(box.width).toBeLessThan(80); // just the rail when collapsed
});

test('clicking the Layers item opens the panel', async ({ page }) => {
  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('sidebar-panel')).toBeVisible();
  await expect(page.getByTestId('layers-panel')).toBeVisible();
});

test('the collapse/open button toggles the panel', async ({ page }) => {
  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('sidebar-panel')).toBeVisible();
  await page.getByTestId('sidebar-toggle').click();
  await expect(page.getByTestId('sidebar-panel')).toHaveCount(0);
  await page.getByTestId('sidebar-toggle').click();
  await expect(page.getByTestId('sidebar-panel')).toBeVisible();
});

test('opening the sidebar narrows the score and the virtualization follows', async ({ page }) => {
  const widthCollapsed = await scoreWidth(page);
  const notesCollapsed = await page.locator('[data-note-id]').count();

  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('sidebar-panel')).toBeVisible();

  const widthOpen = await scoreWidth(page);
  // The score lost roughly the panel's width to the sidebar.
  expect(widthOpen).toBeLessThan(widthCollapsed - 150);
  // With a narrower viewport the virtualization windows fewer bars into the DOM.
  await expect
    .poll(() => page.locator('[data-note-id]').count())
    .toBeLessThan(notesCollapsed);
});
