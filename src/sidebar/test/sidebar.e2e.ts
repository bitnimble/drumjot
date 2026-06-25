import { expect, test, type Page } from '@playwright/test';

/**
 * Black-box coverage of the collapsible right sidebar: the rail is always
 * visible and full-height; clicking a rail item opens its panel **floating**
 * over the score (the score keeps its full width); the rail's topmost button
 * **pins** the panel, docking it so the score narrows and its scroll
 * virtualization stops rendering the bars under it; and a floating panel is
 * dismissed by an outside click / Escape but survives the score interactions
 * that feed it (note clicks, marquee drags) and its own popups.
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

test('the Layers panel lists the song\'s tracks grouped by layer', async ({ page }) => {
  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('layers-tree')).toBeVisible();
  // This single-||-layer song shows one layer with one instrument track,
  // labelled from the instrument mapping (HiHat).
  await expect(page.getByTestId('layers-layer')).toHaveCount(1);
  const track = page.getByTestId('layers-track');
  await expect(track).toHaveCount(1);
  await expect(track).toHaveAttribute('data-track-kind', 'instrument');
  await expect(track).toContainText('HiHat');
});

const TWO_LAYER =
  '{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"}, k:{name:"Kick"} } }} | h h h h | || | k . . . |';

async function loadDsl(page: Page, dsl: string): Promise<void> {
  await page.evaluate(
    (s) => (window as unknown as { drumjot: { loadDsl(x: string): void } }).drumjot.loadDsl(s),
    dsl
  );
  // The kick lane only exists in TWO_LAYER (not the beforeEach LONG_SONG), so
  // waiting for it ensures the new jot has rendered before we assert (the
  // panel/score read it reactively; without this we race the stale song).
  await page.waitForSelector('[data-testid="instrument-track-k"]');
}

test('a two-||-layer song shows two layer bands in the panel', async ({ page }) => {
  await loadDsl(page, TWO_LAYER);
  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('layers-layer')).toHaveCount(2);
});

test('renaming a layer via its ⋯ menu updates the header', async ({ page }) => {
  await loadDsl(page, TWO_LAYER);
  await page.getByTestId('sidebar-item-layers').click();
  const firstLayer = page.getByTestId('layers-layer').first();
  await page.getByTitle('Options for Layer 1').click();
  const input = page.getByTestId('layer-rename-input');
  await input.fill('Hands');
  await input.press('Enter');
  await expect(firstLayer).toContainText('Hands');
});

test('a panel opens floating by default', async ({ page }) => {
  await page.getByTestId('sidebar-item-layers').click();
  const panel = page.getByTestId('sidebar-panel');
  await expect(panel).toBeVisible();
  await expect(panel).toHaveAttribute('data-sidebar-mode', 'floating');
});

test('the topmost button toggles pin/float and never closes the panel', async ({ page }) => {
  await page.getByTestId('sidebar-item-layers').click();
  const panel = page.getByTestId('sidebar-panel');
  await expect(panel).toHaveAttribute('data-sidebar-mode', 'floating');

  await page.getByTestId('sidebar-pin-toggle').click();
  await expect(panel).toHaveAttribute('data-sidebar-mode', 'pinned');
  await page.getByTestId('sidebar-pin-toggle').click();
  await expect(panel).toHaveAttribute('data-sidebar-mode', 'floating');

  // The panel's own rail item is what closes it (VS Code-style toggle).
  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('sidebar-panel')).toHaveCount(0);
});

test('the topmost button opens the panel pinned when the sidebar is closed', async ({ page }) => {
  await expect(page.getByTestId('sidebar-panel')).toHaveCount(0);
  await page.getByTestId('sidebar-pin-toggle').click();
  await expect(page.getByTestId('sidebar-panel')).toHaveAttribute('data-sidebar-mode', 'pinned');
});

test('floating leaves the score full-width; pinning narrows it and the virtualization follows', async ({
  page,
}) => {
  const widthCollapsed = await scoreWidth(page);
  const notesCollapsed = await page.locator('[data-note-id]').count();

  // Floating overlays the score, so its measured width is unchanged.
  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('sidebar-panel')).toHaveAttribute('data-sidebar-mode', 'floating');
  expect(Math.abs((await scoreWidth(page)) - widthCollapsed)).toBeLessThan(2);

  // Pinning docks it: the score loses ~the panel's width and windows fewer bars.
  await page.getByTestId('sidebar-pin-toggle').click();
  await expect(page.getByTestId('sidebar-panel')).toHaveAttribute('data-sidebar-mode', 'pinned');
  expect(await scoreWidth(page)).toBeLessThan(widthCollapsed - 150);
  await expect
    .poll(() => page.locator('[data-note-id]').count())
    .toBeLessThan(notesCollapsed);

  // Unpinning floats it again and restores the score width.
  await page.getByTestId('sidebar-pin-toggle').click();
  await expect(page.getByTestId('sidebar-panel')).toHaveAttribute('data-sidebar-mode', 'floating');
  expect(Math.abs((await scoreWidth(page)) - widthCollapsed)).toBeLessThan(2);
});

test('clicking empty score dismisses a floating panel', async ({ page }) => {
  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('sidebar-panel')).toBeVisible();
  // Click the score away from any note and from the panel (which floats right).
  const box = (await page.locator('[data-jot-scroller]').boundingBox())!;
  await page.mouse.click(box.x + 16, box.y + 8);
  await expect(page.getByTestId('sidebar-panel')).toHaveCount(0);
});

test('Escape dismisses a floating panel', async ({ page }) => {
  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('sidebar-panel')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('sidebar-panel')).toHaveCount(0);
});

test('clicking a note keeps a floating panel open', async ({ page }) => {
  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('sidebar-panel')).toBeVisible();
  await page.locator('[data-note-id]').first().click();
  await expect(page.getByTestId('sidebar-panel')).toBeVisible();
});

test('a marquee drag on the score keeps a floating panel open', async ({ page }) => {
  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('sidebar-panel')).toBeVisible();
  const box = (await page.locator('[data-jot-scroller]').boundingBox())!;
  // A drag past the marquee threshold (vs a plain click) must not dismiss.
  await page.mouse.move(box.x + 16, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + 140, box.y + box.height / 2 + 40, { steps: 6 });
  await page.mouse.up();
  await expect(page.getByTestId('sidebar-panel')).toBeVisible();
});

test('interacting inside a floating panel does not dismiss it', async ({ page }) => {
  await page.getByTestId('sidebar-item-layers').click();
  const panel = page.getByTestId('sidebar-panel');
  await expect(panel).toBeVisible();
  await panel.click({ position: { x: 8, y: 8 } });
  await expect(panel).toBeVisible();
});

test('a pinned panel is not dismissed by clicking the score', async ({ page }) => {
  await page.getByTestId('sidebar-item-layers').click();
  await page.getByTestId('sidebar-pin-toggle').click();
  await expect(page.getByTestId('sidebar-panel')).toHaveAttribute('data-sidebar-mode', 'pinned');
  const box = (await page.locator('[data-jot-scroller]').boundingBox())!;
  await page.mouse.click(box.x + 16, box.y + 8);
  await expect(page.getByTestId('sidebar-panel')).toBeVisible();
});
