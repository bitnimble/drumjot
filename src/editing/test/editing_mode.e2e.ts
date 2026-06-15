import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';

// The smoke fixture's Crash lane (`c`) carries only two notes (both on bar
// downbeats), so a mid-row click lands on empty space, a deterministic spot
// to drop a new note without hitting an existing glyph.
const SONG_FIXTURE = fileURLToPath(
  new URL('../../../tests/fixtures/song.jot', import.meta.url),
);

const elementCount = (page: import('@playwright/test').Page) =>
  page.evaluate(() => (window as any).drumjot.jotEditorStore.jot.elements.size as number);

test('insert mode: hover previews a placeholder, click commits a note', async ({ page }) => {
  await page.goto('/');
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByRole('button', { name: 'Open .jot file' }).click(),
  ]);
  await chooser.setFiles(SONG_FIXTURE);
  await expect(page.locator('h2')).toContainText('Smoke Test Song');

  const before = await elementCount(page);

  // Enter insert mode via the floating toolbar.
  await page.getByTestId('mode-insert').click();
  await expect(page.getByTestId('mode-insert')).toHaveAttribute('aria-pressed', 'true');

  // Hovering the sparse Crash lane's bars row shows the placeholder preview.
  const crashBars = page.locator('[data-testid="instrument-track-c"] [data-bars-row]');
  await crashBars.hover({ position: { x: 250, y: 15 } });
  await expect(page.getByTestId('placeholder-note')).toBeVisible();

  // Clicking there commits exactly one new note into the document.
  await crashBars.click({ position: { x: 250, y: 15 } });
  await expect.poll(() => elementCount(page)).toBe(before + 1);

  // Back in select mode the placeholder clears.
  await page.getByTestId('mode-select').click();
  await expect(page.getByTestId('mode-select')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('placeholder-note')).toHaveCount(0);
});
