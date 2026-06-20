import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';

// A small but non-trivial multi-bar chart (crash, hi-hat with accents +
// open hits, backbeat snare, kick, a tom fill), enough to exercise the
// real render path (multiple instrument rows, patterns, bars, modifiers)
// rather than a one-bar toy. See tests/fixtures/song.jot.
const SONG_FIXTURE = fileURLToPath(
  new URL('../../../tests/fixtures/song.jot', import.meta.url),
);

test('boots to the empty state with the probe surface wired', async ({ page }) => {
  await page.goto('/');
  // Boot no longer auto-loads an example; it lands on the empty-state
  // welcome screen until the user picks something (src/index.tsx).
  await expect(page.locator('h2')).toContainText('Start a new jot or open a file');
  const wired = await page.evaluate(
    () => typeof (window as any).jotPlayer === 'object' && !!(window as any).drumjot,
  );
  expect(wired).toBe(true);
});

test('loads a .jot song from disk and renders it', async ({ page }) => {
  await page.goto('/');
  // Real load path from the empty state: the "Open .jot file" CTA drives a
  // hidden file input.
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByRole('button', { name: 'Open .jot file' }).click(),
  ]);
  await chooser.setFiles(SONG_FIXTURE);

  await expect(page.locator('h2')).toContainText('Smoke Test Song');
  // The chart uses six kit pieces; each becomes its own instrument row.
  const rows = page.locator('[data-testid^="instrument-track-"]');
  await expect.poll(() => rows.count()).toBeGreaterThanOrEqual(4);
  // And it actually rendered notes (not just empty lanes).
  await expect.poll(() => page.locator('[data-noseek="true"]').count()).toBeGreaterThan(8);
});

test('loads a built-in example from the empty-state picker', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Simple rock loop' }).click();
  await expect(page.locator('h2')).toContainText('Simple rock loop');
  await page.waitForSelector('[data-testid^="instrument-track-"]');
});
