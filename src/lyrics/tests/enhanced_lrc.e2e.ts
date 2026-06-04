import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';

/**
 * E2E round-trip for word-aligned lyrics through an enhanced-LRC file.
 *
 * Drives the real UI both ways: imports a word-tagged `.lrc` (our
 * `<start>text<end>` duration extension + an `[offset:]` header) via
 * File → Lyrics → "Load from file…", then exports it back through the
 * lyrics row's ⋯ menu and captures the download. Because our format is
 * canonical (serialize ∘ parse is identity on a serialized document),
 * the exported bytes must equal the imported bytes, proving parse-on-
 * import and serialize-on-export preserve per-word start/end durations,
 * the silent gap between words, the offset nudge, and plain lines.
 */

const ENHANCED_LRC =
  '[offset:500]\n' +
  '[00:01.000]<00:01.000>Hello<00:01.400> <00:01.500>world<00:01.900>\n' +
  '[00:05.000]A plain line\n';

/** Open File → Lyrics so the lyrics loaders are reachable. */
async function openLyricsMenu(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'File', exact: true }).click();
  await page.getByRole('button', { name: 'Lyrics', exact: true }).click();
}

test('enhanced-LRC import → export round-trips word durations + offset', async ({
  page,
}) => {
  await page.goto('/');
  // Need a score on screen for the mixer (and the lyrics row) to exist.
  await page.getByRole('button', { name: 'Simple rock loop' }).click();
  await page.waitForSelector('[data-testid^="instrument-row-"]');

  // Import the word-aligned file through the existing file loader.
  await openLyricsMenu(page);
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByTestId('lyrics-menu-load-file').click(),
  ]);
  await chooser.setFiles({
    name: 'roundtrip.lrc',
    mimeType: 'text/plain',
    buffer: Buffer.from(ENHANCED_LRC, 'utf-8'),
  });

  const row = page.getByTestId('lyrics-row');
  await expect(row).toBeVisible();
  await expect(row.getByText(/File · roundtrip\.lrc/)).toBeVisible();

  // Open the per-row overflow menu (portaled panel) and export.
  await row
    .locator('button[title="More actions for this lyrics track"]')
    .click();
  const exportButton = page.locator('[data-testid^="lyrics-export-"]');
  await expect(exportButton).toBeVisible();

  const [download] = await Promise.all([
    page.waitForEvent('download'),
    exportButton.click(),
  ]);

  expect(download.suggestedFilename()).toBe('roundtrip.lrc');
  const path = await download.path();
  const text = await readFile(path, 'utf-8');
  expect(text).toBe(ENHANCED_LRC);
});
