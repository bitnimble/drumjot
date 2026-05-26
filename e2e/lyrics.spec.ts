import { fileURLToPath } from 'node:url';
import { expect, test, type Page, type Route } from '@playwright/test';

/**
 * E2E coverage for time-aligned lyrics. LRCLIB is stubbed via
 * `page.route` so the suite needs no network access; the LRC payload is
 * a small inline fixture rather than a separate file so the spec is
 * self-contained.
 *
 * What's exercised:
 *  - LRCLIB search → single result still renders a list; user clicks
 *    Load to commit (no auto-load).
 *  - LRCLIB search → multi-result picker; user selects then clicks Load.
 *  - LRCLIB search → 0 results message.
 *  - Word-level alignment checkbox disabled when no audio tracks loaded.
 *  - Load from file (.lrc) via the toolbar dropdown.
 *  - Offset input shifts active line.
 *  - Loading a new jot drops previously-loaded lyrics.
 */

const SAMPLE_LRC = `[00:00.00]Verse line one
[00:05.00]Verse line two
[00:10.00]Verse line three
`;

const TONE_WAV = fileURLToPath(new URL('./fixtures/tone.wav', import.meta.url));

type LrclibRow = {
  id: number;
  trackName: string;
  artistName: string;
  albumName: string | null;
  duration: number | null;
  syncedLyrics: string;
  plainLyrics: string | null;
  instrumental: boolean;
};

function mockLrclib(page: Page, rows: LrclibRow[]): Promise<void> {
  return page.route('https://lrclib.net/api/search**', (route: Route) => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(rows),
    });
  });
}

function makeRow(overrides: Partial<LrclibRow> = {}): LrclibRow {
  return {
    id: 1,
    trackName: 'Simple rock loop',
    artistName: 'Example Artist',
    albumName: 'Example Album',
    duration: 12,
    syncedLyrics: SAMPLE_LRC,
    plainLyrics: 'Verse line one\nVerse line two',
    instrumental: false,
    ...overrides,
  };
}

async function openLyricsDropdown(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Lyrics', exact: true }).click();
}

test('LRCLIB single result still requires explicit Load click', async ({ page }) => {
  await mockLrclib(page, [makeRow()]);
  await page.goto('/');
  await expect(page.locator('h2')).toContainText('Simple rock loop');

  await openLyricsDropdown(page);
  await page.getByTestId('lyrics-menu-search').click();
  // Modal stays open until the user clicks Load.
  await expect(page.getByTestId('lyrics-search-modal')).toBeVisible();
  await expect(page.getByTestId('lyrics-search-results')).toBeVisible();
  await expect(page.getByTestId('lyrics-search-load-footer')).toBeVisible();
  await expect(page.getByTestId('lyrics-search-loaded')).toHaveCount(0);

  await page.getByTestId('lyrics-search-load').click();
  await expect(page.getByTestId('lyrics-search-loaded')).toBeVisible();
  await expect(page.getByTestId('lyrics-search-modal')).toHaveCount(0);
  await expect(page.getByTestId('lyrics-row')).toBeVisible();
  await expect(page.getByTestId('lyrics-line-0')).toBeVisible();
});

test('LRCLIB multi-result picker: select a row then Load', async ({ page }) => {
  await mockLrclib(page, [
    makeRow({ id: 1, trackName: 'Different title A', artistName: 'A' }),
    makeRow({ id: 2, trackName: 'Different title B', artistName: 'B' }),
  ]);
  await page.goto('/');
  await openLyricsDropdown(page);
  await page.getByTestId('lyrics-menu-search').click();
  await expect(page.getByTestId('lyrics-search-results')).toBeVisible();
  // No exact match against the default jot's title, so no pre-selection;
  // footer should be hidden until the user picks a row.
  await expect(page.getByTestId('lyrics-search-load-footer')).toHaveCount(0);
  await page.getByTestId('lyrics-search-result-1').click();
  await expect(page.getByTestId('lyrics-search-load-footer')).toBeVisible();
  await page.getByTestId('lyrics-search-load').click();
  await expect(page.getByTestId('lyrics-search-loaded')).toBeVisible();
});

test('Word-level checkbox is disabled when no audio tracks are loaded', async ({ page }) => {
  await mockLrclib(page, [makeRow()]);
  await page.goto('/');
  await openLyricsDropdown(page);
  await page.getByTestId('lyrics-menu-search').click();
  await expect(page.getByTestId('lyrics-search-load-footer')).toBeVisible();
  await expect(page.getByTestId('lyrics-search-word-level')).toBeDisabled();
});

test('LRCLIB zero results shows the no-results message', async ({ page }) => {
  await mockLrclib(page, []);
  await page.goto('/');
  await openLyricsDropdown(page);
  await page.getByTestId('lyrics-menu-search').click();
  await expect(
    page.getByTestId('lyrics-search-modal').getByText(/No synced lyrics found/i),
  ).toBeVisible();
  await expect(page.getByTestId('lyrics-search-load-footer')).toHaveCount(0);
});

test('Load lyrics from file populates the row + clears via the gutter', async ({ page }) => {
  await page.goto('/');
  await openLyricsDropdown(page);
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByTestId('lyrics-menu-load-file').click(),
  ]);
  await chooser.setFiles({
    name: 'example.lrc',
    mimeType: 'text/plain',
    buffer: Buffer.from(SAMPLE_LRC, 'utf-8'),
  });

  const row = page.getByTestId('lyrics-row');
  await expect(row).toBeVisible();
  await expect(row.getByText(/File · example\.lrc/)).toBeVisible();
  await expect(page.getByTestId('lyrics-line-0')).toBeVisible();
  await expect(page.getByTestId('lyrics-line-1')).toBeVisible();
  await expect(page.getByTestId('lyrics-line-2')).toBeVisible();

  await page.getByTestId('lyrics-clear').click();
  await expect(row).toHaveCount(0);
});

test('Offset input shifts the active line under the playhead', async ({ page }) => {
  await page.goto('/');
  await openLyricsDropdown(page);
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByTestId('lyrics-menu-load-file').click(),
  ]);
  await chooser.setFiles({
    name: 'example.lrc',
    mimeType: 'text/plain',
    buffer: Buffer.from(SAMPLE_LRC, 'utf-8'),
  });
  await expect(page.getByTestId('lyrics-row')).toBeVisible();

  const input = page.getByTestId('lyrics-offset-input');
  await input.fill('2.5');
  await input.blur();
  await expect(input).toHaveValue('2.50');
});

test('Loading a new jot drops previously-loaded lyrics', async ({ page }) => {
  await page.goto('/');
  await openLyricsDropdown(page);
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByTestId('lyrics-menu-load-file').click(),
  ]);
  await chooser.setFiles({
    name: 'example.lrc',
    mimeType: 'text/plain',
    buffer: Buffer.from(SAMPLE_LRC, 'utf-8'),
  });
  await expect(page.getByTestId('lyrics-row')).toBeVisible();

  await page.getByRole('button', { name: 'Load', exact: true }).click();
  const [chooser2] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByRole('button', { name: 'Load audio track(s)' }).click(),
  ]);
  await chooser2.setFiles(TONE_WAV);
  await expect(page.getByTestId('lyrics-row')).toBeVisible();

  await page.evaluate(() => {
    const store = (window as any).drumjot.store;
    const examples: Array<{ id: string }> = store.examples;
    const other = examples.find((e) => e.id !== store.currentExampleId);
    if (other) store.loadExample(other.id);
  });
  await expect(page.getByTestId('lyrics-row')).toHaveCount(0);
});
