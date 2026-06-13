import { fileURLToPath } from 'node:url';
import { expect, test, type Page, type Route } from '@playwright/test';

/**
 * E2E coverage for time-aligned lyrics. LRCLIB is stubbed via
 * `page.route` so the suite needs no network access; the LRC payload is
 * a small inline fixture rather than a separate file so the spec is
 * self-contained.
 *
 * The app boots to the empty state, so every test first loads the
 * `Simple rock loop` example (a ~4s, 2-bar/120bpm score, the lyric
 * timestamps below all fall inside that span so the lines render).
 * Lyrics loaders live under the toolbar's File → Lyrics submenu; the
 * per-row offset / clear / export controls live in the row's ⋯ overflow.
 *
 * What's exercised:
 *  - LRCLIB search → single result still renders a list; user clicks
 *    Load to commit (no auto-load).
 *  - LRCLIB search → multi-result picker; user selects then clicks Load.
 *  - LRCLIB search → 0 results message.
 *  - Word-level alignment checkbox disabled when no audio tracks loaded.
 *  - Load from file (.lrc) via the File → Lyrics submenu.
 *  - Offset input (in the ⋯ overflow) accepts a value.
 *  - Loading a new jot drops previously-loaded lyrics.
 */

const SAMPLE_LRC = `[00:00.00]Verse line one
[00:01.50]Verse line two
[00:03.00]Verse line three
`;

const TONE_WAV = fileURLToPath(
  new URL('../../../tests/fixtures/tone.wav', import.meta.url),
);

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

async function mockLrclib(page: Page, rows: LrclibRow[]): Promise<void> {
  await page.route('https://lrclib.net/api/search**', (route: Route) => {
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

/** Load the built-in `Simple rock loop` example from the empty-state
 *  picker and wait for the score to render. The toolbar (and thus the
 *  lyrics loaders) only exists once a jot is loaded. */
async function loadRockLoop(page: Page): Promise<void> {
  await page.goto('/');
  await page.getByRole('button', { name: 'Simple rock loop' }).click();
  await page.waitForSelector('[data-testid^="instrument-row-"]');
}

/** Open File → Lyrics so the lyrics loaders are reachable. */
async function openLyricsMenu(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'File', exact: true }).click();
  await page.getByRole('button', { name: 'Lyrics', exact: true }).click();
}

/** Open File → Load (the score / audio loaders submenu). */
async function openLoadMenu(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'File', exact: true }).click();
  await page.getByRole('button', { name: 'Load', exact: true }).click();
}

/** Open the per-row ⋯ overflow menu on the lyrics row (offset / export /
 *  remove live in this portaled panel). */
async function openLyricsOverflow(page: Page): Promise<void> {
  await page
    .getByTestId('lyrics-row')
    .locator('button[title="More actions for this lyrics track"]')
    .click();
}

/** Load the inline SAMPLE_LRC through File → Lyrics → Load from file. */
async function loadSampleLrcFromFile(page: Page): Promise<void> {
  await openLyricsMenu(page);
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByTestId('lyrics-menu-load-file').click(),
  ]);
  await chooser.setFiles({
    name: 'example.lrc',
    mimeType: 'text/plain',
    buffer: Buffer.from(SAMPLE_LRC, 'utf-8'),
  });
}

test('LRCLIB single result still requires explicit Load click', async ({ page }) => {
  await mockLrclib(page, [makeRow()]);
  await loadRockLoop(page);
  await expect(page.locator('h2')).toContainText('Simple rock loop');

  await openLyricsMenu(page);
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
  await loadRockLoop(page);
  await openLyricsMenu(page);
  await page.getByTestId('lyrics-menu-search').click();
  await expect(page.getByTestId('lyrics-search-results')).toBeVisible();
  // Footer is always shown; without a selection the Load button is
  // disabled until the user picks a row.
  await expect(page.getByTestId('lyrics-search-load-footer')).toBeVisible();
  await expect(page.getByTestId('lyrics-search-load')).toBeDisabled();
  await page.getByTestId('lyrics-search-result-1').click();
  await expect(page.getByTestId('lyrics-search-load')).toBeEnabled();
  await page.getByTestId('lyrics-search-load').click();
  await expect(page.getByTestId('lyrics-search-loaded')).toBeVisible();
});

test('Word-level checkbox is disabled when no audio tracks are loaded', async ({ page }) => {
  await mockLrclib(page, [makeRow()]);
  await loadRockLoop(page);
  await openLyricsMenu(page);
  await page.getByTestId('lyrics-menu-search').click();
  await expect(page.getByTestId('lyrics-search-load-footer')).toBeVisible();
  await expect(page.getByTestId('lyrics-search-word-level')).toBeDisabled();
});

test('LRCLIB zero results shows the no-results message', async ({ page }) => {
  await mockLrclib(page, []);
  await loadRockLoop(page);
  await openLyricsMenu(page);
  await page.getByTestId('lyrics-menu-search').click();
  await expect(
    page.getByTestId('lyrics-search-modal').getByText(/No synced lyrics found/i),
  ).toBeVisible();
  // Footer always renders; Load is disabled because there's nothing to load.
  await expect(page.getByTestId('lyrics-search-load-footer')).toBeVisible();
  await expect(page.getByTestId('lyrics-search-load')).toBeDisabled();
});

test('Load lyrics from file populates the row + clears via the overflow menu', async ({
  page,
}) => {
  await loadRockLoop(page);
  await loadSampleLrcFromFile(page);

  const row = page.getByTestId('lyrics-row');
  await expect(row).toBeVisible();
  await expect(row.getByText(/File · example\.lrc/)).toBeVisible();
  await expect(page.getByTestId('lyrics-line-0')).toBeVisible();
  await expect(page.getByTestId('lyrics-line-1')).toBeVisible();
  await expect(page.getByTestId('lyrics-line-2')).toBeVisible();

  await openLyricsOverflow(page);
  await page.getByTestId('lyrics-clear').click();
  await expect(row).toHaveCount(0);
});

test('Offset input accepts a value', async ({ page }) => {
  await loadRockLoop(page);
  await loadSampleLrcFromFile(page);
  await expect(page.getByTestId('lyrics-row')).toBeVisible();

  await openLyricsOverflow(page);
  const input = page.locator('input[data-testid^="lyrics-offset-input-"]');
  await input.fill('2.5');
  await input.blur();
  await expect(input).toHaveValue('2.50');
});

test('Loading a new jot drops previously-loaded lyrics', async ({ page }) => {
  await loadRockLoop(page);
  await loadSampleLrcFromFile(page);
  await expect(page.getByTestId('lyrics-row')).toBeVisible();

  // Loading an audio track is additive and must NOT clear lyrics.
  await openLoadMenu(page);
  const [chooser2] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByRole('button', { name: 'Load audio track(s)' }).click(),
  ]);
  await chooser2.setFiles(TONE_WAV);
  await expect(page.getByTestId('lyrics-row')).toBeVisible();

  // A wholesale song change (loading a different example) clears them.
  await page.evaluate(() => {
    const presenter = (window as any).drumjot.presenter;
    const doc = (window as any).drumjot.document;
    const examples: Array<{ id: string }> = doc.examples;
    const other = examples.find((e) => e.id !== doc.currentExampleId);
    if (other) presenter.loadExample(other.id);
  });
  await expect(page.getByTestId('lyrics-row')).toHaveCount(0);
});
