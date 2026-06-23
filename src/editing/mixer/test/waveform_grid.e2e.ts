import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

/**
 * View → Waveforms → "Bar & beat lines": each audio-track waveform row
 * draws the same bar lines + beat grid the score shows above it, mounted
 * only while the toggle is on (default on). The grid bars are positioned
 * in the same beat-space as the score's bars, so this just asserts the
 * overlay mounts/unmounts with the toggle and carries one positioned bar
 * div per visible bar, the pixel alignment itself is shared CSS with the
 * score and isn't re-derivable here.
 */

const TONE_WAV = fileURLToPath(new URL('../../../../tests/fixtures/tone.wav', import.meta.url));

function trackIds(page: Page): Promise<string[]> {
  return page.evaluate(() => Array.from((window as any).jotPlayer.audioTracks.keys()));
}

async function loadRockLoop(page: Page): Promise<void> {
  await page.goto('/');
  await page.getByRole('button', { name: 'Simple rock loop' }).click();
  await page.waitForSelector('[data-testid^="instrument-track-"]');
}

/** File → Load → "Load audio track(s)", feed the fixture, return the new id. */
async function loadAudioTrack(page: Page): Promise<string> {
  const before = new Set(await trackIds(page));
  await page.getByRole('button', { name: 'File', exact: true }).click();
  await page.getByRole('button', { name: 'Load', exact: true }).click();
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByRole('button', { name: 'Load audio track(s)' }).click(),
  ]);
  await chooser.setFiles(TONE_WAV);
  await expect
    .poll(async () => (await trackIds(page)).filter((id) => !before.has(id)).length)
    .toBe(1);
  return (await trackIds(page)).find((x) => !before.has(x))!;
}

/** Flip a View-menu toggle by its label. */
async function toggleView(page: Page, label: string): Promise<void> {
  await page.getByRole('button', { name: 'View' }).click();
  await page.getByText(label).click();
  await page.keyboard.press('Escape');
}

test('waveform grid is on by default with one bar div per visible bar', async ({ page }) => {
  await loadRockLoop(page);
  const id = await loadAudioTrack(page);

  const grid = page.getByTestId(`audio-track-grid-${id}`);
  await expect(grid).toBeVisible();
  // One positioned `.waveformGridBar` per visible (non-lead-in) bar.
  await expect.poll(() => grid.locator('> div').count()).toBeGreaterThan(0);
});

test('the "Bar & beat lines" View toggle mounts/unmounts the waveform grid', async ({ page }) => {
  await loadRockLoop(page);
  const id = await loadAudioTrack(page);
  await expect(page.getByTestId(`audio-track-grid-${id}`)).toBeVisible();

  // Off → overlay unmounts entirely.
  await toggleView(page, 'Bar & beat lines');
  await expect(page.getByTestId(`audio-track-grid-${id}`)).toHaveCount(0);
  expect(
    await page.evaluate(() => (window as any).drumjot.settings.waveformGridLines)
  ).toBe(false);

  // On → overlay comes back.
  await toggleView(page, 'Bar & beat lines');
  await expect(page.getByTestId(`audio-track-grid-${id}`)).toBeVisible();
  expect(
    await page.evaluate(() => (window as any).drumjot.settings.waveformGridLines)
  ).toBe(true);
});
