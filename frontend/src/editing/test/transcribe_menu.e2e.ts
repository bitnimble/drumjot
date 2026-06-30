import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

/**
 * Coverage for the per-track transcribe flow that replaced the header
 * Transcribe dropdown:
 *  - the audio-track ⋯ menu has a "Transcribe…" item that opens the dialog in
 *    *append* mode (options only, no resume-stage picker);
 *  - File → Recent opens the dialog in *replace* mode (options + a resume-stage
 *    picker), with confirm gated on a resumable stage.
 *
 * Mocks `/api/transcribe/list` so the recent picker is populated without the
 * Python transcriber running. No transcription is actually started (that needs
 * the GPU backend); these lock the UI wiring.
 */

const TONE_WAV = fileURLToPath(new URL('../../../tests/fixtures/tone.wav', import.meta.url));

function trackIds(page: Page): Promise<string[]> {
  return page.evaluate(() => Array.from((window as any).jotPlayer.audioTracks.keys()));
}

async function loadRockLoop(page: Page): Promise<void> {
  await page.goto('/');
  await page.getByRole('button', { name: 'Simple rock loop' }).click();
  await page.waitForSelector('[data-testid^="instrument-track-"]');
}

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

const RECENT_RUN = {
  folder: 'fake_run',
  original_filename: 'fake.wav',
  requested_at: '2026-05-01T00:00:00Z',
  last_run_at: null,
  last_resume_stage: null,
  resumable_stages: ['beats', 'transcribe'],
};

async function mockRecentList(page: Page): Promise<void> {
  await page.route('**/api/transcribe/list', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([RECENT_RUN]),
    });
  });
}

test('audio-track ⋯ → Transcribe opens the append dialog (options, no resume stage)', async ({
  page,
}) => {
  await mockRecentList(page);
  await loadRockLoop(page);
  const id = await loadAudioTrack(page);

  await page
    .getByTestId(`audio-track-row-${id}`)
    .locator('button[title^="More actions"]')
    .click();
  await page.getByTestId(`audio-track-transcribe-${id}`).click();

  const dialog = page.getByTestId('transcribe-dialog');
  await expect(dialog).toBeVisible();
  // Append mode: the option selects are present, the resume-stage picker is not.
  await expect(dialog.getByLabel(/^Model/)).toBeVisible();
  await expect(dialog.getByLabel('Onset detector')).toBeVisible();
  await expect(dialog.getByLabel('From stage')).toHaveCount(0);
  // Confirm is always available in append mode (it has a target track).
  await expect(page.getByTestId('transcribe-dialog-confirm')).toBeEnabled();

  // Option edits persist on the shared store.
  await dialog.getByLabel(/^Model/).selectOption('claude-opus-4-7');
  await expect(dialog.getByLabel(/^Model/)).toHaveValue('claude-opus-4-7');

  await page.getByTestId('transcribe-dialog-cancel').click();
  await expect(dialog).toHaveCount(0);
});

test('File → Recent opens the replace dialog with a resume-stage picker', async ({ page }) => {
  await mockRecentList(page);
  await loadRockLoop(page);

  await page.getByRole('button', { name: 'File', exact: true }).click();
  await page.getByRole('button', { name: 'Recent', exact: true }).click();
  await page.getByRole('button', { name: 'fake.wav' }).click();

  const dialog = page.getByTestId('transcribe-dialog');
  await expect(dialog).toBeVisible();
  // Replace mode adds the resume-stage picker; confirm is gated until a
  // resumable stage is chosen.
  const stage = dialog.getByLabel('From stage');
  await expect(stage).toBeVisible();
  await expect(page.getByTestId('transcribe-dialog-confirm')).toBeDisabled();

  await stage.selectOption('transcribe');
  await expect(page.getByTestId('transcribe-dialog-confirm')).toBeEnabled();
});
