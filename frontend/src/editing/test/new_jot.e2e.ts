import { expect, test, type Page } from '@playwright/test';

/**
 * File → New jot + the empty-state "New blank jot" CTA: a fresh jot starts with
 * the standard kit declared as empty lanes (no notes, no audio), and a
 * wholesale replace of an edited session prompts before discarding.
 */

// A small loaded song to replace. Distinct title so we can tell it apart from
// the blank jot's "New Jot".
const SONG_DSL = `{{ bpm: 120, time: "4/4", title: "Existing Song",
  instrumentMapping: { h: { name: "HiHat" }, s: { name: "Snare" }, k: { name: "Kick" } } }}
${Array.from({ length: 2 }, () => '(k+h s+h k+h s+h)').join('\n')}
`;

const KIT_LANES = ['c', 'd', 'h', 's', 'k'];

async function loadSong(page: Page): Promise<void> {
  await page.waitForFunction(() => typeof (window as any).drumjot?.loadDsl === 'function');
  await page.evaluate((dsl) => (window as any).drumjot.loadDsl(dsl), SONG_DSL);
  await expect(page.locator('h2')).toContainText('Existing Song');
}

async function openFileNewJot(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'File', exact: true }).click();
  await page.getByTestId('file-menu-new').click();
}

async function expectBlankKit(page: Page): Promise<void> {
  await expect(page.locator('h2')).toContainText('New Jot');
  // Every kit lane renders as its own (empty) instrument row...
  for (const lane of KIT_LANES) {
    await expect(page.getByTestId(`instrument-track-${lane}`)).toBeVisible();
  }
  await expect.poll(() => page.locator('[data-testid^="instrument-track-"]').count()).toBe(
    KIT_LANES.length
  );
  // ...and the chart has no notes yet (`data-note-id` is the per-note marker).
  await expect(page.locator('[data-note-id]')).toHaveCount(0);
}

test('empty-state "New blank jot" creates a fresh kit with no notes', async ({ page }) => {
  await page.goto('/');
  await page.getByTestId('empty-state-new-jot').click();
  await expectBlankKit(page);
});

test('File → New jot replaces a clean session without prompting', async ({ page }) => {
  await page.goto('/');
  await loadSong(page);

  await openFileNewJot(page);

  // No unsaved edits, so no confirm dialog; the blank jot loads straight away.
  await expect(page.getByTestId('new-jot-confirm-modal')).toHaveCount(0);
  await expectBlankKit(page);
});

test('File → New jot prompts on unsaved edits: cancel keeps the song, discard replaces it', async ({
  page,
}) => {
  await page.goto('/');
  await loadSong(page);

  // Edit the document so the session is dirty (a committed change marks it).
  // Use the audio lead-in register (a plain mutation that doesn't change the
  // displayed title asserted below).
  await page.evaluate(() => {
    (window as any).drumjot.jotEditorStore.jot.songLeadIn = -1.5;
  });
  await expect
    .poll(() => page.evaluate(() => (window as any).drumjot.jotEditorStore.dirty))
    .toBe(true);

  // First attempt: the confirm dialog appears; cancelling keeps the song.
  await openFileNewJot(page);
  await expect(page.getByTestId('new-jot-confirm-modal')).toBeVisible();
  await page.getByTestId('new-jot-confirm-cancel').click();
  await expect(page.getByTestId('new-jot-confirm-modal')).toHaveCount(0);
  await expect(page.locator('h2')).toContainText('Existing Song');

  // Second attempt: discard actually starts the new jot.
  await openFileNewJot(page);
  await expect(page.getByTestId('new-jot-confirm-modal')).toBeVisible();
  await page.getByTestId('new-jot-confirm-discard').click();
  await expectBlankKit(page);
});
