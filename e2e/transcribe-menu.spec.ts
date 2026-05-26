import { expect, test } from '@playwright/test';

/**
 * Regression lock for the Transcribe-dropdown redesign: the Beat-input
 * and Model selects sit ABOVE the New/Resume tab strip and apply to
 * both flows. The original toolbar grouped them visually with the fresh
 * "Select file" button so it wasn't obvious they also fed a resume run;
 * the redesign moves them outside the tab body. This test asserts the
 * shared identity by changing the Model picker on the New tab, switching
 * to Resume, and confirming the picker still reflects the chosen value.
 *
 * Mocks `/api/transcribe/list` so the Resume tab is enabled without
 * needing the Python transcriber to be up.
 */
test('Model selection persists across New ↔ Resume tab switch', async ({ page }) => {
  await page.route('**/api/transcribe/list', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([
        {
          folder: 'fake_run',
          original_filename: 'fake.wav',
          requested_at: '2026-05-01T00:00:00Z',
          last_run_at: null,
          last_resume_stage: null,
          resumable_stages: [],
        },
      ]),
    });
  });

  await page.goto('/');
  await page.getByRole('button', { name: 'Transcribe', exact: false }).click();

  // Default tab is New; the Resume tab is initially disabled (no recent
  // runs yet) and only enables after the mocked /list call resolves.
  const resumeTab = page.getByTestId('transcribe-tab-resume');
  await expect(resumeTab).toBeEnabled();

  // Flip Model away from its default so the assertion below proves the
  // value rode through the tab switch rather than just matching the
  // default on both sides.
  const modelSelect = page.getByLabel('Model');
  await modelSelect.selectOption('claude-haiku-4-5-20251001');
  await expect(modelSelect).toHaveValue('claude-haiku-4-5-20251001');

  await resumeTab.click();

  // The Model select is rendered above the tab strip, so it's still
  // mounted with the same DOM node and value. If a future redesign moves
  // it inside one of the tab bodies, this assertion catches that
  // regression, either the locator goes stale (mount/unmount) or the
  // value resets to the default.
  await expect(modelSelect).toHaveValue('claude-haiku-4-5-20251001');
});
