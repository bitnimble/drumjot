import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';

const JOT_FIXTURE = fileURLToPath(new URL('./fixtures/loop.jot', import.meta.url));

test('app boots and renders the default example', async ({ page }) => {
  await page.goto('/');
  // Bootstrap loads the first registered example ("Simple rock loop").
  await expect(page.locator('h2')).toContainText('Simple rock loop');
  // Probe surface is wired.
  const wired = await page.evaluate(
    () => typeof (window as any).jotPlayer === 'object' && !!(window as any).drumjot,
  );
  expect(wired).toBe(true);
});

test('loads a .jot file from disk', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Load', exact: true }).click();
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByRole('button', { name: 'Load .jot file' }).click(),
  ]);
  await chooser.setFiles(JOT_FIXTURE);
  await expect(page.locator('h2')).toContainText('E2E Fixture Loop');
});

// The transcribe-flow e2e was removed when the DSL-output backend was
// deleted (May 2026): the new MIDI-bundle pathway needs a real .zip
// containing a valid prediction.mid to round-trip through the auto-load
// path, which requires more mock infrastructure than the unit-style smoke
// surface was built for. Re-add when a small MIDI/zip fixture is wired up.
