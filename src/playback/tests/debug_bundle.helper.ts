import path from 'node:path';
import { type Page } from '@playwright/test';

/**
 * Absolute path to a full transcriber debug bundle (`.zip`), taken from
 * the `E2E_DEBUG_BUNDLE` env var (set in `.env`; see AGENTS.md). The
 * bundle is large and machine-local, so it's never committed. Null when
 * the var is unset; callers gate with `test.skip(!DEBUG_BUNDLE_PATH, …)`.
 */
export const DEBUG_BUNDLE_PATH = process.env.E2E_DEBUG_BUNDLE
  ? path.resolve(process.env.E2E_DEBUG_BUNDLE)
  : null;

/**
 * Load the env-var debug bundle through the real toolbar path and wait
 * for the viewer to finish applying it (score + per-stem audio + debug
 * manifest). Boots a throwaway rock-loop first so the toolbar and its
 * hidden file inputs mount; loading the bundle then replaces the song
 * wholesale.
 *
 * The bundle is loaded via File → Load → "Load debug bundle (.zip)"
 * rather than a direct input selector: several hidden inputs (ParaDB
 * map / score / debug bundle) share the same `.zip` `accept`, so the menu
 * is the only unambiguous handle.
 *
 * Throws if `E2E_DEBUG_BUNDLE` is unset; gate the calling test on
 * {@link DEBUG_BUNDLE_PATH} instead of relying on this.
 */
export async function loadDebugBundle(page: Page): Promise<void> {
  if (!DEBUG_BUNDLE_PATH) {
    throw new Error(
      'E2E_DEBUG_BUNDLE is not set; gate the test with test.skip(!DEBUG_BUNDLE_PATH, …).',
    );
  }

  await page.goto('/');
  await page.evaluate(() =>
    (
      window as unknown as { drumjot: { loadTestJot(): void } }
    ).drumjot.loadTestJot(),
  );
  await page.waitForSelector('[data-testid^="instrument-row-"]');

  await page.getByRole('button', { name: 'File', exact: true }).click();
  await page.getByRole('button', { name: 'Load', exact: true }).click();
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByRole('button', { name: 'Load debug bundle (.zip)' }).click(),
  ]);
  await chooser.setFiles(DEBUG_BUNDLE_PATH);

  // Applying the bundle is async (zip unpack + parallel audio decode).
  // Wait for the manifest to mount and at least one audio track to come
  // online before handing control back to the test.
  await page.waitForFunction(
    () => !!(window as unknown as { drumjot: { store: { lastDebugBundle: unknown } } }).drumjot.store.lastDebugBundle,
    null,
    { timeout: 90_000 },
  );
  await page.waitForFunction(
    () => (window as unknown as { jotPlayer: { audioTracks: Map<string, unknown> } }).jotPlayer.audioTracks.size >= 1,
    null,
    { timeout: 90_000 },
  );
}
