import path from 'node:path';
import { type Page } from '@playwright/test';

/**
 * Absolute path to a ParaDB / Paradiddle map pack (`.zip`), taken from the
 * `E2E_PARADB_ZIP` env var (set in `.env`; see AGENTS.md). Like the debug
 * bundle it's large + machine-local, so it's never committed; null when the
 * var is unset and callers gate with `test.skip(!PARADB_ZIP_PATH, …)`.
 */
export const PARADB_ZIP_PATH = process.env.E2E_PARADB_ZIP
  ? path.resolve(process.env.E2E_PARADB_ZIP)
  : null;

/**
 * Load the env-var ParaDB pack through the real toolbar path (File → Load →
 * "Load ParaDB map (.zip)") and wait for it to replace the song. Boots a
 * throwaway rock loop first so the toolbar + its hidden file inputs mount;
 * loading the pack then swaps in the converted chart + its audio tracks.
 *
 * The pack is loaded via the menu rather than a direct input selector
 * because several hidden inputs (ParaDB map / ParaDB score / debug bundle)
 * share the same `.zip` `accept`, so the menu item is the only unambiguous
 * handle. Throws if `E2E_PARADB_ZIP` is unset; gate the calling test on
 * {@link PARADB_ZIP_PATH} instead of relying on this.
 */
export async function loadParadbZip(page: Page): Promise<void> {
  if (!PARADB_ZIP_PATH) {
    throw new Error(
      'E2E_PARADB_ZIP is not set; gate the test with test.skip(!PARADB_ZIP_PATH, …).',
    );
  }

  await page.goto('/');
  // Boot is async (reactive-doc WASM init); wait for the debug global.
  await page.waitForFunction(
    () =>
      typeof (window as unknown as { drumjot?: { loadTestJot?: unknown } }).drumjot?.loadTestJot ===
      'function',
  );
  await page.evaluate(() =>
    (window as unknown as { drumjot: { loadTestJot(): void } }).drumjot.loadTestJot(),
  );
  await page.waitForSelector('[data-testid^="instrument-row-"]');

  await page.getByRole('button', { name: 'File', exact: true }).click();
  await page.getByRole('button', { name: 'Load', exact: true }).click();
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByRole('button', { name: 'Load ParaDB map (.zip)' }).click(),
  ]);
  await chooser.setFiles(PARADB_ZIP_PATH);

  // Applying the pack is async (zip unpack + .rlrr → jot + parallel audio
  // decode). The seeded test jot is the rock loop ("Simple rock loop");
  // wait until the pack's chart has replaced it.
  await page.waitForFunction(
    () => {
      const title = (
        window as unknown as { drumjot: { jotViewStore: { source?: { title: string } } } }
      ).drumjot.jotViewStore.source?.title;
      return !!title && title !== 'Simple rock loop';
    },
    null,
    { timeout: 90_000 },
  );
  await page.waitForSelector('[data-testid^="instrument-row-"]');
}
