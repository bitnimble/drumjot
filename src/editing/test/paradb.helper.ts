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
 * Absolute path to a *full-length-song* ParaDB map pack (`.zip`), taken from
 * the `E2E_PARADB_MAP` env var. Distinct from {@link PARADB_ZIP_PATH}: this
 * one points at a real, complete song (multiple audio tracks, a real lead-in
 * including a synthetic/virtual one), used to exercise lead-in / epoch maths
 * end-to-end. Machine-local, never committed; null when unset and callers gate
 * with `test.skip(!PARADB_MAP_PATH, …)`.
 */
export const PARADB_MAP_PATH = process.env.E2E_PARADB_MAP
  ? path.resolve(process.env.E2E_PARADB_MAP)
  : null;

/**
 * The {@link Epochs} anchors (jot seconds) for the `E2E_PARADB_MAP` fixture,
 * hand-verified by ear against the running app: a real audio lead-in
 * (`songLeadIn`) plus a longer synthetic/virtual rendered lead-in
 * (`fullLeadIn`), so `fullLeadIn < songLeadIn < drums`. These are specific to
 * that one pack; pointing `E2E_PARADB_MAP` at a different song will not match.
 */
export const PARADB_MAP_EXPECTED_EPOCHS = {
  drums: 0,
  songLeadIn: -1.36328125,
  fullLeadIn: -1.6901408450704225,
} as const;

/**
 * Load the `E2E_PARADB_ZIP` pack through the real toolbar path. See
 * {@link loadParadbMapFromPath}; gate the calling test on
 * {@link PARADB_ZIP_PATH}.
 */
export async function loadParadbZip(page: Page): Promise<void> {
  if (!PARADB_ZIP_PATH) {
    throw new Error(
      'E2E_PARADB_ZIP is not set; gate the test with test.skip(!PARADB_ZIP_PATH, …).',
    );
  }
  await loadParadbMapFromPath(page, PARADB_ZIP_PATH);
}

/**
 * Load the `E2E_PARADB_MAP` full-song pack through the real toolbar path. See
 * {@link loadParadbMapFromPath}; gate the calling test on
 * {@link PARADB_MAP_PATH}.
 */
export async function loadParadbMap(page: Page): Promise<void> {
  if (!PARADB_MAP_PATH) {
    throw new Error(
      'E2E_PARADB_MAP is not set; gate the test with test.skip(!PARADB_MAP_PATH, …).',
    );
  }
  await loadParadbMapFromPath(page, PARADB_MAP_PATH);
}

/**
 * Load a ParaDB pack at `zipPath` through the real toolbar path (File → Load →
 * "Load ParaDB map (.zip)") and wait for it to replace the song. Boots a
 * throwaway rock loop first so the toolbar + its hidden file inputs mount;
 * loading the pack then swaps in the converted chart + its audio tracks.
 *
 * The pack is loaded via the menu rather than a direct input selector
 * because several hidden inputs (ParaDB map / ParaDB score / debug bundle)
 * share the same `.zip` `accept`, so the menu item is the only unambiguous
 * handle.
 */
async function loadParadbMapFromPath(page: Page, zipPath: string): Promise<void> {
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
  await page.waitForSelector('[data-testid^="instrument-track-"]');

  await page.getByRole('button', { name: 'File', exact: true }).click();
  await page.getByRole('button', { name: 'Load', exact: true }).click();
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByRole('button', { name: 'Load ParaDB map (.zip)' }).click(),
  ]);
  await chooser.setFiles(zipPath);

  // Applying the pack is async (zip unpack + .rlrr → jot + parallel audio
  // decode). The seeded test jot is the rock loop ("Simple rock loop");
  // wait until the pack's chart has replaced it.
  await page.waitForFunction(
    () => {
      const title = (
        window as unknown as { drumjot: { jotEditorStore: { source?: { title: string } } } }
      ).drumjot.jotEditorStore.source?.title;
      return !!title && title !== 'Simple rock loop';
    },
    null,
    { timeout: 90_000 },
  );
  await page.waitForSelector('[data-testid^="instrument-track-"]');
}
