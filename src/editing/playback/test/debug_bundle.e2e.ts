import { expect, test } from '@playwright/test';
import { DEBUG_BUNDLE_PATH, loadDebugBundle } from './debug_bundle.helper';

/**
 * Opt-in "complete viewer" smoke test. Loads a full transcriber debug
 * bundle (real score + per-stem audio + provenance + logs) pointed to by
 * the `E2E_DEBUG_BUNDLE` env var, exercising a representative viewer
 * instead of the tiny 4s rock-loop example. The bundle is large and
 * machine-local (never committed), so this is skipped when the var is
 * unset, the default `bun run e2e` stays deterministic.
 */
test('full debug bundle loads and renders the complete viewer', async ({
  page,
}) => {
  test.skip(!DEBUG_BUNDLE_PATH, 'E2E_DEBUG_BUNDLE not set');
  test.setTimeout(120_000); // large zip unpack + multi-track audio decode

  await loadDebugBundle(page);

  // Score rendered: at least one instrument row.
  const instrumentRows = page.locator('[data-testid^="instrument-row-"]');
  await expect.poll(() => instrumentRows.count()).toBeGreaterThanOrEqual(1);

  // Per-stem audio mounted: at least one audio-track row.
  const audioRows = page.locator('[data-testid^="audio-track-row-"]');
  await expect.poll(() => audioRows.count()).toBeGreaterThanOrEqual(1);

  // Debug manifest mounted (drives the DebugPanel + filter overlays).
  const hasManifest = await page.evaluate(
    () =>
      !!(
        window as unknown as { drumjot: { provenance: { lastDebugBundle: unknown } } }
      ).drumjot.provenance.lastDebugBundle,
  );
  expect(hasManifest).toBe(true);
});
