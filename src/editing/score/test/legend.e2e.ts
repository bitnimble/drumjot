import { expect, test } from '@playwright/test';

/**
 * Guards the per-jot colour/name legend (`Legend` in score_header.tsx,
 * fed by `jot.palette.legend`). This pins the exact equivalence surface a
 * palette-decomposition step could silently break:
 *   - the LANE ORDER of the chips,
 *   - the instrument NAME shown on each chip, and
 *   - that mapped lanes get distinct, non-empty swatch colours.
 * No prior spec asserted legend order/colour, so a refactor touching the
 * legend's source could pass CI while changing what's painted.
 *
 * Legend order comes from the structure walk's per-bar track-record
 * insertion order (here: s, h, k), which is NOT the same as the jot-wide
 * lane list (`PaletteStore.jotLanes`) or the mixer row order, so
 * `PaletteStore.legend` deliberately walks the structure to preserve it.
 */
const JOT = `{{ bpm: 120, time: "4/4", title: "Legend Order",
  instrumentMapping: { h: { name: "HiHat" }, s: { name: "Snare" }, k: { name: "Kick" } } }}
(h s k h s k)
`;

test('legend renders lanes in order, with names and distinct colours', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => typeof (window as any).drumjot?.loadDsl === 'function');
  await page.evaluate((src) => (window as any).drumjot.loadDsl(src), JOT);
  await page.waitForSelector('[data-testid^="instrument-track-"]');

  const chips = page.locator('[data-testid="legend-chip"]');
  await expect(chips).toHaveCount(3);

  // Lane order as the structure walk currently emits it: s, h, k.
  const lanes = await chips.evaluateAll((els) =>
    els.map((el) => el.getAttribute('data-lane'))
  );
  expect(lanes).toEqual(['s', 'h', 'k']);

  // Names come through from the instrument mapping, per chip, in that order.
  const names = await chips
    .locator('[data-testid="legend-name"]')
    .evaluateAll((els) => els.map((el) => el.textContent));
  expect(names).toEqual(['Snare', 'HiHat', 'Kick']);

  // Each swatch has a real colour and the three are mutually distinct
  // (the palette assigns a different slot per jot-wide lane).
  const colours = await chips
    .locator('[data-testid="legend-swatch"]')
    .evaluateAll((els) => els.map((el) => getComputedStyle(el).backgroundColor));
  for (const c of colours) {
    expect(c).toMatch(/^rgb/);
    expect(c).not.toBe('rgba(0, 0, 0, 0)');
  }
  expect(new Set(colours).size).toBe(3);

  // The rendered order matches the getter that feeds it (DOM ↔ model),
  // so a future change to the getter can't drift from what's painted.
  const fromModel = await page.evaluate(() =>
    (window as any).drumjot.jotEditorStore.palette.legend.map(
      ([lane]: [string, unknown]) => lane
    )
  );
  expect(fromModel).toEqual(['s', 'h', 'k']);
});
