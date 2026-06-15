import { expect, test } from '@playwright/test';

// Regression guard: a song whose lead-in is carried only by `drumsT0Sec`
// (an audio pre-roll, no explicit `leadBars` rest bars, the shape RLRR /
// ParaDB maps produce) must still render a lead-in section. The
// RenderedJot→StructureStore refactor dropped the synthetic lead-in bar; the
// structure now re-synthesizes it from drumsT0Sec.
test('renders a lead-in section for a drumsT0Sec-only song', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => typeof (window as any).drumjot?.loadDsl === 'function');

  // No leading rest bars, just an audio pre-roll declared via drumsT0Sec.
  await page.evaluate(() =>
    (window as any).drumjot.loadDsl(
      '{{ drumsT0Sec: 3 }}\n{{ bpm: 120, time: "4/4", instrumentMapping: { k: { name: "Kick" } } }}\n| k . s . k . s . |'
    )
  );
  await page.waitForSelector('[data-testid^="instrument-track-"]');

  // The lead-in caption only renders when the structure carries a
  // negative-indexed (lead-in) bar.
  await expect(page.getByText('lead-in')).toBeVisible();
});
