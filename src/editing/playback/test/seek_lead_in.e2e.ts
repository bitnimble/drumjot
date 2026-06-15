import { expect, test } from '@playwright/test';

// The score always shows at least one bar of (virtual) lead-in before bar 1,
// at negative jot time. Seeking must be able to reach that left edge, not
// clamp at jot time 0 (bar 1), even when the song has no audio pre-roll
// (drumsT0Sec === 0), where the old `-drumsT0Sec` floor stranded the playhead.
test('scrubs the playhead all the way to the lead-in left edge', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => typeof (window as any).drumjot?.loadDsl === 'function');
  await page.evaluate(() =>
    (window as any).drumjot.loadDsl(
      '{{ bpm: 120, time: "4/4", instrumentMapping: { k: { name: "Kick" } } }}\n| k . . . |'
    )
  );
  await page.waitForSelector('[data-testid^="instrument-track-"]');

  const r = await page.evaluate(() => {
    const w = window as any;
    const tempo = w.drumjot.jotEditorStore.tempo;
    const start = tempo.timeline.bars[0].startSec; // lead-in left edge (negative)
    w.jotPlayer.seek(tempo, -9999); // try to scrub far past the left edge
    return { start, currentTime: w.jotPlayer.currentTime as number };
  });

  expect(r.start).toBeLessThan(0); // there is lead-in before bar 1
  // The playhead reaches the rendered left edge instead of clamping at 0.
  expect(r.currentTime).toBeCloseTo(r.start, 3);
});
