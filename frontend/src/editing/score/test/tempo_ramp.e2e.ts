import { expect, test } from '@playwright/test';

/**
 * Covers the gradual-tempo (`BpmTransition`) ramp in the timeline gutter
 * (`timeline_header.tsx`): a solid bar captioned `<start> bpm ... <end> bpm`
 * spanning the ramp's beat range. Asserts the ramp renders with the right
 * end labels, that the model derivation (`TempoPresenter.tempoRamps`) agrees,
 * and that the redundant flat bpm pill at the ramp's anchor is suppressed.
 */
// Two bars at the initial 120, then a ramp that jumps to 150 and rises to
// 180 over 8 beats. The jump-to-150 sits past bar 1 (so it's distinct from
// the lead-in's initial-tempo pill), letting us assert the redundant flat
// "150 bpm" pill at the ramp anchor is suppressed in favour of the caption.
const JOT = `{{ bpm: 120, time: "4/4", title: "Ramp" }}
| k . s . |
| k . s . |
{{ bpm: { start: 150, end: 180, duration: 8 } }}
| k . s . |
| k . s . |`;

test('renders a gradual bpm transition as a labelled ramp', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => typeof (window as any).drumjot?.loadDsl === 'function');
  await page.evaluate((src) => (window as any).drumjot.loadDsl(src), JOT);
  await page.waitForSelector('[data-testid^="instrument-track-"]');

  // One ramp, with start/end captions taken from the transition.
  const ramp = page.locator('[data-testid="bpm-ramp"]');
  await expect(ramp).toHaveCount(1);
  await expect(ramp.locator('[data-testid="bpm-ramp-start"]')).toHaveText('150 bpm');
  await expect(ramp.locator('[data-testid="bpm-ramp-end"]')).toHaveText('180 bpm');

  // The model derivation agrees: anchored at the third real bar (global
  // beat 12 = the 4-beat view-only lead-in + two 4-beat bars, the same space
  // the header ticks use), spanning the transition's 8-beat duration, 150 -> 180.
  const ramps = await page.evaluate(() =>
    (window as any).drumjot.jotEditorStore.tempo.tempoRamps.map((r: any) => ({
      startBeat: r.startBeat,
      endBeat: r.endBeat,
      startBpm: r.startBpm,
      endBpm: r.endBpm,
    }))
  );
  expect(ramps).toEqual([{ startBeat: 12, endBeat: 20, startBpm: 150, endBpm: 180 }]);

  // The flat "150 bpm" downbeat pill the segment walk would paint at the
  // ramp anchor is suppressed; the only "150 bpm" in the header is the
  // ramp's start caption (it would be 2 without suppression).
  const header = page.locator('[data-bars-row]');
  const startMatches = await header.getByText('150 bpm', { exact: true }).count();
  expect(startMatches).toBe(1);
});
