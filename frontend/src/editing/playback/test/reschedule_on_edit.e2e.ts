import { expect, test } from '@playwright/test';
import { loadRockLoop } from './audio_capture.helper';

/**
 * Regression: editing notes mid-playback must be heard immediately, not only
 * after a stop+start. The engine snapshots its drum schedule (`jotPlayer.events`
 * + the scheduled web-audio events) at `play()`; a `reaction` in
 * `PlaybackPresenter` re-derives + reschedules it whenever the jot's notes
 * change. Pre-fix the snapshot was stale until the next stop/seek.
 *
 * We assert on the schedule snapshot rather than captured audio (covered in
 * audio_capture.e2e.ts): moving a note while `state === 'playing'` must change
 * the set of scheduled event times in place, with playback never interrupted.
 *
 * Reaching `playing` needs the smplr sample CDN, like the other audio specs.
 */

/** The scheduled drum-event times, sorted (the engine's live snapshot). */
function scheduledTimes(page: import('@playwright/test').Page): Promise<number[]> {
  return page.evaluate(
    () =>
      ((window as any).jotPlayer.events as { time: number }[])
        .map((e) => e.time)
        .sort((a, b) => a - b)
  );
}

test('moving a note during playback reschedules the engine without a stop/start', async ({
  page,
}) => {
  await loadRockLoop(page);

  // Start from inside the musical content and play; wait for samples to load.
  await page.evaluate(() => {
    const w = window as any;
    w.jotPlayer.seek(w.drumjot.jotEditorStore.tempo, 0.5);
  });
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => (window as any).jotPlayer.state), { timeout: 35_000 })
    .toBe('playing');

  const before = await scheduledTimes(page);
  expect(before.length).toBeGreaterThan(0);

  // Move one note a full beat later, through the real edit path, while playing.
  await page.evaluate(() => {
    const dj = (window as any).drumjot;
    const layers = dj.jotEditorStore.structural.musicalLayers;
    let note: any;
    outer: for (const layer of layers)
      for (const bar of layer.bars)
        for (const lane of Object.keys(bar.tracks))
          for (const n of bar.tracks[lane].notes) {
            note = n;
            break outer;
          }
    dj.selectionPresenter.replace(note);
    dj.editingPresenter.moveSelection(note, 1.0);
  });

  // The engine's schedule updated in place (same number of events, different
  // times) and playback was never interrupted.
  await expect(page.evaluate(() => (window as any).jotPlayer.state)).resolves.toBe('playing');
  const after = await scheduledTimes(page);
  expect(after.length).toBe(before.length);
  expect(JSON.stringify(after)).not.toBe(JSON.stringify(before));

  await page.getByRole('button', { name: 'Stop', exact: true }).click();
});
