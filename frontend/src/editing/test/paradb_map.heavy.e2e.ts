import { expect, test } from '@playwright/test';
import {
  PARADB_MAP_EXPECTED_EPOCHS,
  PARADB_MAP_PATH,
  loadParadbMap,
} from './paradb.helper';

/**
 * Smoke + lead-in coverage on a real, full-length ParaDB-map song
 * (`E2E_PARADB_MAP`): a complete chart with multiple audio tracks and a real
 * lead-in (audio pre-roll plus a longer synthetic/virtual rendered lead-in).
 * Drives the whole browser import pipeline, unlike the deterministic
 * `playback/test/epochs.test.ts` unit test which asserts the same anchors
 * against the store in isolation. Skipped when the env var is unset (the pack
 * is large + machine-local, never committed), same convention as the debug
 * bundle.
 */

// Read the rendered musical structure + the loaded audio tracks + the live
// epoch anchors in one round-trip. `musicalLayers` excludes the view-only
// virtual lead-in bar, so the bar/note counts reflect the real chart.
const PROBE = `(() => {
  const w = window;
  const structural = w.drumjot.jotEditorStore.structural;
  const layers = structural.musicalLayers;
  let bars = 0;
  let notes = 0;
  for (const layer of layers) {
    bars += layer.bars.length;
    for (const bar of layer.bars) {
      for (const lane of Object.keys(bar.tracks)) notes += bar.tracks[lane].notes.length;
    }
  }
  return {
    bars,
    notes,
    audioTrackCount: w.jotPlayer.audioTracks.size,
    epochs: w.drumjot.playback.epochs,
  };
})()`;

test('a full ParaDB-map song loads, renders, exposes correct lead-in epochs, and plays', async ({
  page,
}) => {
  test.skip(!PARADB_MAP_PATH, 'E2E_PARADB_MAP not set');
  test.setTimeout(120_000); // large zip unpack + .rlrr -> jot + multi-track audio decode

  await loadParadbMap(page);

  // The chart (title + tracks) lands before its backing audio finishes
  // decoding (the pack's tracks decode in parallel after the jot is set), so
  // wait for at least one audio track to register on the player.
  await page.waitForFunction(() => (window as any).jotPlayer.audioTracks.size > 0, null, {
    timeout: 60_000,
  });

  const probe = (await page.evaluate(PROBE)) as {
    bars: number;
    notes: number;
    audioTrackCount: number;
    epochs: { drums: number; songLeadIn: number; fullLeadIn: number };
  };

  // A real full-length song: lots of bars + notes, and the pack ships its
  // backing audio tracks (song + drums), so the chart actually loaded.
  expect(probe.bars).toBeGreaterThan(8);
  expect(probe.notes).toBeGreaterThan(50);
  expect(probe.audioTrackCount).toBeGreaterThan(0);

  // The whole real pipeline (rlrr -> jot -> structure/tempo + the presenter's
  // seed-on-load reaction) produces the hand-verified anchors, with the
  // ordering `fullLeadIn <= songLeadIn <= drums` the seek/audio engines rely
  // on. This is the real-browser counterpart to the unit test.
  expect(probe.epochs).toEqual({ ...PARADB_MAP_EXPECTED_EPOCHS });
  expect(probe.epochs.fullLeadIn).toBeLessThanOrEqual(probe.epochs.songLeadIn);
  expect(probe.epochs.songLeadIn).toBeLessThanOrEqual(probe.epochs.drums);

  // Basic transport: Play -> playing -> the clock advances -> Stop -> idle.
  // `exact` so this doesn't match the "Playback" toolbar menu button. Reaching
  // 'playing' fetches the smplr TR-808 samples from GitHub Pages; in a
  // network-restricted box that can fail with a real load error (not flake),
  // so we assert it with a timeout longer than the player's 30s load budget.
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => (window as any).jotPlayer.state), { timeout: 35_000 })
    .toBe('playing');

  const t1 = await page.evaluate(() => (window as any).jotPlayer.currentTime);
  await page.waitForTimeout(500);
  const t2 = await page.evaluate(() => (window as any).jotPlayer.currentTime);
  expect(t2).toBeGreaterThanOrEqual(t1);

  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect.poll(() => page.evaluate(() => (window as any).jotPlayer.state)).toBe('idle');
});
