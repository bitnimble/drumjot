import { expect, test } from '@playwright/test';
import {
  applyMasters,
  loadAudioTrack,
  loadRockLoop,
  makeSineWavFile,
  measure,
} from './audio_capture.helper';

/**
 * Real-audio verification. Instead of asserting only that notes are
 * *scheduled*, these tap the player's page-master output with an AnalyserNode
 * (a pure observer on the node feeding `ctx.destination`) and assert on the
 * captured windowed RMS:
 *
 *   - playback actually produces audible (non-silent) sound, with drum
 *     transients (max RMS >> mean), and muting the section silences it;
 *   - mute/solo across the two track *types* (drum bus vs audio-track bus)
 *     raises and lowers the master level the way the rules say it should.
 *
 * See `audio_capture.helper.ts` for the capture/measure plumbing and the smplr
 * sample-CDN dependency these share with `audio_tracks.e2e.ts`.
 */

test('playback produces audible output with transients; muting the drum bus silences it', async ({
  page,
}) => {
  await loadRockLoop(page);

  await applyMasters(page, {}); // everything audible (drums only, no audio track)
  const playing = await measure(page);
  // Real sound came out, and it's percussive (onset peaks well above the mean).
  expect(playing.mean).toBeGreaterThan(0.005);
  expect(playing.max).toBeGreaterThan(playing.mean * 2);

  await applyMasters(page, { drumMute: true });
  const muted = await measure(page);
  // Muting the only audible bus drops the output to ~silence.
  expect(muted.mean).toBeLessThan(0.002);
  expect(playing.mean).toBeGreaterThan(muted.mean * 20);
});

test('mute/solo across the drum and audio-track buses raises and lowers the level', async ({
  page,
}) => {
  test.slow(); // several realtime play/capture cycles
  const tone = makeSineWavFile(5, 220, 0.25);
  await loadRockLoop(page);
  const trackId = await loadAudioTrack(page, tone);

  await applyMasters(page, {});
  const both = await measure(page);

  await applyMasters(page, { audioMute: true }); // drums only
  const drumsOnly = await measure(page);

  await applyMasters(page, { drumMute: true }); // audio only
  const audioOnly = await measure(page);

  await applyMasters(page, { drumMute: true, audioMute: true });
  const silent = await measure(page);

  // Both buses together are louder than either alone (uncorrelated power adds).
  expect(both.mean).toBeGreaterThan(drumsOnly.mean);
  expect(both.mean).toBeGreaterThan(audioOnly.mean);
  // Each single bus is still clearly audible, well above the muted floor.
  expect(silent.mean).toBeLessThan(0.002);
  expect(drumsOnly.mean).toBeGreaterThan(silent.mean + 0.004);
  expect(audioOnly.mean).toBeGreaterThan(silent.mean + 0.004);
  expect(both.mean).toBeGreaterThan(silent.mean * 20);

  // Soloing a section excludes the other one, i.e. it matches that section
  // playing alone, not both together.
  await applyMasters(page, { drumSolo: true });
  const drumSolo = await measure(page);
  expect(Math.abs(drumSolo.mean - drumsOnly.mean)).toBeLessThan(
    Math.abs(drumSolo.mean - both.mean)
  );

  await applyMasters(page, { audioSolo: true });
  const audioSolo = await measure(page);
  expect(Math.abs(audioSolo.mean - audioOnly.mean)).toBeLessThan(
    Math.abs(audioSolo.mean - both.mean)
  );

  // Per-track mute, isolated by muting the drum bus so only the tone remains:
  // muting the single audio track then drops the output to silence.
  await applyMasters(page, { drumMute: true });
  await page.evaluate((id) => {
    const w = window as any;
    if (w.drumjot.mixer.isAudioTrackAudible(id)) w.drumjot.mixerPresenter.toggleAudioTrackMute(id);
  }, trackId);
  const trackMuted = await measure(page);
  expect(trackMuted.mean).toBeLessThan(0.002);
});
