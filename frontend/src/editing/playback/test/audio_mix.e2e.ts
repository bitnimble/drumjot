import { expect, test } from '@playwright/test';
import {
  applyMasters,
  captureRaw,
  countOnsets,
  loadAudioTrack,
  loadRockLoop,
  makeSineWavFile,
  measure,
  resetMix,
  setAudioMasterVolume,
  setDrumMasterVolume,
  setLaneSolo,
  setLaneVolume,
  setMasterVolume,
  setSpeed,
  setTrackMute,
  setTrackVolume,
} from './audio_capture.helper';

/**
 * Real-audio coverage (windowed-RMS of the captured page-master output) for the
 * mixer's level controls beyond on/off:
 *   - volume faders scale the output (page master, audio bus, per-track,
 *     drum bus, per-lane);
 *   - master mute/solo interacting with individual lane/track mute/solo;
 *   - playback speed changing the audible onset density.
 *
 * Linear-scaling is asserted tightly against a STEADY TONE (identical signal
 * every play, so a 0.5× gain gives a 0.5× RMS exactly); the drum bus is
 * onset-based, so across two plays its captured energy varies a little and we
 * only assert it's clearly reduced. All `test.slow()` and dependent on the
 * smplr sample CDN (see audio_capture.helper.ts).
 */

test('page-master, audio-bus and per-track volume scale the tone linearly', async ({ page }) => {
  test.slow();
  const tone = makeSineWavFile(6, 220, 0.3);
  await loadRockLoop(page);
  const id = await loadAudioTrack(page, tone);
  await resetMix(page);
  await applyMasters(page, { drumMute: true }); // isolate the steady tone

  const halfClean = (ratio: number) => {
    expect(ratio).toBeGreaterThan(0.45);
    expect(ratio).toBeLessThan(0.55);
  };

  await setMasterVolume(page, 1.0);
  const full = await measure(page);
  expect(full.mean).toBeGreaterThan(0.005);

  await setMasterVolume(page, 0.5);
  halfClean((await measure(page)).mean / full.mean); // page master
  await setMasterVolume(page, 1.0);

  await setAudioMasterVolume(page, 0.5);
  halfClean((await measure(page)).mean / full.mean); // audio bus
  await setAudioMasterVolume(page, 1.0);

  await setTrackVolume(page, id, 0.5);
  halfClean((await measure(page)).mean / full.mean); // per-track
});

test('drum-bus and per-lane volume reduce the drum level', async ({ page }) => {
  test.slow();
  await loadRockLoop(page);
  await resetMix(page);
  // Solo the hi-hat (plays every 8th, so always in the capture window) so its
  // level drives the signal. Drum energy is onset-based, so assert "clearly
  // reduced, roughly halved" rather than an exact ratio.
  await setLaneSolo(page, 'h', true);
  const reduced = (ratio: number) => {
    expect(ratio).toBeGreaterThan(0.3);
    expect(ratio).toBeLessThan(0.7);
  };

  await setDrumMasterVolume(page, 1.0);
  await setLaneVolume(page, 'h', 1.0);
  const full = await measure(page, 1200);
  // Non-silence sanity floor only (the load-independent `reduced` ratio below is
  // the real assertion). Kept low so a CPU-starved box, which depresses the
  // measured mean, doesn't flake it, silence still reads ~0.
  expect(full.mean).toBeGreaterThan(0.001);

  await setDrumMasterVolume(page, 0.5);
  reduced((await measure(page, 1200)).mean / full.mean); // drum bus
  await setDrumMasterVolume(page, 1.0);

  await setLaneVolume(page, 'h', 0.5);
  reduced((await measure(page, 1200)).mean / full.mean); // per-lane
});

test('master mute/solo combine with individual lane/track mute/solo', async ({ page }) => {
  test.slow();
  const tone = makeSineWavFile(5, 220, 0.25);
  await loadRockLoop(page);
  const id = await loadAudioTrack(page, tone);

  // The hi-hat lane plays every 8th, so a soloed lane is reliably in the
  // capture window (the kick is too sparse).

  // (a) Drum master mute wins over a soloed lane: the section stays silent.
  await resetMix(page);
  await setLaneSolo(page, 'h', true);
  await applyMasters(page, { drumMute: true, audioMute: true });
  expect((await measure(page)).mean).toBeLessThan(0.002);

  // (b) Soloing a drum lane is a GLOBAL solo: it excludes the audio bus, so
  // additionally muting the audio track changes nothing.
  await resetMix(page);
  await setLaneSolo(page, 'h', true);
  const laneSolo = await measure(page);
  await setTrackMute(page, id, true);
  const laneSoloTrackMuted = await measure(page);
  expect(laneSolo.mean).toBeGreaterThan(0.002); // hi-hat audible
  expect(Math.abs(laneSoloTrackMuted.mean - laneSolo.mean)).toBeLessThan(laneSolo.mean * 0.3);

  // (c) A soloed lane PLUS a soloed audio master: both pass, louder than either.
  await resetMix(page);
  await setLaneSolo(page, 'h', true);
  const hatOnly = await measure(page);
  await applyMasters(page, { audioSolo: true }); // keep the lane solo, add audio-master solo
  const hatPlusAudio = await measure(page);
  expect(hatPlusAudio.mean).toBeGreaterThan(hatOnly.mean);
});

test('playback speed changes the audible onset density', async ({ page }) => {
  test.slow();
  await loadRockLoop(page);
  await resetMix(page);

  await setSpeed(page, 1.0);
  const base = countOnsets(await captureRaw(page, 1500));
  await setSpeed(page, 2.0);
  const fast = countOnsets(await captureRaw(page, 1500));
  await setSpeed(page, 0.5);
  const slow = countOnsets(await captureRaw(page, 1500));
  await setSpeed(page, 1.0);

  // More onsets per real second when faster, fewer when slower.
  expect(base).toBeGreaterThan(2);
  expect(fast).toBeGreaterThan(base * 1.4);
  expect(slow).toBeLessThan(base * 0.8);
});
