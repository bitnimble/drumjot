import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, test, type Page } from '@playwright/test';

/**
 * Real-audio verification. Instead of asserting only that notes are
 * *scheduled*, these tap the player's page-master output with an AnalyserNode
 * (`jotPlayer.startOutputCapture`, a pure observer on the node feeding
 * `ctx.destination`) and assert on the captured windowed RMS:
 *
 *   - playback actually produces audible (non-silent) sound, with drum
 *     transients (max RMS >> mean), and muting the section silences it;
 *   - mute/solo across the two track *types* (drum bus vs audio-track bus)
 *     raises and lowers the master level the way the rules say it should.
 *
 * Reaching `playing` needs the smplr TR-808 samples (smpldsnds.github.io), so
 * like `audio_tracks.e2e.ts::playback starts…` these depend on that CDN; a
 * load failure there is a real environment signal, asserted with a generous
 * timeout rather than masked.
 */

// ---- helpers ---------------------------------------------------------------

/** Synthesise a mono 16-bit PCM sine WAV and write it to a temp file, returning
 *  the path. A few seconds long so the tone plays continuously through every
 *  capture window (the committed `tone.wav` fixture is only 0.5s). */
function makeSineWavFile(seconds: number, freq: number, amp: number): string {
  const sampleRate = 22050;
  const n = Math.floor(seconds * sampleRate);
  const buf = Buffer.alloc(44 + n * 2);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + n * 2, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(1, 22); // mono
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) {
    const s = Math.sin((2 * Math.PI * freq * i) / sampleRate) * amp;
    buf.writeInt16LE(Math.round(s * 32767), 44 + i * 2);
  }
  const path = join(mkdtempSync(join(tmpdir(), 'drumjot-tone-')), 'tone_long.wav');
  writeFileSync(path, buf);
  return path;
}

async function loadRockLoop(page: Page): Promise<void> {
  await page.goto('/');
  await page.getByRole('button', { name: 'Simple rock loop' }).click();
  await page.waitForSelector('[data-testid^="instrument-track-"]');
}

function trackIds(page: Page): Promise<string[]> {
  return page.evaluate(() => Array.from((window as any).jotPlayer.audioTracks.keys()) as string[]);
}

/** Load an audio-track file via File → Load and return the player-allocated id. */
async function loadAudioTrack(page: Page, file: string): Promise<string> {
  const before = new Set(await trackIds(page));
  await page.getByRole('button', { name: 'File', exact: true }).click();
  await page.getByRole('button', { name: 'Load', exact: true }).click();
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByRole('button', { name: 'Load audio track(s)' }).click(),
  ]);
  await chooser.setFiles(file);
  await expect
    .poll(async () => (await trackIds(page)).filter((id) => !before.has(id)).length)
    .toBe(1);
  return (await trackIds(page)).find((x) => !before.has(x))!;
}

type Mix = { drumMute?: boolean; audioMute?: boolean; drumSolo?: boolean; audioSolo?: boolean };

/** Apply a full master mute/solo config idempotently via the real presenter
 *  toggles (solos first: soloing a section clears its mute, so mutes win). */
async function applyMasters(page: Page, cfg: Mix): Promise<void> {
  await page.evaluate((c) => {
    const m = (window as any).drumjot.mixer;
    const p = (window as any).drumjot.mixerPresenter;
    const set = (cur: boolean, want: boolean, toggle: () => void) => {
      if (cur !== !!want) toggle();
    };
    set(m.drumMasterSoloed, !!c.drumSolo, () => p.toggleDrumMasterSolo());
    set(m.audioMasterSoloed, !!c.audioSolo, () => p.toggleAudioMasterSolo());
    set(m.drumMasterMuted, !!c.drumMute, () => p.toggleDrumMasterMute());
    set(m.audioMasterMuted, !!c.audioMute, () => p.toggleAudioMasterMute());
  }, cfg);
}

/** Mean RMS over the captured series (edges trimmed to drop start/stop
 *  transients) plus the peak, for transient detection. */
function summarise(samples: { t: number; rms: number }[]): { mean: number; max: number } {
  if (samples.length === 0) return { mean: 0, max: 0 };
  const trim = Math.floor(samples.length * 0.15);
  const core = samples.slice(trim, samples.length - trim);
  const xs = (core.length > 0 ? core : samples).map((s) => s.rms);
  return { mean: xs.reduce((a, b) => a + b, 0) / xs.length, max: Math.max(...xs) };
}

/** Seek to a beat well inside the musical content, play, capture ~800ms of the
 *  page-master output, then stop. Same window every call, so only the mute/solo
 *  config differs between measurements. */
async function measure(page: Page): Promise<{ mean: number; max: number }> {
  await page.evaluate(() => {
    const w = window as any;
    w.jotPlayer.seek(w.drumjot.jotEditorStore.tempo, 0.5);
  });
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => (window as any).jotPlayer.state), { timeout: 35_000 })
    .toBe('playing');
  await page.evaluate(() => (window as any).jotPlayer.startOutputCapture());
  await page.waitForTimeout(800);
  const samples = await page.evaluate(
    () => (window as any).jotPlayer.stopOutputCapture() as { t: number; rms: number }[]
  );
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect.poll(() => page.evaluate(() => (window as any).jotPlayer.state)).toBe('idle');
  return summarise(samples);
}

// ---- tests -----------------------------------------------------------------

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
