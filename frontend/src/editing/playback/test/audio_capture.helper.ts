import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, type Page } from '@playwright/test';

/**
 * Shared helpers for the real-audio e2e specs (`audio_capture`, `audio_mix`).
 * They drive the app, then tap the player's page-master output via
 * `jotPlayer.startOutputCapture` (an AnalyserNode observing the node feeding
 * `ctx.destination`) and read back a windowed-RMS series to assert on actual
 * sound, non-silence, mute/solo levels, volume scaling, playback speed.
 *
 * Reaching `playing` needs the smplr TR-808 sample CDN (smpldsnds.github.io),
 * so these depend on it like `audio_tracks.e2e.ts::playback starts…`.
 */

/** Synthesise a mono 16-bit PCM sine WAV to a temp file. A few seconds long so
 *  the tone plays continuously through every capture window (the committed
 *  `tone.wav` is only 0.5s). */
export function makeSineWavFile(seconds: number, freq: number, amp: number): string {
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

export async function loadRockLoop(page: Page): Promise<void> {
  await page.goto('/');
  await page.getByRole('button', { name: 'Simple rock loop' }).click();
  await page.waitForSelector('[data-testid^="instrument-track-"]');
}

export function trackIds(page: Page): Promise<string[]> {
  return page.evaluate(() => Array.from((window as any).jotPlayer.audioTracks.keys()) as string[]);
}

/** Load an audio-track file via File → Load and return the player-allocated id. */
export async function loadAudioTrack(page: Page, file: string): Promise<string> {
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

export type Mix = {
  drumMute?: boolean;
  audioMute?: boolean;
  drumSolo?: boolean;
  audioSolo?: boolean;
};

/** Apply a full master mute/solo config idempotently via the real presenter
 *  toggles (solos first: soloing a section clears its mute, so mutes win). */
export async function applyMasters(page: Page, cfg: Mix): Promise<void> {
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

/** Clear ALL mute/solo state (masters + per-lane + per-track) back to "every
 *  row audible", so each measurement starts from a known baseline. */
export async function resetMix(page: Page): Promise<void> {
  await page.evaluate(() => {
    const m = (window as any).drumjot.mixer;
    const p = (window as any).drumjot.mixerPresenter;
    if (m.drumMasterSoloed) p.toggleDrumMasterSolo();
    if (m.audioMasterSoloed) p.toggleAudioMasterSolo();
    if (m.drumMasterMuted) p.toggleDrumMasterMute();
    if (m.audioMasterMuted) p.toggleAudioMasterMute();
    // Drum mute/solo are keyed per-track (`layerId/lane`); the sets already hold
    // those composite keys, so toggle them back off directly.
    [...m.soloedTracks].forEach((k: string) => p.toggleSolo(k));
    [...m.mutedTracks].forEach((k: string) => p.toggleMute(k));
    [...m.soloedAudioTracks].forEach((id: string) => p.toggleAudioTrackSolo(id));
    [...m.mutedAudioTracks].forEach((id: string) => p.toggleAudioTrackMute(id));
  });
}

// The per-track mute/solo/volume filter is keyed by `layerId/lane`, not the
// bare lane; these setters resolve the lane's owning layer (single-layer songs
// -> `v0/<lane>`) to build that key.

/** Idempotent per-lane / per-track solo + mute setters. */
export const setLaneSolo = (page: Page, lane: string, on: boolean) =>
  page.evaluate(
    ({ lane, on }) => {
      const dj = (window as any).drumjot;
      const key = `${dj.jotEditorStore.structural?.ownerLayerFor(lane) ?? 'v0'}/${lane}`;
      if (dj.mixer.soloedTracks.has(key) !== on) dj.mixerPresenter.toggleSolo(key);
    },
    { lane, on }
  );
export const setLaneMute = (page: Page, lane: string, on: boolean) =>
  page.evaluate(
    ({ lane, on }) => {
      const dj = (window as any).drumjot;
      const key = `${dj.jotEditorStore.structural?.ownerLayerFor(lane) ?? 'v0'}/${lane}`;
      if (dj.mixer.mutedTracks.has(key) !== on) dj.mixerPresenter.toggleMute(key);
    },
    { lane, on }
  );
export const setTrackMute = (page: Page, id: string, on: boolean) =>
  page.evaluate(
    ({ id, on }) => {
      const m = (window as any).drumjot.mixer;
      if (m.mutedAudioTracks.has(id) !== on)
        (window as any).drumjot.mixerPresenter.toggleAudioTrackMute(id);
    },
    { id, on }
  );

/** Volume setters (all linear gains; RMS scales with them). */
export const setMasterVolume = (page: Page, v: number) =>
  page.evaluate((v) => (window as any).jotPlayer.setMasterVolume(v), v);
export const setDrumMasterVolume = (page: Page, v: number) =>
  page.evaluate((v) => (window as any).jotPlayer.setDrumMasterVolume(v), v);
export const setAudioMasterVolume = (page: Page, v: number) =>
  page.evaluate((v) => (window as any).jotPlayer.setAudioTrackMasterVolume(v), v);
export const setLaneVolume = (page: Page, lane: string, v: number) =>
  page.evaluate(
    ({ lane, v }) => {
      const dj = (window as any).drumjot;
      const key = `${dj.jotEditorStore.structural?.ownerLayerFor(lane) ?? 'v0'}/${lane}`;
      dj.mixerPresenter.setTrackVolume(key, v);
    },
    { lane, v }
  );
export const setTrackVolume = (page: Page, id: string, v: number) =>
  page.evaluate(
    ({ id, v }) => (window as any).drumjot.mixerPresenter.setAudioTrackVolume(id, v),
    { id, v }
  );
export const setSpeed = (page: Page, speed: number) =>
  page.evaluate((s) => (window as any).jotPlayer.setPlaybackSpeed(s), speed);

export type RmsSample = { t: number; rms: number };

/** Mean RMS over the captured series (edges trimmed to drop start/stop
 *  transients) plus the peak. */
export function summarise(samples: RmsSample[]): { mean: number; max: number } {
  if (samples.length === 0) return { mean: 0, max: 0 };
  const trim = Math.floor(samples.length * 0.15);
  const core = samples.slice(trim, samples.length - trim);
  const xs = (core.length > 0 ? core : samples).map((s) => s.rms);
  return { mean: xs.reduce((a, b) => a + b, 0) / xs.length, max: Math.max(...xs) };
}

/** Count onset peaks in the RMS series: rising crossings above a fraction of
 *  the peak, with a refractory gap so one hit's decay isn't double-counted.
 *  Used to verify playback speed changes the audible onset density. */
export function countOnsets(samples: RmsSample[], frac = 0.4, refractoryMs = 60): number {
  if (samples.length === 0) return 0;
  const peak = Math.max(...samples.map((s) => s.rms));
  const thresh = peak * frac;
  let count = 0;
  let armed = true;
  let lastT = -Infinity;
  for (const s of samples) {
    if (armed && s.rms >= thresh && s.t - lastT > refractoryMs / 1000) {
      count++;
      armed = false;
      lastT = s.t;
    } else if (s.rms < thresh * 0.6) {
      armed = true;
    }
  }
  return count;
}

/** Seek to a beat well inside the musical content, play, capture `ms` of the
 *  page-master output, then stop. Same window every call, so only the mix
 *  config differs between measurements. */
export async function measure(page: Page, ms = 800): Promise<{ mean: number; max: number }> {
  const samples = await captureRaw(page, ms);
  return summarise(samples);
}

/** Like {@link measure} but returns the raw {t, rms} series (for onset rate). */
export async function captureRaw(page: Page, ms = 1500): Promise<RmsSample[]> {
  await page.evaluate(() => {
    const w = window as any;
    w.jotPlayer.seek(w.drumjot.jotEditorStore.tempo, 0.5);
  });
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => (window as any).jotPlayer.state), { timeout: 35_000 })
    .toBe('playing');
  await page.evaluate(() => (window as any).jotPlayer.startOutputCapture());
  await page.waitForTimeout(ms);
  const samples = await page.evaluate(
    () => (window as any).jotPlayer.stopOutputCapture() as RmsSample[]
  );
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect.poll(() => page.evaluate(() => (window as any).jotPlayer.state)).toBe('idle');
  return samples;
}
