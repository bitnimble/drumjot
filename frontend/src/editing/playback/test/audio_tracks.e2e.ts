import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

const TONE_WAV = fileURLToPath(
  new URL('../../../../tests/fixtures/tone.wav', import.meta.url),
);

/**
 * Coverage for the audio-track feature. Loading + decoding + waveform +
 * mute/solo are CDN-independent (only `decodeAudioData` on the local
 * fixture, no smplr), so these are the reliable core. The play-sync
 * assertion at the end needs the smplr sample CDN and is written
 * tolerantly — see its comment.
 *
 * Audio tracks have no fixed `music`/`drums` ids — every load appends a
 * track with a fresh id (`track-1`, `track-2`, …) in load order — so the
 * helper returns the id the player allocated for the file it just fed.
 */

function trackIds(page: Page): Promise<string[]> {
  return page.evaluate(() =>
    Array.from((window as any).jotPlayer.audioTracks.keys()),
  );
}

/** Load the built-in `Simple rock loop` example from the empty-state
 *  picker and wait for the score to render. The toolbar (and thus the
 *  File → Load submenu the audio-track loader lives in) only exists once a
 *  jot is loaded. Mirrors `lyrics.e2e.ts::loadRockLoop`. */
async function loadRockLoop(page: Page): Promise<void> {
  await page.goto('/');
  await page.getByRole('button', { name: 'Simple rock loop' }).click();
  await page.waitForSelector('[data-testid^="instrument-track-"]');
}

/**
 * Walk File → Load → "Load audio track(s)", feed the fixture into the file
 * chooser, then wait for the async decode to land a *new* track on the
 * player and return its id. The change handler is fire-and-forget (decode
 * is async), so the track map only grows a tick or two after `setFiles`
 * resolves.
 */
async function loadAudioTrack(page: Page): Promise<string> {
  const before = new Set(await trackIds(page));
  await page.getByRole('button', { name: 'File', exact: true }).click();
  await page.getByRole('button', { name: 'Load', exact: true }).click();
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByRole('button', { name: 'Load audio track(s)' }).click(),
  ]);
  await chooser.setFiles(TONE_WAV);
  await expect
    .poll(async () => (await trackIds(page)).filter((id) => !before.has(id)).length)
    .toBe(1);
  const id = (await trackIds(page)).find((x) => !before.has(x))!;
  await expect
    .poll(() =>
      page.evaluate((tid) => {
        const t = (window as any).jotPlayer.audioTracks.get(tid);
        return t ? t.buffer.duration : 0;
      }, id),
    )
    .toBeGreaterThan(0.4); // 0.5s tone fixture
  return id;
}

/** Open a track's ⋯ overflow menu and click "Remove track". The menu
 *  panel is portaled out of the row, so the trigger is scoped to the
 *  track's row (every `tone.wav` track shares the same label) while the
 *  "Remove track" item is reached by its unique global testid. */
async function removeTrack(page: Page, id: string): Promise<void> {
  await page
    .getByTestId(`audio-track-row-${id}`)
    .locator('button[title^="More actions"]')
    .click();
  await page.getByTestId(`audio-track-clear-${id}`).click();
}

test('a loaded audio track appears in the Layers panel (in ordering, groupable)', async ({ page }) => {
  await loadRockLoop(page);
  await loadAudioTrack(page);
  await page.getByTestId('sidebar-item-layers').click();
  await expect(page.getByTestId('layers-tree')).toBeVisible();
  // The audio track is now a first-class track in `ordering`, so it shows in
  // the panel as an audio row (groupable with instrument tracks via DnD).
  await expect(
    page.locator('[data-testid="layers-track"][data-track-kind="audio"]')
  ).toHaveCount(1);
});

test('loads a track: row appears, buffer decodes, waveform renders', async ({ page }) => {
  await loadRockLoop(page);
  await expect(page.locator('h2')).toContainText('Simple rock loop');

  const id = await loadAudioTrack(page);

  const row = page.getByTestId(`audio-track-row-${id}`);
  await expect(row).toBeVisible();
  // Label is the filename without its extension; the full name shows
  // on the second line.
  await expect(row.getByText('tone', { exact: true })).toBeVisible();
  await expect(row.getByText('tone.wav')).toBeVisible();

  // Waveform painted. The chunk canvas transfers control to the waveform
  // worker (`transferControlToOffscreen`), so the main thread can no
  // longer `getContext` it, and `drawImage` of the placeholder canvas
  // reads back blank. A Playwright element screenshot is compositor-level
  // though, so it captures what the worker actually painted; decode that
  // PNG through an `Image` (a normal bitmap, not a transferred canvas) and
  // count pixels that differ from the uniform background. A blank/unpainted
  // canvas would be uniform (~0 differing); a rendered waveform is not.
  const canvas = page.getByTestId(`audio-track-waveform-${id}`);
  await expect(canvas).toBeVisible();
  const shot = await canvas.screenshot();
  const dataUrl = `data:image/png;base64,${shot.toString('base64')}`;
  const differing = await page.evaluate(async (url) => {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('waveform screenshot failed to decode'));
      img.src = url;
    });
    const off = document.createElement('canvas');
    off.width = img.naturalWidth;
    off.height = img.naturalHeight;
    const ctx = off.getContext('2d')!;
    ctx.drawImage(img, 0, 0);
    const { data } = ctx.getImageData(0, 0, off.width, off.height);
    const [br, bg, bb] = [data[0], data[1], data[2]];
    let count = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (
        Math.abs(data[i] - br) > 16 ||
        Math.abs(data[i + 1] - bg) > 16 ||
        Math.abs(data[i + 2] - bb) > 16
      ) {
        count++;
      }
    }
    return count;
  }, dataUrl);
  expect(differing).toBeGreaterThan(0);
});

test('mute/solo on a track flips audibility live', async ({ page }) => {
  await loadRockLoop(page);
  const a = await loadAudioTrack(page);

  // Assert via the store's public audibility API, not player internals.
  const audible = (id: string) =>
    page.evaluate((tid) => (window as any).drumjot.mixer.isAudioTrackAudible(tid), id);

  await expect.poll(() => audible(a)).toBe(true);
  await page.getByTestId(`audio-track-mute-${a}`).click();
  await expect.poll(() => audible(a)).toBe(false);
  await page.getByTestId(`audio-track-mute-${a}`).click();
  await expect.poll(() => audible(a)).toBe(true);

  // Solo a second track (with no other solo) should exclude the first.
  const b = await loadAudioTrack(page);
  await page.getByTestId(`audio-track-solo-${b}`).click();
  await expect.poll(() => audible(a)).toBe(false);
  await expect.poll(() => audible(b)).toBe(true);
});

test('multiple tracks load independently and clear individually', async ({ page }) => {
  await loadRockLoop(page);
  const a = await loadAudioTrack(page);
  const b = await loadAudioTrack(page);
  const c = await loadAudioTrack(page);

  expect(new Set([a, b, c]).size).toBe(3); // distinct ids
  await expect(page.getByTestId(`audio-track-row-${a}`)).toBeVisible();
  await expect(page.getByTestId(`audio-track-row-${b}`)).toBeVisible();
  await expect(page.getByTestId(`audio-track-row-${c}`)).toBeVisible();

  await removeTrack(page, b);
  await expect(page.getByTestId(`audio-track-row-${b}`)).toHaveCount(0);
  await expect(page.getByTestId(`audio-track-row-${a}`)).toBeVisible();
  await expect(page.getByTestId(`audio-track-row-${c}`)).toBeVisible();
});

test('playback starts with a track loaded and stops cleanly', async ({ page }) => {
  await loadRockLoop(page);
  await loadAudioTrack(page);

  // `exact` so this doesn't also match the "Playback" toolbar menu button;
  // the transport play/pause toggle is aria-labelled exactly "Play" when idle.
  await page.getByRole('button', { name: 'Play', exact: true }).click();

  // Reaching 'playing' requires the smplr TR-808 samples to fetch from
  // smpldsnds.github.io (GitHub Pages). In a network-restricted box
  // this can fail with a load error — that is a real environment
  // signal, not a flaky test, so we assert it directly with a timeout
  // longer than the player's 30s load budget rather than masking it.
  await expect
    .poll(() => page.evaluate(() => (window as any).jotPlayer.state), {
      timeout: 35_000,
    })
    .toBe('playing');

  const t1 = await page.evaluate(() => (window as any).jotPlayer.currentTime);
  await page.waitForTimeout(500);
  const t2 = await page.evaluate(() => (window as any).jotPlayer.currentTime);
  expect(t2).toBeGreaterThanOrEqual(t1);

  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => (window as any).jotPlayer.state))
    .toBe('idle');
});
