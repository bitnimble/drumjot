import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

const TONE_WAV = fileURLToPath(new URL('./fixtures/tone.wav', import.meta.url));

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

/**
 * Open the header "Load" dropdown, click the single audio-track menu
 * item, feed the fixture into the file chooser, then wait for the async
 * decode to land a *new* track on the player and return its id. The
 * change handler is fire-and-forget (decode is async), so the track map
 * only grows a tick or two after `setFiles` resolves.
 */
async function loadAudioTrack(page: Page): Promise<string> {
  const before = new Set(await trackIds(page));
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

test('loads a track: row appears, buffer decodes, waveform renders', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('h2')).toContainText('Simple rock loop');

  const id = await loadAudioTrack(page);

  const row = page.getByTestId(`audio-track-row-${id}`);
  await expect(row).toBeVisible();
  // Label is the filename without its extension; the full name shows
  // on the second line.
  await expect(row.getByText('tone', { exact: true })).toBeVisible();
  await expect(row.getByText('tone.wav')).toBeVisible();

  // Waveform canvas painted: some non-transparent pixels in the early
  // portion where the 0.5s tone sits (rest of the 4s timeline is silent).
  const canvas = page.getByTestId(`audio-track-waveform-${id}`);
  await expect(canvas).toBeVisible();
  const painted = await canvas.evaluate((el: HTMLCanvasElement) => {
    const ctx = el.getContext('2d');
    if (!ctx) return 0;
    const { data } = ctx.getImageData(0, 0, el.width, el.height);
    let nonTransparent = 0;
    for (let i = 3; i < data.length; i += 4) if (data[i] !== 0) nonTransparent++;
    return nonTransparent;
  });
  expect(painted).toBeGreaterThan(0);
});

test('mute/solo on a track flips audibility live', async ({ page }) => {
  await page.goto('/');
  const a = await loadAudioTrack(page);

  // Assert via the store's public audibility API, not player internals.
  const audible = (id: string) =>
    page.evaluate((tid) => (window as any).drumjot.store.isAudioTrackAudible(tid), id);

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
  await page.goto('/');
  const a = await loadAudioTrack(page);
  const b = await loadAudioTrack(page);
  const c = await loadAudioTrack(page);

  expect(new Set([a, b, c]).size).toBe(3); // distinct ids
  await expect(page.getByTestId(`audio-track-row-${a}`)).toBeVisible();
  await expect(page.getByTestId(`audio-track-row-${b}`)).toBeVisible();
  await expect(page.getByTestId(`audio-track-row-${c}`)).toBeVisible();

  await page.getByTestId(`audio-track-clear-${b}`).click();
  await expect(page.getByTestId(`audio-track-row-${b}`)).toHaveCount(0);
  await expect(page.getByTestId(`audio-track-row-${a}`)).toBeVisible();
  await expect(page.getByTestId(`audio-track-row-${c}`)).toBeVisible();
});

test('playback starts with a track loaded and stops cleanly', async ({ page }) => {
  await page.goto('/');
  await loadAudioTrack(page);

  await page.getByRole('button', { name: /Play/ }).click();

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

  await page.getByRole('button', { name: /Stop/ }).click();
  await expect
    .poll(() => page.evaluate(() => (window as any).jotPlayer.state))
    .toBe('idle');
});
