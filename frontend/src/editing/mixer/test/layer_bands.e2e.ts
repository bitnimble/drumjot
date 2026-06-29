import { expect, test, type Page } from '@playwright/test';

/**
 * The score renders instrument rows layer-first: one band per `||` layer,
 * each holding its per-track rows, so the same lane in two layers shows two
 * independent rows. A two-layer screenshot is written to /tmp for an eyeball;
 * the assertions cover the structure.
 */

async function load(page: Page, dsl: string): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(
    () => typeof (window as unknown as { drumjot?: { loadDsl?: unknown } }).drumjot?.loadDsl === 'function'
  );
  await page.evaluate(
    (s) => (window as unknown as { drumjot: { loadDsl(x: string): void } }).drumjot.loadDsl(s),
    dsl
  );
  await page.waitForSelector('[data-note-id]');
}

const META = '{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"}, s:{name:"Snare"}, k:{name:"Kick"} } }}';

test('single-layer song renders one (untinted) band', async ({ page }) => {
  await load(page, `${META} | k h s h |`);
  await expect(page.getByTestId('score-layer-v0')).toBeVisible();
  await expect(page.locator('[data-testid^="score-layer-"]')).toHaveCount(1);
});

test('lead-in caption renders on the topmost layer-first instrument row', async ({ page }) => {
  await load(
    page,
    '{{ songLeadIn: -3 }}\n{{ bpm: 120, time: "4/4", instrumentMapping: { k: { name: "Kick" } } }}\n| k . s . k . s . |'
  );
  await expect(page.getByText('lead-in')).toBeVisible();
});

test('muting one layer\'s snare leaves the other layer\'s snare audible (per-track)', async ({ page }) => {
  await load(page, `${META} | s s | || | s s |`);
  const probe = () =>
    page.evaluate(() => {
      const m = (window as unknown as { drumjot: { mixer: { isTrackAudible(k: string): boolean; mutedTracks: Set<string> } } }).drumjot.mixer;
      return { v0: m.isTrackAudible('v0/s'), v1: m.isTrackAudible('v1/s'), muted: [...m.mutedTracks] };
    });
  expect(await probe()).toEqual({ v0: true, v1: true, muted: [] });
  await page.evaluate(() =>
    (window as unknown as { drumjot: { mixerPresenter: { toggleMute(k: string): void } } }).drumjot.mixerPresenter.toggleMute('v0/s')
  );
  expect(await probe()).toEqual({ v0: false, v1: true, muted: ['v0/s'] });
});

test('the gutter M button on a layer row mutes that layer\'s track only', async ({ page }) => {
  await load(page, `${META} | s s | || | s s |`);
  // Click Mute on the snare row inside layer 2's band.
  await page.getByTestId('score-layer-v1').getByTitle('Mute s').click();
  const muted = await page.evaluate(
    () => [...(window as unknown as { drumjot: { mixer: { mutedTracks: Set<string> } } }).drumjot.mixer.mutedTracks]
  );
  expect(muted).toEqual(['v1/s']);
});

test('the View menu "Visually merge layers" toggle collapses same-lane rows', async ({ page }) => {
  // v0 = {HiHat, Snare}; v1 = {Snare, Kick}. Unmerged: two bands, two snare rows.
  await load(page, `${META} | h s | || | s k |`);
  await expect(page.locator('[data-testid^="score-layer-"]')).toHaveCount(2);
  await expect(page.locator('[data-testid="instrument-track-s"]')).toHaveCount(2);

  // Flip the toggle via the real View menu.
  await page.getByRole('button', { name: 'View' }).click();
  await page.getByText('Visually merge layers').click();
  await page.keyboard.press('Escape');

  // Merged: no layer bands, the two snares collapse to one row; h/k stay once.
  await expect(page.locator('[data-testid^="score-layer-"]')).toHaveCount(0);
  await expect(page.locator('[data-testid="instrument-track-s"]')).toHaveCount(1);
  await expect(page.locator('[data-testid="instrument-track-h"]')).toHaveCount(1);
  await expect(page.locator('[data-testid="instrument-track-k"]')).toHaveCount(1);
  await page.screenshot({ path: '/tmp/score_merged.png', fullPage: false });
});

test('two-||-layer song renders two bands; same lane appears in both', async ({ page }) => {
  // lane s in BOTH layers; v0 = {s}, v1 = {s, k}.
  await load(page, `${META} | s . s . | || | s . k . |`);
  await expect(page.locator('[data-testid^="score-layer-"]')).toHaveCount(2);
  await expect(page.getByTestId('score-layer-v0')).toBeVisible();
  await expect(page.getByTestId('score-layer-v1')).toBeVisible();
  // Two snare rows: one per layer.
  await expect(page.locator('[data-testid="instrument-track-s"]')).toHaveCount(2);
  // The kick lives only in layer 2's band.
  await expect(
    page.getByTestId('score-layer-v1').locator('[data-testid="instrument-track-k"]')
  ).toHaveCount(1);
  await expect(
    page.getByTestId('score-layer-v0').locator('[data-testid="instrument-track-k"]')
  ).toHaveCount(0);
  await page.screenshot({ path: '/tmp/score_two_layer.png', fullPage: false });
});
