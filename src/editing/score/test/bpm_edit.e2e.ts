import { expect, test, type Page } from '@playwright/test';

/**
 * BPM editing in the timeline header (PDB-44): inline-editable bpm pills, the
 * right-click "Change BPM here" / "Delete BPM change" context menu, and the
 * underlying reactive `tempoEvents` / initial-bpm mutations.
 */

// Two bars: initial 120, then a flat change to 140 on bar 2's downbeat.
const JOT = `{{ bpm: 120, time: "4/4", title: "BPM edit" }}
| k s k s |
{{ bpm: 140 }}
| k s k s |`;

async function load(page: Page) {
  await page.goto('/');
  await page.waitForFunction(() => typeof (window as any).drumjot?.loadDsl === 'function');
  await page.evaluate((src) => (window as any).drumjot.loadDsl(src), JOT);
  await page.waitForSelector('[data-testid^="instrument-track-"]');
}

/** The flat tempo-event bpms currently in the reactive document. */
async function eventBpms(page: Page): Promise<number[]> {
  return page.evaluate(() =>
    [...(window as any).drumjot.jotEditorStore.jot.tempoEvents.values()].map((e: any) => e.bpm)
  );
}

async function initialBpm(page: Page): Promise<number> {
  return page.evaluate(() => (window as any).drumjot.jotEditorStore.jot.bpm);
}

test('edits an existing bpm pill in place', async ({ page }) => {
  await load(page);
  const pill = page.getByTestId('bpm-pill').filter({ hasText: '140 bpm' });
  await expect(pill).toHaveCount(1);
  await pill.click();
  const input = page.getByTestId('bpm-pill-input');
  await expect(input).toBeVisible();
  await input.fill('165');
  await input.press('Enter');

  // The single tempo event mutated in place (no second event added).
  expect(await eventBpms(page)).toEqual([165]);
  await expect(page.getByTestId('bpm-pill').filter({ hasText: '165 bpm' })).toHaveCount(1);
});

test('clamps an edit to the 20-400 range', async ({ page }) => {
  await load(page);
  const pill = page.getByTestId('bpm-pill').filter({ hasText: '120 bpm' });
  await pill.click();
  const input = page.getByTestId('bpm-pill-input');
  await input.fill('9999');
  await input.press('Enter');
  expect(await initialBpm(page)).toBe(400);
});

test('clearing an event pill deletes the tempo change', async ({ page }) => {
  await load(page);
  await page.getByTestId('bpm-pill').filter({ hasText: '140 bpm' }).click();
  const input = page.getByTestId('bpm-pill-input');
  await input.fill('');
  await input.press('Enter');
  expect(await eventBpms(page)).toEqual([]);
});

test('right-click → "Change BPM here" drops a new editable change', async ({ page }) => {
  await load(page);
  expect((await eventBpms(page)).length).toBe(1);

  // Right-click empty header space (well past the sticky gutter, between pills).
  await page.getByTestId('timeline-bars-row').click({ button: 'right', position: { x: 320, y: 14 } });
  await page.getByTestId('bpm-menu-change').click();

  // The new pill mounts in edit mode; type its value and commit.
  const input = page.getByTestId('bpm-pill-input');
  await expect(input).toBeVisible();
  await input.fill('99');
  await input.press('Enter');

  const bpms = await eventBpms(page);
  expect(bpms).toContain(99);
  expect(bpms.length).toBe(2);
});

test('right-click → "Delete BPM change" removes an event', async ({ page }) => {
  await load(page);
  await page
    .getByTestId('bpm-pill')
    .filter({ hasText: '140 bpm' })
    .click({ button: 'right' });
  await page.getByTestId('bpm-menu-delete').click();
  expect(await eventBpms(page)).toEqual([]);
});

test('the initial-tempo pill is editable but not deletable', async ({ page }) => {
  await load(page);
  await page
    .getByTestId('bpm-pill')
    .filter({ hasText: '120 bpm' })
    .click({ button: 'right' });
  // The menu offers Edit but no Delete for the song's base tempo.
  await expect(page.getByTestId('bpm-menu-edit')).toBeVisible();
  await expect(page.getByTestId('bpm-menu-delete')).toHaveCount(0);
  await page.getByTestId('bpm-menu-edit').click();
  const input = page.getByTestId('bpm-pill-input');
  await input.fill('128');
  await input.press('Enter');
  expect(await initialBpm(page)).toBe(128);
});
