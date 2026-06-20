import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * Black-box coverage of the editable Note properties panel: it reflects the
 * selection's fields, edits write through to the score, per-lane modifier
 * gating + the Roll conflict apply, and a mixed multi-selection shows `--` /
 * the count hint. Selection is driven by clicking note glyphs (ctrl-click to
 * extend), the way a user does it.
 */

// h has two notes (beats 1,2), s one (beat 3), k one (beat 4).
const SONG =
  '{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"}, s:{name:"Snare"}, k:{name:"Kick"} } }} | h h s k |';

async function load(page: Page): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(
    () => typeof (window as unknown as { drumjot?: { loadDsl?: unknown } }).drumjot?.loadDsl === 'function'
  );
  await page.evaluate(
    (s) => (window as unknown as { drumjot: { loadDsl(x: string): void } }).drumjot.loadDsl(s),
    SONG
  );
  await page.waitForSelector('[data-note-id]');
  await page.getByTestId('sidebar-item-note_properties').click();
  await expect(page.getByTestId('note-properties-panel')).toBeVisible();
}

function laneNote(page: Page, lane: string, nth = 0): Locator {
  return page.locator(`[data-testid="instrument-track-${lane}"] [data-note-id]`).nth(nth);
}

test.beforeEach(async ({ page }) => {
  await load(page);
});

test('with nothing selected the panel shows the empty hint', async ({ page }) => {
  await expect(page.getByTestId('note-properties-empty')).toBeVisible();
  await expect(page.getByTestId('note-properties')).toHaveCount(0);
});

test('selecting a note shows its id, lane, and bar/beat', async ({ page }) => {
  const id = (await laneNote(page, 's').getAttribute('data-note-id'))!;
  await laneNote(page, 's').click();
  await expect(page.getByTestId('note-properties-id')).toHaveText(`id: ${id}`);
  await expect(page.getByTestId('np-lane')).toHaveValue('s');
  await expect(page.getByTestId('np-barbeat-bar')).toHaveValue('1');
  await expect(page.getByTestId('np-barbeat-beat')).toHaveValue('3'); // snare on beat 3
});

test('changing the lane re-homes the note to the new instrument row', async ({ page }) => {
  const id = (await laneNote(page, 's').getAttribute('data-note-id'))!;
  await laneNote(page, 's').click();
  await page.getByTestId('np-lane').selectOption('k');
  // The same note now lives in the kick row and is gone from the snare row.
  await expect(page.locator(`[data-testid="instrument-track-k"] [data-note-id="${id}"]`)).toHaveCount(1);
  await expect(page.locator(`[data-testid="instrument-track-s"] [data-note-id="${id}"]`)).toHaveCount(0);
});

test('stepping the volume changes the value', async ({ page }) => {
  await laneNote(page, 's').click();
  const input = page.getByTestId('np-volume-input');
  const before = Number(await input.inputValue());
  await page.getByLabel('Volume: increase').click();
  await expect(input).toHaveValue(String(before + 1));
});

test('the articulation dropdown summarises selected options and toggles them', async ({ page }) => {
  await laneNote(page, 's').click();
  // Collapsed: nothing on yet.
  await expect(page.getByTestId('np-articulation-summary')).toHaveText('None');
  await page.getByTitle('Articulation').click();
  const rimshot = page.getByTestId('np-modifier-r');
  await expect(rimshot).not.toBeChecked();
  await rimshot.click();
  await expect(rimshot).toBeChecked();
  // The collapsed summary now lists it.
  await expect(page.getByTestId('np-articulation-summary')).toHaveText('Rimshot');
});

test('modifiers irrelevant to the lane are disabled', async ({ page }) => {
  await laneNote(page, 'h').click(); // hi-hat: open valid, rimshot not
  await page.getByTitle('Articulation').click();
  await expect(page.getByTestId('np-modifier-o')).toBeEnabled();
  await expect(page.getByTestId('np-modifier-r')).toBeDisabled();
});

test('enabling Roll disables the modifiers it conflicts with', async ({ page }) => {
  await laneNote(page, 's').click();
  await page.getByTitle('Articulation').click();
  await expect(page.getByTestId('np-modifier-fl')).toBeEnabled();
  await page.getByTestId('np-roll').click();
  await expect(page.getByTestId('np-roll')).toBeChecked();
  await expect(page.getByTestId('np-modifier-fl')).toBeDisabled();
});

test('setting sticking marks the radio option', async ({ page }) => {
  await laneNote(page, 's').click();
  const right = page.getByTestId('np-sticking-r');
  await expect(right).toHaveAttribute('aria-checked', 'false');
  await right.click();
  await expect(right).toHaveAttribute('aria-checked', 'true');
});

test('a mixed multi-selection shows the count hint and -- for differing fields', async ({ page }) => {
  await laneNote(page, 'h', 0).click();
  await laneNote(page, 'h', 1).click({ modifiers: ['ControlOrMeta'] });
  await expect(page.getByTestId('note-properties-id')).toHaveText('(multiple notes selected)');
  // Same lane + bar, differing beats -> beat segment blank (placeholder --).
  await expect(page.getByTestId('np-lane')).toHaveValue('h');
  await expect(page.getByTestId('np-barbeat-bar')).toHaveValue('1');
  await expect(page.getByTestId('np-barbeat-beat')).toHaveValue('');
});
