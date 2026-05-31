import { expect, test, type Page } from '@playwright/test';

/**
 * Regression coverage for selection popovers being clipped / obscured by
 * the tracks below them.
 *
 * The score virtualizes bars with `content-visibility: auto` (→
 * `contain: paint`), which clips any descendant that overflows the bar's
 * box. A note's label popover (`.noteLabel`) hangs well below the note, * past the bar's `overflow-clip-margin` and into the next instrument
 * row, so without the `.bar:has([data-note-label-open])` opt-out
 * (score.module.css) it gets clipped and the row below shows through.
 * Separately the row carrying the popover is lifted above its siblings
 * via `:has([data-note-label-open])` (mixer.module.css) so it wins the
 * z-order. This spec asserts the end-to-end result: the popover is the
 * top, fully-painted element across its whole height, including the part
 * that overlaps the track beneath it.
 *
 * Visibility is asserted with `document.elementFromPoint`: the popover
 * sets `pointer-events: auto`, so if it is the painted-on-top element at a
 * point it is also the hit-test result there. A clipped or z-obscured
 * popover yields the underlying lane/bar instead.
 */

async function loadRockJot(page: Page): Promise<void> {
  await page.goto('/');
  // Boot starts on the empty state; load the built-in rock jot (h/s/k
  // instrument rows) through the exposed debug global.
  await page.evaluate(() => (window as unknown as { drumjot: { loadTestJot(): void } }).drumjot.loadTestJot());
  await expect(page.locator('h2')).toContainText('Simple rock loop');
  await page.waitForSelector('[data-testid^="instrument-row-"]');
}

/**
 * Click the first note in `rowTestId`, then return the popover geometry
 * plus, for a set of vertical sample points down the popover, whether the
 * popover (or one of its children) is the element painted on top there.
 */
async function selectNoteAndProbe(page: Page, rowTestId: string) {
  const noteCenter = await page.evaluate((rid) => {
    const row = document.querySelector(`[data-testid="${rid}"]`)!;
    const note = row.querySelector('[data-noseek="true"]')!;
    const r = note.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }, rowTestId);

  await page.mouse.click(noteCenter.x, noteCenter.y);
  await page.waitForSelector('[data-note-label-open]');

  return page.evaluate((rid) => {
    const row = document.querySelector(`[data-testid="${rid}"]`)!;
    const sel = document.querySelector('[data-note-label-open]')!;
    const label = Array.from(sel.querySelectorAll('*')).find((e) =>
      (e.getAttribute('class') ?? '').toLowerCase().includes('notelabel'),
    )!;
    const lr = label.getBoundingClientRect();
    const rowRect = row.getBoundingClientRect();

    const samples = [0.2, 0.5, 0.8, 0.95].map((frac) => {
      const px = lr.left + lr.width / 2;
      const py = lr.top + lr.height * frac;
      const top = document.elementFromPoint(px, py);
      return {
        frac,
        onTop: !!top && (top === label || label.contains(top)),
        // Is this sample point below the selected row (i.e. overlapping
        // the track beneath)? Those are the points that regress.
        belowRow: py > rowRect.bottom,
      };
    });

    return {
      labelTop: lr.top,
      labelBottom: lr.bottom,
      labelHeight: lr.height,
      rowBottom: rowRect.bottom,
      samples,
    };
  }, rowTestId);
}

test('note label popover paints over the track below it (top row)', async ({ page }) => {
  await loadRockJot(page);
  const probe = await selectNoteAndProbe(page, 'instrument-row-h');

  // Sanity: the popover must actually extend past its own row, otherwise
  // the "overlaps the track below" assertion is vacuous.
  expect(probe.labelBottom).toBeGreaterThan(probe.rowBottom);
  // At least one sample must fall over the track below.
  expect(probe.samples.some((s) => s.belowRow)).toBe(true);
  // Every sample down the popover must have the popover painted on top.
  for (const s of probe.samples) {
    expect(s.onTop, `popover on top at frac ${s.frac} (belowRow=${s.belowRow})`).toBe(true);
  }
});

test('note label popover paints over the track below it (middle row)', async ({ page }) => {
  await loadRockJot(page);
  const probe = await selectNoteAndProbe(page, 'instrument-row-s');

  expect(probe.labelBottom).toBeGreaterThan(probe.rowBottom);
  expect(probe.samples.some((s) => s.belowRow)).toBe(true);
  for (const s of probe.samples) {
    expect(s.onTop, `popover on top at frac ${s.frac} (belowRow=${s.belowRow})`).toBe(true);
  }
});
