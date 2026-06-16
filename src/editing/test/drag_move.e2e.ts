import { expect, test } from '@playwright/test';
import {
  disableSnapping,
  dragMouse,
  dragPreviews,
  laneCentreY,
  laneNotes,
  laneRowCentreY,
  loadRockJot,
  noteGeom,
  noteLeft,
  previewGeom,
  selectedNotes,
  selectionFrame,
} from './editing_e2e_utils';

/**
 * Black-box coverage of drag-moving notes. The drag is state-driven: while
 * held, the real glyph hides and a top-down preview placeholder is rendered on
 * whichever lane row the cursor is over (no DOM is measured to position it).
 * These specs assert the live preview mid-drag AND the committed result on
 * release. Snapping is disabled so they exercise raw free-drag geometry (grid
 * snapping is covered in snapping.e2e.ts).
 */

test.beforeEach(async ({ page }) => {
  await loadRockJot(page);
  await disableSnapping(page);
});

test('dragging a note horizontally moves it within its lane', async ({ page }) => {
  const n = await noteGeom(page, 'h', 1);
  await dragMouse(page, { x: n.x, y: n.y }, { x: n.x + 70, y: n.y });
  const left = (await noteLeft(page, n.id))!;
  expect(left - n.left).toBeGreaterThan(40);
});

test('dragging a note onto another lane re-homes it there', async ({ page }) => {
  const src = await noteGeom(page, 'h', 1);
  const kY = await laneCentreY(page, 'k');
  await dragMouse(page, { x: src.x, y: src.y }, { x: src.x, y: kY });

  // The same note now lives in the kick row and no longer in the hi-hat row.
  await expect(
    page.locator(`[data-testid="instrument-track-k"] [data-note-id="${src.id}"]`)
  ).toHaveCount(1);
  await expect(
    page.locator(`[data-testid="instrument-track-h"] [data-note-id="${src.id}"]`)
  ).toHaveCount(0);
});

test('the real glyph hides and a preview stands in for it while dragging', async ({ page }) => {
  const n = await noteGeom(page, 'h', 1);
  await page.mouse.move(n.x, n.y);
  await page.mouse.down();
  await page.mouse.move(n.x + 30, n.y, { steps: 5 });
  // Mid-drag: the real note is hidden (visibility:hidden, so no box) and a
  // single preview placeholder represents it on the hi-hat row.
  await expect(page.locator(`[data-note-id="${n.id}"]`)).toBeHidden();
  await expect(dragPreviews(page, 'h')).toHaveCount(1);
  await page.mouse.up();
  // Released: the real glyph is back, the preview is gone.
  await expect(page.locator(`[data-note-id="${n.id}"]`)).toBeVisible();
  await expect(dragPreviews(page, 'h')).toHaveCount(0);
});

test('the drag preview is vertically centred on the row (no down-shift)', async ({ page }) => {
  const src = await noteGeom(page, 'h', 1);
  const hRowCentre = await laneRowCentreY(page, 'h');
  await page.mouse.move(src.x, src.y);
  await page.mouse.down();
  await page.mouse.move(src.x + 35, src.y, { steps: 5 });
  const p = await previewGeom(page, 'h');
  await page.mouse.up();
  expect(p).not.toBeNull();
  // The preview sits at the row's vertical centre, not dropped by half its
  // height (the regression the old transform-based drag had).
  expect(Math.abs(p!.y - hRowCentre)).toBeLessThan(8);
});

test('the drag preview follows the cursor onto each track it crosses', async ({ page }) => {
  const src = await noteGeom(page, 'h', 1);
  const rows = { h: src.y, s: await laneCentreY(page, 's'), k: await laneCentreY(page, 'k') };
  await page.mouse.move(src.x, src.y);
  await page.mouse.down();
  await page.mouse.move(src.x + 6, src.y, { steps: 2 }); // cross threshold → begin

  // Walk the cursor through every lane row; the preview must appear ONLY on the
  // row under the cursor (the gap the earlier DOM-measured fix didn't cover).
  for (const lane of ['h', 's', 'k', 's', 'h'] as const) {
    await page.mouse.move(src.x + 12, rows[lane], { steps: 3 });
    expect(await previewGeom(page, lane), `preview on ${lane}`).not.toBeNull();
    for (const other of ['h', 's', 'k'] as const) {
      if (other !== lane) {
        expect(await previewGeom(page, other), `no preview on ${other}`).toBeNull();
      }
    }
  }

  // Release over the kick row → committed there.
  await page.mouse.move(src.x + 12, rows.k, { steps: 3 });
  await page.mouse.up();
  await expect(
    page.locator(`[data-testid="instrument-track-k"] [data-note-id="${src.id}"]`)
  ).toHaveCount(1);
});

test('releasing a note drag over empty bars does not move the playhead', async ({ page }) => {
  const playheadTime = () => page.evaluate(() => (window as any).jotPlayer.currentTime as number);
  const before = await playheadTime();
  const n = await noteGeom(page, 'h', 1);
  // Drag right into empty bar space and release there; pre-fix this fell
  // through to click-to-seek and parked the playhead at the release x.
  await dragMouse(page, { x: n.x, y: n.y }, { x: n.x + 45, y: n.y });
  const after = await playheadTime();
  expect(after).toBeCloseTo(before, 3);
});

test('dragging a multi-note selection moves the whole group, preserving spacing', async ({ page }) => {
  const a = await noteGeom(page, 'h', 1);
  const b = await noteGeom(page, 'h', 3);
  const gap = b.left - a.left;

  await laneNotes(page, 'h').nth(1).click();
  await laneNotes(page, 'h').nth(3).click({ modifiers: ['Control'] });
  await expect(selectedNotes(page)).toHaveCount(2);

  // Drag the group by grabbing one of its members.
  await dragMouse(page, { x: a.x, y: a.y }, { x: a.x + 60, y: a.y });

  const aLeft = (await noteLeft(page, a.id))!;
  const bLeft = (await noteLeft(page, b.id))!;
  // Both shifted right...
  expect(aLeft - a.left).toBeGreaterThan(30);
  expect(bLeft - b.left).toBeGreaterThan(30);
  // ...by the same amount, so their spacing is preserved.
  expect(Math.abs(bLeft - aLeft - gap)).toBeLessThan(3);
  // Still a live multi-selection.
  await expect(selectedNotes(page)).toHaveCount(2);
  await expect(selectionFrame(page)).toBeVisible();
});

test('pressing the selection frame (not a notehead) moves the whole group', async ({ page }) => {
  const a = await noteGeom(page, 'h', 1);
  const b = await noteGeom(page, 'h', 3);
  const gap = b.left - a.left;

  await laneNotes(page, 'h').nth(1).click();
  await laneNotes(page, 'h').nth(3).click({ modifiers: ['Control'] });
  await expect(selectedNotes(page)).toHaveCount(2);
  await expect(selectionFrame(page)).toBeVisible();

  // Grab the FRAME at a point between the two selected notes, over the
  // intervening (unselected) hi-hat that the frame overlays. Without the frame
  // intercepting, this press would re-select that single note; instead it drags
  // the whole group.
  const mid = { x: (a.x + b.x) / 2, y: a.y };
  await dragMouse(page, mid, { x: mid.x + 60, y: mid.y });

  // Still the same two-note selection (the in-between note was never selected).
  await expect(selectedNotes(page)).toHaveCount(2);
  const aLeft = (await noteLeft(page, a.id))!;
  const bLeft = (await noteLeft(page, b.id))!;
  // Both shifted right by the same amount, spacing preserved.
  expect(aLeft - a.left).toBeGreaterThan(30);
  expect(bLeft - b.left).toBeGreaterThan(30);
  expect(Math.abs(bLeft - aLeft - gap)).toBeLessThan(3);
});

test('a multi-note drag previews every note on its target track (frame hidden)', async ({ page }) => {
  // Two hi-hats; drag the group onto the snare row. Both shift by the same one
  // row, so both previews land on the snare lane.
  const a = await noteGeom(page, 'h', 1);
  const sY = await laneCentreY(page, 's');
  await laneNotes(page, 'h').nth(1).click();
  await laneNotes(page, 'h').nth(3).click({ modifiers: ['Control'] });
  await expect(selectedNotes(page)).toHaveCount(2);

  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(a.x, sY, { steps: 6 });
  // Both previews on the snare row; none left on the hi-hat row.
  await expect(dragPreviews(page, 's')).toHaveCount(2);
  await expect(dragPreviews(page, 'h')).toHaveCount(0);
  // The dashed selection frame is suppressed mid-drag (the previews represent
  // the group), rather than stranded at the pre-drag bounding box.
  await expect(selectionFrame(page)).toHaveCount(0);
  await page.mouse.up();
  await expect(selectionFrame(page)).toBeVisible();
});
