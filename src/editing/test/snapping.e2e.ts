import { expect, test } from '@playwright/test';
import {
  disableSnapping,
  dragMouse,
  enableSnapping,
  laneNoteIds,
  loadRockJot,
  noteGeom,
  noteLeft,
  previewGeom,
  setMode,
} from './editing_e2e_utils';

/**
 * Black-box coverage of grid snapping (Edit menu). When on, inserting and
 * moving notes snaps to the grid; positions are observed via the notes' on-
 * screen x. The hi-hat at index 1 sits on beat 1, a 16th-grid line, so it is a
 * convenient on-grid reference: a snapped note near it lands at the same x.
 */

test.beforeEach(async ({ page }) => {
  await loadRockJot(page);
});

test('Edit menu toggles the snapping checkmark (on by default)', async ({ page }) => {
  await page.getByRole('button', { name: 'Edit' }).click();
  const item = page.getByTestId('edit-menu-snapping');
  await expect(item).toHaveAttribute('aria-checked', 'true');
  await item.click();
  await expect(item).toHaveAttribute('aria-checked', 'false');
});

// Insert a snare note just past beat 3, an EMPTY grid line in the snare lane,
// so the click lands on bare bars-row (no glyph to intercept it). Returns the
// inserted note's left and the computed left edge of the beat-3 grid line.
// `frac16` is how far past beat 3 to click, in 16ths.
async function insertPastEmptyBeat3(page: import('@playwright/test').Page, frac16: number) {
  const s2 = await noteGeom(page, 's', 0); // beat 2
  const s6 = await noteGeom(page, 's', 1); // beat 6
  const pxPerBeat = (s6.x - s2.x) / 4;
  const sixteenth = pxPerBeat / 4;
  const beat3LeftEdge = s2.left + pxPerBeat; // snare is silent on beat 3
  const x = s2.x + pxPerBeat + sixteenth * frac16; // centre of beat 3 + offset
  const before = await laneNoteIds(page, 's');
  await setMode(page, 'insert');
  await page.mouse.move(x, s2.y);
  await page.mouse.click(x, s2.y);
  const after = await laneNoteIds(page, 's');
  const newId = after.find((id) => !before.includes(id))!;
  return { left: (await noteLeft(page, newId))!, beat3LeftEdge };
}

test('with snapping OFF an inserted note stays where it was clicked', async ({ page }) => {
  await disableSnapping(page);
  // ~0.45 of a 16th past beat 3, clearly off any grid line.
  const { left, beat3LeftEdge } = await insertPastEmptyBeat3(page, 0.45);
  expect(left - beat3LeftEdge).toBeGreaterThan(5);
});

test('with snapping ON an inserted note lands on the grid line', async ({ page }) => {
  await enableSnapping(page);
  // The same off-grid click snaps back onto the beat-3 grid line.
  const { left, beat3LeftEdge } = await insertPastEmptyBeat3(page, 0.45);
  expect(Math.abs(left - beat3LeftEdge)).toBeLessThan(2);
});

test('with snapping ON a small drag snaps back to the same grid line', async ({ page }) => {
  await enableSnapping(page);
  const ref = await noteGeom(page, 'h', 1);
  // Drag past the click threshold but less than half a 16th, snapping returns
  // it to its original grid line, so its x is unchanged.
  await dragMouse(page, { x: ref.x, y: ref.y }, { x: ref.x + 10, y: ref.y });
  const left = (await noteLeft(page, ref.id))!;
  expect(Math.abs(left - ref.left)).toBeLessThan(3);
});

test('with snapping ON the live drag preview jumps between grid lines', async ({ page }) => {
  await enableSnapping(page);
  // Hi-hats sit on every 8th (beats 0, 0.5, 1, …), so h[0] and h[2] are exactly
  // one beat apart, giving the on-screen px-per-beat and thus the 16th pitch.
  const h0 = await noteGeom(page, 'h', 0); // beat 0
  const h2 = await noteGeom(page, 'h', 2); // beat 1
  const sixteenth = (h2.x - h0.x) / 4;
  const ref = await noteGeom(page, 'h', 1); // on beat 0.5, an on-grid reference

  await page.mouse.move(ref.x, ref.y);
  await page.mouse.down();
  // Under half a 16th: the live preview snaps back to the original grid line.
  await page.mouse.move(ref.x + sixteenth * 0.4, ref.y, { steps: 4 });
  const near = await previewGeom(page, 'h');
  expect(Math.abs(near!.left - ref.left)).toBeLessThan(2);
  // Past half a 16th: it jumps a full grid step rather than moving continuously.
  await page.mouse.move(ref.x + sixteenth * 0.7, ref.y, { steps: 4 });
  const next = await previewGeom(page, 'h');
  expect(Math.abs(next!.left - (ref.left + sixteenth))).toBeLessThan(2);
  await page.mouse.up();
});

test('with snapping OFF the same small drag moves the note freely', async ({ page }) => {
  await disableSnapping(page);
  const ref = await noteGeom(page, 'h', 1);
  await dragMouse(page, { x: ref.x, y: ref.y }, { x: ref.x + 10, y: ref.y });
  const left = (await noteLeft(page, ref.id))!;
  expect(left - ref.left).toBeGreaterThan(4);
});
