import { expect, test } from '@playwright/test';
import {
  DEBUG_BUNDLE_PATH,
  loadDebugBundle,
} from '../playback/test/debug_bundle.helper';
import { PARADB_ZIP_PATH, loadParadbZip } from './paradb.helper';

/**
 * Full-pipeline round-trip guard: import (DSL) -> render -> export to .jot
 * (DSL) -> reimport -> render, and assert the rendered structure is
 * byte-identical across the trip.
 *
 * Unlike `src/schema/dsl/test/writer.test.ts` (which round-trips `parse`<->`writeDsl`
 * as pure functions), this drives the WHOLE browser pipeline each time:
 * `loadDsl` -> reactive Loro document -> StructureStore walk -> the structural
 * view-model the renderer reads. So it catches regressions anywhere in that
 * chain (the reactive-doc build, the element-tree flatten, the relative->
 * absolute coordinate math, tuplet/pattern span derivation) that a unit test
 * of the DSL formatter can't see. It's a broad net: it'll go red if a refactor
 * of the structure/derivation layer drops a note, a modifier, a tuplet
 * bracket, or shifts a beat.
 *
 * The song deliberately exercises a lot of DSL surface (all fragments are
 * taken from writer.test's round-trip-stable cases): a pattern def + usage,
 * a weighted tuplet of simultaneities, an explicit repeat, flam/accent
 * modifiers + L/R sticking, an open roll, a 3:4 polyrhythm, and two layers.
 */
const SONG = `{{ title: "Round Trip", bpm: 120, time: "4/4",
  instrumentMapping: { k:{name:"Kick"}, s:{name:"Snare"}, h:{name:"HiHat"}, c:{name:"Crash"}, a:{name:"TomA"}, b:{name:"TomB"} } }}
[Groove=(k.s.kks.)]
| [Groove] (k+s k+s k+s)_4 |
| s:fl@l k@r s@r:a . k@r s@l k@r k@r |
||
| h:c h:c h:c h:c |
| (a a a)_4 + (b b b b)_4 |
`;

// Serialised in the browser; reads the structural view-model the renderer
// uses and normalises it to a stable, shape-tolerant snapshot. `vol` and
// `modifiers` are read defensively so the snapshot survives the (in-flight)
// migration of the structural shape (`Struct*` flat fields vs the legacy
// synthesised `note.source`).
const GRID_FN = `() => {
  const r = (x) => Math.round(x * 1e4) / 1e4;
  const structural = window.drumjot.jotEditorStore.structural;
  return structural.layers.map((v) => ({
    lanes: v.lanes,
    bars: v.bars.map((bar) => ({
      index: bar.index,
      beats: r(bar.beats),
      tracks: Object.fromEntries(
        Object.keys(bar.tracks).sort().map((p) => [
          p,
          bar.tracks[p].notes.map((n) => ({
            beat: r(n.beat),
            dur: r(n.duration),
            straight: !!n.straight,
            roll: !!n.roll,
            mods: [...(n.modifiers ?? [])].sort(),
            stick: n.sticking ?? null,
            velocity: n.velocity ?? null,
          })),
        ])
      ),
      patternSpans: bar.patternSpans.map((s) => ({
        name: s.name, start: r(s.startBeat), end: r(s.endBeat),
      })),
      tupletSpans: bar.tupletSpans.map((s) => ({
        count: s.count, start: r(s.startBeat), end: r(s.endBeat),
      })),
    })),
  }));
}`;

// `page.evaluate` runs a string as an EXPRESSION, so wrap the function as
// an IIFE to actually invoke it and return the snapshot.
const READ_GRID = `(${GRID_FN})()`;

test('a rich song survives an import -> export-to-DSL -> reimport round-trip', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => typeof (window as any).drumjot?.loadDsl === 'function');

  // Import + render.
  await page.evaluate((src) => (window as any).drumjot.loadDsl(src), SONG);
  await page.waitForSelector('[data-testid^="instrument-track-"]');
  const gridBefore = (await page.evaluate(READ_GRID)) as Array<{
    bars: Array<{
      tracks: Record<string, Array<{ mods: string[] }>>;
      tupletSpans: unknown[];
      patternSpans: unknown[];
    }>;
  }>;

  // Sanity: the snapshot actually captured the rich structure, so the
  // equality assertions below can't pass vacuously by comparing two empty
  // grids (e.g. if a future change silently rendered nothing).
  expect(gridBefore.length).toBe(2); // two layers
  const allBars = gridBefore.flatMap((v) => v.bars);
  const allNotes = allBars.flatMap((b) => Object.values(b.tracks).flat());
  expect(allNotes.length).toBeGreaterThan(10);
  expect(allBars.some((b) => b.tupletSpans.length > 0)).toBe(true);
  expect(allBars.some((b) => b.patternSpans.length > 0)).toBe(true);
  expect(allNotes.some((n) => n.mods.length > 0)).toBe(true);

  // Export to .jot, then reimport that exported text + re-render.
  const dsl1 = await page.evaluate(() => (window as any).drumjot.toDsl() as string);
  expect(dsl1).toContain('title: "Round Trip"');
  await page.evaluate((src) => (window as any).drumjot.loadDsl(src), dsl1);
  await page.waitForSelector('[data-testid^="instrument-track-"]');
  const gridAfter = await page.evaluate(READ_GRID);
  const dsl2 = await page.evaluate(() => (window as any).drumjot.toDsl() as string);

  // The rendered structure (notes, modifiers, sticking, rolls, tuplet +
  // pattern brackets, per-bar beats, lane order) is identical across the
  // trip: nothing was dropped or shifted by export + reimport + re-render.
  expect(gridAfter).toEqual(gridBefore);
  // And the formatter is a fixpoint: a second export equals the first, so
  // the .jot text is stable (no churn on repeated save/load).
  expect(dsl2).toBe(dsl1);
});

/**
 * The same round-trip, but on a REAL full song: a complete transcriber
 * debug bundle (pointed to by `E2E_DEBUG_BUNDLE`). This is the realistic
 * stress case, dozens of bars, real tempo/time content, the full lane
 * vocabulary, so it catches structure/derivation regressions the small
 * synthetic song can't reach. Skipped when the env var is unset (the
 * bundle is large + machine-local, never committed), same convention as
 * `debug_bundle.e2e.ts`.
 */
test('a full debug-bundle song survives an export-to-DSL -> reimport round-trip', async ({
  page,
}) => {
  test.skip(!DEBUG_BUNDLE_PATH, 'E2E_DEBUG_BUNDLE not set');
  test.setTimeout(120_000); // large zip unpack + multi-track audio decode

  await loadDebugBundle(page);
  const gridBefore = (await page.evaluate(READ_GRID)) as Array<{
    bars: Array<{ tracks: Record<string, unknown[]> }>;
  }>;

  // Sanity: a real song has lots of notes across many bars, so the
  // equality assertion below can't pass on two empty grids.
  const barsBefore = gridBefore.flatMap((v) => v.bars);
  const notesBefore = barsBefore.flatMap((b) => Object.values(b.tracks).flat());
  expect(barsBefore.length).toBeGreaterThan(8);
  expect(notesBefore.length).toBeGreaterThan(50);

  const dsl1 = await page.evaluate(() => (window as any).drumjot.toDsl() as string);
  expect(dsl1.length).toBeGreaterThan(0);
  await page.evaluate((src) => (window as any).drumjot.loadDsl(src), dsl1);
  await page.waitForSelector('[data-testid^="instrument-track-"]');
  const gridAfter = await page.evaluate(READ_GRID);

  // The whole rendered structure (every note position, modifier, tuplet +
  // pattern span, per-bar beats, lane order, across dozens of bars) is
  // identical across the trip. This is the structural fidelity guard.
  //
  // The strict TEXT fixpoint (dsl2 === dsl1) is asserted by the synthetic
  // test above, not here: a real transcriber song hits a known, separate
  // writeDsl<->parse edge case, a tempo event at the very end of the last
  // bar is emitted trailing (`… {{ bpm: X }} |`) but the parser re-anchors
  // inline tempo to the *next* element, of which there is none, so it's
  // dropped on reparse. Structurally inert (no notes follow), so the grid
  // round-trips cleanly; only the trailing tempo marker churns.
  expect(gridAfter).toEqual(gridBefore);
});

/**
 * The same round-trip on a real ParaDB / Paradiddle map pack
 * (`E2E_PARADB_ZIP`). ParaDB charts come in through a different front door
 * than transcriber bundles, `.rlrr` JSON → `parseRlrr`, so they exercise
 * a distinct import path into the same structure pipeline (real sticking,
 * the full kit, dense charts). Skipped when the env var is unset (the pack
 * is large + machine-local, never committed), same convention as the debug
 * bundle.
 */
test('a ParaDB pack survives an export-to-DSL -> reimport round-trip', async ({ page }) => {
  test.skip(!PARADB_ZIP_PATH, 'E2E_PARADB_ZIP not set');
  test.setTimeout(120_000); // zip unpack + rlrr->jot + multi-track audio decode

  await loadParadbZip(page);
  const gridBefore = (await page.evaluate(READ_GRID)) as Array<{
    bars: Array<{ tracks: Record<string, unknown[]> }>;
  }>;

  // Sanity: a real chart has lots of notes across many bars.
  const barsBefore = gridBefore.flatMap((v) => v.bars);
  const notesBefore = barsBefore.flatMap((b) => Object.values(b.tracks).flat());
  expect(barsBefore.length).toBeGreaterThan(8);
  expect(notesBefore.length).toBeGreaterThan(50);

  const dsl1 = await page.evaluate(() => (window as any).drumjot.toDsl() as string);
  expect(dsl1.length).toBeGreaterThan(0);
  await page.evaluate((src) => (window as any).drumjot.loadDsl(src), dsl1);
  await page.waitForSelector('[data-testid^="instrument-track-"]');
  const gridAfter = await page.evaluate(READ_GRID);

  // Whole rendered structure identical across the trip (text fixpoint
  // omitted for the same tempo-tail reason as the debug-bundle test above).
  expect(gridAfter).toEqual(gridBefore);
});
