import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';

const TONE_WAV = fileURLToPath(new URL('./fixtures/tone.wav', import.meta.url));

/**
 * Visual capture of the debug-details timing visualization. The
 * popover only renders the diff-row layout we care about screenshotting
 * when the selected note has a matching `NoteProvenanceEntry`, which
 * normally only ships inside a transcriber debug bundle. We synthesize
 * the minimum state here (rock loop + tone fixture + a hand-rolled
 * provenance entry on the first hi-hat note) so the test boots fast,
 * captures the new layout, and never relies on the transcriber backend.
 */
test('captures the debug-details timing visualization', async ({ page }) => {
  // The popover is wide; give it a viewport that won't clip it on the
  // left when the selected note sits a few hundred px from the gutter.
  await page.setViewportSize({ width: 1600, height: 900 });
  await page.goto('/');
  await page.evaluate(() =>
    (window as unknown as { drumjot: { loadTestJot(): void } }).drumjot.loadTestJot()
  );
  await expect(page.locator('h2')).toContainText('Simple rock loop');
  await page.waitForSelector('[data-testid="instrument-row-h"]');

  // Load the 0.5 s tone fixture as the only audio track. We set files
  // directly on the hidden `<input type="file" accept="audio/*,.flac">`
  // instead of walking the toolbar's File → Load → Load audio track(s)
  // submenu chain, same code path, no submenu plumbing in the test.
  await page
    .locator('input[type=file][accept="audio/*,.flac"][multiple]')
    .setInputFiles(TONE_WAV);
  await expect
    .poll(() =>
      page.evaluate(() => (window as any).jotPlayer.audioTracks.size as number)
    )
    .toBe(1);

  // Tag the first hi-hat note with a known MIDI tick and seed a
  // matching provenance file. Mutating the plain `Note` source isn't
  // MobX-observable, but the subsequent `noteProvenance =` assignment
  // is, and that triggers the re-render where the new tick is read.
  // Values are picked so every pipeline stage emits a visible diff row:
  // a non-zero envelope refine, both coarse + fine beat alignment, all
  // four quantise passes, and a MIDI snap residual.
  await page.evaluate(() => {
    const store = (window as any).drumjot.store;
    const jot = store.currentJot;
    if (!jot) throw new Error('no rendered jot');
    // Walk the structural cache: voices → bars → tracks[pitch] → notes,
    // matching how the renderer locates notes (see `src/jot.ts`).
    // Pick a hi-hat note well past the gutter so the centered popover
    // doesn't clip on the left edge of the viewport. Falls back to the
    // first available `h` note if there aren't enough.
    const hiHats: any[] = [];
    for (const voice of jot.structure.voices) {
      for (const bar of voice.bars) {
        const track = bar.tracks?.h;
        if (!track) continue;
        for (const n of track.notes) hiHats.push(n);
      }
    }
    if (hiHats.length === 0) throw new Error('no hi-hat note found in current jot');
    const targetNote = hiHats[Math.min(6, hiHats.length - 1)];
    const sourceNote = targetNote.source;
    sourceNote.metadata = {
      ...(sourceNote.metadata ?? {}),
      midi: { ...(sourceNote.metadata?.midi ?? {}), tick: 100 },
    };

    store.noteProvenance = {
      format: 3,
      lead_bars: 0,
      beat_alignment_offset_sec: 0.011,
      beat_align_coarse_offset_sec: 0.006,
      beat_align_fine_offset_sec: 0.005,
      per_pitch: {
        h: [
          {
            pitch: 'h',
            midi_note: 42,
            tick: 100,
            detected_time_sec: 0.512,
            raw_model_time_sec: 0.498,
            quantised_time_sec: 0.531,
            quantised_shift_slots: 2,
            geometric_shift_slots: 1,
            envelope_shift_slots: 1,
            grid_shift_slots: 0,
            llm_shift_slots: 0,
            quantised_residual_slots: 0.18,
            off_grid: false,
            strength: 0.82,
            bar: 0,
            beat_in_bar: 1,
            out_of_range: false,
            kept: true,
            rejected_by: null,
          },
        ],
      },
    };
  });

  // Click the same hi-hat note we tagged with provenance. The DOM
  // doesn't carry the structural note's identity, so we match by tick:
  // the rendered note's `source` is the same `Note` object we mutated,
  // and the data-* attributes carry the beat, but the easiest path is
  // to click the Nth `[data-noseek="true"]` inside the hi-hat row, where
  // N matches the index we picked above (6, capped to row length).
  await page.evaluate(() => {
    const row = document.querySelector('[data-testid="instrument-row-h"]')!;
    const notes = row.querySelectorAll('[data-noseek="true"]');
    if (notes.length === 0) throw new Error('no hi-hat note in row');
    const idx = Math.min(6, notes.length - 1);
    (notes[idx] as HTMLElement).click();
  });
  await page.waitForSelector('[data-note-label-open]');

  // The label panel is a descendant of the selected note (the one
  // carrying `data-note-label-open`); its class includes the CSS-modules
  // mangled `noteLabel` substring. Resolve to that element so the
  // screenshot frames the popover itself, not the 14 px note marker.
  // The Debug details toggle starts expanded for kept notes
  // (`<NoteProvenanceDetails … startOpen />`), so the timing-viz mounts
  // as soon as the popover opens; no extra click required.
  await page.waitForSelector('[data-note-label-open] [class*="noteLabel"]');
  // Wait until the off-thread waveform compute has painted something
  // into the timing-viz canvas; otherwise we race the screenshot.
  await page.waitForFunction(() => {
    const label = document.querySelector(
      '[data-note-label-open] [class*="noteLabel"]'
    );
    if (!label) return false;
    const canvas = label.querySelector('canvas') as HTMLCanvasElement | null;
    return !!canvas && canvas.width > 0 && canvas.height > 0;
  });

  const popover = page
    .locator('[data-note-label-open] [class*="noteLabel"]')
    .first();
  await expect(popover).toBeVisible();

  // Real coverage for the redesigned timing-visualization:
  //
  // 1. Multiple diff rows render. The seeded provenance has non-zero
  //    deltas at every pipeline stage that produces a row (envelope
  //    refine, beat-align coarse + fine, three quantise passes, MIDI
  //    snap, bar-anchor drift, unknown-drift residual), so a healthy
  //    layout shows at least five. The exact count depends on the
  //    chain math and may shift; `>= 5` is the floor that proves the
  //    multi-row stack is rendering, not collapsing to one bar.
  // 2. Every diff row carries the percentage-based `--bar-left` /
  //    `--bar-width` CSS vars the inverted-text clip depends on. If
  //    these regressed to pixels (or were dropped), the inverted
  //    overlay would no longer track the colored bar.
  // 3. The inverted-text twin layer exists for each row. The overlay
  //    is what keeps the label legible where the colored bar paints
  //    over it; without this element the redesign's whole readability
  //    story is gone.
  const diffRows = popover.locator('[class*="timingVizDiffRow"]').filter({
    hasNot: page.locator('[class*="timingVizDiffRowText"]'),
  });
  const diffRowCount = await diffRows.count();
  expect(diffRowCount).toBeGreaterThanOrEqual(5);

  const cssVars = await diffRows.evaluateAll((els) =>
    els.map((el) => ({
      barLeft: (el as HTMLElement).style.getPropertyValue('--bar-left'),
      barWidth: (el as HTMLElement).style.getPropertyValue('--bar-width'),
    }))
  );
  for (const { barLeft, barWidth } of cssVars) {
    expect(barLeft).toMatch(/^-?\d+(\.\d+)?%$/);
    expect(barWidth).toMatch(/^\d+(\.\d+)?%$/);
  }

  const invertedCount = await popover
    .locator('[class*="timingVizDiffRowTextInverted"]')
    .count();
  expect(invertedCount).toBe(diffRowCount);

  await popover.screenshot({ path: 'e2e/debug-details-popover.png' });
});
