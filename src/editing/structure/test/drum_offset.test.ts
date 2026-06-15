/**
 * Tests for the drum beat-grid offset (`StructuralPresenter.drumOffsetBeats`):
 * sliding every drum note across the bars to realign a consistently
 * mis-detected groove. The bar grid stays fixed, so the shift shows up both
 * as moved note positions in the structural layers and as shifted times out
 * of `jotToEvents`.
 */
import { describe, expect, it } from 'bun:test';
import { parse } from 'src/schema/dsl/parser/parser';
import { buildStructural } from 'src/editing/jot_editor_store';
import { StructuralPresenter } from 'src/editing/structure/structural_presenter';
import { jotToEvents } from 'src/editing/playback/events';

// Two 4/4 bars at 120 BPM (0.5 s/beat, 2.0 s/bar), one kick on each
// downbeat: abs beats 0 and 4, event times 0.0 and 2.0 s.
const SRC =
  '{{ bpm: 120, time: "4/4", instrumentMapping: { k:{name:"Kick"} } }} | k . . . | k . . . |';

function render(offsetBeats: number): StructuralPresenter {
  const structural = buildStructural(parse(SRC));
  structural.setDrumOffset(offsetBeats);
  return structural;
}

describe('drum beat-grid offset', () => {
  it('is identity at offset 0', () => {
    const evs = jotToEvents(render(0));
    expect(evs.map((e) => e.time)).toEqual([0, 2]);
  });

  it('shifts notes later within the fixed bar grid (+1 beat)', () => {
    const rendered = render(1);
    // abs 0 -> bar 1 beat 1; abs 4 -> bar 2 beat 1.
    const bars = rendered.layers[0].bars;
    expect(bars[0].tracks['k'].notes[0].beat).toBeCloseTo(1, 6);
    expect(bars[1].tracks['k'].notes[0].beat).toBeCloseTo(1, 6);
    // One beat at 120 BPM = 0.5 s, so both events slide by 0.5 s.
    expect(jotToEvents(rendered).map((e) => e.time)).toEqual([0.5, 2.5]);
    // Bar count (and thus the timeline) is untouched.
    expect(bars).toHaveLength(2);
  });

  it('drops notes pushed before the first beat (-1 beat)', () => {
    const rendered = render(-1);
    // abs 0 -> -1 (dropped); abs 4 -> 3 (bar 1 beat 3).
    const evs = jotToEvents(rendered);
    expect(evs).toHaveLength(1);
    expect(evs[0].time).toBeCloseTo(1.5, 6);
    expect(rendered.layers[0].bars[0].tracks['k'].notes[0].beat).toBeCloseTo(3, 6);
  });

  it('keeps straightness for dyadic shifts', () => {
    const note = render(1.5).layers[0].bars[0].tracks['k'].notes[0];
    // abs 0 -> 1.5, still on the binary grid.
    expect(note.beat).toBeCloseTo(1.5, 6);
    expect(note.straight).toBe(true);
  });

  it('cancels the control value when the baseline matches it', () => {
    // Hydrates the way `applyDebugBundle` does after a transcriber run:
    // the control shows the alignment value, but baseline matching it
    // means the score still renders at the source positions.
    const rendered = buildStructural(parse(SRC));
    rendered.setDrumOffsetBaseline(0.5);
    rendered.setDrumOffset(0.5);
    expect(rendered.effectiveDrumOffsetBeats).toBe(0);
    expect(jotToEvents(rendered).map((e) => e.time)).toEqual([0, 2]);
  });

  it('shifts by the delta when the control diverges from the baseline', () => {
    // Reset-to-zero exposes the pre-alignment positions when a baseline
    // is set: effective offset = 0 - 0.5 = -0.5 beats earlier.
    const rendered = buildStructural(parse(SRC));
    rendered.setDrumOffsetBaseline(0.5);
    rendered.setDrumOffset(0);
    expect(rendered.effectiveDrumOffsetBeats).toBe(-0.5);
    // abs 0 -> -0.5 (dropped); abs 4 -> 3.5 in bar 1.
    const evs = jotToEvents(rendered);
    expect(evs).toHaveLength(1);
    expect(evs[0].time).toBeCloseTo(1.75, 6);
  });
});
