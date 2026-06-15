/**
 * Tests for the view-only virtual lead-in on StructuralPresenter: the score
 * always shows at least one bar of lead-in so the first note never clips,
 * sized to the audio pre-roll (drumsT0Sec) when that already exceeds a bar.
 * The virtual bar is present only on `layers` (the view), never on
 * `musicalLayers` (export / playback / tempo).
 */
import { describe, expect, it } from 'bun:test';
import { parse } from 'src/schema/dsl/parser/parser';
import { buildStructural } from 'src/editing/jot_editor_store';
import { LEAD_IN_BAR_ID } from 'src/editing/structure/structure_store';

const META = '{{ bpm: 120, time: "4/4", instrumentMapping: { k:{name:"Kick"} } }}';

describe('virtual lead-in (view layer)', () => {
  it('adds a 1-bar virtual lead-in to a song with no pre-roll', () => {
    const s = buildStructural(parse(`${META}\n| k . . . |`));
    // Musical structure: just the one real bar, no lead-in.
    expect(s.musicalLayers[0].bars.map((b) => b.index)).toEqual([1]);
    // View structure: a virtual lead-in bar (index -1) of one full bar.
    const view = s.layers[0].bars;
    expect(view.map((b) => b.index)).toEqual([-1, 1]);
    expect(view[0].id).toBe(LEAD_IN_BAR_ID);
    expect(view[0].beats).toBeCloseTo(4, 6); // a full 4/4 bar
  });

  it('sizes the virtual lead-in to the audio pre-roll when it exceeds a bar', () => {
    // 3s @ 120bpm = 6 quarter-note beats (> one 4/4 bar).
    const s = buildStructural(parse(`{{ drumsT0Sec: 3 }}\n${META}\n| k . . . |`));
    expect(s.musicalLayers[0].bars.map((b) => b.index)).toEqual([1]);
    const view = s.layers[0].bars;
    expect(view[0].index).toBe(-1);
    expect(view[0].beats).toBeCloseTo(6, 6);
  });

  it('rounds a sub-bar pre-roll up to a full bar', () => {
    // 1s @ 120bpm = 2 beats (< one 4/4 bar) -> rounded up to 4.
    const s = buildStructural(parse(`{{ drumsT0Sec: 1 }}\n${META}\n| k . . . |`));
    expect(s.layers[0].bars[0].beats).toBeCloseTo(4, 6);
  });

  it('adds no virtual bar when the song already has an explicit lead-in bar', () => {
    const s = buildStructural(parse(`{{ bpm: 120, time: "4/4", leadBars: 1, instrumentMapping: { k:{name:"Kick"} } }}\n| . . . . | k . . . |`));
    // No LEAD_IN_BAR_ID; the real lead bar already provides the room.
    expect(s.layers[0].bars.some((b) => b.id === LEAD_IN_BAR_ID)).toBe(false);
    expect(s.layers[0].bars.map((b) => b.index)).toEqual([-1, 1]);
  });
});
