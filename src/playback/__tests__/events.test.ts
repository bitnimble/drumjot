/**
 * Sanity tests for the jot → playback-event timing formula.
 *
 * The formula must match `toMidi`'s tick math (which the old playback
 * path round-tripped through), so a 160 BPM 4/4 jot with notes on the
 * downbeats should produce events at 0.0, 0.375, 0.75, 1.125 s — same
 * as `toMidi`'s ticks would decode to.
 */
import { describe, expect, it } from 'bun:test';
import { parse } from 'src/parser';
import { RenderedJot } from 'src/jot';
import { jotToEvents } from '../events';

function events(src: string) {
  const jot = parse(src);
  const rendered = new RenderedJot(jot);
  return jotToEvents(rendered);
}

describe('jotToEvents timing', () => {
  it('4 quarter-note kicks at 160 BPM 4/4 land at 0/0.375/0.75/1.125', () => {
    const evs = events(
      '{{ bpm: 160, time: "4/4", instrumentMapping: { k:{name:"Kick"} } }} | k k k k |'
    );
    expect(evs).toHaveLength(4);
    expect(evs[0].time).toBeCloseTo(0, 6);
    expect(evs[1].time).toBeCloseTo(0.375, 6);
    expect(evs[2].time).toBeCloseTo(0.75, 6);
    expect(evs[3].time).toBeCloseTo(1.125, 6);
  });

  it('two 4/4 bars at 160 BPM cover 3.0 seconds total', () => {
    const evs = events(
      '{{ bpm: 160, time: "4/4", instrumentMapping: { k:{name:"Kick"} } }} | k . . . | . . . k |'
    );
    // First kick at t=0, last kick at the last quarter-note of bar 2.
    expect(evs[0].time).toBeCloseTo(0, 6);
    // Bar 2 starts at 1.5s; beat 3 (0-indexed) of bar 2 is at 1.5 + 3*0.375.
    expect(evs[evs.length - 1].time).toBeCloseTo(1.5 + 3 * 0.375, 6);
  });

  it('falls back to 120 BPM when bpm is missing', () => {
    const evs = events(
      '{{ time: "4/4", instrumentMapping: { k:{name:"Kick"} } }} | k k k k |'
    );
    // 120 BPM → 60/120 = 0.5s per beat.
    expect(evs[1].time).toBeCloseTo(0.5, 6);
  });
});
