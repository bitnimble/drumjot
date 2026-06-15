/**
 * Sanity tests for the jot → playback-event timing formula.
 *
 * The formula must match `toMidi`'s tick math (which the old playback
 * path round-tripped through), so a 160 BPM 4/4 jot with notes on the
 * downbeats should produce events at 0.0, 0.375, 0.75, 1.125 s — same
 * as `toMidi`'s ticks would decode to.
 */
import { describe, expect, it } from 'bun:test';
import { parse } from 'src/schema/dsl/parser/parser';
import { Jot } from 'src/schema/dsl/dsl';
import { buildJotModel } from 'src/editing/jot_editor_store';
import { buildTimeline } from '../timeline';
import { jotToEvents } from '../events';

function events(src: string) {
  const jot = parse(src);
  return jotToEvents(buildJotModel(jot).structural);
}

function eventsFor(jot: Jot) {
  return jotToEvents(buildJotModel(jot).structural);
}

/** A 4/4 @ 120 BPM jot whose only note is a kick on beat 1 with `offset` ms. */
function kickWithOffset(offset: number | undefined): Jot {
  return {
    title: '',
    globalMetadata: { bpm: 120, time: { count: 4, unit: 4 }, instrumentMapping: { k: { kind: 'kick', name: 'Kick' } } },
    voices: [
      {
        bars: [
          {
            elements: [
              { kind: 'note', pitch: 'k', ...(offset !== undefined ? { offset } : {}) },
              { kind: 'rest' },
              { kind: 'rest' },
              { kind: 'rest' },
            ],
          },
        ],
      },
    ],
  };
}

describe('jotToEvents note.offset', () => {
  it('shifts a note by its offset in ms (positive = later)', () => {
    const evs = eventsFor(kickWithOffset(25));
    expect(evs).toHaveLength(1);
    // beat 1 at 120 BPM = 0.0s, +25 ms.
    expect(evs[0].time).toBeCloseTo(0.025, 6);
  });

  it('shifts a note earlier for a negative offset', () => {
    const evs = eventsFor(kickWithOffset(-15));
    expect(evs[0].time).toBeCloseTo(-0.015, 6);
  });

  it('leaves a note on its slot when there is no offset', () => {
    const evs = eventsFor(kickWithOffset(undefined));
    expect(evs[0].time).toBeCloseTo(0, 6);
  });
});

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

  it('honours a mid-bar bpm change at the next-element onset', () => {
    // 8-slot 4/4 bar at 60 bpm: each slot is 0.5 beats = 0.5s at 60 bpm.
    // {{bpm:120}} after the 4th element (slot index 4 is at beat 2.0,
    // 2.0s into the bar) anchors at the next element (slot 5, beat 2.5).
    // Until that anchor: 60 bpm. From that anchor: 120 bpm (0.25s/slot).
    //   slot 0 (k) → 0.0s   (60 bpm region)
    //   slot 1 (.) → 0.5s
    //   slot 2 (s) → 1.0s
    //   slot 3 (.) → 1.5s
    //   slot 4 (k) → 2.0s   (last 60-bpm element)
    //   slot 5 (.) → 2.5s   (anchor: tempo flips to 120 here)
    //   slot 6 (s) → 2.75s  (120 bpm region: 0.25s/slot)
    //   slot 7 (.) → 3.0s
    const evs = events(
      '{{ bpm: 60, time: "4/4", instrumentMapping: { k:{name:"Kick"}, s:{name:"Snare"} } }} ' +
        '| k . s . k {{ bpm: 120 }} . s . |'
    );
    expect(evs.map((e) => e.time)).toEqual([
      0.0,
      expect.closeTo(1.0, 6) as unknown as number,
      expect.closeTo(2.0, 6) as unknown as number,
      expect.closeTo(2.75, 6) as unknown as number,
    ]);
  });

  it('honours a {bpm} modifier on a note at that note onset', () => {
    // Same shape as above but the change is attached directly to the
    // post-change rest's beat via a note modifier instead of a
    // freestanding marker.
    const evs = events(
      '{{ bpm: 60, time: "4/4", instrumentMapping: { k:{name:"Kick"}, s:{name:"Snare"} } }} ' +
        '| k . s . k . s{bpm: 120} . |'
    );
    // The second snare is at slot 6 (beat 3.0). Before slot 6: tempo 60.
    // From slot 6 onward: tempo 120.
    //   slot 0 (k) → 0.0s
    //   slot 2 (s) → 1.0s
    //   slot 4 (k) → 2.0s
    //   slot 6 (s) → 3.0s (tempo flips to 120 here, but no later events)
    expect(evs.map((e) => e.time)).toEqual([0.0, 1.0, 2.0, 3.0]);
  });

  it('buildTimeline reports a half-tempo bar duration when bpm doubles at the midpoint', () => {
    // 4/4 bar at 60 bpm = 4.0s if uniform. With a tempo doubling halfway
    // through (beat 2): first half = 2*1.0 = 2.0s, second half =
    // 2*0.5 = 1.0s, total = 3.0s.
    const jot = parse(
      '{{ bpm: 60, time: "4/4", instrumentMapping: { k:{name:"Kick"}, s:{name:"Snare"} } }} ' +
        '| k . s . {{ bpm: 120 }} k . s . |'
    );
    const structural = buildJotModel(jot).structural;
    const timeline = buildTimeline(structural);
    expect(timeline.bars[0].durationSec).toBeCloseTo(3.0, 6);
  });

  it('anchors bar 1 at jot 0 when leadBars shifts bars[0] into the pre-drum window', () => {
    // Two empty lead-in bars then a kick on bar-1 downbeat at 120 BPM:
    // bar 1 (= voice.bars[2]) lives at jot 0, so the kick fires at t=0,
    // even though the rendered jot has two preceding (empty) bars at
    // negative jot time.
    const jot = parse(
      '{{ bpm: 120, time: "4/4", leadBars: 2, instrumentMapping: { k:{name:"Kick"} } }} | . . . . | . . . . | k . . . |',
    );
    const structural = buildJotModel(jot).structural;
    const evs = jotToEvents(structural);
    expect(evs).toHaveLength(1);
    expect(evs[0].time).toBeCloseTo(0, 6);
    // The timeline mirrors that anchor: pre-drum bars sit at negative
    // startSec, bar 1 at 0, total playable jot duration spans the drum
    // bars only (the lead-in is reachable via negative seek but not
    // counted toward the end-of-playback sentinel).
    const timeline = buildTimeline(structural);
    expect(timeline.bars[0].startSec).toBeCloseTo(-4, 6);
    expect(timeline.bars[1].startSec).toBeCloseTo(-2, 6);
    expect(timeline.bars[2].startSec).toBeCloseTo(0, 6);
    expect(timeline.totalDurationSec).toBeCloseTo(2, 6);
  });
});
