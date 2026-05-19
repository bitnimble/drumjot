import { describe, expect, it } from 'bun:test';
import { RlrrFile, rlrrToJot } from 'src/rlrr';

/**
 * Regression for the ParaDB "drums offset from audio" bug: a placeholder
 * `{ bpm: 120, time: 0 }` followed by the real tempo at the first downbeat
 * must be collapsed into a `startOffset` lead-in with the grid starting at
 * the real tempo, instead of being treated as 120-bpm music (which shifted
 * the whole chart against the audio).
 */
function makeRlrr(bpmEvents: RlrrFile['bpmEvents'], firstHitSec: number): RlrrFile {
  return {
    version: 0.7,
    recordingMetadata: {},
    instruments: [],
    events: [{ name: 'BP_HiHat_C_8', vel: 100, loc: 0, time: firstHitSec }],
    bpmEvents,
  };
}

describe('rlrrToJot lead-in handling', () => {
  it('collapses a spurious leading 120 bpm into startOffset (user-reported case)', () => {
    const jot = rlrrToJot(
      makeRlrr(
        [
          { bpm: 120, time: 0 },
          { bpm: 90.859977722167969, time: 2.625 },
        ],
        2.625,
      ),
    );

    expect(jot.globalMetadata.startOffset).toBeCloseTo(2.625, 6);
    expect(jot.globalMetadata.bpm).toBeCloseTo(90.859977722167969, 6);
    // The hit rebases to beat 0 — first slot of bar 0 is the note, not a rest.
    expect(jot.voices[0].bars[0].elements[0].kind).toBe('note');
  });

  it('collapses the plan example (120 placeholder then 174 at 1.0s)', () => {
    const jot = rlrrToJot(
      makeRlrr(
        [
          { bpm: 120, time: 0 },
          { bpm: 174, time: 1.0 },
        ],
        1.0,
      ),
    );

    expect(jot.globalMetadata.startOffset).toBeCloseTo(1.0, 6);
    expect(jot.globalMetadata.bpm).toBe(174);
    expect(jot.voices[0].bars[0].elements[0].kind).toBe('note');
  });

  it('leaves a chart whose real tempo starts at time 0 unchanged (control)', () => {
    const jot = rlrrToJot(makeRlrr([{ bpm: 120, time: 0 }], 0.5));

    expect(jot.globalMetadata.startOffset).toBeUndefined();
    expect(jot.globalMetadata.bpm).toBe(120);
  });

  it('does not treat a later bpm event as lead-in when a note precedes it (pickup)', () => {
    // First note at 0.5s precedes the real bpm event at 1.0s, so the 1.0s
    // event is after the first onset and must not become the lead-in.
    const jot = rlrrToJot(
      makeRlrr(
        [
          { bpm: 120, time: 0 },
          { bpm: 174, time: 1.0 },
        ],
        0.5,
      ),
    );

    expect(jot.globalMetadata.startOffset).toBeUndefined();
    expect(jot.globalMetadata.bpm).toBe(120);
  });
});
