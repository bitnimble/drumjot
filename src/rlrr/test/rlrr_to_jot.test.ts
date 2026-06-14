import { describe, expect, it } from 'bun:test';
import { rlrrToJot } from 'src/rlrr/rlrr_to_jot';
import { RlrrFile } from 'src/rlrr/schema';

/**
 * Regression for the ParaDB "drums offset from audio" bug: a placeholder
 * `{ bpm: 120, time: 0 }` followed by the real tempo at the first downbeat
 * must be collapsed into a `drumsT0Sec` lead-in with the grid starting at
 * the real tempo, instead of being treated as 120-bpm music (which shifted
 * the whole chart against the audio).
 *
 * Under the three-epoch model, `drumsT0Sec` is literally the audio time of
 * the first drum onset (not the last-bpm-event-before-drums heuristic the
 * pre-three-epoch code used). The starting bpm for bar 1 is still picked
 * from the latest bpm event at or before that point.
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

    expect(jot.globalMetadata.drumsT0Sec).toBeCloseTo(2.625, 6);
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

    expect(jot.globalMetadata.drumsT0Sec).toBeCloseTo(1.0, 6);
    expect(jot.globalMetadata.bpm).toBe(174);
    expect(jot.voices[0].bars[0].elements[0].kind).toBe('note');
  });

  it('uses first-drum-onset as the lead-in even when only the placeholder bpm exists', () => {
    // bpm 120 at 0 (the authoring-tool placeholder) and the first drum at
    // 0.5s: drumsT0Sec is literally the first drum onset (0.5s), bar 1's
    // starting bpm is the latest event at or before that — here, 120.
    const jot = rlrrToJot(makeRlrr([{ bpm: 120, time: 0 }], 0.5));

    expect(jot.globalMetadata.drumsT0Sec).toBeCloseTo(0.5, 6);
    expect(jot.globalMetadata.bpm).toBe(120);
  });

  it('drum at time 0 yields no lead-in', () => {
    const jot = rlrrToJot(makeRlrr([{ bpm: 120, time: 0 }], 0));

    expect(jot.globalMetadata.drumsT0Sec).toBeUndefined();
    expect(jot.globalMetadata.bpm).toBe(120);
  });

  it('takes drumsT0Sec from the first drum even when a later bpm event would shift it (pickup-shaped chart)', () => {
    // First drum at 0.5s with a later bpm event at 1.0s. Under the
    // three-epoch model the bar grid anchors on the literal first drum
    // (0.5s), and the 1.0s bpm event becomes a positive jot-time tempo
    // change (drops into the next bar boundary it lands at). The
    // pre-three-epoch heuristic kept startOffset at 0 here; the new model
    // chose simplicity over the pickup-aware special case.
    const jot = rlrrToJot(
      makeRlrr(
        [
          { bpm: 120, time: 0 },
          { bpm: 174, time: 1.0 },
        ],
        0.5,
      ),
    );

    expect(jot.globalMetadata.drumsT0Sec).toBeCloseTo(0.5, 6);
    expect(jot.globalMetadata.bpm).toBe(120);
  });
});
