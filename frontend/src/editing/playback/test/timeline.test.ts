/**
 * Playhead mapping (`timeToX` / `xToTime`) must be exact through a gradual
 * tempo ramp, not bar-level-linear. The score x-axis is beats, so within a
 * ramp bar the time→x curve is nonlinear: a 60→240 accelerando spends most
 * of the bar's seconds in its slow first half, so the bar's TIME-midpoint
 * lands well before its BEAT-midpoint.
 */
import { describe, expect, it } from 'bun:test';
import { parse } from 'src/schema/dsl/parser/parser';
import { buildStructural } from 'src/editing/jot_editor_store';
import { buildTimeline, timeToX, xToTime } from '../timeline';

function rampTimeline() {
  const jot = parse(
    '{{ bpm: 60, time: "4/4", instrumentMapping: { k:{name:"K"} } }}\n' +
      '{{ bpm: { start: 60, end: 240, duration: 4 } }}\n' +
      '| k k k k |\n| k k k k |'
  );
  return buildTimeline(buildStructural(jot));
}

describe('timeToX / xToTime with a tempo ramp', () => {
  it('round-trips x -> time -> x across the score', () => {
    const tl = rampTimeline();
    const pxPerBeat = tl.rendered!.pxPerBeat;
    for (const x of [pxPerBeat, pxPerBeat * 3, pxPerBeat * 5, pxPerBeat * 7]) {
      expect(timeToX(tl, xToTime(tl, x))).toBeCloseTo(x, 3);
    }
  });

  it('maps time nonlinearly within a ramp bar (exact, not bar-linear)', () => {
    const tl = rampTimeline();
    // For each bar, the time at its BEAT-midpoint (x midpoint) vs its
    // TIME-midpoint. Bar-level-linear mapping makes this deviation zero for
    // every bar; a real ramp bar shows a large nonzero deviation.
    let maxDev = 0;
    for (const bar of tl.bars) {
      if (bar.durationSec <= 0) continue;
      const tMid = bar.startSec + bar.durationSec / 2;
      const xMid = (timeToX(tl, bar.startSec) + timeToX(tl, bar.startSec + bar.durationSec)) / 2;
      maxDev = Math.max(maxDev, Math.abs(xToTime(tl, xMid) - tMid));
    }
    expect(maxDev).toBeGreaterThan(0.05);
  });
});
