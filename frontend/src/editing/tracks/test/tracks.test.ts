import { describe, expect, test } from 'bun:test';
import {
  resolveAudioInheritedColor,
  type MixerContext,
} from 'src/editing/tracks/tracks';

/** A synthetic {@link MixerContext}: `lanes` is what
 *  `groupInstrumentLanesForAudio` reports for the audio row under test (the
 *  grouped-instrument lanes, in slot order), `colors` the per-lane instrument
 *  colours the inheritance chain bottoms out in. */
function ctxWith(lanes: string[], colors: Record<string, string>): MixerContext {
  return {
    groupInstrumentLanesForAudio: () => lanes,
    getInstrumentTrack: (lane: string) =>
      ({ color: colors[lane] ?? '#000000' }) as ReturnType<MixerContext['getInstrumentTrack']>,
  };
}

describe('resolveAudioInheritedColor', () => {
  test('inherits the matched-lane instrument colour as the tiebreaker', () => {
    // The audio is grouped with both `c` and `d`; its own lane link is `d`,
    // so it picks d's colour rather than the first group member's.
    expect(resolveAudioInheritedColor('a1', 'd', ctxWith(['c', 'd'], { c: '#111', d: '#222' }))).toBe(
      '#222',
    );
  });

  test('falls back to the first grouped instrument when the lane is not a member', () => {
    expect(resolveAudioInheritedColor('a1', 'zzz', ctxWith(['c'], { c: '#111' }))).toBe('#111');
  });

  test('falls back to the first grouped instrument when the audio has no lane link', () => {
    expect(resolveAudioInheritedColor('a1', undefined, ctxWith(['c'], { c: '#111' }))).toBe('#111');
  });

  test('an ungrouped audio row with a lane link still tints to that lane', () => {
    // Default placement for a synced per-lane stem is a loose run (no group),
    // so the load-time lane mapping keeps it tinted to its instrument.
    expect(resolveAudioInheritedColor('a1', 'c', ctxWith([], { c: '#111' }))).toBe('#111');
  });

  test('returns undefined (neutral) for an ungrouped audio row with no lane link', () => {
    expect(resolveAudioInheritedColor('a1', undefined, ctxWith([], { c: '#111' }))).toBeUndefined();
  });
});
