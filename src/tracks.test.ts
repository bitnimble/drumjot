import { describe, expect, test } from 'bun:test';
import { buildDebugBundleTrackOrder } from './tracks';

/** Collapse the order to a compact `audio:<id>` / `instr:<pitch>` form
 *  so assertions read as the rendered row sequence. */
function asKeys(order: ReturnType<typeof buildDebugBundleTrackOrder>): string[] {
  return order.map((k) => {
    if (k.kind === 'audio') return `audio:${k.id}`;
    if (k.kind === 'instrument') return `instr:${k.pitch}`;
    return `lyrics:${k.id}`;
  });
}

function dupes(keys: string[]): string[] {
  return keys.filter((v, i) => keys.indexOf(v) !== i);
}

describe('buildDebugBundleTrackOrder', () => {
  test('pairs each per-pitch stem above its instrument, unmatched stems on top', () => {
    const pitches = ['k', 's', 'h'];
    const loadedByKey = new Map<string, string>([
      ['no_drums', 'track-1'],
      ['k', 'track-2'],
      ['s', 'track-3'],
      ['h', 'track-4'],
    ]);
    expect(asKeys(buildDebugBundleTrackOrder(pitches, loadedByKey))).toEqual([
      'audio:track-1',
      'audio:track-2',
      'instr:k',
      'audio:track-3',
      'instr:s',
      'audio:track-4',
      'instr:h',
    ]);
  });

  test('a stem shared by two jot pitches renders once, both pitches under it', () => {
    // Cymbal split: stem_c.mp3 declared for both crash (c) and ride (d),
    // and the jot contains both.
    const pitches = ['c', 'd'];
    const loadedByKey = new Map<string, string>([
      ['c', 'track-5'],
      ['d', 'track-5'],
    ]);
    const keys = asKeys(buildDebugBundleTrackOrder(pitches, loadedByKey));
    expect(dupes(keys)).toEqual([]);
    expect(keys).toEqual(['audio:track-5', 'instr:c', 'instr:d']);
  });

  test('a stem shared by a present and an absent jot pitch is not duplicated', () => {
    // Regression: stem_c.mp3 declared for crash (c) AND ride (d), but the
    // song has crash hits and no ride, so the jot contains only `c`. The
    // `d` key is then "unmatched"; the stem must still pair with `c` and
    // appear exactly once, not once as an unmatched top stem AND once
    // paired with `c`.
    const pitches = ['c'];
    const loadedByKey = new Map<string, string>([
      ['no_drums', 'track-1'],
      ['c', 'track-2'],
      ['d', 'track-2'],
    ]);
    const keys = asKeys(buildDebugBundleTrackOrder(pitches, loadedByKey));
    expect(dupes(keys)).toEqual([]);
    expect(keys).toEqual(['audio:track-1', 'audio:track-2', 'instr:c']);
  });

  test('a stem mapped only to non-jot keys stays a single unmatched top row', () => {
    const pitches = ['k'];
    const loadedByKey = new Map<string, string>([
      ['no_drums', 'track-1'],
      ['x', 'track-9'],
      ['y', 'track-9'], // same unused stem under two non-jot keys
      ['k', 'track-2'],
    ]);
    const keys = asKeys(buildDebugBundleTrackOrder(pitches, loadedByKey));
    expect(dupes(keys)).toEqual([]);
    expect(keys).toEqual(['audio:track-1', 'audio:track-9', 'audio:track-2', 'instr:k']);
  });
});
