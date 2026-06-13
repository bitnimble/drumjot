import { describe, expect, test } from 'bun:test';
import { buildDebugBundleTrackOrder, reorderTrackOrder, type TrackKey } from './tracks';

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

/** The reorder behind drag-and-drop. `toIdx` is the insertion gap in the
 *  current list (0 = before first row, length = after last). */
describe('reorderTrackOrder (drag-and-drop)', () => {
  const I = (pitch: string, groupId?: string): TrackKey => ({ kind: 'instrument', pitch, groupId });
  // `kind:id-or-pitch` plus `#groupId` when grouped, so assertions read as
  // the rendered row sequence + grouping.
  const summary = (order: readonly TrackKey[]): string[] =>
    order.map((k) => {
      const tag =
        k.kind === 'instrument' ? `instr:${k.pitch}` : k.kind === 'audio' ? `audio:${k.id}` : `lyrics:${k.id}`;
      return k.groupId ? `${tag}#${k.groupId}` : tag;
    });

  test('moves a row down to the target gap', () => {
    const order = [I('k'), I('s'), I('h'), I('c')];
    expect(summary(reorderTrackOrder(order, 0, 3))).toEqual(['instr:s', 'instr:h', 'instr:k', 'instr:c']);
  });

  test('moves a row up to the target gap', () => {
    const order = [I('k'), I('s'), I('h'), I('c')];
    expect(summary(reorderTrackOrder(order, 3, 1))).toEqual(['instr:k', 'instr:c', 'instr:s', 'instr:h']);
  });

  test('a no-op move (in place, adjacent gap, or out-of-range) returns the same reference', () => {
    const order = [I('k'), I('s'), I('h')];
    expect(reorderTrackOrder(order, 1, 1)).toBe(order); // dropped exactly where it was
    expect(reorderTrackOrder(order, 1, 2)).toBe(order); // dropped into its own trailing gap
    expect(reorderTrackOrder(order, 5, 0)).toBe(order); // fromIdx out of range
    expect(reorderTrackOrder(order, -1, 0)).toBe(order);
  });

  test('joins a group when dropped strictly between two rows that share a groupId', () => {
    const order = [I('a', 'g1'), I('b', 'g1'), I('z')];
    // Drop z into the gap between a and b → it adopts g1.
    expect(summary(reorderTrackOrder(order, 2, 1))).toEqual(['instr:a#g1', 'instr:z#g1', 'instr:b#g1']);
  });

  test('goes solo when dropped at a group edge (top/bottom/boundary)', () => {
    // Pull a out of its 2-row group to the bottom → solo (groupId cleared).
    const order = [I('a', 'g1'), I('b', 'g1')];
    expect(summary(reorderTrackOrder(order, 0, 2))).toEqual(['instr:b#g1', 'instr:a']);
  });

  test('goes solo when dropped between two DIFFERENT groups', () => {
    const order = [I('a', 'g1'), I('b', 'g2'), I('z')];
    expect(summary(reorderTrackOrder(order, 2, 1))).toEqual(['instr:a#g1', 'instr:z', 'instr:b#g2']);
  });

  test('preserves the moved row kind + identity (audio id, not just pitch)', () => {
    const order = [{ kind: 'audio', id: 'a1' } as TrackKey, I('k'), I('s')];
    const out = reorderTrackOrder(order, 0, 3);
    expect(out[2]).toMatchObject({ kind: 'audio', id: 'a1' });
    expect(summary(out)).toEqual(['instr:k', 'instr:s', 'audio:a1']);
  });
});
