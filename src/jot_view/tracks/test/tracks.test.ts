import { describe, expect, test } from 'bun:test';
import {
  buildDebugBundleTrackOrder,
  groupInstrumentPitches,
  reorderTrackOrder,
  resolveAudioInheritedColor,
  type MixerContext,
  type TrackKey,
} from 'src/jot_view/tracks/tracks';

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

describe('groupInstrumentPitches', () => {
  const ctx = (order: TrackKey[]): MixerContext => ({
    trackOrder: order,
    // Unused by groupInstrumentPitches; throws if a test accidentally relies on it.
    getInstrumentTrack: () => {
      throw new Error('getInstrumentTrack should not be called here');
    },
  });

  test('returns the paired instrument pitch for a grouped audio row', () => {
    const order: TrackKey[] = [
      { kind: 'audio', id: 'a1', groupId: 'pair:k' },
      { kind: 'instrument', pitch: 'k', groupId: 'pair:k' },
    ];
    expect(groupInstrumentPitches('a1', ctx(order))).toEqual(['k']);
  });

  test('returns every instrument sharing the group, in row order', () => {
    const order: TrackKey[] = [
      { kind: 'audio', id: 'a1', groupId: 'pair:c' },
      { kind: 'instrument', pitch: 'c', groupId: 'pair:c' },
      { kind: 'instrument', pitch: 'd', groupId: 'pair:c' },
    ];
    expect(groupInstrumentPitches('a1', ctx(order))).toEqual(['c', 'd']);
  });

  test('is empty for a solo (ungrouped) audio row', () => {
    const order: TrackKey[] = [
      { kind: 'audio', id: 'a1' },
      { kind: 'instrument', pitch: 'k', groupId: 'pair:k' },
    ];
    expect(groupInstrumentPitches('a1', ctx(order))).toEqual([]);
  });

  test('is empty when the group holds no instrument rows', () => {
    const order: TrackKey[] = [
      { kind: 'audio', id: 'a1', groupId: 'g' },
      { kind: 'audio', id: 'a2', groupId: 'g' },
    ];
    expect(groupInstrumentPitches('a1', ctx(order))).toEqual([]);
  });

  test('is empty when the audio id is absent', () => {
    const order: TrackKey[] = [{ kind: 'instrument', pitch: 'k', groupId: 'pair:k' }];
    expect(groupInstrumentPitches('missing', ctx(order))).toEqual([]);
  });
});

describe('resolveAudioInheritedColor', () => {
  const ctxWith = (order: TrackKey[], colors: Record<string, string>): MixerContext => ({
    trackOrder: order,
    getInstrumentTrack: (pitch: string) =>
      ({ color: colors[pitch] ?? '#000000' }) as ReturnType<MixerContext['getInstrumentTrack']>,
  });

  test('inherits the matched-pitch instrument colour as the tiebreaker', () => {
    const order: TrackKey[] = [
      { kind: 'audio', id: 'a1', groupId: 'pair:c' },
      { kind: 'instrument', pitch: 'c', groupId: 'pair:c' },
      { kind: 'instrument', pitch: 'd', groupId: 'pair:c' },
    ];
    // audioPitch 'd' is a group member -> picks d's colour, not the first.
    expect(resolveAudioInheritedColor('a1', 'd', ctxWith(order, { c: '#111', d: '#222' }))).toBe(
      '#222',
    );
  });

  test('falls back to the first grouped instrument when the pitch is not a member', () => {
    const order: TrackKey[] = [
      { kind: 'audio', id: 'a1', groupId: 'pair:c' },
      { kind: 'instrument', pitch: 'c', groupId: 'pair:c' },
    ];
    expect(resolveAudioInheritedColor('a1', 'zzz', ctxWith(order, { c: '#111' }))).toBe('#111');
  });

  test('returns undefined (no inheritance) for a solo audio row', () => {
    const order: TrackKey[] = [{ kind: 'audio', id: 'a1' }];
    expect(resolveAudioInheritedColor('a1', 'c', ctxWith(order, { c: '#111' }))).toBeUndefined();
  });
});
