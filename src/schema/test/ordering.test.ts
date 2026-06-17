import { describe, expect, it } from 'bun:test';
import { dslToMutable } from 'src/schema/dsl/from_dsl';
import { parse } from 'src/schema/dsl/parser/parser';
import {
  groupIdOfTrack,
  laneForNote,
  layerIdOfTrack,
  trackLaneOf,
} from 'src/schema/ordering';
import type { MutableJot, NoteElement } from 'src/schema/schema';

function jotFrom(src: string): MutableJot {
  return dslToMutable(parse(src)).model;
}

/** All placed top-level notes, in id order. */
function notesOf(jot: MutableJot): NoteElement[] {
  return [...jot.elements.values()].filter((e) => e.kind === 'note') as NoteElement[];
}

/** The lone loose slot's track ids for a layer in the default ordering. */
function looseTrackIds(jot: MutableJot, layerId: string): string[] {
  const layer = [...jot.ordering].find((l) => l.layerId === layerId);
  if (!layer) return [];
  return [...layer.slots].flatMap((s) => [...s.tracks].map((t) => t.trackId));
}

describe('track / ordering model from DSL conversion', () => {
  it('mints one instrument track per (layer, lane) with a default loose ordering', () => {
    const jot = jotFrom('| h h h h | || | k . s . |');

    // Two ordered layers, v0 then v1.
    const order = [...jot.ordering];
    expect(order.map((l) => l.layerId)).toEqual(['v0', 'v1']);

    // Each layer has exactly one loose run (groupId null).
    for (const layer of order) {
      const slots = [...layer.slots];
      expect(slots).toHaveLength(1);
      expect(slots[0].groupId).toBeNull();
    }

    // v0 → one track (h); v1 → two tracks (k, s).
    const v0 = looseTrackIds(jot, 'v0');
    const v1 = looseTrackIds(jot, 'v1');
    expect(v0).toHaveLength(1);
    expect(v1).toHaveLength(2);
    expect(trackLaneOf(jot, v0[0])).toBe('h');
    expect(new Set(v1.map((id) => trackLaneOf(jot, id)))).toEqual(new Set(['k', 's']));

    // Every placed note resolves its lane via its trackId.
    for (const note of notesOf(jot)) {
      expect(note.trackId).toBeDefined();
      expect(laneForNote(jot, note)).toBe(note.lane);
    }
  });

  it('keeps the same lane in two layers as two distinct tracks', () => {
    const jot = jotFrom('| s s | || | s s |');
    const v0 = looseTrackIds(jot, 'v0');
    const v1 = looseTrackIds(jot, 'v1');
    expect(v0).toHaveLength(1);
    expect(v1).toHaveLength(1);
    expect(v0[0]).not.toBe(v1[0]);
    expect(trackLaneOf(jot, v0[0])).toBe('s');
    expect(trackLaneOf(jot, v1[0])).toBe('s');
  });

  it('reverse-looks-up a track to its layer and (null) group', () => {
    const jot = jotFrom('| h h h h | || | k . s . |');
    const v0Track = looseTrackIds(jot, 'v0')[0];
    const v1Track = looseTrackIds(jot, 'v1')[0];
    expect(layerIdOfTrack(jot, v0Track)).toBe('v0');
    expect(layerIdOfTrack(jot, v1Track)).toBe('v1');
    expect(groupIdOfTrack(jot, v0Track)).toBeNull();
    expect(layerIdOfTrack(jot, 'nope')).toBeUndefined();
    expect(groupIdOfTrack(jot, 'nope')).toBeUndefined();
  });

  it('laneForNote falls back to note.lane for a track-less (template) note', () => {
    const jot = jotFrom('| k |');
    const fake = { kind: 'note', id: 'x', beat: 0, duration: 1, lane: 'q', modifiers: [] } as NoteElement;
    expect(laneForNote(jot, fake)).toBe('q');
  });
});
