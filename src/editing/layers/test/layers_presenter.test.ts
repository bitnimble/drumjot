import { describe, expect, it } from 'bun:test';
import { LayersPresenter } from 'src/editing/layers/layers_presenter';
import { dslToReactive } from 'src/schema/dsl/from_dsl';
import { groupIdOfTrack, groupSiblingInstrumentLanes, layerIdOfTrack } from 'src/schema/ordering';
import { parse } from 'src/schema/dsl/parser/parser';
import type { Jot } from 'src/schema/schema';

function setup(src: string): { jot: Jot; p: LayersPresenter } {
  const jot = dslToReactive(parse(src)).model;
  return { jot, p: new LayersPresenter(() => jot) };
}

/** Track ids placed in a layer, in render order. */
function tracksIn(jot: Jot, layerId: string): string[] {
  const layer = [...jot.ordering].find((l) => l.layerId === layerId);
  return layer ? [...layer.slots].flatMap((s) => [...s.tracks].map((t) => t.trackId)) : [];
}

describe('LayersPresenter', () => {
  it('sets and clears a layer colour, preserving the name', () => {
    const { jot, p } = setup('| h s | || | k |');
    p.setLayerName('v0', 'Hands');
    p.setLayerColor('v0', '#abcdef');
    expect(jot.layers.get('v0')?.name).toBe('Hands');
    expect(jot.layers.get('v0')?.color).toBe('#abcdef');
    p.setLayerColor('v0', undefined);
    expect(jot.layers.get('v0')?.color).toBeUndefined();
    expect(jot.layers.get('v0')?.name).toBe('Hands'); // name survives
  });

  it('reorders whole layers', () => {
    const { jot, p } = setup('| h | || | k |');
    expect([...jot.ordering].map((l) => l.layerId)).toEqual(['v0', 'v1']);
    p.reorderLayer(0, 1);
    expect([...jot.ordering].map((l) => l.layerId)).toEqual(['v1', 'v0']);
  });

  it('moves a track to another layer', () => {
    const { jot, p } = setup('| h s | || | k |');
    const [h, s] = tracksIn(jot, 'v0');
    p.moveTrack(s, 'v1', null);
    expect(layerIdOfTrack(jot, s)).toBe('v1');
    expect(tracksIn(jot, 'v0')).toEqual([h]);
    expect(tracksIn(jot, 'v1')).toContain(s);
  });

  it('reorders a track within its layer via beforeTrackId', () => {
    const { jot, p } = setup('| h s | || | k |');
    const [h, s] = tracksIn(jot, 'v0');
    expect(tracksIn(jot, 'v0')).toEqual([h, s]);
    p.moveTrack(s, 'v0', h); // drop s above h
    expect(tracksIn(jot, 'v0')).toEqual([s, h]);
  });

  it('creates a group from a track, lets another join, then ungroups', () => {
    const { jot, p } = setup('| h s | || | k |');
    const [h, s] = tracksIn(jot, 'v0');
    const groupId = p.createGroup(s, 'Cymbals');
    expect(groupId).toBeDefined();
    expect(jot.trackGroups.get(groupId!)?.name).toBe('Cymbals');
    expect(groupIdOfTrack(jot, s)).toBe(groupId);
    expect(groupIdOfTrack(jot, h)).toBeNull(); // still loose
    expect(tracksIn(jot, 'v0')).toEqual([h, s]); // order preserved

    // Drop h before s -> joins the group.
    p.moveTrack(h, 'v0', s);
    expect(groupIdOfTrack(jot, h)).toBe(groupId);
    expect(tracksIn(jot, 'v0')).toEqual([h, s]);

    // Ungroup -> tracks stay, group metadata gone.
    p.ungroup(groupId!);
    expect(jot.trackGroups.get(groupId!)).toBeUndefined();
    expect(groupIdOfTrack(jot, s)).toBeNull();
    expect(groupIdOfTrack(jot, h)).toBeNull();
  });

  it('moves a whole group within its layer', () => {
    const { jot, p } = setup('| h s k |');
    const [h, s, k] = tracksIn(jot, 'v0');
    const groupId = p.createGroup(s)!; // slots: [loose h,k][group s] -> [h,k,s]
    expect(tracksIn(jot, 'v0')).toEqual([h, k, s]);
    // Move the group's slot before h -> [group s][loose h,k] -> [s,h,k].
    p.moveGroup(groupId, 'v0', h);
    expect(tracksIn(jot, 'v0')).toEqual([s, h, k]);
    expect(groupIdOfTrack(jot, s)).toBe(groupId); // still grouped
  });

  it('groupSiblingInstrumentLanes lists instrument lanes sharing a non-loose group', () => {
    const { jot, p } = setup('| h s |');
    const [h, s] = tracksIn(jot, 'v0');
    // Loose run → no siblings (the audio-colour inheritance reads this).
    expect(groupSiblingInstrumentLanes(jot, h)).toEqual([]);
    const g = p.createGroup(h)!; // group [h]
    p.moveTrack(s, 'v0', h); // s joins above h → slot [s, h]
    expect(groupSiblingInstrumentLanes(jot, h)).toEqual(['s', 'h']);
    expect(groupSiblingInstrumentLanes(jot, s)).toEqual(['s', 'h']); // queried by either member
    p.ungroup(g);
    expect(groupSiblingInstrumentLanes(jot, h)).toEqual([]); // back to loose
  });

  it('moveTrack / moveGroup with an unknown target layer is a no-op (no orphaning)', () => {
    const { jot, p } = setup('| h s | || | k |');
    const [h, s] = tracksIn(jot, 'v0');
    // Bad target: the track must stay placed where it was, not get removed from
    // its source slot and left unplaced (which would orphan its notes).
    p.moveTrack(s, 'v99', null);
    expect(layerIdOfTrack(jot, s)).toBe('v0');
    expect(tracksIn(jot, 'v0')).toEqual([h, s]);
    // Same guard for a whole group.
    const g = p.createGroup(s)!;
    p.moveGroup(g, 'v99', null);
    expect(groupIdOfTrack(jot, s)).toBe(g); // still grouped + placed
    expect(layerIdOfTrack(jot, s)).toBe('v0');
  });

  it('prunes an emptied loose run when its last track leaves', () => {
    const { jot, p } = setup('| h | || | k |');
    const [h] = tracksIn(jot, 'v0');
    p.moveTrack(h, 'v1', null);
    // v0 now has no tracks (its only loose slot was pruned).
    expect(tracksIn(jot, 'v0')).toEqual([]);
    const v0 = [...jot.ordering].find((l) => l.layerId === 'v0')!;
    expect([...v0.slots]).toHaveLength(0);
  });
});
