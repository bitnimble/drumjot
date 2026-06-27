import { describe, expect, it } from 'bun:test';
import { LayersPresenter } from 'src/editing/layers/layers_presenter';
import { dslToMutable } from 'src/schema/dsl/from_dsl';
import { groupIdOfTrack, groupSiblingInstrumentLanes, layerIdOfTrack } from 'src/schema/ordering';
import { parse } from 'src/schema/dsl/parser/parser';
import type { MutableJot } from 'src/schema/schema';

function setup(src: string): { jot: MutableJot; p: LayersPresenter } {
  const jot = dslToMutable(parse(src)).model;
  return { jot, p: new LayersPresenter(() => jot) };
}

/** Track ids placed in a layer, in render order. */
function tracksIn(jot: MutableJot, layerId: string): string[] {
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

  it('moveTrackAfter inserts after the anchor, joining its slot', () => {
    const { jot, p } = setup('| h s | || | k |');
    const [h, s] = tracksIn(jot, 'v0');
    p.moveTrackAfter(h, 'v0', s); // h after s -> [s, h]
    expect(tracksIn(jot, 'v0')).toEqual([s, h]);
    // Dropping after a grouped track joins that group.
    const g = p.createGroup(s)!; // group [s]
    p.moveTrackAfter(h, 'v0', s); // h joins after s in the group
    expect(groupIdOfTrack(jot, h)).toBe(g);
    // After itself is a no-op (no orphaning).
    p.moveTrackAfter(h, 'v0', h);
    expect(groupIdOfTrack(jot, h)).toBe(g);
  });

  it('moveTrackAfter with a mismatched layer/anchor is a no-op (no orphaning)', () => {
    const { jot, p } = setup('| h s | || | k |');
    const [h, s] = tracksIn(jot, 'v0');
    const [k] = tracksIn(jot, 'v1');
    // Anchor `k` lives in v1, not v0: the track must stay put, never get
    // removed-but-unplaced.
    p.moveTrackAfter(s, 'v0', k);
    expect(layerIdOfTrack(jot, s)).toBe('v0');
    expect(tracksIn(jot, 'v0')).toEqual([h, s]);
    expect(tracksIn(jot, 'v1')).toEqual([k]);
  });

  it('deleteGroup refuses a non-empty group (no orphaning)', () => {
    const { jot, p } = setup('| h s | || | k |');
    const [, s] = tracksIn(jot, 'v0');
    const g = p.createGroup(s)!; // group [s], still populated
    p.deleteGroup(g); // must be a no-op while it holds a track
    expect(jot.trackGroups.get(g)).toBeDefined();
    expect(groupIdOfTrack(jot, s)).toBe(g);
    expect(layerIdOfTrack(jot, s)).toBe('v0');
  });

  it('groupTracks wraps a loose target + folds into an existing group', () => {
    const { jot, p } = setup('| h s k |');
    const [h, s, k] = tracksIn(jot, 'v0');
    // Drop h onto loose s -> a fresh group {s, h}.
    p.groupTracks(h, s);
    const g = groupIdOfTrack(jot, s);
    expect(g).not.toBeNull();
    expect(groupIdOfTrack(jot, h)).toBe(g);
    expect(groupIdOfTrack(jot, k)).toBeNull(); // k untouched
    // Drop k onto a now-grouped member -> k joins the same group.
    p.groupTracks(k, s);
    expect(groupIdOfTrack(jot, k)).toBe(g);
    // Onto itself is a no-op.
    p.groupTracks(s, s);
    expect(groupIdOfTrack(jot, s)).toBe(g);
  });

  it('deleteGroup removes an emptied group slot and its entry', () => {
    const { jot, p } = setup('| h s | || | k |');
    const [h, s] = tracksIn(jot, 'v0');
    const g = p.createGroup(s)!; // group [s] in v0
    // Drag the lone member out -> the group slot is left empty.
    p.moveTrack(s, 'v1', null);
    expect(groupIdOfTrack(jot, s)).toBeNull();
    expect(jot.trackGroups.get(g)).toBeDefined(); // empty group persists
    // Delete it: slot + entry gone, the other track is undisturbed.
    p.deleteGroup(g);
    expect(jot.trackGroups.get(g)).toBeUndefined();
    expect(tracksIn(jot, 'v0')).toEqual([h]);
    const v0 = [...jot.ordering].find((l) => l.layerId === 'v0')!;
    expect([...v0.slots].some((sl) => sl.groupId === g)).toBe(false);
  });

  it('moves a whole group within its layer', () => {
    const { jot, p } = setup('| h s k |');
    const [h, s, k] = tracksIn(jot, 'v0');
    const groupId = p.createGroup(s)!; // wraps s in place: [loose h][group s][loose k]
    expect(tracksIn(jot, 'v0')).toEqual([h, s, k]);
    // Move the group's slot before h -> [group s][loose h][loose k] -> [s,h,k].
    p.moveGroup(groupId, 'v0', h);
    expect(tracksIn(jot, 'v0')).toEqual([s, h, k]);
    expect(groupIdOfTrack(jot, s)).toBe(groupId); // still grouped
  });

  it('createGroup wraps a mid-run track in place (not at the run edge)', () => {
    const { jot, p } = setup('| h s k |');
    const [h, s, k] = tracksIn(jot, 'v0');
    const g = p.createGroup(s)!; // s is in the middle: group lands between h and k
    expect(tracksIn(jot, 'v0')).toEqual([h, s, k]); // order preserved
    expect(groupIdOfTrack(jot, h)).toBeNull();
    expect(groupIdOfTrack(jot, s)).toBe(g);
    expect(groupIdOfTrack(jot, k)).toBeNull();
  });

  it('moveGroup lands between loose tracks, splitting the run', () => {
    const { jot, p } = setup('| h s k |'); // one loose run [h,s,k]
    const [h, s, k] = tracksIn(jot, 'v0');
    const g = p.createGroup(h)!; // [group h][loose s,k]
    p.moveGroup(g, 'v0', k); // drop before k -> between s and k
    expect(tracksIn(jot, 'v0')).toEqual([s, h, k]);
    expect(groupIdOfTrack(jot, h)).toBe(g);
    expect(groupIdOfTrack(jot, s)).toBeNull();
    expect(groupIdOfTrack(jot, k)).toBeNull();
  });

  it('moveGroup refuses to nest inside another group (no orphaning)', () => {
    const { jot, p } = setup('| h s k |');
    const [h, s, k] = tracksIn(jot, 'v0');
    const gA = p.createGroup(h)!; // group A = {h}
    p.moveTrackAfter(s, 'v0', h); // s joins A after h -> A = {h, s}
    expect(gA).toBeDefined();
    const gB = p.createGroup(k)!; // group B = {k}
    // Try to drop B before `s`, which is a non-first track inside group A.
    p.moveGroup(gB, 'v0', s);
    expect(groupIdOfTrack(jot, k)).toBe(gB); // B intact, not nested into A
    expect(tracksIn(jot, 'v0')).toEqual([h, s, k]); // order unchanged
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

  it('clusters a per-lane audio stem into a named group above its instrument', () => {
    const { jot, p } = setup('| h s k |');
    const [, , k] = tracksIn(jot, 'v0');
    // A transcribe stem for lane `k` lands as a runtime audio track.
    jot.tracks.set('audio:k1', { id: 'audio:k1', kind: 'audio', audioId: 'k1' });
    p.placeRuntimeAudioTrack('audio:k1', ['k']);
    // The stem + its instrument now share a group, stem directly above.
    const g = groupIdOfTrack(jot, k);
    expect(g).not.toBeNull();
    expect(groupIdOfTrack(jot, 'audio:k1')).toBe(g);
    expect(jot.trackGroups.get(g!)?.name).toBe('Kick'); // named after the instrument
    // Slot order is [stem, instrument] so the stem renders flush above its row.
    const slot = [...[...jot.ordering][0].slots].find((s) => s.groupId === g)!;
    expect([...slot.tracks].map((t) => t.trackId)).toEqual(['audio:k1', k]);
    // The instrument lane is now a sibling of the stem (audio colour inherits it).
    expect(groupSiblingInstrumentLanes(jot, 'audio:k1')).toEqual(['k']);
  });

  it('folds every dependent instrument of a shared stem into one group', () => {
    // A cymbal-split stem backs both crash (`c`) and ride (`d`).
    const { jot, p } = setup('| c d k |');
    const [c, d] = tracksIn(jot, 'v0');
    jot.tracks.set('audio:cym', { id: 'audio:cym', kind: 'audio', audioId: 'cym' });
    p.placeRuntimeAudioTrack('audio:cym', ['c', 'd']);
    const g = groupIdOfTrack(jot, c);
    expect(g).not.toBeNull();
    expect(jot.trackGroups.get(g!)?.name).toBe('Crash'); // named after the primary lane
    // Both instruments + the stem share the group: [stem, crash, ride].
    expect(groupIdOfTrack(jot, d)).toBe(g);
    expect(groupIdOfTrack(jot, 'audio:cym')).toBe(g);
    const slot = [...[...jot.ordering][0].slots].find((s) => s.groupId === g)!;
    expect([...slot.tracks].map((t) => t.trackId)).toEqual(['audio:cym', c, d]);
    expect(groupSiblingInstrumentLanes(jot, 'audio:cym')).toEqual(['c', 'd']);
  });

  it('drops a laneless / unmatched audio stem loose at the top', () => {
    const { jot, p } = setup('| h s k |');
    // A drumless backing stem carries no lane.
    jot.tracks.set('audio:mix', { id: 'audio:mix', kind: 'audio', audioId: 'mix' });
    p.placeRuntimeAudioTrack('audio:mix', []);
    // A stem for a lane the song doesn't contain stays loose too.
    jot.tracks.set('audio:ride', { id: 'audio:ride', kind: 'audio', audioId: 'ride' });
    p.placeRuntimeAudioTrack('audio:ride', ['d']);
    expect(groupIdOfTrack(jot, 'audio:mix')).toBeNull();
    expect(groupIdOfTrack(jot, 'audio:ride')).toBeNull();
    // Both sit at the very top of the layer (most-recent-first).
    expect(tracksIn(jot, 'v0').slice(0, 2)).toEqual(['audio:ride', 'audio:mix']);
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
