/**
 * `StructuralPresenter.barsForTrack(layerId, lane)` is the per-track render
 * source: unlike `barsForLane` (which merges a lane's notes across every
 * layer), it returns ONLY the given layer's notes on the lane, so the same
 * lane living in two layers renders two independent rows.
 */
import { describe, expect, it } from 'bun:test';
import { buildStructural, JotEditorStore } from 'src/editing/jot_editor_store';
import { LayersPresenter } from 'src/editing/layers/layers_presenter';
import type { LaneBars } from 'src/editing/structure/structural_presenter';
import { parse } from 'src/schema/dsl/parser/parser';
import { layerIdOfTrack } from 'src/schema/ordering';
import type { NoteElement, Track } from 'src/schema/schema';

const META = '{{ bpm: 120, time: "4/4", instrumentMapping: { s:{name:"Snare"} } }}';

function noteCount(lb: LaneBars, lane: string): number {
  return lb.bars.reduce((n, b) => n + (b.tracks[lane]?.notes.length ?? 0), 0);
}

describe('barsForTrack', () => {
  it('scopes a lane to one layer; same lane in two layers stays separate', () => {
    // lane s: two hits in layer v0, one hit in layer v1.
    const s = buildStructural(parse(`${META}\n| s . s . | || | s . . . |`));
    expect(noteCount(s.barsForTrack('v0', 's'), 's')).toBe(2);
    expect(noteCount(s.barsForTrack('v1', 's'), 's')).toBe(1);
    // barsForLane merges both layers -> all three hits.
    expect(noteCount(s.barsForLane('s'), 's')).toBe(3);
  });

  it('returns an empty track for a lane the layer does not carry', () => {
    const s = buildStructural(parse(`${META}\n| s . s . | || | s . . . |`));
    // v1 has no notes on a non-existent lane.
    expect(noteCount(s.barsForTrack('v1', 'k'), 'k')).toBe(0);
  });

  it('matches barsForLane for a single-layer song', () => {
    const s = buildStructural(parse(`${META}\n| s . s . |`));
    expect(noteCount(s.barsForTrack('v0', 's'), 's')).toBe(2);
    expect(noteCount(s.barsForLane('s'), 's')).toBe(2);
  });

  it('a note follows its track when the track moves to another layer (no per-note rewrite)', () => {
    // The contraction's payoff: a placed note stores no layer; its layer derives
    // from its trackId's placement in `ordering`. Moving the track re-homes the
    // note in the score with the note element untouched.
    const store = new JotEditorStore();
    store.loadSource(
      parse(
        '{{ time: "4/4", instrumentMapping: { s:{name:"Snare"}, k:{name:"Kick"} } }}\n' +
          '| s . . . | || | k . . . |'
      )
    );
    const s = store.structural!;
    const jot = store.jot!;
    // Kick lives in v1 to start.
    expect(noteCount(s.barsForTrack('v1', 'k'), 'k')).toBe(1);
    expect(noteCount(s.barsForTrack('v0', 'k'), 'k')).toBe(0);

    const kTrackId = [...jot.tracks.entries()].find(
      ([, t]) => (t as Track).kind === 'instrument' && (t as Track & { lane: string }).lane === 'k'
    )![0];
    const kNote = [...jot.elements.values()].find(
      (e) => (e as NoteElement).kind === 'note' && (e as NoteElement).lane === 'k'
    ) as NoteElement;
    const noteTrackBefore = kNote.trackId;

    // Move the kick track from v1 to v0 (ordering-only mutation).
    new LayersPresenter(() => store.jot).moveTrack(kTrackId, 'v0', null);

    // The note now renders in v0; its own `trackId` is unchanged.
    expect(noteCount(s.barsForTrack('v0', 'k'), 'k')).toBe(1);
    expect(noteCount(s.barsForTrack('v1', 'k'), 'k')).toBe(0);
    expect(kNote.trackId).toBe(noteTrackBefore);
    expect(layerIdOfTrack(jot, kTrackId)).toBe('v0');
  });
});
