import { computed, makeObservable } from 'mobx';
import { computedFn } from 'mobx-utils';
import { groupIdOfTrack, layerIdOfTrack } from 'src/schema/ordering';
import type { Jot, Track } from 'src/schema/schema';
import { PICKER_PALETTE } from 'src/editing/tracks/tracks';

/**
 * Default band colour for the layer at order `index`: the first layer is
 * transparent (undefined), the rest rotate through the shared palette, mirroring
 * the instrument-track colour defaults. A stored `layer.color` overrides this.
 */
export function defaultLayerColor(index: number): string | undefined {
  if (index <= 0) return undefined;
  return PICKER_PALETTE[(index - 1) % PICKER_PALETTE.length];
}

/**
 * Reactive read-model over the track / ordering schema (see `schema.ts` +
 * `ordering.ts`): turns `jot.ordering` + `jot.tracks` + `jot.trackGroups` +
 * `jot.layers` into the layer → slot → track structure the score's gutter and
 * the Layers panel both render, plus memoised reverse-lookups (`trackId →
 * layer / group`).
 *
 * Data only (observables + computeds), per the store/presenter split: the
 * single writer of `ordering`/`tracks`/`trackGroups` is the Layers presenter.
 * Display name + colour are NOT resolved here (they're functions of the
 * palette + instrument mapping, which the view composes); this store stays
 * dependency-light (just the jot) so it's unit-testable in isolation.
 */

export type LayersTrackView = Track;

export type LayersGroupSlot = {
  kind: 'group';
  id: string;
  name: string;
  color?: string;
  tracks: LayersTrackView[];
};
export type LayersLooseSlot = { kind: 'loose'; tracks: LayersTrackView[] };
export type LayersSlotView = LayersGroupSlot | LayersLooseSlot;

export type LayersLayerView = {
  id: string;
  name?: string;
  /** Effective band colour (stored override, else the palette default). */
  color?: string;
  /** Whether `color` is an explicit stored override (vs the derived default);
   *  drives the colour picker's Reset state. */
  hasColorOverride: boolean;
  slots: LayersSlotView[];
};

export class LayersStore {
  constructor(private readonly getJot: () => Jot | undefined) {
    makeObservable(this, { layout: computed, mergedLayout: computed });
  }

  /**
   * The ordered layer → slot → track structure, resolving each `trackId`
   * against `jot.tracks` (dangling refs dropped), each `groupId` against
   * `jot.trackGroups`, and each `layerId` against `jot.layers` for its
   * name/colour. Empty when no jot is loaded.
   */
  get layout(): LayersLayerView[] {
    const jot = this.getJot();
    if (!jot) return [];
    const out: LayersLayerView[] = [];
    let index = 0;
    for (const layer of jot.ordering) {
      const meta = jot.layers.get(layer.layerId);
      const slots: LayersSlotView[] = [];
      for (const slot of layer.slots) {
        const tracks: LayersTrackView[] = [];
        for (const ref of slot.tracks) {
          const track = jot.tracks.get(ref.trackId) as Track | undefined;
          if (track) tracks.push(track);
        }
        if (slot.groupId !== null) {
          const g = jot.trackGroups.get(slot.groupId);
          slots.push({
            kind: 'group',
            id: slot.groupId,
            name: g?.name ?? slot.groupId,
            color: g?.color,
            tracks,
          });
        } else {
          slots.push({ kind: 'loose', tracks });
        }
      }
      out.push({
        id: layer.layerId,
        name: meta?.name,
        color: meta?.color ?? defaultLayerColor(index),
        hasColorOverride: meta?.color !== undefined,
        slots,
      });
      index++;
    }
    return out;
  }

  /**
   * The "Visually merge layers" view: all tracks of the same lane collapsed to
   * a single row, in the order/grouping of their TOPMOST occurrence (a lane
   * grouped in the top layer keeps that group; later layers' matching lanes
   * fold in). Audio/lyrics tracks are kept as-is. A flat list of slots (no
   * layer bands). The row's note data is the lane's union across layers
   * (`barsForLane`), and edits route per-note.
   */
  get mergedLayout(): LayersSlotView[] {
    const seen = new Set<string>();
    const out: LayersSlotView[] = [];
    for (const layer of this.layout) {
      for (const slot of layer.slots) {
        const tracks = slot.tracks.filter((t) => {
          if (t.kind !== 'instrument') return true;
          if (seen.has(t.lane)) return false;
          seen.add(t.lane);
          return true;
        });
        if (tracks.length === 0) continue;
        out.push(
          slot.kind === 'group'
            ? { kind: 'group', id: slot.id, name: slot.name, color: slot.color, tracks }
            : { kind: 'loose', tracks }
        );
      }
    }
    return out;
  }

  /** Layer ids (top-to-bottom) that carry an instrument track on `lane`. The
   *  merged view's row aggregates these (mute/solo/volume act on all). */
  layerIdsForLane(lane: string): string[] {
    const out: string[] = [];
    for (const layer of this.layout) {
      if (layer.slots.some((s) => s.tracks.some((t) => t.kind === 'instrument' && t.lane === lane))) {
        out.push(layer.id);
      }
    }
    return out;
  }

  /** Id of the layer holding `trackId` (reactive, memoised per id). */
  layerIdOfTrack = computedFn((trackId: string): string | undefined => {
    const jot = this.getJot();
    return jot ? layerIdOfTrack(jot, trackId) : undefined;
  });

  /** Group (or `null` for a loose run) holding `trackId`; `undefined` if
   *  unplaced. Reactive, memoised per id. */
  groupIdOfTrack = computedFn((trackId: string): string | null | undefined => {
    const jot = this.getJot();
    return jot ? groupIdOfTrack(jot, trackId) : undefined;
  });

  /** Display name for an instrument lane from the jot's instrument mapping
   *  (a read accessor reshaping store data; the view falls back to the lane
   *  letter). */
  instrumentName(lane: string): string | undefined {
    return this.getJot()?.instruments.get(lane)?.name;
  }
}
