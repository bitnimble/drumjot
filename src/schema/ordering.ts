/**
 * Pure helpers over the track / ordering model (see `schema.ts`):
 *
 * - {@link laneForNote} / {@link trackLaneOf}: resolve a note's lane via its
 *   `trackId` -> {@link Track} (falling back to the transitional `note.lane`,
 *   which is still the home for pattern-body template notes).
 * - {@link layerIdOfTrack} / {@link groupIdOfTrack}: reverse-lookup a track's
 *   placement (which layer / group) from `jot.ordering`.
 * - {@link TrackBuilder}: a converter-side allocator that mints one instrument
 *   track per `(layerId, lane)` and emits a sane default {@link
 *   Ordering} (one loose run per layer, tracks in first-appearance order).
 *
 * These are plain functions today; the reactive, memoised forms live on the
 * Layers store (added when consumers read them). A later augmented
 * `ReactiveJot` will fold `laneForNote` onto the note itself.
 */
import type { Jot, NoteElement, Track } from './schema';

/** Separates layerId from lane in a TrackBuilder cache key. Layer ids are
 *  converter slugs and lanes are single letters, so neither contains a slash;
 *  this can't produce an ambiguous key. */
const KEY_SEP = '/';

/** Deterministic track-entity id for a session-only runtime audio track. The
 *  audio/lyrics runtime ids are folded into `tracks`/`ordering` under these
 *  stable prefixes so the sync is idempotent and lookups (colour inheritance,
 *  panel rendering) can address them without a separate registry. */
export const audioTrackEntityId = (audioId: string): string => `audio:${audioId}`;
export const lyricsTrackEntityId = (lyricsId: string): string => `lyrics:${lyricsId}`;

/**
 * Instrument-track lanes sharing a (non-loose) group with `trackId`, in slot
 * order. Empty when the track sits in a loose run, isn't grouped, or isn't
 * placed. The single source of truth for "which instrument(s) is this track
 * paired with"; the audio-track colour inheritance and derived `lane` read it.
 * Pure, but reads `jot.ordering` / `jot.tracks`, so calling it inside a MobX
 * derivation makes that derivation react to regroups.
 */
export function groupSiblingInstrumentLanes(jot: Jot, trackId: string): string[] {
  for (const layer of jot.ordering) {
    for (const slot of layer.slots) {
      let inSlot = false;
      for (const t of slot.tracks) {
        if (t.trackId === trackId) {
          inSlot = true;
          break;
        }
      }
      if (!inSlot) continue;
      if (slot.groupId === null) return [];
      const out: string[] = [];
      for (const t of slot.tracks) {
        const track = jot.tracks.get(t.trackId) as Track | undefined;
        if (track && track.kind === 'instrument') out.push(track.lane);
      }
      return out;
    }
  }
  return [];
}

/** A track's lane, or undefined if it isn't an instrument track / not found. */
export function trackLaneOf(jot: Jot, trackId: string): string | undefined {
  const t = jot.tracks.get(trackId) as Track | undefined;
  return t && t.kind === 'instrument' ? t.lane : undefined;
}

/** A note's lane: via its `trackId` -> track, else the transitional `note.lane`
 *  (pattern-body template notes carry no `trackId`). */
export function laneForNote(jot: Jot, note: NoteElement): string {
  if (note.trackId !== undefined) {
    const lane = trackLaneOf(jot, note.trackId);
    if (lane !== undefined) return lane;
  }
  return note.lane;
}

/** The id of the layer that holds `trackId`, by reverse-lookup in the
 *  ordering. `undefined` if the track isn't placed. */
export function layerIdOfTrack(jot: Jot, trackId: string): string | undefined {
  for (const layer of jot.ordering) {
    for (const slot of layer.slots) {
      for (const t of slot.tracks) if (t.trackId === trackId) return layer.layerId;
    }
  }
  return undefined;
}

/** The group (`groupId`, or `null` for a loose run) that holds `trackId`.
 *  `undefined` if the track isn't placed at all. */
export function groupIdOfTrack(jot: Jot, trackId: string): string | null | undefined {
  for (const layer of jot.ordering) {
    for (const slot of layer.slots) {
      for (const t of slot.tracks) if (t.trackId === trackId) return slot.groupId;
    }
  }
  return undefined;
}

// ---------- Converter-side allocation ----------

/** Plain-object init shape of one ordering layer (movableList element). */
export type OrderLayerInit = {
  layerId: string;
  slots: Array<{ groupId: string | null; tracks: Array<{ trackId: string }> }>;
};

/** Plain-object init shape of an instrument track entity. */
export type InstrumentTrackInit = { id: string; kind: 'instrument'; lane: string };

/**
 * Allocates instrument tracks per `(layerId, lane)` while a converter walks
 * its notes, then emits the matching `tracks` map + default `ordering`. A
 * layer holds at most one track per lane (the DSL constraint), so repeated
 * `(layerId, lane)` calls return the same track id.
 */
export class TrackBuilder {
  private readonly byKey = new Map<string, string>();
  private readonly orderByLayer = new Map<string, string[]>();
  private n = 0;
  /** Allocated instrument tracks, keyed by id, ready as an idMap init. */
  readonly tracks: Record<string, InstrumentTrackInit> = {};

  /** @param prefix track-id prefix. Owns its own counter (NOT the shared
   *   element-id counter) so adding tracks doesn't shift element/bar ids. */
  constructor(private readonly prefix: string = 'tk') {}

  /** Get (or create) the instrument track id for `(layerId, lane)`. */
  track(layerId: string, lane: string): string {
    const key = `${layerId}${KEY_SEP}${lane}`;
    let id = this.byKey.get(key);
    if (id === undefined) {
      id = `${this.prefix}${this.n++}`;
      this.byKey.set(key, id);
      this.tracks[id] = { id, kind: 'instrument', lane };
      const list = this.orderByLayer.get(layerId) ?? [];
      list.push(id);
      this.orderByLayer.set(layerId, list);
    }
    return id;
  }

  /**
   * Default ordering for `layerIds` (in the given order): one loose run per
   * layer. Tracks are sorted by `compareLanes` when given (the converter passes
   * the default mixer kind-order so a fresh jot matches the mixer's familiar
   * top-of-kit-first layout), else first-appearance order. Empty layers still
   * appear so the layer survives a round-trip even before it has notes.
   */
  ordering(
    layerIds: readonly string[],
    compareLanes?: (a: string, b: string) => number
  ): OrderLayerInit[] {
    return layerIds.map((layerId) => {
      const ids = (this.orderByLayer.get(layerId) ?? []).slice();
      if (compareLanes) {
        ids.sort((x, y) => compareLanes(this.tracks[x].lane, this.tracks[y].lane));
      }
      return { layerId, slots: [{ groupId: null, tracks: ids.map((trackId) => ({ trackId })) }] };
    });
  }
}
