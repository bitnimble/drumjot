import { reaction } from 'mobx';
import { defaultKindForLane, getInstrumentMetadata } from 'src/instruments/instruments';
import { jotPlayer } from 'src/editing/playback/player';
import { lyricsStore } from 'src/lyrics/store';
import { audioTrackEntityId, lyricsTrackEntityId } from 'src/schema/ordering';
import type { MutableJot, OrderLayer, OrderSlot, Track } from 'src/schema/schema';

/** Deterministic track-entity ids for the session-only runtime tracks, so the
 *  sync is idempotent (re-running finds the existing entity). */
const audioTrackId = audioTrackEntityId;
const lyricsTrackId = lyricsTrackEntityId;

/**
 * The single writer of the layers model: layer metadata (`jot.layers`
 * name/colour), group metadata (`jot.trackGroups`), and the `jot.ordering`
 * arrangement (layer order, per-layer slot/track order, group membership). The
 * Layers panel and the score's gutter drag both call these; both surfaces stay
 * in sync because there is one source of truth and one writer.
 *
 * Per the store/presenter split, all `ordering`/`layers`/`trackGroups` mutation
 * lives here; `LayersStore` only reads. Mutations go straight through the
 * reactive Loro façade (nested `ReactiveList` insert/delete/move), each its own
 * commit. Also owns the reaction that folds the session-only audio/lyrics
 * tracks into `tracks` + `ordering` so they render + group alongside the
 * instrument tracks.
 */
export class LayersPresenter {
  constructor(private readonly getJot: () => MutableJot | undefined) {
    // Keep audio/lyrics tracks present in the doc model: add a track entity +
    // place it at the top of layer 0 when one loads, drop it when it unloads.
    // Existing placements (e.g. a user-grouped audio row) are left alone.
    reaction(
      () => ({
        // Reading the jot reference re-syncs after a song load (fresh ordering).
        jot: this.getJot(),
        audioIds: [...jotPlayer.audioTracks.keys()],
        lyricsIds: lyricsStore.trackIds.slice(),
      }),
      ({ audioIds, lyricsIds }) => this.syncRuntimeTracks(audioIds, lyricsIds),
      { fireImmediately: true }
    );
  }

  /**
   * Reconcile the session-only audio/lyrics tracks with `tracks` + `ordering`:
   * create + place any that are newly loaded, remove any whose runtime track is
   * gone. Idempotent; respects an existing placement. Lyrics drop loose at the
   * top of layer 0; a per-lane audio stem instead clusters into a named group
   * directly above its instrument row (see {@link placeRuntimeAudioTrack}).
   */
  private syncRuntimeTracks(audioIds: readonly string[], lyricsIds: readonly string[]): void {
    const jot = this.getJot();
    if (!jot) return;
    const desired = new Set<string>([
      ...audioIds.map(audioTrackId),
      ...lyricsIds.map(lyricsTrackId),
    ]);
    // Drop entities for runtime tracks that no longer exist.
    for (const [tid, t] of [...jot.tracks.entries()] as [string, Track][]) {
      if ((t.kind === 'audio' || t.kind === 'lyrics') && !desired.has(tid)) {
        this.removeFromOrdering(tid);
        jot.tracks.delete(tid);
      }
    }
    // Add + place newcomers (lyrics first, then audio, mirroring the old order).
    for (const id of lyricsIds) {
      const tid = lyricsTrackId(id);
      if (!jot.tracks.has(tid)) jot.tracks.set(tid, { id: tid, kind: 'lyrics', lyricsId: id });
      if (!this.locate(tid)) this.placeAtTopOfFirstLayer(tid);
    }
    for (const id of audioIds) {
      const tid = audioTrackId(id);
      if (!jot.tracks.has(tid)) jot.tracks.set(tid, { id: tid, kind: 'audio', audioId: id });
      // The load-time lane mapping (a transcribe / debug-bundle stem) clusters
      // the stem with its instrument row(s); read the full set off the runtime
      // track (a shared stem backs several lanes, the cymbal split).
      if (!this.locate(tid)) {
        this.placeRuntimeAudioTrack(tid, jotPlayer.audioTracks.get(id)?.mappedLanes ?? []);
      }
    }
  }

  /**
   * Place a freshly-loaded audio stem. When its load-time `lanes` back one or
   * more instrument rows, cluster the stem and EVERY one of those instruments
   * into a single named group, with the stem directly above its primary
   * instrument, restoring the post-transcribe "group by instrument" layout the
   * debug-bundle loader used to apply. A shared stem (the cymbal split's one
   * file backing both crash and ride) thus folds all its dependent rows under
   * it. A laneless / unmatched stem (a backing mix, the drumless `no_drums`
   * stem, or a stem for a lane this song lacks) drops loose at the top instead.
   */
  placeRuntimeAudioTrack(trackId: string, lanes: readonly string[]): void {
    if (this.clusterAudioWithInstruments(trackId, lanes)) return;
    this.placeAtTopOfFirstLayer(trackId);
  }

  /** The first instrument track carrying `lane`, anywhere in the ordering, or
   *  undefined when no row plays it. */
  private findInstrumentTrackByLane(lane: string): string | undefined {
    const jot = this.getJot();
    if (!jot) return undefined;
    for (const layer of jot.ordering) {
      for (const slot of layer.slots) {
        for (const t of slot.tracks) {
          const tr = jot.tracks.get(t.trackId) as Track | undefined;
          if (tr && tr.kind === 'instrument' && tr.lane === lane) return t.trackId;
        }
      }
    }
    return undefined;
  }

  /**
   * Cluster `audioTrackId` with the instrument rows for `lanes` into one named
   * group: the stem sits directly above its primary instrument (the first lane
   * with a row), the group takes that instrument's position + label, and every
   * other dependent instrument is folded in beneath. Reuses the primary's group
   * when it already sits in one. Returns false (caller drops the stem loose)
   * when none of the lanes back a placed instrument row.
   */
  private clusterAudioWithInstruments(audioTrackId: string, lanes: readonly string[]): boolean {
    const jot = this.getJot();
    if (!jot) return false;
    // Resolve the placed instrument rows for these lanes, primary first, deduped
    // (a shared stem can repeat a lane; distinct lanes never share a track).
    const instr: Array<{ lane: string; id: string }> = [];
    for (const lane of lanes) {
      const id = this.findInstrumentTrackByLane(lane);
      if (id !== undefined && !instr.some((x) => x.id === id)) instr.push({ lane, id });
    }
    if (instr.length === 0) return false;
    const primary = instr[0];

    const loc = this.locate(primary.id);
    if (!loc) return false;
    if (jot.ordering.at(loc.li)!.slots.at(loc.si)!.groupId === null) {
      if (this.createGroup(primary.id, this.instrumentLabel(primary.lane)) === undefined) return false;
    }
    // Re-locate: createGroup moved the instrument into its own group slot. Slot
    // the stem directly above it.
    const gloc = this.locate(primary.id);
    if (!gloc) return false;
    const layer = jot.ordering.at(gloc.li)!;
    const slot = layer.slots.at(gloc.si)!;
    if (slot.groupId === null) return false;
    slot.tracks.insert(gloc.ti, { trackId: audioTrackId });

    // Fold each dependent instrument into the group, in order, beneath the
    // primary (each lands right after the previously-placed member).
    let anchor = primary.id;
    for (let i = 1; i < instr.length; i++) {
      this.moveTrackAfter(instr[i].id, layer.layerId, anchor);
      anchor = instr[i].id;
    }
    return true;
  }

  /** A lane's display label: its `Instrument.name`, else the kind's default
   *  label (recovering the kind from the lane letter when it's `custom`). */
  private instrumentLabel(lane: string): string {
    const instrument = this.getJot()?.instruments.get(lane);
    if (instrument?.name) return instrument.name;
    let kind = instrument?.kind ?? 'custom';
    if (kind === 'custom') kind = defaultKindForLane(lane);
    return getInstrumentMetadata(kind).label;
  }

  /** Insert a track at the very top of layer 0 (the loose run there, or a new
   *  one). Used for freshly-loaded audio/lyrics before the user arranges them. */
  private placeAtTopOfFirstLayer(trackId: string): void {
    const jot = this.getJot();
    const layer0 = jot?.ordering.at(0);
    if (!layer0) return;
    const first = layer0.slots.at(0);
    if (first && first.groupId === null) {
      first.tracks.insert(0, { trackId });
    } else {
      layer0.slots.insert(0, { groupId: null, tracks: [{ trackId }] } as unknown as OrderSlot);
    }
  }

  /** Remove a track ref from wherever it sits in `ordering`, pruning an emptied
   *  loose run. (Group slots are left even if emptied.) */
  private removeFromOrdering(trackId: string): void {
    const loc = this.locate(trackId);
    const jot = this.getJot();
    if (!loc || !jot) return;
    const layer = jot.ordering.at(loc.li)!;
    const slot = layer.slots.at(loc.si)!;
    slot.tracks.delete(loc.ti);
    if (slot.tracks.length === 0 && slot.groupId === null) layer.slots.delete(loc.si);
  }

  // ---------- Layer metadata ----------

  /** Set (or clear, with `undefined`) a layer's band colour. */
  setLayerColor(layerId: string, color: string | undefined): void {
    const jot = this.getJot();
    const layer = jot?.layers.get(layerId);
    if (!jot || !layer) return;
    jot.layers.set(layerId, {
      id: layerId,
      ...(layer.name !== undefined ? { name: layer.name } : {}),
      ...(color !== undefined ? { color } : {}),
    });
  }

  /** Rename a layer (empty/undefined clears the name). */
  setLayerName(layerId: string, name: string | undefined): void {
    const jot = this.getJot();
    const layer = jot?.layers.get(layerId);
    if (!jot || !layer) return;
    const trimmed = name?.trim();
    jot.layers.set(layerId, {
      id: layerId,
      ...(trimmed ? { name: trimmed } : {}),
      ...(layer.color !== undefined ? { color: layer.color } : {}),
    });
  }

  // ---------- Group metadata ----------

  /**
   * Wrap `trackId` in a fresh named group **in place** (the start of a group the
   * user then drags more tracks into). Mints a group id + a
   * {@link TrackGroupSchema} entry and replaces the track, where it sits, with a
   * one-track group slot, splitting its loose run so the group lands at exactly
   * the track's visual position (not at the run's edge). Only loose tracks are
   * wrappable; returns undefined if the track isn't placed or is already grouped.
   */
  createGroup(trackId: string, name?: string): string | undefined {
    const jot = this.getJot();
    if (!jot) return undefined;
    const loc = this.locate(trackId);
    if (!loc) return undefined;
    const layer = jot.ordering.at(loc.li)!;
    const slot = layer.slots.at(loc.si)!;
    if (slot.groupId !== null) return undefined; // already grouped

    // Mint `g<n>` above the highest existing numeric group id.
    let n = 0;
    for (const id of jot.trackGroups.keys()) {
      const m = /^g(\d+)$/.exec(id);
      if (m) n = Math.max(n, Number(m[1]) + 1);
    }
    const groupId = `g${n}`;
    jot.trackGroups.set(groupId, { id: groupId, name: name ?? `Group ${n + 1}` });

    // Split the loose run around `trackId`: head [0..ti-1] stays in `slot`, the
    // new group slot takes its place, and any tail [ti+1..] becomes a fresh
    // loose slot after the group, so the group sits exactly where the track was.
    const tail: { trackId: string }[] = [];
    for (let k = loc.ti + 1; k < slot.tracks.length; k++) {
      tail.push({ trackId: slot.tracks.at(k)!.trackId });
    }
    for (let k = slot.tracks.length - 1; k >= loc.ti; k--) slot.tracks.delete(k);
    const groupSlot = { groupId, tracks: [{ trackId }] } as unknown as OrderSlot;
    if (loc.ti === 0) {
      // Head is empty: the group takes the slot's position; drop the empty head.
      layer.slots.insert(loc.si, groupSlot);
      layer.slots.delete(loc.si + 1);
      if (tail.length) {
        layer.slots.insert(loc.si + 1, { groupId: null, tracks: tail } as unknown as OrderSlot);
      }
    } else {
      layer.slots.insert(loc.si + 1, groupSlot);
      if (tail.length) {
        layer.slots.insert(loc.si + 2, { groupId: null, tracks: tail } as unknown as OrderSlot);
      }
    }
    return groupId;
  }

  /**
   * Insert `slotInit` into `layer` so it lands immediately before `beforeTrackId`
   * in the visual (flattened) order, splitting a loose run when the anchor sits
   * mid-run. `beforeTrackId === null` appends at the layer's end. Returns false
   * WITHOUT mutating when the anchor is inside a group but isn't its first track:
   * slots don't nest, so a group can only land at a top-level boundary, never
   * inside another group. The caller must pre-validate (see
   * {@link canInsertSlotBefore}) before any destructive step to avoid orphaning.
   */
  private insertSlotBefore(
    layer: OrderLayer,
    beforeTrackId: string | null,
    slotInit: OrderSlot
  ): boolean {
    if (beforeTrackId === null) {
      layer.slots.insert(layer.slots.length, slotInit);
      return true;
    }
    for (let si = 0; si < layer.slots.length; si++) {
      const slot = layer.slots.at(si)!;
      for (let ti = 0; ti < slot.tracks.length; ti++) {
        if (slot.tracks.at(ti)!.trackId !== beforeTrackId) continue;
        if (slot.groupId !== null) {
          if (ti !== 0) return false; // inside a group, not a legal boundary
          layer.slots.insert(si, slotInit);
          return true;
        }
        if (ti === 0) {
          layer.slots.insert(si, slotInit);
          return true;
        }
        // Split the loose run: [0..ti-1] stays, slotInit between, [ti..] tail.
        const tail: { trackId: string }[] = [];
        for (let k = ti; k < slot.tracks.length; k++) {
          tail.push({ trackId: slot.tracks.at(k)!.trackId });
        }
        for (let k = slot.tracks.length - 1; k >= ti; k--) slot.tracks.delete(k);
        layer.slots.insert(si + 1, slotInit);
        layer.slots.insert(si + 2, { groupId: null, tracks: tail } as unknown as OrderSlot);
        return true;
      }
    }
    return false; // anchor not in this layer
  }

  /**
   * Whether {@link insertSlotBefore} would accept `beforeTrackId` as a drop
   * boundary in `layer` (a loose track at any position, the first track of a
   * group, or `null` for append). False for a non-first track inside a group:
   * dropping a whole group there would nest it, which this UI disallows.
   */
  private canInsertSlotBefore(layer: OrderLayer, beforeTrackId: string | null): boolean {
    if (beforeTrackId === null) return true;
    for (let si = 0; si < layer.slots.length; si++) {
      const slot = layer.slots.at(si)!;
      for (let ti = 0; ti < slot.tracks.length; ti++) {
        if (slot.tracks.at(ti)!.trackId === beforeTrackId) {
          return slot.groupId === null || ti === 0;
        }
      }
    }
    return false;
  }

  /** Dissolve a group: its slot becomes a loose run (tracks stay in place) and
   *  the {@link TrackGroupSchema} entry is removed. */
  ungroup(groupId: string): void {
    const jot = this.getJot();
    if (!jot) return;
    for (const layer of jot.ordering) {
      for (const slot of layer.slots) {
        if (slot.groupId === groupId) slot.groupId = null;
      }
    }
    jot.trackGroups.delete(groupId);
  }

  setGroupName(groupId: string, name: string): void {
    const jot = this.getJot();
    const g = jot?.trackGroups.get(groupId);
    if (!jot || !g) return;
    jot.trackGroups.set(groupId, { id: groupId, name, ...(g.color !== undefined ? { color: g.color } : {}) });
  }

  setGroupColor(groupId: string, color: string | undefined): void {
    const jot = this.getJot();
    const g = jot?.trackGroups.get(groupId);
    if (!jot || !g) return;
    jot.trackGroups.set(groupId, { id: groupId, name: g.name, ...(color !== undefined ? { color } : {}) });
  }

  // ---------- Ordering ----------

  /** Reorder whole layers (top↔bottom in the score). */
  reorderLayer(fromIndex: number, toIndex: number): void {
    const jot = this.getJot();
    if (!jot) return;
    const n = jot.ordering.length;
    if (fromIndex < 0 || fromIndex >= n || toIndex < 0 || toIndex >= n || fromIndex === toIndex) return;
    jot.ordering.move(fromIndex, toIndex);
  }

  /**
   * Move a track within / across layers. `beforeTrackId` (when given and found
   * in the target layer) inserts the track immediately before it, joining that
   * track's slot, so dropping above a track that's in a group joins the group,
   * and dropping into a loose run keeps it loose. `beforeTrackId === null`
   * appends to the target layer's last loose run (creating one if needed).
   * Emptied loose slots are pruned.
   */
  moveTrack(trackId: string, targetLayerId: string, beforeTrackId: string | null): void {
    const jot = this.getJot();
    if (!jot) return;
    const loc = this.locate(trackId);
    if (!loc) return;

    // Resolve the target layer FIRST, before any destructive mutation: an
    // unknown `targetLayerId` must bail cleanly, never leave the track removed
    // from its source slot but unplaced (which would orphan its notes, they'd
    // fall out of `membership` into the empty-layer bucket and stop rendering).
    // Deleting a slot below never shifts the layer ordering, so this index
    // stays valid.
    let tli = -1;
    for (let i = 0; i < jot.ordering.length; i++) {
      if (jot.ordering.at(i)!.layerId === targetLayerId) {
        tli = i;
        break;
      }
    }
    if (tli < 0) return;

    // Remove from the source slot, pruning an emptied loose run.
    const srcLayer = jot.ordering.at(loc.li)!;
    const srcSlot = srcLayer.slots.at(loc.si)!;
    srcSlot.tracks.delete(loc.ti);
    if (srcSlot.tracks.length === 0 && srcSlot.groupId === null) {
      srcLayer.slots.delete(loc.si);
    }

    const tLayer = jot.ordering.at(tli)!;

    if (beforeTrackId !== null) {
      for (let si = 0; si < tLayer.slots.length; si++) {
        const slot = tLayer.slots.at(si)!;
        for (let ti = 0; ti < slot.tracks.length; ti++) {
          if (slot.tracks.at(ti)!.trackId === beforeTrackId) {
            slot.tracks.insert(ti, { trackId });
            return;
          }
        }
      }
    }

    // Append: into the last loose run, or a fresh loose run.
    let lastLoose = -1;
    for (let si = tLayer.slots.length - 1; si >= 0; si--) {
      if (tLayer.slots.at(si)!.groupId === null) {
        lastLoose = si;
        break;
      }
    }
    if (lastLoose >= 0) {
      const slot = tLayer.slots.at(lastLoose)!;
      slot.tracks.insert(slot.tracks.length, { trackId });
    } else {
      // The write surface is typed with the read shape (a nested `ReactiveList`)
      // but the engine deep-creates nested containers from plain Init data, so
      // pass the plain slot through a cast.
      const slot = { groupId: null, tracks: [{ trackId }] } as unknown as OrderSlot;
      tLayer.slots.insert(tLayer.slots.length, slot);
    }
  }

  /**
   * Like {@link moveTrack} but inserts the track immediately *after*
   * `afterTrackId` (joining that track's slot, so dropping below a grouped
   * track joins the group). Used by the panel's bottom-edge drop zone. No-op
   * when dropping a track after itself, or when the target layer is unknown
   * (never orphans, same guard as {@link moveTrack}).
   */
  moveTrackAfter(trackId: string, targetLayerId: string, afterTrackId: string): void {
    const jot = this.getJot();
    if (!jot || trackId === afterTrackId) return;
    const loc = this.locate(trackId);
    if (!loc) return;

    // Resolve the target layer AND confirm the anchor exists in it before any
    // destructive mutation (see moveTrack): a mismatched layer/anchor must bail
    // cleanly, never leave the track removed from its source slot but unplaced
    // (which would orphan its notes). `afterTrackId !== trackId` (guarded
    // above), so the source delete below can't invalidate this.
    let tli = -1;
    for (let i = 0; i < jot.ordering.length && tli < 0; i++) {
      const layer = jot.ordering.at(i)!;
      if (layer.layerId !== targetLayerId) continue;
      for (const slot of layer.slots) {
        for (const t of slot.tracks) {
          if (t.trackId === afterTrackId) {
            tli = i;
            break;
          }
        }
      }
    }
    if (tli < 0) return;

    // Remove from the source slot, pruning an emptied loose run.
    const srcLayer = jot.ordering.at(loc.li)!;
    const srcSlot = srcLayer.slots.at(loc.si)!;
    srcSlot.tracks.delete(loc.ti);
    if (srcSlot.tracks.length === 0 && srcSlot.groupId === null) {
      srcLayer.slots.delete(loc.si);
    }

    // Re-find the anchor by id (its indices may have shifted after the delete)
    // and insert right after it, joining its slot. The anchor is guaranteed
    // present (confirmed above, and the delete can't have removed it).
    const tLayer = jot.ordering.at(tli)!;
    for (let si = 0; si < tLayer.slots.length; si++) {
      const slot = tLayer.slots.at(si)!;
      for (let ti = 0; ti < slot.tracks.length; ti++) {
        if (slot.tracks.at(ti)!.trackId === afterTrackId) {
          slot.tracks.insert(ti + 1, { trackId });
          return;
        }
      }
    }
  }

  /**
   * Group `draggedId` with `targetId` (the panel's centre drop zone). When the
   * target is already in a group, the dragged track joins that group right
   * after it; when the target is loose, a fresh group wrapping just the two is
   * minted. No-op when dropping a track onto itself.
   */
  groupTracks(draggedId: string, targetId: string): void {
    const jot = this.getJot();
    if (!jot || draggedId === targetId) return;
    const tLoc = this.locate(targetId);
    if (!tLoc) return;
    const targetLayerId = jot.ordering.at(tLoc.li)!.layerId;
    const targetSlot = jot.ordering.at(tLoc.li)!.slots.at(tLoc.si)!;
    // Wrap a loose target in a new group first; a grouped target already has a
    // group slot to fold the dragged track into.
    if (targetSlot.groupId === null && !this.createGroup(targetId)) return;
    this.moveTrackAfter(draggedId, targetLayerId, targetId);
  }

  /**
   * Delete an *empty* group outright: removes its slot from `ordering` and the
   * {@link TrackGroupSchema} entry. The panel offers this instead of Ungroup
   * once a group has no tracks left. A no-op (leaving the group intact) if the
   * slot still holds tracks, so a direct caller can't orphan them, non-empty
   * groups must go through {@link ungroup}.
   */
  deleteGroup(groupId: string): void {
    const jot = this.getJot();
    if (!jot) return;
    for (let li = 0; li < jot.ordering.length; li++) {
      const layer = jot.ordering.at(li)!;
      for (let si = 0; si < layer.slots.length; si++) {
        const slot = layer.slots.at(si)!;
        if (slot.groupId === groupId) {
          if (slot.tracks.length > 0) return; // safety: only delete empty groups
          layer.slots.delete(si);
          jot.trackGroups.delete(groupId);
          return;
        }
      }
    }
    // No slot carries this group (a dangling entry); drop the metadata.
    jot.trackGroups.delete(groupId);
  }

  /**
   * Move a whole group (its slot) within / across layers. `beforeTrackId` (found
   * in the target layer) places the group immediately before that track in the
   * visual order, splitting a loose run when needed so it can land anywhere, not
   * just at a run's edge; `null` appends it to the target layer. The group keeps
   * its tracks; only the slot's position (and layer) changes. No-op (no nesting)
   * when the anchor sits inside another group, or is one of the group's own
   * tracks; validated before the destructive delete so nothing is orphaned.
   */
  moveGroup(groupId: string, targetLayerId: string, beforeTrackId: string | null): void {
    const jot = this.getJot();
    if (!jot) return;
    // Locate the group's slot.
    let srcLi = -1;
    let srcSi = -1;
    for (let li = 0; li < jot.ordering.length && srcLi < 0; li++) {
      const layer = jot.ordering.at(li)!;
      for (let si = 0; si < layer.slots.length; si++) {
        if (layer.slots.at(si)!.groupId === groupId) {
          srcLi = li;
          srcSi = si;
          break;
        }
      }
    }
    if (srcLi < 0) return;

    const srcLayer = jot.ordering.at(srcLi)!;
    const slot = srcLayer.slots.at(srcSi)!;
    const trackIds = [...slot.tracks].map((t) => t.trackId);
    // Dropping the group onto one of its own tracks is a no-op.
    if (beforeTrackId !== null && trackIds.includes(beforeTrackId)) return;

    // Resolve the target layer + validate the anchor BEFORE the destructive
    // delete: an unknown layer or a nesting drop (anchor inside another group)
    // must bail cleanly, never leave the group's tracks removed from `ordering`
    // but unplaced (which would orphan their notes).
    let tli = -1;
    for (let i = 0; i < jot.ordering.length; i++) {
      if (jot.ordering.at(i)!.layerId === targetLayerId) {
        tli = i;
        break;
      }
    }
    if (tli < 0) return;
    const tLayer = jot.ordering.at(tli)!;
    if (!this.canInsertSlotBefore(tLayer, beforeTrackId)) return;

    srcLayer.slots.delete(srcSi);
    const moved = { groupId, tracks: trackIds.map((trackId) => ({ trackId })) } as unknown as OrderSlot;
    this.insertSlotBefore(tLayer, beforeTrackId, moved);
  }

  // ---------- Instrument-track provisioning (editing) ----------

  /**
   * Find the instrument track for `(layerId, lane)`, minting + placing one when
   * the layer has none yet. The editing presenter calls this to resolve a
   * placed note's `trackId` on insert / cross-lane move, so a note's home is
   * always a real track and its layer is derived from `ordering` (never stored
   * on the note). A layer holds at most one track per lane, so the lookup is
   * unambiguous. Returns the track id (undefined only when there's no jot).
   */
  ensureInstrumentTrack(layerId: string, lane: string): string | undefined {
    const jot = this.getJot();
    if (!jot) return undefined;
    // Reuse the existing instrument track for this (layer, lane) when placed.
    for (const layer of jot.ordering) {
      if (layer.layerId !== layerId) continue;
      for (const slot of layer.slots) {
        for (const t of slot.tracks) {
          const tr = jot.tracks.get(t.trackId) as Track | undefined;
          if (tr && tr.kind === 'instrument' && tr.lane === lane) return t.trackId;
        }
      }
    }
    // Mint a new instrument track above the highest existing numeric `tk<n>`.
    let n = 0;
    for (const id of jot.tracks.keys()) {
      const m = /^tk(\d+)$/.exec(id);
      if (m) n = Math.max(n, Number(m[1]) + 1);
    }
    const trackId = `tk${n}`;
    jot.tracks.set(trackId, { id: trackId, kind: 'instrument', lane });
    this.appendTrackToLayer(layerId, trackId);
    return trackId;
  }

  /** Append a track to a layer's last loose run (creating the run, or the whole
   *  layer slot list, when absent). */
  private appendTrackToLayer(layerId: string, trackId: string): void {
    const jot = this.getJot();
    if (!jot) return;
    let tLayer: OrderLayer | undefined;
    for (const l of jot.ordering) {
      if (l.layerId === layerId) {
        tLayer = l;
        break;
      }
    }
    if (!tLayer) {
      jot.ordering.insert(jot.ordering.length, {
        layerId,
        slots: [{ groupId: null, tracks: [{ trackId }] }],
      } as unknown as OrderLayer);
      return;
    }
    let lastLoose = -1;
    for (let si = tLayer.slots.length - 1; si >= 0; si--) {
      if (tLayer.slots.at(si)!.groupId === null) {
        lastLoose = si;
        break;
      }
    }
    if (lastLoose >= 0) {
      const slot = tLayer.slots.at(lastLoose)!;
      slot.tracks.insert(slot.tracks.length, { trackId });
    } else {
      tLayer.slots.insert(tLayer.slots.length, {
        groupId: null,
        tracks: [{ trackId }],
      } as unknown as OrderSlot);
    }
  }

  /** Locate a track's `{layer, slot, track}` indices in `jot.ordering`. */
  private locate(trackId: string): { li: number; si: number; ti: number } | undefined {
    const jot = this.getJot();
    if (!jot) return undefined;
    for (let li = 0; li < jot.ordering.length; li++) {
      const layer = jot.ordering.at(li)!;
      for (let si = 0; si < layer.slots.length; si++) {
        const slot = layer.slots.at(si)!;
        for (let ti = 0; ti < slot.tracks.length; ti++) {
          if (slot.tracks.at(ti)!.trackId === trackId) return { li, si, ti };
        }
      }
    }
    return undefined;
  }
}
