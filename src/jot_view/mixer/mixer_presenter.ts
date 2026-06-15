import { comparer, makeAutoObservable, reaction } from 'mobx';
import { lyricsStore, LyricsTrackId } from 'src/lyrics/store';
import { AudioTrackId } from 'src/jot_view/playback/audio_tracks';
import { jotPlayer } from 'src/jot_view/playback/player';
import { buildDebugBundleTrackOrder, reorderTrackOrder, trackKeyEq, type TrackKey } from 'src/jot_view/tracks/tracks';
import { JotViewStore } from '../jot_view_store';
import { MixerStore, clampVolume, collectJotPitches } from './mixer_store';
import { toastStore } from '../../ui/toasts/toasts';

/**
 * Mutations over {@link MixerStore}, per-row + master mute/solo/volume,
 * the user-customisable row order, and the audio-track split stubs. Owns
 * the reaction that keeps the row order synced with the live
 * track/pitch/lyrics set and the instrument-track prune, and wires the
 * store in as the player's {@link MixerContext} for audio-track colour
 * inheritance. The engine's mute/solo/volume filter is no longer pushed
 * from here; the player pulls it off the PlaybackStore computeds (see
 * `PlaybackPresenter`).
 */
export class MixerPresenter {
  readonly mixer: MixerStore;
  readonly jotViewStore: JotViewStore;

  constructor(mixer: MixerStore, jotViewStore: JotViewStore) {
    this.mixer = mixer;
    this.jotViewStore = jotViewStore;
    makeAutoObservable(this, { mixer: false, jotViewStore: false });

    // Wire the mixer store in as the player's mixer context so freshly-
    // constructed AudioTracks can resolve grouped-instrument colour
    // inheritance. Done before any reactions fire so loadAudioTrack calls
    // made during the same tick see a populated context.
    jotPlayer.attachMixerContext(this.mixer);

    // Prune instrument-track view-models for pitches no longer present in
    // the active jot. The override is store-owned and survives jot
    // reloads, so a pitch that comes back in a later jot picks up its
    // previous override; we only forget pitches that disappeared from the
    // current jot to keep the map from growing unboundedly across a long
    // session. `fireImmediately` runs the prune once on boot.
    reaction(
      () => new Set(this.mixer.jotPitches),
      (pitches) => {
        for (const p of Array.from(this.mixer.instrumentTracks.keys())) {
          if (!pitches.has(p)) this.mixer.instrumentTracks.delete(p);
        }
      },
      { fireImmediately: true, equals: comparer.structural }
    );

    // NOTE: the engine no longer has mixer state pushed into it here. The
    // player PULLS its mute/solo/volume filter + section-audibility off the
    // PlaybackStore computeds (which delegate to this store's `pitchFilter`
    // / `audioTrackFilter` / `isAudio|DrumSectionAudible`); the reactions
    // that fire the imperative audio-graph re-apply live in
    // `PlaybackPresenter`.

    // Keep `trackOrder` synced with the live audio-track set and the
    // current jot's pitches. Dropped rows are removed; newly-discovered
    // rows are slotted at a sensible default position so the user's
    // drag-and-drop ordering of surviving rows is preserved.
    reaction(
      () => ({
        audioIds: Array.from(jotPlayer.audioTracks.keys()),
        pitches: this.mixer.jotPitches,
        lyricsIds: lyricsStore.trackIds.slice(),
      }),
      ({ audioIds, pitches, lyricsIds }) => this.syncTrackOrder(audioIds, pitches, lyricsIds),
      { fireImmediately: true }
    );
  }

  /**
   * Drop entries from {@link MixerStore.trackOrder} that no longer
   * correspond to a live audio track, jot pitch, or lyrics track; then
   * append the missing ones at a sensible default position so the row
   * appears immediately:
   *   - new audio track  → after the last existing audio entry (or top of
   *     the list if no audio entries exist yet)
   *   - new pitch        → end of the list
   *   - new lyrics row   → just after the last existing lyrics row,
   *     keeping the lyrics group contiguous (top of list when none exist).
   *
   * Existing entries keep their relative order so a user drag survives an
   * audio-track add/remove or a jot reload that didn't change the pitches.
   */
  private syncTrackOrder(
    audioIds: AudioTrackId[],
    pitches: readonly string[],
    lyricsIds: readonly LyricsTrackId[]
  ): void {
    const wanted: TrackKey[] = [
      ...lyricsIds.map((id) => ({ kind: 'lyrics' as const, id })),
      ...audioIds.map((id) => ({ kind: 'audio' as const, id })),
      ...pitches.map((pitch) => ({ kind: 'instrument' as const, pitch })),
    ];
    const next: TrackKey[] = this.mixer.trackOrder.filter((k) =>
      wanted.some((w) => trackKeyEq(w, k))
    );
    for (const w of wanted) {
      if (next.some((k) => trackKeyEq(k, w))) continue;
      if (w.kind === 'lyrics') {
        let insertAt: number | undefined;
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].kind === 'lyrics') {
            insertAt = i + 1;
            break;
          }
        }
        if (insertAt === undefined) {
          next.unshift(w);
        } else {
          next.splice(insertAt, 0, w);
        }
      } else if (w.kind === 'audio') {
        let insertAt = 0;
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].kind === 'audio') {
            insertAt = i + 1;
            break;
          }
        }
        next.splice(insertAt, 0, w);
      } else {
        next.push(w);
      }
    }
    // The reaction fires whenever its data fn returns a new object, even
    // when the underlying sets are unchanged. Skip the observable
    // assignment when the order is structurally identical (identity AND
    // groupId) so the mixer renderer doesn't pay an unnecessary re-render
    // but a real grouping change still propagates.
    if (
      next.length === this.mixer.trackOrder.length &&
      next.every(
        (k, i) =>
          trackKeyEq(k, this.mixer.trackOrder[i]) && k.groupId === this.mixer.trackOrder[i].groupId
      )
    ) {
      return;
    }
    this.mixer.trackOrder = next;
  }

  /**
   * Reorder the mixer by moving the row at `fromIdx` to position `toIdx`.
   * Both indices refer to positions in the *current* trackOrder; a no-op
   * move is silently dropped. On drop the moved row's `groupId` becomes
   * the adjacent rows' shared group (dropped inside a group → join it;
   * dropped at a boundary / top / bottom → becomes solo).
   */
  moveTrack(fromIdx: number, toIdx: number): void {
    const prev = this.mixer.trackOrder;
    const next = reorderTrackOrder(prev, fromIdx, toIdx);
    // `reorderTrackOrder` returns the same array reference on a no-op move,
    // so this only writes (and wakes observers) on a real reorder.
    if (next === prev) return;
    // If the reorder pulled an audio row out of its group (grouped ->
    // solo), bake the group-derived pitch into the track's own state
    // before the group is gone, so it keeps its instrument association.
    const movedPrev = prev[fromIdx];
    if (movedPrev?.kind === 'audio' && movedPrev.groupId !== undefined) {
      const movedNext = next.find((k) => k.kind === 'audio' && k.id === movedPrev.id);
      if (movedNext && movedNext.groupId === undefined) {
        jotPlayer.audioTracks.get(movedPrev.id)?.detachPitch();
      }
    }
    this.mixer.trackOrder = next as TrackKey[];
  }

  toggleAudioMasterMute() {
    this.mixer.audioMasterMuted = !this.mixer.audioMasterMuted;
  }

  toggleDrumMasterMute() {
    this.mixer.drumMasterMuted = !this.mixer.drumMasterMuted;
  }

  /** Enabling solo clears the matching master-mute so the section can
   * actually be heard; mirrors `toggleSolo` for per-row state. */
  toggleAudioMasterSolo() {
    if (this.mixer.audioMasterSoloed) {
      this.mixer.audioMasterSoloed = false;
    } else {
      this.mixer.audioMasterSoloed = true;
      this.mixer.audioMasterMuted = false;
    }
  }

  toggleDrumMasterSolo() {
    if (this.mixer.drumMasterSoloed) {
      this.mixer.drumMasterSoloed = false;
    } else {
      this.mixer.drumMasterSoloed = true;
      this.mixer.drumMasterMuted = false;
    }
  }

  toggleMute(pitch: string) {
    if (this.mixer.mutedPitches.has(pitch)) this.mixer.mutedPitches.delete(pitch);
    else this.mixer.mutedPitches.add(pitch);
  }

  toggleSolo(pitch: string) {
    if (this.mixer.soloedPitches.has(pitch)) {
      this.mixer.soloedPitches.delete(pitch);
    } else {
      this.mixer.soloedPitches.add(pitch);
      this.mixer.mutedPitches.delete(pitch);
    }
  }

  setPitchVolume(pitch: string, v: number) {
    this.mixer.pitchVolumes.set(pitch, clampVolume(v));
  }

  toggleAudioTrackMute(id: AudioTrackId) {
    if (this.mixer.mutedAudioTracks.has(id)) this.mixer.mutedAudioTracks.delete(id);
    else this.mixer.mutedAudioTracks.add(id);
  }

  toggleAudioTrackSolo(id: AudioTrackId) {
    if (this.mixer.soloedAudioTracks.has(id)) {
      this.mixer.soloedAudioTracks.delete(id);
    } else {
      this.mixer.soloedAudioTracks.add(id);
      this.mixer.mutedAudioTracks.delete(id);
    }
  }

  setAudioTrackVolume(id: AudioTrackId, v: number) {
    this.mixer.audioTrackVolumes.set(id, clampVolume(v));
  }

  /** Mute a batch of audio tracks. Used by the song loaders
   * (JotViewPresenter) to default per-pitch stems / drum tracks to
   * muted so the audible drums come from the score scheduler. */
  muteAudioTracks(ids: readonly AudioTrackId[]): void {
    for (const id of ids) this.mixer.mutedAudioTracks.add(id);
  }

  /** Drop a removed audio track's mute/solo/volume so it doesn't linger
   * (ids are never reused), and so clearing the only soloed track doesn't
   * leave a phantom solo silencing everything else. The colour override
   * lives on the AudioTrack instance and is freed by the player. */
  clearAudioTrack(id: AudioTrackId): void {
    jotPlayer.clearAudioTrack(id);
    this.mixer.mutedAudioTracks.delete(id);
    this.mixer.soloedAudioTracks.delete(id);
    this.mixer.audioTrackVolumes.delete(id);
  }

  /** Mark an audio track as currently being split. Drives the per-row
   *  spinner; safe to call multiple times (latest call wins). */
  beginAudioTrackSplit(id: AudioTrackId, kind: 'mix' | 'pieces'): void {
    this.mixer.audioTrackSplitStatuses.set(id, { phase: 'splitting', kind });
  }

  /** Clear the splitting status once the work has finished. */
  endAudioTrackSplit(id: AudioTrackId): void {
    this.mixer.audioTrackSplitStatuses.delete(id);
  }

  /** Stub: "Split into drums + backing" from the audio-track overflow
   * menu. The transcriber-side wiring is deferred; surface a status pill
   * so the click visibly does something in the meantime. */
  splitAudioTrackFromMix(id: AudioTrackId): void {
    const track = jotPlayer.audioTracks.get(id);
    const name = track ? track.filename : id;
    toastStore.showError(`Split into drums + backing on "${name}" isn't wired up yet.`);
  }

  /** Stub: "Split into kick / snare / hi-hat / cymbals". See
   * {@link splitAudioTrackFromMix}. */
  splitAudioTrackDrumPieces(id: AudioTrackId): void {
    const track = jotPlayer.audioTracks.get(id);
    const name = track ? track.filename : id;
    toastStore.showError(`Split into drum pieces on "${name}" isn't wired up yet.`);
  }

  /** Drop every loaded audio track. Used when a new source replaces the
   * current song so the previous song's tracks don't linger. */
  clearAllAudioTracks(): void {
    for (const id of Array.from(jotPlayer.audioTracks.keys())) {
      this.clearAudioTrack(id);
    }
  }

  /** Reset the per-pitch mixer (mute/solo/volume). Keyed by DSL pitch
   * letter, not by song, so without this a setting on one song bleeds onto
   * the next song's matching rows. */
  resetPitchMixer(): void {
    this.mixer.mutedPitches.clear();
    this.mixer.soloedPitches.clear();
    this.mixer.pitchVolumes.clear();
  }

  /**
   * Re-order the mixer after a debug bundle is loaded so each per-pitch
   * audio stem sits immediately above its instrument row, with any
   * unmatched audio at the top. The ordering itself is the pure
   * {@link buildDebugBundleTrackOrder}; this just feeds it the current
   * jot's pitches. Called by the bundle loader (JotViewPresenter).
   */
  applyDebugBundleTrackOrder(loadedByKey: ReadonlyMap<string, AudioTrackId>): void {
    const pitches = collectJotPitches(this.jotViewStore.structural);
    this.mixer.trackOrder = buildDebugBundleTrackOrder(pitches, loadedByKey);
  }
}
