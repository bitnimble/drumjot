import { comparer, makeAutoObservable, reaction, runInAction } from 'mobx';
import { loadDebugZip, NO_DRUMS_KEY } from 'src/debug_zip';
import { ExampleJot } from 'src/fakes';
import { px, RenderedJot } from 'src/jot';
import {
  AlignLyricsRequest,
  LyricLine,
  LyricsSource,
  LyricsTrackId,
  alignLyricsForced,
  lyricsStore,
  nameLooksLikeVocals,
  parseEnhancedLrc,
  stripLyricNoise,
} from 'src/lyrics';
import { fromMidi } from 'src/midi';
import { parse, ParseError } from 'src/parser';
import {
  AudioTrackId,
  AudioTrackRole,
  jotPlayer,
  buildTimeline,
  xToTime,
} from 'src/playback';
import { pickDominantBpmAndTime } from 'src/playback/timeline';
import { loadParadbZip } from 'src/rlrr';
import {
  BeatInput,
  DrumSeparator,
  LlmModel,
  stemUrl,
  titleFromFilename,
  transcriber,
  TranscribeProgress,
  TranscribeStage,
} from 'src/transcriber';
import { transcribeSuccessToastMessage } from './toasts_messages';
import { toastStore } from './toasts';
import { SettingsStore } from './stores/settings_store';
import { DocumentStore } from './stores/document_store';
import { TranscribeStore } from './stores/transcribe_store';
import { ProvenanceStore } from './stores/provenance_store';
import { LyricsAlignStore } from './stores/lyrics_align_store';
import { PlaybackStore } from './stores/playback_store';
import { ViewportStore } from './stores/viewport_store';
import { MixerStore, clampVolume, collectJotPitches } from './stores/mixer_store';
import {
  BASE_BAR_WIDTH,
  MAX_GUTTER_WIDTH,
  MAX_ZOOM,
  MIN_GUTTER_WIDTH,
  MIN_ZOOM,
  snapToDevicePx,
} from './stores/viewport_store';

import { buildDebugBundleTrackOrder, trackKeyEq, type TrackKey } from 'src/tracks';

/**
 * Dependencies the presenter orchestrates over. Every store is a plain
 * data container; the presenter is the single place that mutates them.
 *
 * This is a TEMPORARY catch-all for all the orchestration that used to
 * live on `JotViewStore`; once split further, each feature gets its own
 * presenter owning the subset of stores it touches.
 */
export type JotViewerPresenterDeps = {
  document: DocumentStore;
  settings: SettingsStore;
  transcribe: TranscribeStore;
  provenance: ProvenanceStore;
  lyricsAlign: LyricsAlignStore;
  playback: PlaybackStore;
  viewport: ViewportStore;
  mixer: MixerStore;
};

/**
 * Catch-all presenter for the jot viewer. Holds the actions, reactions,
 * and orchestration that mutate the data-only stores; React components
 * bind its methods to UI callbacks and read store state for rendering.
 */
export class JotViewerPresenter {
  // Mute/solo/volume, masters, instrumentTracks, trackOrder, split
  // statuses + their computeds moved to `MixerStore` (this.mixer).
  // Debug-bundle + per-note provenance + DebugPanel chrome moved to
  // `ProvenanceStore` (this.provenance).
  // Playhead-follow + transport state moved to `PlaybackStore`
  // (this.playback).
  // DebugPanel open/height moved to `ProvenanceStore` (this.provenance).
  // Lyrics modal visibility + per-track align status moved to
  // `LyricsAlignStore` (this.lyricsAlign).
  // Zoom / scroll offsets / viewport+content extents / gutter width
  // moved to `ViewportStore` (this.viewport).
  /**
   * Controller for the in-flight `/transcribe` request, if any. The
   * "Stop" toolbar button calls `.abort()` here; the request's
   * AbortSignal is passed into `transcriber.transcribe` which forwards
   * it to `fetch` so the request is genuinely cancelled at the
   * network layer rather than just discarding the response.
   */
  transcribeController: AbortController | undefined;

  /**
   * Wrap an async file-load with the modal overlay's bookkeeping (the
   * loading counter / label live on {@link DocumentStore}). Errors
   * propagate; the finally block guarantees the counter decrements even if
   * the inner promise rejects, so a failed load never leaves the overlay
   * stuck on screen.
   */
  private async withLoading<T>(label: string, fn: () => Promise<T>): Promise<T> {
    runInAction(() => {
      if (this.document.loadingCount === 0) this.document.loadingLabel = label;
      this.document.loadingCount += 1;
    });
    try {
      return await fn();
    } finally {
      runInAction(() => {
        this.document.loadingCount -= 1;
        if (this.document.loadingCount === 0) this.document.loadingLabel = undefined;
      });
    }
  }

  /** Data-only stores carved out of this (transitional) store. Held as
   * references so the orchestration still living here can read/write them;
   * excluded from observability (they're already-observable stores). */
  readonly document: DocumentStore;
  readonly settings: SettingsStore;
  readonly transcribe: TranscribeStore;
  readonly provenance: ProvenanceStore;
  readonly lyricsAlign: LyricsAlignStore;
  readonly playback: PlaybackStore;
  readonly viewport: ViewportStore;
  readonly mixer: MixerStore;

  constructor(deps: JotViewerPresenterDeps) {
    this.document = deps.document;
    this.settings = deps.settings;
    this.transcribe = deps.transcribe;
    this.provenance = deps.provenance;
    this.lyricsAlign = deps.lyricsAlign;
    this.playback = deps.playback;
    this.viewport = deps.viewport;
    this.mixer = deps.mixer;
    makeAutoObservable(this, {
      transcribeController: false,
      lyricsAlignControllers: false,
      document: false,
      settings: false,
      transcribe: false,
      provenance: false,
      lyricsAlign: false,
      playback: false,
      viewport: false,
      mixer: false,
    });
    // Wire the mixer store in as the player's mixer context so freshly-
    // constructed AudioTracks can resolve grouped-instrument colour
    // inheritance. Done before any reactions fire so loadAudioTrack
    // calls made during the same tick see a populated context.
    jotPlayer.attachMixerContext(this.mixer);

    // Prune instrument-track view-models for pitches no longer present
    // in the active jot. The override is store-owned and survives jot
    // reloads, so a pitch that comes back in a later jot picks up its
    // previous override; we only forget pitches that disappeared from
    // the current jot to keep the map from growing unboundedly across
    // a long session. `fireImmediately` runs the prune once on boot
    // (a no-op when the map is empty).
    reaction(
      () => new Set(this.mixer.jotPitches),
      (pitches) => {
        for (const p of Array.from(this.mixer.instrumentTracks.keys())) {
          if (!pitches.has(p)) this.mixer.instrumentTracks.delete(p);
        }
      },
      { fireImmediately: true, equals: comparer.structural }
    );

    // Push mute / solo state to the player whenever it changes. While
    // playback is in flight, the player cancels and reschedules events
    // so the toggle takes effect immediately (including bringing
    // previously-muted rows back mid-song). When idle, the filter is
    // just stored for the next play().
    //
    // This MUST be a `reaction`, not an `autorun`: `setFilter` both
    // reads (`scheduleEvents` → `isAudibleUnder(..., this.currentFilter)`)
    // and writes (`this.currentFilter = filter`) an observable on the
    // MobX-observable player while playing. An `autorun` tracks reads
    // made during the effect, so it would depend on the very observable
    // it writes — a non-converging reaction that MobX bails on, after
    // which the UI (observer components reading store state directly)
    // keeps updating but the filter stops reaching the player (e.g.
    // un-solo visually clears yet audio stays soloed). `reaction` only
    // tracks the data selector; the effect runs untracked, so the
    // player's internal reads/writes stay out of the dependency graph.
    // The data fn returns the `pitchFilter` computed; that getter
    // snapshots its Sets/Maps so the structural comparer can actually
    // detect mute/solo/volume changes. Sharing the live Set/Map
    // references would defeat the comparer (prev and next cached
    // snapshots would point to the same mutated instance, so a deep
    // walk sees no diff) and the reaction would silently stop firing
    // after the initial seed; see the `pitchFilter` getter's doc
    // comment.
    reaction(
      () => this.mixer.pitchFilter,
      (filter) => jotPlayer.setFilter(filter),
      { fireImmediately: true, equals: comparer.structural }
    );
    // Same shape for audio tracks; observed mutations push immediately
    // so toggling M/S on a track is sample-accurate during playback
    // (per-track GainNode flip, no source recreation). Same
    // read-and-write-the-same-observable hazard as above
    // (`setAudioTrackFilter` reads/writes `currentAudioTrackFilter`), so this is a
    // `reaction` for the same reason.
    reaction(
      () => this.mixer.audioTrackFilter,
      (filter) => jotPlayer.setAudioTrackFilter(filter),
      { fireImmediately: true, equals: comparer.structural }
    );
    // Push the section-audibility booleans to the player so master mute
    // and master solo can flip the bus gain to 0 without touching the
    // per-row M/S sets. fireImmediately to seed the initial unmuted state.
    reaction(
      () => this.mixer.isAudioSectionAudible,
      (audible) => jotPlayer.setAudioMasterAudible(audible),
      { fireImmediately: true }
    );
    reaction(
      () => this.mixer.isDrumSectionAudible,
      (audible) => jotPlayer.setDrumMasterAudible(audible),
      { fireImmediately: true }
    );
    // Seed the player's live drum↔audio offset from each loaded jot's
    // transcribed lead-in (`globalMetadata.drumsT0Sec`). Tracking
    // `currentJot` (an observable reference) re-fires whenever a new jot
    // is loaded, resetting the offset to that recording's value; manual
    // nudges via the Offset control persist until the next load. We read
    // `globalMetadata` (the raw source) rather than `resolved` so seeding
    // doesn't force a layout pass.
    reaction(
      () => {
        const raw = this.document.currentJot?.globalMetadata.drumsT0Sec;
        return typeof raw === 'number' && raw > 0 ? raw : 0;
      },
      (offsetSec) => jotPlayer.setDrumsT0Sec(offsetSec),
      { fireImmediately: true }
    );

    // Keep `trackOrder` synced with the live audio-track set and the
    // current jot's pitches. The reaction fires whenever either changes;
    // dropped rows are removed and newly-discovered rows are slotted at
    // a sensible default position (new audio tracks → end of the audio
    // block, new pitches → end of the list) so the user's drag-and-drop
    // ordering of surviving rows is preserved. `fireImmediately` seeds
    // the initial ordering on construction.
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
   * Drop entries from {@link trackOrder} that no longer correspond to a
   * live audio track, jot pitch, or lyrics track; then append the
   * missing ones at a sensible default position so the row appears
   * immediately:
   *   - new audio track  → after the last existing audio entry (or top
   *     of the list if no audio entries exist yet)
   *   - new pitch        → end of the list
   *   - new lyrics row   → just after the last existing lyrics row,
   *     keeping the lyrics group contiguous. The very first lyrics row
   *     (when none exist yet) goes to the top of the list. User can drag
   *     it elsewhere; its position survives subsequent reactions because
   *     the filter step preserves surviving entries.
   *
   * Existing entries keep their relative order so a user drag survives
   * an audio-track add/remove or a jot reload that didn't change the
   * pitch set.
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
    const next: TrackKey[] = this.mixer.trackOrder.filter((k) => wanted.some((w) => trackKeyEq(w, k)));
    for (const w of wanted) {
      if (next.some((k) => trackKeyEq(k, w))) continue;
      if (w.kind === 'lyrics') {
        // Slot a new lyrics row just after the last existing lyrics
        // entry so lyrics rows stay contiguous by default. When no
        // lyrics rows exist yet, default to the very top of the mixer
        // (above any audio / instrument rows); the filter step above
        // preserves the position the user drags it to on subsequent
        // runs.
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
        // Slot a new audio track in just after the last existing audio
        // entry so audio rows stay contiguous by default.
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
    // when the underlying sets are unchanged (zoom-driven layout passes
    // mint a fresh `pitches` array, audio-track mute toggles re-emit
    // `audioIds` of the same content, etc.). Skip the observable
    // assignment when the order is structurally identical — identity
    // *and* groupId — so downstream consumers (the mixer renderer)
    // don't pay an unnecessary re-render but a real grouping change
    // still propagates.
    if (
      next.length === this.mixer.trackOrder.length &&
      next.every(
        (k, i) => trackKeyEq(k, this.mixer.trackOrder[i]) && k.groupId === this.mixer.trackOrder[i].groupId
      )
    ) {
      return;
    }
    this.mixer.trackOrder = next;
  }

  /**
   * Reorder the mixer by moving the row at `fromIdx` to position
   * `toIdx`. Both indices refer to positions in the *current*
   * `trackOrder` (so `toIdx` is interpreted before the removal); a
   * no-op move (`fromIdx === toIdx`) is silently dropped.
   *
   * Drives the gutter drag-and-drop: the dragged row's start index is
   * the source; the drop target's "insert before me" index is the
   * destination. Used by the keyboard reorder shortcuts too (Alt+
   * Up/Down on a focused row).
   *
   * Group semantics on drop: the moved row's `groupId` is replaced with
   * the post-move adjacent rows' shared group, if any:
   *   - dropped inside an existing group (above + below in same group)
   *     → join that group
   *   - dropped at a group boundary, between solos, or at top/bottom
   *     → becomes solo (`groupId = undefined`)
   * This way a row can be dropped into a group by aiming for the middle
   * of it, and out of a group by aiming for a boundary or the end —
   * without any explicit "leave group" / "join group" UI.
   */
  moveTrack(fromIdx: number, toIdx: number): void {
    if (fromIdx < 0 || fromIdx >= this.mixer.trackOrder.length) return;
    const clamped = Math.max(0, Math.min(this.mixer.trackOrder.length, toIdx));
    if (clamped === fromIdx || clamped === fromIdx + 1) return;
    const next = this.mixer.trackOrder.slice();
    const [moved] = next.splice(fromIdx, 1);
    // After the removal a `toIdx` that was past the source shifts down
    // by one; before the source it's unaffected.
    const adjusted = clamped > fromIdx ? clamped - 1 : clamped;
    const above = adjusted > 0 ? next[adjusted - 1] : undefined;
    const below = adjusted < next.length ? next[adjusted] : undefined;
    const newGroupId =
      above && below && above.groupId !== undefined && above.groupId === below.groupId
        ? above.groupId
        : undefined;
    // Spread keeps the discriminant intact while overwriting groupId; a
    // direct assignment would widen the type and lose narrowing.
    let repositioned: TrackKey;
    if (moved.kind === 'audio') {
      repositioned = { kind: 'audio', id: moved.id, groupId: newGroupId };
    } else if (moved.kind === 'instrument') {
      repositioned = { kind: 'instrument', pitch: moved.pitch, groupId: newGroupId };
    } else {
      repositioned = { kind: 'lyrics', id: moved.id, groupId: newGroupId };
    }
    next.splice(adjusted, 0, repositioned);
    this.mixer.trackOrder = next;
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

  /**
   * Load an audio file as a new audio track and update the status pill
   * on failure. Decoding goes through the shared `AudioContext`, so the
   * call has to occur inside a user gesture (the file-picker click
   * satisfies that). Every call appends an independent track — load N
   * files to get N tracks. Returns the new track's id, or `undefined`
   * if the load failed (so callers can e.g. default it to muted).
   */
  async loadAudioTrack(
    file: File,
    pitch?: string,
    role?: AudioTrackRole
  ): Promise<AudioTrackId | undefined> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      try {
        return await jotPlayer.loadAudioTrack(file, pitch, role);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Audio track load failed: ${message}`);
        return undefined;
      }
    });
  }

  clearAudioTrack(id: AudioTrackId): void {
    jotPlayer.clearAudioTrack(id);
    // Drop the removed track's mute/solo/volume so it doesn't linger
    // (ids are never reused, so the entries would be dead weight); and
    // critically so clearing the only soloed audio track doesn't leave
    // a phantom solo silencing everything else. The colour override
    // lives on the AudioTrack instance itself and is freed alongside it
    // when the player drops the track.
    this.mixer.mutedAudioTracks.delete(id);
    this.mixer.soloedAudioTracks.delete(id);
    this.mixer.audioTrackVolumes.delete(id);
  }


  /** Mark an audio track as currently being split. Drives the per-row
   *  spinner; safe to call multiple times (latest call wins). */
  beginAudioTrackSplit(id: AudioTrackId, kind: 'mix' | 'pieces'): void {
    this.mixer.audioTrackSplitStatuses.set(id, { phase: 'splitting', kind });
  }

  /** Clear the splitting status for an audio track once the work has
   *  finished (success, failure, or cancellation). */
  endAudioTrackSplit(id: AudioTrackId): void {
    this.mixer.audioTrackSplitStatuses.delete(id);
  }

  /**
   * Stub: invoked from the audio-track overflow menu's "Split into
   * drums + backing" item. The actual transcriber-side wiring (POST the
   * track's PCM to a single-stage `stems_all` endpoint, then auto-load
   * the resulting drum stem + drumless backing as fresh audio tracks)
   * is deferred to a follow-up change; surface a status pill so the
   * click visibly does something in the meantime.
   */
  splitAudioTrackFromMix(id: AudioTrackId): void {
    const track = jotPlayer.audioTracks.get(id);
    const name = track ? track.filename : id;
    toastStore.showError(`Split into drums + backing on "${name}" isn't wired up yet.`);
  }

  /**
   * Stub: invoked from the audio-track overflow menu's "Split into
   * kick / snare / hi-hat / cymbals" item. See
   * {@link splitAudioTrackFromMix} for the same TODO.
   */
  splitAudioTrackDrumPieces(id: AudioTrackId): void {
    const track = jotPlayer.audioTracks.get(id);
    const name = track ? track.filename : id;
    toastStore.showError(`Split into drum pieces on "${name}" isn't wired up yet.`);
  }

  /** Drop every loaded audio track. Used when a new source (e.g. a
   * ParaDB pack) replaces the current song, otherwise the previous
   * song's tracks linger and play over the new one. */
  clearAllAudioTracks(): void {
    for (const id of Array.from(jotPlayer.audioTracks.keys())) {
      this.clearAudioTrack(id);
    }
  }

  /** Reset the per-pitch mixer (mute/solo/volume). These are keyed by
   * DSL pitch letter, not by song, so without this a mute/solo/fader
   * set on one song silently bleeds onto the next song's matching rows
   * when a new source replaces the current one. */
  resetPitchMixer(): void {
    this.mixer.mutedPitches.clear();
    this.mixer.soloedPitches.clear();
    this.mixer.pitchVolumes.clear();
  }

  setJot(jot: RenderedJot | undefined) {
    this.document.currentJot = jot;
    // External setJot calls invalidate the example pointer + any
    // previously-loaded debug provenance (provenance is per-bundle and
    // doesn't survive a wholesale jot replacement).
    this.document.currentExampleId = undefined;
    this.clearNoteProvenance();
    // Lyrics are tied to a specific recording; a new jot means they no
    // longer apply. See `src/lyrics/store.ts` for the lifecycle rationale.
    this.clearLyrics();
    // Replace the song wholesale: stop any in-flight playback so the
    // playhead, scheduled drum events, and idle cue from the previous
    // jot don't leak onto the new one.
    jotPlayer.stop();
  }

  /** Drop the debug bundle's per-note provenance + reset the toolbar
   * visibility toggle. Called from every loader that replaces the
   * current song outside the bundle path so stale debug info from a
   * previous bundle doesn't leak onto the new score. */
  private clearNoteProvenance() {
    this.provenance.noteProvenance = undefined;
    this.provenance.showFilteredOnsets = false;
  }

  toggleFollowPlayhead() {
    this.setFollowPlayhead(!this.playback.followPlayhead);
  }

  /**
   * Set {@link followPlayhead} and tag whether the off-state is
   * transient (set while playing) or deliberate (set while idle/paused).
   * Idempotent: redundant calls don't reshuffle the transient tag so e.g.
   * a pan during playback can't promote an already-deliberate off-state
   * into a transient one.
   */
  setFollowPlayhead(on: boolean) {
    if (on === this.playback.followPlayhead) return;
    this.playback.followPlayhead = on;
    this.playback.followDisabledIsTransient = on ? false : jotPlayer.state === 'playing';
  }

  setExamples(examples: readonly ExampleJot[]) {
    this.document.examples = examples;
  }

  loadExample(id: string) {
    const example = this.document.examples.find((e) => e.id === id);
    if (!example) return;
    this.document.currentJot = new RenderedJot(example.jot, this.document.viewConfig);
    this.document.currentExampleId = id;
    this.clearNoteProvenance();
    this.clearLyrics();
    jotPlayer.stop();
  }


  /**
   * Upload an audio file to the transcriber service, parse the returned
   * Drumjot DSL, and load the resulting Jot. Updates `transcribeStatus`
   * so the toolbar can show progress / errors.
   *
   * A single in-flight transcription is tracked via `transcribeController`.
   * Calling `cancelTranscribe()` aborts the underlying `fetch` request and
   * surfaces a cancelled state on the toolbar; starting a new
   * transcription while one is in flight will abort the previous one
   * first (defensive - the UI disables the button during upload, but the
   * console-level `loadDsl` API doesn't).
   */
  async transcribeAudio(file: File): Promise<void> {
    if (this.transcribeController) {
      this.transcribeController.abort();
    }
    const controller = new AbortController();
    this.transcribeController = controller;
    runInAction(() => {
      this.transcribe.transcribeStatus = { phase: 'uploading', filename: file.name };
    });
    try {
      const response = await transcriber.transcribe(file, {
        debug: this.transcribe.transcribeOptions.debug,
        beatInput: this.transcribe.transcribeOptions.beatInput,
        drumSeparator: this.transcribe.transcribeOptions.drumSeparator,
        llmModel: this.transcribe.transcribeOptions.llmModel,
        quantise: this.transcribe.transcribeOptions.quantise,
        quantiseUseLlm: this.transcribe.transcribeOptions.quantiseUseLlm,
        signal: controller.signal,
        onProgress: (event) => this.applyProgress(file.name, event),
      });
      await this.applyTranscribeResponse(response, file.name, controller.signal);
    } catch (err) {
      this.handleTranscribeError(err, controller, 'Transcribe');
    } finally {
      if (this.transcribeController === controller) {
        this.transcribeController = undefined;
      }
      // The folder list has a new entry (the just-finished run); refresh
      // best-effort so the picker is up to date without the operator
      // having to reopen the dropdown.
      void this.refreshRecentTranscriptions();
    }
  }

  /**
   * Re-run the transcribe pipeline from a chosen stage against a
   * previously-cached debug folder. Same status / auto-load semantics as
   * {@link transcribeAudio}: progress pill while in flight, the response
   * either parses straight (DSL mode) or auto-loads the rebuilt debug
   * bundle (filter mode), and the resume controller shares
   * `transcribeController` so the Stop button cancels both flows.
   */
  async resumeTranscribe(folder: string, stage: TranscribeStage): Promise<void> {
    if (this.transcribeController) {
      this.transcribeController.abort();
    }
    const controller = new AbortController();
    this.transcribeController = controller;
    const label = `${folder} from ${stage}`;
    runInAction(() => {
      this.transcribe.transcribeStatus = { phase: 'uploading', filename: label };
    });
    try {
      const response = await transcriber.resume({
        resumeFolder: folder,
        resumeStage: stage,
        beatInput: this.transcribe.transcribeOptions.beatInput,
        drumSeparator: this.transcribe.transcribeOptions.drumSeparator,
        llmModel: this.transcribe.transcribeOptions.llmModel,
        quantise: this.transcribe.transcribeOptions.quantise,
        quantiseUseLlm: this.transcribe.transcribeOptions.quantiseUseLlm,
        signal: controller.signal,
        onProgress: (event) => this.applyProgress(label, event),
      });
      // The resumed run reuses the original folder, so the original
      // upload filename is the most informative pill label — fall back
      // to the resume folder name when the server doesn't know it.
      const fallbackName =
        this.transcribe.recentTranscriptions.find((t) => t.folder === folder)?.original_filename ?? folder;
      await this.applyTranscribeResponse(response, fallbackName, controller.signal);
    } catch (err) {
      this.handleTranscribeError(err, controller, 'Resume');
    } finally {
      if (this.transcribeController === controller) {
        this.transcribeController = undefined;
      }
      void this.refreshRecentTranscriptions();
    }
  }

  /**
   * Shared post-transcribe handling. The backend produces a MIDI
   * prediction; we auto-load the bundled debug.zip so the score (via
   * `from_midi.ts`), audio tracks, and note provenance hydrate in one
   * go without the user having to download and re-load the zip by hand.
   */
  private async applyTranscribeResponse(
    response: Awaited<ReturnType<typeof transcriber.transcribe>>,
    fallbackName: string,
    signal: AbortSignal
  ): Promise<void> {
    const bundleUrl = stemUrl(response.debug_zip_url ?? null);
    if (!bundleUrl) {
      runInAction(() => {
        this.transcribe.transcribeStatus = { phase: 'idle' };
      });
      toastStore.showError('Transcriber returned no debug bundle.');
      return;
    }
    const ok = await this.autoLoadDebugBundle(bundleUrl, fallbackName, signal);
    if (!ok) {
      // The auto-loader already surfaced the specific failure as an
      // error toast; clear the busy pill back to idle and bail.
      runInAction(() => {
        this.transcribe.transcribeStatus = { phase: 'idle' };
      });
      return;
    }
    runInAction(() => {
      this.transcribe.transcribeStatus = { phase: 'idle' };
    });
    toastStore.showSuccess(
      transcribeSuccessToastMessage({
        filename: fallbackName,
        tempo: response.metadata.initial_tempo,
        hasTempoChanges: response.metadata.has_tempo_changes,
        hasTimeSigChanges: response.metadata.has_time_sig_changes,
        barCount: response.metadata.bars.length,
        debugDir: response.debug_dir ?? null,
        debugZipUrl: bundleUrl,
      }),
      {
        title: response.debug_dir
          ? `Debug artifacts saved to ${response.debug_dir} (under ./debug/ on the host with the default docker-compose mount).`
          : undefined,
      }
    );
  }

  /**
   * Fetch the debug zip from `url`, parse it, and load every artifact
   * via {@link applyDebugBundle}. The predicted-MIDI score, audio
   * tracks, note provenance, and stage timings / logs all come along
   * in one round trip.
   *
   * Returns `true` on success, `false` if either the fetch or the
   * parse failed (in which case the caller surfaces an error pill).
   */
  private async autoLoadDebugBundle(
    url: string,
    fallbackName: string,
    signal: AbortSignal
  ): Promise<boolean> {
    let bundle: Awaited<ReturnType<typeof loadDebugZip>>;
    try {
      const res = await fetch(url, { signal });
      if (!res.ok) {
        throw new Error(`fetch ${url} failed (${res.status})`);
      }
      const blob = await res.blob();
      const file = new File([blob], `${fallbackName}.debug.zip`, {
        type: 'application/zip',
      });
      bundle = await loadDebugZip(file);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') throw err;
      // eslint-disable-next-line no-console
      console.warn('Auto-load debug bundle failed:', err);
      return false;
    }
    try {
      const ok = await this.applyDebugBundle(bundle, fallbackName);
      return ok;
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') throw err;
      // eslint-disable-next-line no-console
      console.warn('Auto-load debug bundle apply failed:', err);
      return false;
    }
  }

  /** Shared transcribe / resume failure handler. Routes aborts to idle
   *  (user cancelled), everything else to the error pill. */
  /**
   * Fold one streamed `TranscribeProgress` event into the live
   * `transcribeStatus` pill so the user sees the pipeline advancing
   * through each stage. `stage` events with `phase='start'` set the
   * current stage and clear any substage label from the previous one;
   * `substage` events overwrite the in-stage detail without changing
   * the stage itself. `phase='end'` is ignored for UI purposes — the
   * pill rolls straight from one stage's `start` to the next stage's
   * `start`, which reads more clearly than briefly showing "(done)".
   */
  private applyProgress(filename: string, event: TranscribeProgress): void {
    runInAction(() => {
      const status = this.transcribe.transcribeStatus;
      // If the request was aborted or already terminal (success/error)
      // before this late event fires, ignore — late progress shouldn't
      // resurrect the spinner over an idle/success/error pill.
      if (status.phase !== 'uploading') return;
      if (event.kind === 'stage' && event.phase === 'start') {
        this.transcribe.transcribeStatus = {
          phase: 'uploading',
          filename,
          stage: event.stage,
        };
      } else if (event.kind === 'substage') {
        this.transcribe.transcribeStatus = {
          phase: 'uploading',
          filename,
          stage: event.stage,
          substage: event.detail,
        };
      }
    });
  }

  private handleTranscribeError(err: unknown, controller: AbortController, verb: string): void {
    // AbortError surfaces as DOMException with name='AbortError' (and
    // wraps as TypeError in some runtimes when the fetch was already
    // aborted at start). Treat the user-initiated cancellation
    // distinctly from real errors so we don't show a scary red pill.
    const isAbort =
      controller.signal.aborted || (err instanceof DOMException && err.name === 'AbortError');
    if (isAbort) {
      runInAction(() => {
        this.transcribe.transcribeStatus = { phase: 'idle' };
      });
      return;
    }
    const message =
      err instanceof ParseError
        ? `${verb} returned invalid DSL: ${err.message}`
        : err instanceof Error
          ? err.message
          : String(err);
    runInAction(() => {
      this.transcribe.transcribeStatus = { phase: 'idle' };
    });
    toastStore.showError(`${verb} failed: ${message}`);
  }

  /**
   * Refresh the recent-transcriptions picker from the server. Failures
   * are logged but never surfaced — the picker just stays as-is, which
   * is the right behaviour when the backend is briefly unavailable.
   * Safe to call from a fire-and-forget context.
   */
  async refreshRecentTranscriptions(): Promise<void> {
    runInAction(() => {
      this.transcribe.recentTranscriptionsLoading = true;
    });
    try {
      const list = await transcriber.listTranscriptions();
      runInAction(() => {
        this.transcribe.recentTranscriptions = list;
        this.transcribe.recentTranscriptionsLoaded = true;
        // Drop the selection if its target folder vanished server-side
        // (e.g. operator pruned the debug dir between dropdown opens).
        if (
          this.transcribe.selectedResumeFolder !== undefined &&
          !list.some((s) => s.folder === this.transcribe.selectedResumeFolder)
        ) {
          this.transcribe.selectedResumeFolder = undefined;
          this.transcribe.selectedResumeStage = undefined;
        }
      });
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn('Could not refresh recent transcriptions:', err);
    } finally {
      runInAction(() => {
        this.transcribe.recentTranscriptionsLoading = false;
      });
    }
  }

  /**
   * Load a previously produced transcription's debug bundle straight from
   * the server's `/outputs/<folder>/debug.zip` without re-running any
   * pipeline stage. The bundle carries the kept-onset MIDI score, the
   * per-stem audio, and the run's logs / stage timings, so this is the
   * cheap way to reopen a finished run.
   *
   * Errors land on the shared status pill, mirroring the explicit
   * "Load debug bundle" file picker. Wrapped in `withLoading` so the
   * modal overlay reads as one continuous load even though the inner
   * `applyDebugBundle` may itself trigger nested audio-track loads.
   */
  async loadRecentTranscription(folder: string): Promise<void> {
    const url = stemUrl(`/outputs/${encodeURIComponent(folder)}/debug.zip`);
    if (!url) return;
    const summary = this.transcribe.recentTranscriptions.find((s) => s.folder === folder);
    const fallbackName = summary?.original_filename ?? folder;
    return this.withLoading(`Loading ${fallbackName}…`, async () => {
      try {
        const res = await fetch(url);
        if (!res.ok) {
          throw new Error(`fetch ${url} failed (${res.status})`);
        }
        const blob = await res.blob();
        const file = new File([blob], `${fallbackName}.debug.zip`, {
          type: 'application/zip',
        });
        const bundle = await loadDebugZip(file);
        const ok = await this.applyDebugBundle(bundle, fallbackName);
        if (!ok) {
          toastStore.showError(`Could not parse score from ${fallbackName}.`);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not load ${fallbackName}: ${message}`);
      }
    });
  }

  /**
   * Abort the in-flight transcription, if any. No-op when nothing is
   * running. The next `transcribeAudio` call resumes normally.
   */
  cancelTranscribe() {
    if (!this.transcribeController) return;
    this.transcribeController.abort();
    this.transcribeController = undefined;
  }

  /**
   * Read a Drumjot DSL file from the user's machine and load it as the
   * current jot. Parse failures surface as error toasts.
   */
  async loadJotFile(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      let text: string;
      try {
        text = await file.text();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not read ${file.name}: ${message}`);
        return;
      }
      try {
        const jot = parse(text);
        if (!jot.title) {
          const derivedTitle = titleFromFilename(file.name);
          if (derivedTitle) jot.title = derivedTitle;
        }
        runInAction(() => {
          this.document.currentJot = new RenderedJot(jot, this.document.viewConfig);
          this.document.currentExampleId = undefined;
          // A bare jot file has no provenance; drop whatever the
          // previous bundle put there so the selection label doesn't
          // surface stale debug data on the new song's notes.
          this.clearNoteProvenance();
          this.clearLyrics();
          jotPlayer.stop();
        });
      } catch (err) {
        const message =
          err instanceof ParseError
            ? `Could not parse ${file.name}: ${err.message}`
            : err instanceof Error
              ? err.message
              : String(err);
        toastStore.showError(message);
      }
    });
  }

  /**
   * Read a Standard MIDI File from the user's machine, convert it to a
   * Jot via {@link fromMidi}, and load it as the current jot. Like
   * {@link loadJotFile}, conversion runs entirely client-side and
   * failures surface through the shared `transcribeStatus` pill.
   */
  async loadMidiFile(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      let bytes: ArrayBuffer;
      try {
        bytes = await file.arrayBuffer();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not read ${file.name}: ${message}`);
        return;
      }
      try {
        const jot = fromMidi(bytes);
        if (!jot.title) {
          const derivedTitle = titleFromFilename(file.name);
          if (derivedTitle) jot.title = derivedTitle;
        }
        runInAction(() => {
          this.document.currentJot = new RenderedJot(jot, this.document.viewConfig);
          this.document.currentExampleId = undefined;
          // Same reasoning as in loadJotFile: a bare MIDI load shouldn't
          // surface stale provenance from a previous debug bundle.
          this.clearNoteProvenance();
          this.clearLyrics();
          jotPlayer.stop();
        });
      } catch (err) {
        const message =
          err instanceof Error ? `Could not convert ${file.name}: ${err.message}` : String(err);
        toastStore.showError(message);
      }
    });
  }

  /**
   * Load a ParaDB / Paradiddle map pack (`.zip`): convert its `.rlrr`
   * chart to a Jot and auto-load its audio tracks so the pack is
   * immediately play-along ready. Audio decoding shares the
   * `AudioContext`, so this must run inside the file-picker's user
   * gesture (the same constraint as {@link loadAudioTrack}). Errors surface
   * through the shared status pill, matching {@link loadJotFile}.
   */
  async loadParadbMap(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      let map: Awaited<ReturnType<typeof loadParadbZip>>;
      try {
        map = await loadParadbZip(file);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not load ${file.name}: ${message}`);
        return;
      }

      const jot = map.jot;
      if (!jot.title) {
        const derivedTitle = titleFromFilename(file.name);
        if (derivedTitle) jot.title = derivedTitle;
      }
      runInAction(() => {
        // Replace the song wholesale: drop any audio tracks from a
        // previously loaded map/transcription so they don't play over
        // the new pack's tracks, and reset the per-pitch mixer so an
        // old song's mute/solo/faders don't bleed onto the new rows.
        this.clearAllAudioTracks();
        this.resetPitchMixer();
        this.document.currentJot = new RenderedJot(jot, this.document.viewConfig);
        this.document.currentExampleId = undefined;
        this.clearNoteProvenance();
        this.clearLyrics();
        jotPlayer.stop();
      });

      // Audio tracks are best-effort: a chart with the score loaded is
      // still useful even if one is absent or fails to decode.
      // loadAudioTrack already reports its own failures on the status pill.
      // Drum tracks load too but start muted; you're playing the drums,
      // so the backing music should be the only thing you hear by default.
      //
      // Lyrics alignment is deliberately NOT auto-fired here: vocals
      // separation (BS-Roformer) eats a chunk of GPU time, and most
      // ParaDB loads don't need lyrics. The user kicks it off explicitly
      // via the Lyrics menu (or the LRCLIB search modal) when they want
      // synced lyrics.
      //
      // Decode in parallel; `decodeAudioData` runs on browser-side
      // codec threads so concurrent calls overlap, cutting the song +
      // drums decode wall time roughly in half. Mirrors the debug-
      // bundle loader's approach.
      const resolved = await Promise.all(
        map.audioTracks.map(async (track) => {
          const id = await this.loadAudioTrack(track.file, undefined, track.role);
          return { id, defaultMuted: track.defaultMuted };
        })
      );
      runInAction(() => {
        for (const { id, defaultMuted } of resolved) {
          if (id && defaultMuted) this.mixer.mutedAudioTracks.add(id);
        }
      });
    });
  }

  /**
   * Score a ParaDB `.zip` map against its own audio via the transcriber's
   * `POST /score`, surfacing the result as a toast (full result to the
   * console). A development test harness for the corpus-filtering scorer
   * (`transcriber/app/scoring`); unlike {@link loadParadbMap} it does NOT
   * touch the current score, it only reports a quality number.
   */
  async scoreParadbMap(file: File): Promise<void> {
    return this.withLoading(`Scoring ${file.name}…`, async () => {
      try {
        const result = await transcriber.scoreParadb(file);
        const offsetMs = (result.offset_sec * 1000).toFixed(0);
        toastStore.showSuccess(
          `${file.name}: ${result.score_corrected}/100 corrected ` +
            `(raw ${result.score}) · offset ${offsetMs} ms · ` +
            `tempo ${result.tempo_ratio.toFixed(3)}× · ${result.audio_reference}`,
          { title: 'See the browser console for the full per-lane breakdown.' }
        );
        // eslint-disable-next-line no-console
        console.log('Alignment score', file.name, result);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not score ${file.name}: ${message}`);
      }
    });
  }

  /**
   * Load a transcriber debug `.zip` bundle: parse the embedded
   * `final.jot`, load every audio track in the manifest's `mapping`, and
   * stash the manifest (stage timings + log stream) on
   * {@link lastDebugBundle} so the {@link DebugPanel} can show it.
   *
   * Behaves like {@link loadParadbMap}: replaces the current song
   * wholesale (drops previously loaded audio tracks, resets the pitch
   * mixer), runs entirely client-side, and surfaces errors on the
   * shared status pill.
   *
   * The `no_drums` entry (drumless backing audio) is auto-defaulted to
   * unmuted; the per-pitch stems are defaulted to muted, mirroring the
   * "drum tracks are reference-only, you're playing them" convention
   * from the ParaDB loader — the drums you hear should be the smplr-
   * scheduled ones from the score, not a re-decoded stem layered on top.
   */
  async loadDebugBundleFile(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      let bundle: Awaited<ReturnType<typeof loadDebugZip>>;
      try {
        bundle = await loadDebugZip(file);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not load ${file.name}: ${message}`);
        return;
      }
      const ok = await this.applyDebugBundle(bundle, file.name);
      if (!ok) {
        toastStore.showError(`Could not parse score from ${file.name}.`);
      }
    });
  }

  /**
   * Apply an already-parsed {@link DebugBundle} to the store: replace the
   * current song with the bundle's score (DSL → MIDI fallback), load each
   * audio track, pair stems with their instrument rows, and mount the
   * manifest on the DebugPanel.
   *
   * Returns `true` if a score was loaded, `false` if neither `final.jot`
   * nor `prediction.mid` could be turned into a jot (the audio tracks
   * still load either way so the operator can at least listen).
   *
   * Status-pill management is left to the caller — `loadDebugBundleFile`
   * sets it to idle/error on completion, while `transcribeAudio` keeps
   * its success pill visible after the auto-load.
   */
  private async applyDebugBundle(
    bundle: Awaited<ReturnType<typeof loadDebugZip>>,
    fallbackName: string
  ): Promise<boolean> {
    runInAction(() => {
      this.clearAllAudioTracks();
      this.resetPitchMixer();
      this.clearLyrics();
      this.provenance.lastDebugBundle = bundle.manifest;
      // Replace (or clear) the per-note debug provenance whenever a
      // new bundle loads. Older bundles may not carry one (e.g. a
      // hand-built or legacy zip); the absent-case clears the previous
      // bundle's provenance so it doesn't leak onto the new score.
      this.provenance.noteProvenance = bundle.noteProvenance ?? undefined;
      // Reset the visibility toggle so a freshly loaded bundle reads
      // as just "the score"; operator opts into the ghost overlays.
      this.provenance.showFilteredOnsets = false;
      // Bundles come from the transcribe pipeline, which routinely
      // emits triplet subdivisions; the 48ths grid is the LCM of 16ths
      // + triplets so it visualises both. Override the store-wide 16ths
      // default for this load specifically.
      this.settings.gridLines = {
        mainBeat: true,
        subBeat16: false,
        subBeatQuarterTriplet: false,
        subBeatTriplet: false,
        subBeat48: true,
      };
    });

    // The bundle's score is the `prediction.mid` produced by the
    // transcribe stage; `src/midi/from_midi.ts` converts it to a Jot.
    let scoreLoaded = false;
    if (bundle.predictionMidi) {
      try {
        const jot = fromMidi(bundle.predictionMidi);
        if (!jot.title) {
          const derivedTitle = titleFromFilename(fallbackName);
          if (derivedTitle) jot.title = derivedTitle;
        }
        // The beats stage's `align_beats_to_*` shift is already baked
        // into `prediction.mid`'s tick grid (see `compute_bar_tick_grid`
        // in `transcriber/app/pipeline/onsets_midi.py`), so the loaded
        // MIDI is at the aligned positions and the Beat control starts
        // at 0. The applied alignment is still visible per-note in the
        // selection popup as the "Beat alignment" row sourced from
        // `noteProvenance.beat_alignment_offset_sec`.
        runInAction(() => {
          const rendered = new RenderedJot(jot, this.document.viewConfig);
          this.document.currentJot = rendered;
          this.document.currentExampleId = undefined;
          jotPlayer.stop();
        });
        scoreLoaded = true;
      } catch (err) {
        const message =
          err instanceof Error ? `Could not convert prediction.mid: ${err.message}` : String(err);
        toastStore.showError(message);
      }
    }

    // Decode every audio track in parallel, `decodeAudioData` runs on
    // browser-side codec threads, so concurrent calls overlap well and
    // turn what used to be a one-by-one wait into a single combined
    // wait. `Promise.all` preserves input order so the resolved array
    // still matches `bundle.audioTracks` (which is already in manifest
    // order; `no_drums` first, then pitch letters), keeping the
    // post-load pair-with-instrument-row logic stable. The bundle
    // loader dedupes by filename, so each `track` here represents one
    // unique file; we bind every key in `track.keys` to the resulting
    // `AudioTrackId` so a shared stem (e.g. `stem_c.mp3` serving both
    // crash and ride after the cymbal split) is loaded once and looked
    // up under either key.
    const resolved = await Promise.all(
      bundle.audioTracks.map(async (track) => {
        // The audio-row's `pitch` (used by the mixer for waveform
        // tinting) takes the first non-`no_drums` key; for a stem
        // shared across pitches, this picks the first-mentioned pitch
        // in the manifest, which is good enough since the tint is
        // cosmetic and both siblings live in the same colour family.
        const primaryKey = track.keys.find((k) => k !== NO_DRUMS_KEY);
        // Role classification: any track whose only key is `no_drums`
        // is the Demucs drumless mix; everything else came from the
        // per-pitch split (a key shared between multiple pitches still
        // counts as a single drum piece for menu purposes).
        const role: AudioTrackRole = primaryKey === undefined ? 'no-drums' : 'drum-piece';
        const id = await this.loadAudioTrack(track.file, primaryKey, role);
        return { keys: track.keys, id };
      })
    );
    const loadedByKey = new Map<string, AudioTrackId>();
    const toMute: AudioTrackId[] = [];
    for (const { keys, id } of resolved) {
      if (!id) continue;
      let muteThis = false;
      for (const key of keys) {
        loadedByKey.set(key, id);
        // Mute the per-pitch stems by default so the (audible) drums
        // come from the smplr score scheduler; the drumless backing
        // stays unmuted. Multiple keys → still one mute, since they
        // share the same `id`.
        if (key !== NO_DRUMS_KEY) muteThis = true;
      }
      if (muteThis) toMute.push(id);
    }

    // Batch the mute updates and the reorder into a single observable
    // mutation so the mixer renders once at the end instead of once
    // per loaded track.
    runInAction(() => {
      for (const id of toMute) this.mixer.mutedAudioTracks.add(id);
      this.applyDebugBundleTrackOrder(loadedByKey);
    });

    return scoreLoaded;
  }

  /**
   * Re-order the mixer after a debug bundle is loaded so each per-pitch
   * audio stem sits immediately above its instrument row, with any
   * unmatched audio (e.g. the `no_drums` backing) at the top. The
   * ordering itself is the pure {@link buildDebugBundleTrackOrder} (see
   * there for the full layout + the shared-stem dedupe); this just feeds
   * it the current jot's pitches.
   *
   * The `syncTrackOrder` reaction won't reshuffle the result, it only
   * ever drops stale entries and appends new ones, both of which are
   * no-ops right after a fresh bundle load.
   */
  private applyDebugBundleTrackOrder(loadedByKey: ReadonlyMap<string, AudioTrackId>): void {
    const pitches = collectJotPitches(this.document.currentJot);
    this.mixer.trackOrder = buildDebugBundleTrackOrder(pitches, loadedByKey);
  }

  /** Resize the {@link DebugPanel}. Clamped so it can't shrink past the
   * header or grow past the viewport (with headroom for the toolbar).
   * Stays here transitionally because the clamp reads `_viewportHeight`
   * (viewport state not yet extracted); moves to the presenter with the
   * viewport slice. */
  setDebugPanelHeight(px: number): void {
    const max = Math.max(120, this.viewport._viewportHeight - 160);
    this.provenance.debugPanelHeight = Math.min(max, Math.max(80, px));
  }

  // Viewport actions (setZoom, scroll setters + clamps, viewport/content
  // size, setGutterWidth) and the visibleBeatRange computed moved to
  // `ViewportStore` (data) + the presenter (actions).

  /**
   * Read a synced-lyrics file (LRC, or a text file in LRC format) from
   * disk and push it into the session lyrics store. Empty / unparseable
   * inputs surface a failure message on the shared status pill instead
   * of silently doing nothing.
   */
  async loadLyricsFile(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      let text: string;
      try {
        text = await file.text();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not read ${file.name}: ${message}`);
        return;
      }
      // Enhanced-LRC aware: word-tagged files load as word-aligned
      // tracks (with per-word durations), plain line-level LRC parses
      // exactly as before. A leading `[offset:±ms]` restores the saved
      // offset nudge.
      const { lines, offsetSec } = parseEnhancedLrc(text);
      if (lines.length === 0) {
        toastStore.showError(`No synced lyrics found in ${file.name}.`);
        return;
      }
      runInAction(() => {
        const id = lyricsStore.add(lines, {
          source: 'file',
          sourceLabel: `File · ${file.name}`,
        });
        if (offsetSec !== 0) lyricsStore.setOffsetSec(id, offsetSec);
      });
    });
  }

  /**
   * Apply a synced-lyrics result the LRCLIB modal picked. The modal
   * parses the candidate's LRC and hands us the lines + the picked
   * match's identifying fields. Source label always reads `LRCLIB · …`;
   * word-level upgrades replace the lines in-place but keep the source.
   *
   * When `opts.wordLevel` is true, the LRCLIB lines load immediately
   * (so the row shows up right away with line-level timing) and a
   * background CTC forced-alignment job runs against an auto-picked
   * audio track. Success replaces the lines with word-timed
   * versions; failure leaves the line-level lines in place and surfaces
   * the error on the status pill.
   */
  applyLrclibResult(
    lines: readonly LyricLine[],
    match: { trackName: string; artistName: string },
    opts: { wordLevel: boolean } = { wordLevel: false }
  ): void {
    const trackId = lyricsStore.add(lines, {
      source: 'lrclib',
      sourceLabel: `LRCLIB · ${match.trackName} - ${match.artistName}`,
    });
    toastStore.showSuccess(`Loaded ${match.trackName} by ${match.artistName} from LRCLIB`, {
      testId: 'lyrics-search-loaded',
    });
    if (opts.wordLevel) {
      void this.runWordLevelAlignmentForLrclib(trackId, lines, match);
    }
  }

  /**
   * Auto-pick an audio track and run CTC forced-alignment against it
   * using the LRCLIB lines as authoritative text. The picked track
   * + kind drive whether the backend's vocals separator runs first
   * (`mix` = run separation; `vocals` = skip it).
   *
   * No-op (with a status pill error) when no audio tracks are loaded;
   * the modal disables the word-level checkbox in that case so this is
   * a programming-error safety net rather than a user-reachable path.
   */
  private async runWordLevelAlignmentForLrclib(
    targetTrackId: LyricsTrackId,
    lines: readonly LyricLine[],
    match: { trackName: string; artistName: string }
  ): Promise<void> {
    const pick = this.pickAudioTrackForAlignment();
    if (!pick) {
      toastStore.showError('Word-level alignment needs an audio track; load one first.');
      return;
    }
    const track = jotPlayer.audioTracks.get(pick.id);
    if (!track) return;
    const file = new File([track.sourceBlob], track.filename, { type: track.sourceBlob.type });
    const label = `${match.trackName} - ${match.artistName}`;
    await this.alignLyricsForced(
      targetTrackId,
      {
        kind: pick.kind,
        file,
        realign: {
          lines: lines.map((l) => ({ startSec: l.startSec, text: l.text })),
        },
      },
      label,
      {
        source: 'lrclib',
        sourceLabel: `LRCLIB · ${match.trackName} - ${match.artistName}`,
      }
    );
  }

  /**
   * Pick the loaded audio track most likely to carry vocals + the
   * separator mode to feed it to the CTC aligner with. Heuristic priority:
   *
   *   1. Any track whose filename looks like vocals → `vocals` (skip
   *      separation).
   *   2. First non-drums track (role ≠ `drums` / `drum-piece`) → `mix`
   *      (separator extracts vocals first).
   *   3. Fallback: first track regardless → `mix` (even a drums-only
   *      track is worth trying once over erroring out; the separator
   *      may still find faint vocal bleed; if not the user gets a
   *      "no speech found" message and can load a better track).
   *
   * Returns undefined only when no audio tracks are loaded.
   */
  private pickAudioTrackForAlignment(): { id: AudioTrackId; kind: 'mix' | 'vocals' } | undefined {
    const tracks = Array.from(jotPlayer.audioTracks.values());
    if (tracks.length === 0) return undefined;
    for (const t of tracks) {
      if (nameLooksLikeVocals(t.filename)) {
        return { id: t.id, kind: 'vocals' };
      }
    }
    for (const t of tracks) {
      if (t.role !== 'drums' && t.role !== 'drum-piece') {
        return { id: t.id, kind: 'mix' };
      }
    }
    return { id: tracks[0].id, kind: 'mix' };
  }

  /**
   * Push pasted / typed plain-text lyrics into the session lyrics store.
   *
   * Plain text has no timestamps, so we synthesise them by spreading
   * the lines evenly across the song's known duration (longest loaded
   * audio track > rendered jot's timeline > 60 s fallback). The spread
   * serves two ends: lines are immediately visible across the row
   * (otherwise they'd all stack at beat 0 and collapse to an invisible
   * point), and `opts.wordLevel`'s re-time path gets non-degenerate
   * starting estimates for wav2vec2 (whose search window for each line
   * is `[startSec, nextLine.startSec]` - all-zero starts collapse every
   * segment to the same audio window).
   *
   * Strips section markers like `[Chorus]` / `[Verse 1]` (any line whose
   * trimmed content is wrapped in a single pair of brackets) because
   * pastes from Genius and similar lyric sites carry them and they
   * aren't sung. Also strips parenthetical asides and music glyphs via
   * {@link stripLyricNoise}, so echo lines like `(I'm screaming…)` and
   * interlude markers like `♪ ♪ ♪` drop out. Returns the number of
   * lines actually loaded so the caller can surface a "nothing usable
   * in this paste" error.
   *
   * When `opts.wordLevel` is true and an audio track is loaded, fires
   * the same background CTC forced-alignment used by the LRCLIB
   * word-level path: the spread lines land immediately, then word-
   * timed versions replace them on success.
   */
  applyPlainTextLyrics(text: string, opts: { wordLevel?: boolean } = {}): number {
    const cleaned: string[] = [];
    for (const raw of text.split(/\r?\n/)) {
      const trimmed = raw.trim();
      if (trimmed.length === 0) continue;
      if (/^\[[^\]]*\]$/.test(trimmed)) continue;
      const stripped = stripLyricNoise(trimmed);
      if (stripped.length === 0) continue;
      cleaned.push(stripped);
    }
    if (cleaned.length === 0) return 0;
    const spreadSec = this.computeLyricsSpreadSec();
    // Linear `i / N` spread (not `i / (N-1)`) leaves the final 1/N of
    // the song as buffer past the last line, which is closer to how
    // real lyrics sit relative to a recording's tail (intro & outro
    // are often instrumental). First line lands at 0.
    const lines: LyricLine[] = cleaned.map((t, i) => ({
      startSec: (spreadSec * i) / cleaned.length,
      text: t,
    }));
    const trackId = lyricsStore.add(lines, {
      source: 'plaintext',
      sourceLabel: 'Plain text',
    });
    if (opts.wordLevel) {
      void this.runWordLevelAlignmentForPlainText(trackId, lines);
    }
    return lines.length;
  }

  /** Best-effort duration in seconds across which to spread untimed
   *  lyric lines. Prefers loaded audio (matches the realign domain),
   *  then the rendered jot's timeline, then a small default. */
  private computeLyricsSpreadSec(): number {
    let longestAudio = 0;
    for (const t of jotPlayer.audioTracks.values()) {
      if (t.durationSec > longestAudio) longestAudio = t.durationSec;
    }
    if (longestAudio > 0) return longestAudio;
    if (this.document.currentJot) {
      const tl = buildTimeline(this.document.currentJot);
      if (tl.totalDurationSec > 0) return tl.totalDurationSec;
    }
    return 60;
  }

  /** Mirror of {@link runWordLevelAlignmentForLrclib} for the plain-
   *  text source. Picks an audio track and runs CTC forced alignment
   *  using the spread lines as authoritative text; on success the
   *  lines are replaced with word-timed versions while the source
   *  label stays "Plain text". */
  private async runWordLevelAlignmentForPlainText(
    targetTrackId: LyricsTrackId,
    lines: readonly LyricLine[]
  ): Promise<void> {
    const pick = this.pickAudioTrackForAlignment();
    if (!pick) {
      toastStore.showError('Word-level alignment needs an audio track; load one first.');
      return;
    }
    const track = jotPlayer.audioTracks.get(pick.id);
    if (!track) return;
    const file = new File([track.sourceBlob], track.filename, { type: track.sourceBlob.type });
    await this.alignLyricsForced(
      targetTrackId,
      {
        kind: pick.kind,
        file,
        realign: {
          lines: lines.map((l) => ({ startSec: l.startSec, text: l.text })),
        },
      },
      'Plain text',
      { source: 'plaintext', sourceLabel: 'Plain text' }
    );
  }

  /**
   * Drop every lyrics row and abort every in-flight align. Called by
   * wholesale-song-reload paths (`loadJotFile`, `loadParadbMap`,
   * `applyDebugBundle`) so stale lyrics + still-running aligns can't
   * leak onto the new song.
   */
  clearLyrics(): void {
    lyricsStore.clear();
    this.cancelAllLyricsAlign();
  }

  /**
   * Remove a single lyrics row, aborting that row's in-flight align if
   * any. Routed through here (rather than `lyricsStore.remove(id)`
   * directly) so the lyrics store stays unaware of the per-track align
   * state held on `JotViewStore`.
   */
  removeLyricsTrack(id: LyricsTrackId): void {
    const ctrl = this.lyricsAlignControllers.get(id);
    if (ctrl) {
      ctrl.abort();
      this.lyricsAlignControllers.delete(id);
    }
    runInAction(() => {
      this.lyricsAlign.lyricsAlignStatuses.delete(id);
    });
    lyricsStore.remove(id);
  }

  /**
   * Per-track Whisper alignment state. Each row aligning at the same
   * time has its own AbortController and status entry; absence of an
   * entry means that row is idle. Per-track concurrency lets users
   * align a duet's two vocal lines without one cancelling the other,
   * and lets the per-row spinner show *which* row is currently working
   * (the toolbar busy pill, in contrast, just shows a generic "any
   * aligning" boolean).
   *
   * The controller map is non-observable; statuses are observable so
   * `lyricsAnyAligning` and the per-row spinner re-render on change.
   */
  lyricsAlignControllers: Map<LyricsTrackId, AbortController> = new Map();

  /**
   * Run CTC forced-alignment against the given input source and
   * upgrade `targetTrackId`'s lines on success. The caller supplies the
   * {@link LyricsSource} and source label to re-apply, so the row's
   * gutter label doesn't get rewritten to a hardcoded LRCLIB string
   * when the plain-text flow runs through here.
   *
   * Per-target concurrency: a second align on the SAME track aborts the
   * first (the newer pick wins). Aligns on DIFFERENT tracks run
   * concurrently from this layer's perspective; the backend serialises
   * them GPU-wise.
   */
  private async alignLyricsForced(
    targetTrackId: LyricsTrackId,
    req: AlignLyricsRequest,
    label: string,
    opts: { source: LyricsSource; sourceLabel: string }
  ): Promise<void> {
    const existing = this.lyricsAlignControllers.get(targetTrackId);
    if (existing) {
      existing.abort();
      this.lyricsAlignControllers.delete(targetTrackId);
    }
    const controller = new AbortController();
    this.lyricsAlignControllers.set(targetTrackId, controller);
    runInAction(() => {
      this.lyricsAlign.lyricsAlignStatuses.set(targetTrackId, { phase: 'aligning', detail: label });
    });
    let lines: LyricLine[];
    try {
      lines = await alignLyricsForced(req, {
        signal: controller.signal,
        onProgress: (event) => {
          // The stream emits `queued` while waiting behind another GPU
          // job, then `running` once alignment starts. Flip the per-row
          // status so the spinner/pill read "Queued…" vs "Aligning…".
          // Guard against a newer align (or a clear) that raced in while
          // we were waiting: only this controller may touch the status.
          if (this.lyricsAlignControllers.get(targetTrackId) !== controller) {
            return;
          }
          runInAction(() => {
            this.lyricsAlign.lyricsAlignStatuses.set(targetTrackId, {
              phase: event.kind === 'queued' ? 'queued' : 'aligning',
              detail: label,
            });
          });
        },
      });
    } catch (err) {
      if (controller.signal.aborted) {
        // A newer align on the same track (or a wholesale jot replace)
        // cancelled us; don't overwrite their state. The newer caller
        // already set either its own aligning status or cleared back to
        // idle for this track.
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      runInAction(() => {
        this.lyricsAlign.lyricsAlignStatuses.delete(targetTrackId);
      });
      toastStore.showError(`Lyrics align failed: ${message}`);
      return;
    } finally {
      if (this.lyricsAlignControllers.get(targetTrackId) === controller) {
        this.lyricsAlignControllers.delete(targetTrackId);
      }
    }
    if (lines.length === 0) {
      runInAction(() => {
        this.lyricsAlign.lyricsAlignStatuses.delete(targetTrackId);
      });
      toastStore.showError(`No lyrics were aligned (the aligner found no speech in ${label}).`);
      return;
    }
    runInAction(() => {
      lyricsStore.replace(targetTrackId, lines, {
        source: opts.source,
        sourceLabel: opts.sourceLabel,
      });
      this.lyricsAlign.lyricsAlignStatuses.delete(targetTrackId);
    });
  }

  /**
   * Abort every in-flight Whisper alignment and clear the statuses.
   * Called by wholesale-song-reload paths so slow aligns from the
   * previous song can't land lines onto the new one.
   */
  private cancelAllLyricsAlign(): void {
    for (const ctrl of this.lyricsAlignControllers.values()) {
      ctrl.abort();
    }
    this.lyricsAlignControllers.clear();
    runInAction(() => {
      this.lyricsAlign.lyricsAlignStatuses.clear();
    });
  }

  async playCurrent(): Promise<void> {
    const jot = this.document.currentJot;
    if (!jot) return;
    // Pass the laid-out RenderedJot (not its source) so the player's
    // timeline reads live bar widths — the playhead then tracks correctly
    // across zoom changes.
    await jotPlayer.play(jot);
  }

  stopPlayback(): void {
    jotPlayer.stop();
  }

  /**
   * Slide every drum note across the bar grid by `beats` quarter-note
   * beats to realign a consistently mis-detected groove (see
   * {@link RenderedJot.drumOffsetBeats}). Reflows the score reactively and
   * reschedules in-flight playback so the change is heard immediately.
   */
  setDrumOffset(beats: number): void {
    const jot = this.document.currentJot;
    if (!jot) return;
    // Slider semantics: the user is re-labeling note positions on the
    // notational grid (e.g. "this hit is on 1/48, not 3/48"), not
    // re-timing the drums against the audio recording. So when the
    // shift moves every note by Δ beats in jot time, compensate the
    // audio offset by the same magnitude in the opposite direction so
    // the audio-track waveform tracks the noteheads instead of sliding
    // out from under them. Uses the dominant bpm (the tempo the song
    // spends the most audio time at, excluding lead-in bars) rather
    // than globalMetadata.bpm, because transcribed bundles store a
    // back-solved lead-in tempo as the first setTempo event and that
    // value can be very different from the song's actual rate. Per-bar
    // tempo variation still leaves a few-ms-per-note residual; same
    // caveat as the Drum-offset row in the debug panel.
    const deltaBeats = beats - jot.drumOffsetBeats;
    if (Math.abs(deltaBeats) > 1e-12) {
      const { dominantBpm } = pickDominantBpmAndTime(jot);
      const bpm = dominantBpm ?? 120;
      const deltaSec = (deltaBeats * 60) / bpm;
      jotPlayer.setDrumsT0Sec(jotPlayer.drumsT0Sec - deltaSec);
    }
    jot.setDrumOffset(beats);
    jotPlayer.refreshDrumSchedule(jot);
  }

  /**
   * Click-to-seek. `x` is a pixel offset within the bars row — the same
   * coordinate space `bar.x` / the playhead use (origin at the left
   * edge of the bars region, after the gutter). While playing this
   * scrubs live; while idle it parks the playhead and the next Play
   * starts from there. Uses the live timeline when one exists so a
   * mid-playback scrub reads the exact bars being played.
   */
  seekToX(x: number): void {
    const jot = this.document.currentJot;
    if (!jot) return;
    const timeline = jotPlayer.timeline.bars.length > 0 ? jotPlayer.timeline : buildTimeline(jot);
    jotPlayer.seek(jot, xToTime(timeline, x));
  }

  /**
   * Single transport action shared by the spacebar shortcut and the
   * toolbar's play/pause button:
   *   idle    -> play the current jot from the start
   *   playing -> pause (freezes the clock, playhead stays put)
   *   paused  -> resume from the same spot
   * `loading` is intentionally a no-op so a double-press during the
   * one-time sample fetch can't stack two `play()` calls.
   */
  async togglePlayPause(): Promise<void> {
    switch (jotPlayer.state) {
      case 'idle':
        this.maybeReenableFollowOnPlay();
        await this.playCurrent();
        break;
      case 'playing':
        await jotPlayer.pause();
        break;
      case 'paused':
        this.maybeReenableFollowOnPlay();
        await jotPlayer.resume();
        break;
    }
  }

  /**
   * Restore {@link followPlayhead} on the idle/paused → playing
   * transition when the off-state was set during the previous playback
   * session (pan, minimap drag, or follow-button toggle while playing).
   * No-op when {@link autoFollowOnPlay} is off, when follow is already
   * on, or when the user deliberately disabled it while idle/paused.
   */
  private maybeReenableFollowOnPlay() {
    if (!this.playback.autoFollowOnPlay) return;
    if (this.playback.followPlayhead) return;
    if (!this.playback.followDisabledIsTransient) return;
    this.setFollowPlayhead(true);
  }

  // --- transcribe (form options + resume picker) ---

  setDebug(enabled: boolean) {
    this.transcribe.transcribeOptions.debug = enabled;
  }

  setBeatInput(input: BeatInput) {
    this.transcribe.transcribeOptions.beatInput = input;
  }

  setDrumSeparator(separator: DrumSeparator) {
    this.transcribe.transcribeOptions.drumSeparator = separator;
  }

  setLlmModel(model: LlmModel) {
    this.transcribe.transcribeOptions.llmModel = model;
  }

  setQuantise(enabled: boolean) {
    this.transcribe.transcribeOptions.quantise = enabled;
  }

  setQuantiseUseLlm(enabled: boolean) {
    this.transcribe.transcribeOptions.quantiseUseLlm = enabled;
  }

  setSelectedResumeFolder(folder: string | undefined) {
    this.transcribe.selectedResumeFolder = folder;
    // Clearing the folder (or picking a different one) invalidates any
    // stage selection, different folders have different `resumable_stages`,
    // so a stale pick could land on a stage missing its prerequisites.
    this.transcribe.selectedResumeStage = undefined;
  }

  setSelectedResumeStage(stage: TranscribeStage | undefined) {
    this.transcribe.selectedResumeStage = stage;
  }

  setTranscribeMode(mode: 'new' | 'resume') {
    this.transcribe.transcribeMode = mode;
  }

  // --- provenance / debug panel ---

  /** Replace the toolbar's `Show filtered` checkbox state. */
  setShowFilteredOnsets(show: boolean) {
    this.provenance.showFilteredOnsets = show;
  }

  setPinnedFilteredOnsetKey(key: string | undefined) {
    this.provenance.pinnedFilteredOnsetKey = key;
  }

  /** Toggle the DebugPanel's open state without forgetting the bundle. */
  toggleDebugPanel() {
    this.provenance.debugPanelOpen = !this.provenance.debugPanelOpen;
  }

  // --- lyrics (modal visibility) ---

  setLyricsSearchOpen(open: boolean) {
    this.lyricsAlign.lyricsSearchOpen = open;
  }

  setLyricsTextOpen(open: boolean) {
    this.lyricsAlign.lyricsTextOpen = open;
  }

  // --- playback / transport ---

  setAutoFollowOnPlay(on: boolean) {
    this.playback.autoFollowOnPlay = on;
  }

  // --- viewport (zoom / scroll / gutter) ---

  setZoom(z: number) {
    const clamped = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
    this.viewport.zoom = clamped;
    this.document.viewConfig.barWidth = px(BASE_BAR_WIDTH * clamped);
  }

  /** Cache the score viewport's pixel dimensions. Fed by a ResizeObserver
   * on `.jotContainer`. Re-clamps scroll so a resize that shrinks the
   * viewport (or grows it past the content) doesn't leave scroll parked
   * off the new end. */
  setViewportSize(width: number, height: number): void {
    this.viewport._viewportWidth = width;
    this.viewport._viewportHeight = height;
    this.viewport.scrollX = this.clampScrollX(this.viewport.scrollX);
    this.viewport.scrollY = this.clampScrollY(this.viewport.scrollY);
  }

  /** Cache the scroll-content's pixel dimensions (the inner
   * `.scrollViewport` wrapper's offset size). Re-clamps as above. */
  setContentSize(width: number, height: number): void {
    this.viewport._contentWidth = width;
    this.viewport._contentHeight = height;
    this.viewport.scrollX = this.clampScrollX(this.viewport.scrollX);
    this.viewport.scrollY = this.clampScrollY(this.viewport.scrollY);
  }

  setScrollX(x: number): void {
    this.viewport.scrollX = this.clampScrollX(snapToDevicePx(x));
  }

  setScrollY(y: number): void {
    this.viewport.scrollY = this.clampScrollY(snapToDevicePx(y));
  }

  setScrollBy(dx: number, dy: number): void {
    this.viewport.scrollX = this.clampScrollX(snapToDevicePx(this.viewport.scrollX + dx));
    this.viewport.scrollY = this.clampScrollY(snapToDevicePx(this.viewport.scrollY + dy));
  }

  /** Reset the horizontal scroll to the score's start (Stop transitions).
   * Deliberately does NOT touch scrollY, the user's vertical view
   * shouldn't snap back on Stop, only the playhead-tracking axis. */
  resetScrollX(): void {
    this.viewport.scrollX = 0;
  }

  /** Clamp a tentative target to `[0, contentSize - viewportSize]`. */
  clampScrollX(x: number): number {
    const max = Math.max(0, this.viewport._contentWidth - this.viewport._viewportWidth);
    if (!(x > 0)) return 0;
    if (x > max) return max;
    return x;
  }

  clampScrollY(y: number): number {
    const max = Math.max(0, this.viewport._contentHeight - this.viewport._viewportHeight);
    if (!(y > 0)) return 0;
    if (y > max) return max;
    return y;
  }

  /** Resize the sticky gutter column, clamped to a sensible range so a
   * runaway drag can't collapse the controls or push the bars row off
   * screen. */
  setGutterWidth(width: number): void {
    if (!Number.isFinite(width)) return;
    this.viewport.gutterWidth = Math.min(MAX_GUTTER_WIDTH, Math.max(MIN_GUTTER_WIDTH, width));
  }
}
