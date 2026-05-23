import { makeAutoObservable, reaction, runInAction } from 'mobx';
import {
  DebugBundleManifest,
  loadDebugZip,
  NO_DRUMS_KEY,
  NoteProvenanceEntry,
  NoteProvenanceFile,
} from 'src/debug_zip';
import { ExampleJot } from 'src/fakes';
import { RenderedJot, ViewConfig, px } from 'src/jot';
import { fromMidi } from 'src/midi';
import { parse, ParseError } from 'src/parser';
import {
  AudioTrackId,
  isAudibleUnder,
  isAudioTrackAudibleUnder,
  jotPlayer,
  buildTimeline,
  xToTime,
} from 'src/playback';
import { loadParadbZip } from 'src/rlrr';
import {
  BeatInput,
  stemUrl,
  titleFromFilename,
  transcriber,
  TranscribeProgress,
  TranscribeStage,
  TranscriptionSummary,
} from 'src/transcriber';

export type TranscribeStatus =
  | { phase: 'idle' }
  | {
      phase: 'uploading';
      filename: string;
      /** Current pipeline stage (`stems_all`, `beats`, `transcribe`, …)
       *  reported by the server's NDJSON progress stream. `undefined`
       *  until the first stage event arrives — the initial "uploading"
       *  read covers everything before the first stage starts. */
      stage?: TranscribeStage;
      /** Optional in-stage detail, e.g. "filtering 3/5 instruments
       *  (latest: snare)". Cleared whenever the stage advances. */
      substage?: string;
    }
  | { phase: 'error'; message: string }
  | {
      phase: 'success';
      filename: string;
      tempo: number;
      hasTempoChanges: boolean;
      hasTimeSigChanges: boolean;
      barCount: number;
      debugDir?: string | null;
      /** Resolved URL of the debug bundle for this run (see
       * `TranscribeResponse.debug_zip_url`). The success pill renders a
       * download link to this; the user can then drop the zip back into
       * "Load debug bundle" to inspect logs in-app. */
      debugZipUrl?: string | null;
    };

export type TranscribeOptions = {
  debug: boolean;
  beatInput: BeatInput;
};

/**
 * Pixels-per-bar at zoom = 1. Same numeric value as `ViewConfig.barWidth`'s
 * own default so existing layouts are unchanged for users who never touch
 * the slider.
 */
export const BASE_BAR_WIDTH = 448;
export const MIN_ZOOM = 0.3;
export const MAX_ZOOM = 3.0;
// Row volume faders are pure attenuation (0 = silent, 1 = unscaled).
// The kit's overall loudness is handled by the drum master gain.
export const VOLUME_STEP = 0.05;

// Sticky gutter column width (px). Default matches the legacy
// hardcoded 132px so existing layouts are unchanged; the user can drag
// the gutter's right edge to widen it when long track names are clipped
// with `…` and `fit-content` would be too jumpy.
export const DEFAULT_GUTTER_WIDTH = 132;
// Floor at the width needed to fit the row gutter's minimum content:
// padding + drag handle + a short volume slider + the X/M/S button trio
// (audio-track rows have all three; pitch rows render an invisible
// spacer where X would sit so both rows share the same geometry).
export const MIN_GUTTER_WIDTH = 128;
export const MAX_GUTTER_WIDTH = 480;

export function clampVolume(v: number): number {
  if (!Number.isFinite(v)) return 1;
  return Math.max(0, Math.min(1, v));
}

/**
 * One row in the unified mixer — either a loaded audio (backing) track
 * or a single drum-instrument pitch. Pitch rows are keyed by DSL pitch
 * letter (the same key mute/solo/volume already use), so identity is
 * stable across jot reloads when the user keeps the same instrument
 * map. The order of these keys in {@link JotViewStore.trackOrder} drives
 * the row order on screen and can be rearranged by drag-and-drop.
 *
 * `groupId` is a UI-only clustering tag — consecutive entries that
 * share the same id render flush (no inter-row gap); a transition to a
 * different id (or to/from `undefined`) draws the small inter-group
 * gap. Identity for sync/move purposes is `kind + id/pitch` only;
 * `trackKeyEq` ignores `groupId`. Today only the debug-bundle loader
 * assigns these (one fresh id per audio↔pitch pair); a future
 * "create group" UI could expose it directly.
 */
export type TrackKey =
  | { kind: 'audio'; id: AudioTrackId; groupId?: string }
  | { kind: 'pitch'; pitch: string; groupId?: string };

export function trackKeyEq(a: TrackKey, b: TrackKey): boolean {
  if (a.kind !== b.kind) return false;
  return a.kind === 'audio'
    ? a.id === (b as { kind: 'audio'; id: AudioTrackId }).id
    : a.pitch === (b as { kind: 'pitch'; pitch: string }).pitch;
}

/**
 * Pitches that appear anywhere in the rendered jot, in voice-then-source
 * order. Multi-voice jots flatten voice-by-voice so the natural default
 * matches the legacy stacked-staves order (e.g. hands then feet); a
 * pitch that shows up in two voices is listed once at its first
 * appearance.
 *
 * Reads the zoom-invariant structural cache (not `jot.resolved`) so the
 * mixer-order reaction that wraps this doesn't re-evaluate on every
 * wheel tick — pitch identity is a function of the source DSL, not the
 * pixel layout.
 */
export function collectJotPitches(jot: RenderedJot | undefined): string[] {
  if (!jot) return [];
  const out: string[] = [];
  for (const voice of jot.structure.voices) {
    for (const p of voice.pitches) {
      if (!out.includes(p)) out.push(p);
    }
  }
  return out;
}

export class JotViewStore {
  currentJot: RenderedJot | undefined;
  examples: readonly ExampleJot[] = [];
  currentExampleId: string | undefined = undefined;
  transcribeStatus: TranscribeStatus = { phase: 'idle' };
  /** UI-controlled options for the next transcribe call. `debug=true`
   *  so the run is resumable. */
  transcribeOptions: TranscribeOptions = {
    debug: true,
    beatInput: 'full_mix',
  };
  /** Server-side picker of recent /transcribe runs that can be resumed.
   *  Populated by {@link refreshRecentTranscriptions}; an empty array
   *  before the first fetch (the picker shows "Loading…" in that state).
   *  Refreshed lazily whenever the toolbar opens the Transcribe dropdown
   *  so the operator sees their just-completed run without needing to
   *  reload the page. */
  recentTranscriptions: TranscriptionSummary[] = [];
  /** Folder name of the currently-selected recent transcription, or
   *  `undefined` when nothing is selected. Drives the stage picker (we
   *  read `resumable_stages` off the matching summary). */
  selectedResumeFolder: string | undefined = undefined;
  /** Stage the user has picked to resume from. `undefined` until they
   *  pick one; reset whenever {@link selectedResumeFolder} changes so
   *  stale picks from one folder can't leak into another folder's
   *  request. */
  selectedResumeStage: TranscribeStage | undefined = undefined;
  /**
   * Shared layout config threaded into every new `RenderedJot` we
   * construct, so the zoom slider mutates a single config object and
   * the layout reflows reactively (ViewConfig is MobX-observable;
   * RenderedJot's `layoutJot` is a computedFn that reads `barWidth`).
   */
  viewConfig: ViewConfig = new ViewConfig();
  /** Horizontal zoom multiplier; 1.0 = `BASE_BAR_WIDTH` pixels per bar. */
  zoom: number = 1;
  /** DSL pitches the user has muted via the row-gutter M button. */
  mutedPitches: Set<string> = new Set();
  /**
   * DSL pitches the user has soloed. When non-empty, ONLY these rows
   * are audible; this and `mutedPitches` are pushed to the player via
   * an autorun so toggles take effect live during playback.
   */
  soloedPitches: Set<string> = new Set();
  /** Audio-track ids the user has muted via the gutter M button. */
  mutedAudioTracks: Set<AudioTrackId> = new Set();
  /** Soloed audio-track ids — same semantics as `soloedPitches`. */
  soloedAudioTracks: Set<AudioTrackId> = new Set();
  /**
   * Per-row volume faders, 0..1 (1 = full). Sparse: a row absent from
   * the map plays at full volume. Pitch volumes scale note velocity in
   * the scheduler; audio-track volumes scale the track's GainNode.
   */
  pitchVolumes: Map<string, number> = new Map();
  audioTrackVolumes: Map<AudioTrackId, number> = new Map();
  /**
   * User-customizable order of mixer rows. Each entry is either a
   * loaded audio track id or a DSL pitch letter; the mixer renders rows
   * top-to-bottom in this exact order, with audio and drum-instrument
   * rows freely interleavable.
   *
   * Kept in sync with the live set of audio tracks (added via
   * {@link loadAudioTrack}, removed via {@link clearAudioTrack}) and
   * the pitches in the current jot through a reaction in the
   * constructor — entries that no longer correspond to anything are
   * dropped, new audio tracks append after the last audio entry (or to
   * the top if none), and new pitches append at the end so manual
   * reorderings survive reloads.
   */
  trackOrder: TrackKey[] = [];
  /**
   * Last loaded transcriber debug bundle (`.zip`), if any. Carries the
   * captured logs + per-stage timings produced server-side during a
   * transcribe run, so the UI's DebugPanel can show what happened end-
   * to-end without requiring a `docker compose logs` round trip.
   * Replaced when a new bundle is loaded; otherwise survives jot/audio
   * changes.
   */
  lastDebugBundle: DebugBundleManifest | undefined = undefined;
  /**
   * Per-note debug provenance from the loaded debug bundle, if the
   * bundle came from a filter-mode transcribe run. Keyed by DSL pitch
   * letter → list of every detected onset (kept and rejected). The
   * NoteView selection label looks up its provenance by matching
   * `note.metadata.midi.tick` against entries' `tick`; the
   * FilteredOnsetView renders the `kept=false` entries as ghost
   * overlays gated by {@link showFilteredOnsets}. `undefined` until a
   * filter-mode bundle is loaded; cleared when a new (non-bundle) song
   * replaces the current one.
   */
  noteProvenance: NoteProvenanceFile | undefined = undefined;
  /**
   * Toolbar checkbox: show rejected onsets as dashed ghost overlays.
   * Only meaningful when {@link noteProvenance} is loaded; the checkbox
   * is hidden when there's nothing to show. Default off so a freshly
   * loaded bundle reads as just "the score" until the operator opts in.
   */
  showFilteredOnsets: boolean = false;
  /** Whether the DebugPanel is expanded — small UI state, kept here so
   * the toolbar toggle and the panel itself stay in sync. */
  debugPanelOpen: boolean = false;
  /** Height of the DebugPanel (px) when expanded; adjusted by dragging
   * the resize handle along its top edge. */
  debugPanelHeight: number = 280;
  /** Width (px) of the sticky mixer/score gutter column; user-resizable
   * by dragging the gutter's right edge. Propagated to every gutter
   * element through the `--gutter-width` CSS variable set on the JotView
   * container. */
  gutterWidth: number = DEFAULT_GUTTER_WIDTH;
  /**
   * Controller for the in-flight `/transcribe` request, if any. The
   * "Stop" toolbar button calls `.abort()` here; the request's
   * AbortSignal is passed into `transcriber.transcribe` which forwards
   * it to `fetch` so the request is genuinely cancelled at the
   * network layer rather than just discarding the response.
   */
  private transcribeController: AbortController | undefined;

  /**
   * In-flight file-load counter. Each top-level loader (jot / midi / paradb
   * map / debug bundle / audio track) enters via {@link withLoading}, which
   * bumps this and surfaces the modal overlay. Nested calls (e.g. the debug
   * bundle loading its per-stem audio tracks) bump the count too but keep
   * the outer label so the overlay reads as one operation. The first loader
   * sets {@link loadingLabel}; later loaders only set it again when the
   * count was zero, so we don't churn the label while nested work runs.
   */
  loadingCount: number = 0;
  loadingLabel: string | undefined = undefined;

  get isLoading(): boolean {
    return this.loadingCount > 0;
  }

  /**
   * Wrap an async file-load with the modal overlay's bookkeeping. Errors
   * propagate; the finally block guarantees the counter decrements even if
   * the inner promise rejects, so a failed load never leaves the overlay
   * stuck on screen.
   */
  private async withLoading<T>(label: string, fn: () => Promise<T>): Promise<T> {
    runInAction(() => {
      if (this.loadingCount === 0) this.loadingLabel = label;
      this.loadingCount += 1;
    });
    try {
      return await fn();
    } finally {
      runInAction(() => {
        this.loadingCount -= 1;
        if (this.loadingCount === 0) this.loadingLabel = undefined;
      });
    }
  }

  constructor() {
    makeAutoObservable(this);
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
    reaction(
      () => ({
        mutedPitches: new Set(this.mutedPitches),
        soloedPitches: new Set(this.soloedPitches),
        soloActive: this.soloActive,
        volumes: new Map(this.pitchVolumes),
      }),
      (filter) => jotPlayer.setFilter(filter),
      { fireImmediately: true },
    );
    // Same shape for audio tracks — observed mutations push immediately
    // so toggling M/S on a track is sample-accurate during playback
    // (per-track GainNode flip, no source recreation). Same
    // read-and-write-the-same-observable hazard as above
    // (`setAudioTrackFilter` reads/writes `currentAudioTrackFilter`), so this is a
    // `reaction` for the same reason.
    reaction(
      () => ({
        mutedAudioTracks: new Set(this.mutedAudioTracks),
        soloedAudioTracks: new Set(this.soloedAudioTracks),
        soloActive: this.soloActive,
        volumes: new Map(this.audioTrackVolumes),
      }),
      (filter) => jotPlayer.setAudioTrackFilter(filter),
      { fireImmediately: true },
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
        const raw = this.currentJot?.globalMetadata.drumsT0Sec;
        return typeof raw === 'number' && raw > 0 ? raw : 0;
      },
      (offsetSec) => jotPlayer.setDrumsT0Sec(offsetSec),
      { fireImmediately: true },
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
        pitches: collectJotPitches(this.currentJot),
      }),
      ({ audioIds, pitches }) => this.syncTrackOrder(audioIds, pitches),
      { fireImmediately: true },
    );
  }

  /**
   * Drop entries from {@link trackOrder} that no longer correspond to a
   * live audio track or jot pitch, then append the missing ones at a
   * sensible default position so the row appears immediately:
   *   - new audio track  → after the last existing audio entry (or top
   *     of the list if no audio entries exist yet)
   *   - new pitch        → end of the list
   *
   * Existing entries keep their relative order so a user drag survives
   * an audio-track add/remove or a jot reload that didn't change the
   * pitch set.
   */
  private syncTrackOrder(audioIds: AudioTrackId[], pitches: string[]): void {
    const wanted: TrackKey[] = [
      ...audioIds.map((id) => ({ kind: 'audio' as const, id })),
      ...pitches.map((pitch) => ({ kind: 'pitch' as const, pitch })),
    ];
    const next: TrackKey[] = this.trackOrder.filter((k) =>
      wanted.some((w) => trackKeyEq(w, k)),
    );
    for (const w of wanted) {
      if (next.some((k) => trackKeyEq(k, w))) continue;
      if (w.kind === 'audio') {
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
      next.length === this.trackOrder.length &&
      next.every(
        (k, i) =>
          trackKeyEq(k, this.trackOrder[i]) && k.groupId === this.trackOrder[i].groupId,
      )
    ) {
      return;
    }
    this.trackOrder = next;
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
    if (fromIdx < 0 || fromIdx >= this.trackOrder.length) return;
    const clamped = Math.max(0, Math.min(this.trackOrder.length, toIdx));
    if (clamped === fromIdx || clamped === fromIdx + 1) return;
    const next = this.trackOrder.slice();
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
    const repositioned: TrackKey =
      moved.kind === 'audio'
        ? { kind: 'audio', id: moved.id, groupId: newGroupId }
        : { kind: 'pitch', pitch: moved.pitch, groupId: newGroupId };
    next.splice(adjusted, 0, repositioned);
    this.trackOrder = next;
  }

  /**
   * Solo is one global mode across both the pitch and audio-track domains: any
   * soloed row (drum *or* music) puts every non-soloed row — in either
   * domain — into the "solo-excluded" state. Without this, soloing a
   * drum to practise it would leave the backing music playing.
   */
  get soloActive(): boolean {
    return this.soloedPitches.size > 0 || this.soloedAudioTracks.size > 0;
  }

  toggleMute(pitch: string) {
    if (this.mutedPitches.has(pitch)) this.mutedPitches.delete(pitch);
    else this.mutedPitches.add(pitch);
  }

  toggleSolo(pitch: string) {
    if (this.soloedPitches.has(pitch)) this.soloedPitches.delete(pitch);
    else this.soloedPitches.add(pitch);
  }

  isPitchAudible(pitch: string): boolean {
    return isAudibleUnder(pitch, {
      mutedPitches: this.mutedPitches,
      soloedPitches: this.soloedPitches,
      soloActive: this.soloActive,
      volumes: this.pitchVolumes,
    });
  }

  pitchVolume(pitch: string): number {
    return this.pitchVolumes.get(pitch) ?? 1;
  }

  setPitchVolume(pitch: string, v: number) {
    this.pitchVolumes.set(pitch, clampVolume(v));
  }

  toggleAudioTrackMute(id: AudioTrackId) {
    if (this.mutedAudioTracks.has(id)) this.mutedAudioTracks.delete(id);
    else this.mutedAudioTracks.add(id);
  }

  toggleAudioTrackSolo(id: AudioTrackId) {
    if (this.soloedAudioTracks.has(id)) this.soloedAudioTracks.delete(id);
    else this.soloedAudioTracks.add(id);
  }

  isAudioTrackAudible(id: AudioTrackId): boolean {
    return isAudioTrackAudibleUnder(id, {
      mutedAudioTracks: this.mutedAudioTracks,
      soloedAudioTracks: this.soloedAudioTracks,
      soloActive: this.soloActive,
      volumes: this.audioTrackVolumes,
    });
  }

  audioTrackVolume(id: AudioTrackId): number {
    return this.audioTrackVolumes.get(id) ?? 1;
  }

  setAudioTrackVolume(id: AudioTrackId, v: number) {
    this.audioTrackVolumes.set(id, clampVolume(v));
  }

  /**
   * Load an audio file as a new audio track and update the status pill
   * on failure. Decoding goes through the shared `AudioContext`, so the
   * call has to occur inside a user gesture (the file-picker click
   * satisfies that). Every call appends an independent track — load N
   * files to get N tracks. Returns the new track's id, or `undefined`
   * if the load failed (so callers can e.g. default it to muted).
   */
  async loadAudioTrack(file: File, pitch?: string): Promise<AudioTrackId | undefined> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      try {
        return await jotPlayer.loadAudioTrack(file, pitch);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        runInAction(() => {
          this.transcribeStatus = {
            phase: 'error',
            message: `Audio track load failed: ${message}`,
          };
        });
        return undefined;
      }
    });
  }

  clearAudioTrack(id: AudioTrackId): void {
    jotPlayer.clearAudioTrack(id);
    // Drop the removed track's mute/solo/volume so it doesn't linger
    // (ids are never reused, so the entries would be dead weight) — and
    // critically so clearing the only soloed audio track doesn't leave a
    // phantom solo silencing everything else.
    this.mutedAudioTracks.delete(id);
    this.soloedAudioTracks.delete(id);
    this.audioTrackVolumes.delete(id);
  }

  /** Drop every loaded audio track. Used when a new source (e.g. a
   * ParaDB pack) replaces the current song — otherwise the previous
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
    this.mutedPitches.clear();
    this.soloedPitches.clear();
    this.pitchVolumes.clear();
  }

  setJot(jot: RenderedJot | undefined) {
    this.currentJot = jot;
    // External setJot calls invalidate the example pointer + any
    // previously-loaded debug provenance (provenance is per-bundle and
    // doesn't survive a wholesale jot replacement).
    this.currentExampleId = undefined;
    this.clearNoteProvenance();
    jotPlayer.clearCue();
  }

  /** Drop the debug bundle's per-note provenance + reset the toolbar
   * visibility toggle. Called from every loader that replaces the
   * current song outside the bundle path so stale debug info from a
   * previous bundle doesn't leak onto the new score. */
  private clearNoteProvenance() {
    this.noteProvenance = undefined;
    this.showFilteredOnsets = false;
  }

  /** Replace the toolbar's `Show filtered` checkbox state. */
  setShowFilteredOnsets(show: boolean) {
    this.showFilteredOnsets = show;
  }

  /**
   * Pre-indexed view onto `noteProvenance` for the per-note selection
   * label lookup. Keyed by `${pitch}:${tick}` so `NoteView` can attach
   * provenance to its note in O(1) instead of scanning the per-pitch
   * list on every render. Recomputed when `noteProvenance` changes.
   */
  get noteProvenanceByTick(): Map<string, NoteProvenanceEntry> {
    const out = new Map<string, NoteProvenanceEntry>();
    const provenance = this.noteProvenance;
    if (!provenance) return out;
    for (const [pitch, entries] of Object.entries(provenance.per_pitch)) {
      for (const entry of entries) {
        if (entry.tick === null || !entry.kept) continue;
        out.set(`${pitch}:${entry.tick}`, entry);
      }
    }
    return out;
  }

  /**
   * Per-pitch list of rejected onsets the {@link FilteredOnsetView}
   * renders. Built once from `noteProvenance` and cached via MobX so
   * the per-pitch row doesn't re-filter on every render. Out-of-range
   * entries (those that fell outside the beat-tracked region) are
   * dropped — they have no displayable bar to anchor against.
   */
  get filteredOnsetsByPitch(): Map<string, NoteProvenanceEntry[]> {
    const out = new Map<string, NoteProvenanceEntry[]>();
    const provenance = this.noteProvenance;
    if (!provenance) return out;
    for (const [pitch, entries] of Object.entries(provenance.per_pitch)) {
      const rejected = entries.filter(
        (e) => !e.kept && !e.out_of_range,
      );
      if (rejected.length > 0) out.set(pitch, rejected);
    }
    return out;
  }

  setExamples(examples: readonly ExampleJot[]) {
    this.examples = examples;
  }

  loadExample(id: string) {
    const example = this.examples.find((e) => e.id === id);
    if (!example) return;
    this.currentJot = new RenderedJot(example.jot, this.viewConfig);
    this.currentExampleId = id;
    this.clearNoteProvenance();
    jotPlayer.clearCue();
  }

  setDebug(enabled: boolean) {
    this.transcribeOptions.debug = enabled;
  }

  setBeatInput(input: BeatInput) {
    this.transcribeOptions.beatInput = input;
  }

  setSelectedResumeFolder(folder: string | undefined) {
    this.selectedResumeFolder = folder;
    // Clearing the folder (or picking a different one) invalidates any
    // stage selection — different folders have different `resumable_stages`,
    // so a stale pick could land on a stage missing its prerequisites.
    this.selectedResumeStage = undefined;
  }

  setSelectedResumeStage(stage: TranscribeStage | undefined) {
    this.selectedResumeStage = stage;
  }

  setZoom(z: number) {
    const clamped = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
    this.zoom = clamped;
    this.viewConfig.barWidth = px(BASE_BAR_WIDTH * clamped);
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
      this.transcribeStatus = { phase: 'uploading', filename: file.name };
    });
    try {
      const response = await transcriber.transcribe(file, {
        debug: this.transcribeOptions.debug,
        beatInput: this.transcribeOptions.beatInput,
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
      this.transcribeStatus = { phase: 'uploading', filename: label };
    });
    try {
      const response = await transcriber.resume({
        resumeFolder: folder,
        resumeStage: stage,
        beatInput: this.transcribeOptions.beatInput,
        signal: controller.signal,
        onProgress: (event) => this.applyProgress(label, event),
      });
      // The resumed run reuses the original folder, so the original
      // upload filename is the most informative pill label — fall back
      // to the resume folder name when the server doesn't know it.
      const fallbackName =
        this.recentTranscriptions.find((t) => t.folder === folder)
          ?.original_filename ?? folder;
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
    signal: AbortSignal,
  ): Promise<void> {
    const bundleUrl = stemUrl(response.debug_zip_url ?? null);
    if (!bundleUrl) {
      runInAction(() => {
        this.transcribeStatus = {
          phase: 'error',
          message: 'Transcriber returned no debug bundle.',
        };
      });
      return;
    }
    const ok = await this.autoLoadDebugBundle(bundleUrl, fallbackName, signal);
    if (!ok) {
      // The auto-loader already populated `transcribeStatus` with the
      // specific failure reason; bail without overwriting it.
      return;
    }
    runInAction(() => {
      this.transcribeStatus = {
        phase: 'success',
        filename: fallbackName,
        tempo: response.metadata.initial_tempo,
        hasTempoChanges: response.metadata.has_tempo_changes,
        hasTimeSigChanges: response.metadata.has_time_sig_changes,
        barCount: response.metadata.bars.length,
        debugDir: response.debug_dir ?? null,
        debugZipUrl: bundleUrl,
      };
    });
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
    signal: AbortSignal,
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
      const status = this.transcribeStatus;
      // If the request was aborted or already terminal (success/error)
      // before this late event fires, ignore — late progress shouldn't
      // resurrect the spinner over an idle/success/error pill.
      if (status.phase !== 'uploading') return;
      if (event.kind === 'stage' && event.phase === 'start') {
        this.transcribeStatus = {
          phase: 'uploading',
          filename,
          stage: event.stage,
        };
      } else if (event.kind === 'substage') {
        this.transcribeStatus = {
          phase: 'uploading',
          filename,
          stage: event.stage,
          substage: event.detail,
        };
      }
    });
  }

  private handleTranscribeError(
    err: unknown,
    controller: AbortController,
    verb: string,
  ): void {
    // AbortError surfaces as DOMException with name='AbortError' (and
    // wraps as TypeError in some runtimes when the fetch was already
    // aborted at start). Treat the user-initiated cancellation
    // distinctly from real errors so we don't show a scary red pill.
    const isAbort =
      controller.signal.aborted ||
      (err instanceof DOMException && err.name === 'AbortError');
    if (isAbort) {
      runInAction(() => {
        this.transcribeStatus = { phase: 'idle' };
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
      this.transcribeStatus = { phase: 'error', message };
    });
  }

  /**
   * Refresh the recent-transcriptions picker from the server. Failures
   * are logged but never surfaced — the picker just stays as-is, which
   * is the right behaviour when the backend is briefly unavailable.
   * Safe to call from a fire-and-forget context.
   */
  async refreshRecentTranscriptions(): Promise<void> {
    try {
      const list = await transcriber.listTranscriptions();
      runInAction(() => {
        this.recentTranscriptions = list;
        // Drop the selection if its target folder vanished server-side
        // (e.g. operator pruned the debug dir between dropdown opens).
        if (
          this.selectedResumeFolder !== undefined &&
          !list.some((s) => s.folder === this.selectedResumeFolder)
        ) {
          this.selectedResumeFolder = undefined;
          this.selectedResumeStage = undefined;
        }
      });
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn('Could not refresh recent transcriptions:', err);
    }
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

  clearTranscribeStatus() {
    this.transcribeStatus = { phase: 'idle' };
  }

  /**
   * Read a Drumjot DSL file from the user's machine and load it as the
   * current jot. Parse failures are surfaced through `transcribeStatus`
   * (same error pill the transcribe flow uses) so the user sees what
   * went wrong rather than getting silent dismissal.
   */
  async loadJotFile(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      let text: string;
      try {
        text = await file.text();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        runInAction(() => {
          this.transcribeStatus = { phase: 'error', message: `Could not read ${file.name}: ${message}` };
        });
        return;
      }
      try {
        const jot = parse(text);
        if (!jot.title) {
          const derivedTitle = titleFromFilename(file.name);
          if (derivedTitle) jot.title = derivedTitle;
        }
        runInAction(() => {
          this.currentJot = new RenderedJot(jot, this.viewConfig);
          this.currentExampleId = undefined;
          // A bare jot file has no provenance — drop whatever the
          // previous bundle put there so the selection label doesn't
          // surface stale debug data on the new song's notes.
          this.clearNoteProvenance();
          jotPlayer.clearCue();
          this.transcribeStatus = { phase: 'idle' };
        });
      } catch (err) {
        const message =
          err instanceof ParseError
            ? `Could not parse ${file.name}: ${err.message}`
            : err instanceof Error
              ? err.message
              : String(err);
        runInAction(() => {
          this.transcribeStatus = { phase: 'error', message };
        });
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
        runInAction(() => {
          this.transcribeStatus = { phase: 'error', message: `Could not read ${file.name}: ${message}` };
        });
        return;
      }
      try {
        const jot = fromMidi(bytes);
        if (!jot.title) {
          const derivedTitle = titleFromFilename(file.name);
          if (derivedTitle) jot.title = derivedTitle;
        }
        runInAction(() => {
          this.currentJot = new RenderedJot(jot, this.viewConfig);
          this.currentExampleId = undefined;
          // Same reasoning as in loadJotFile: a bare MIDI load shouldn't
          // surface stale provenance from a previous debug bundle.
          this.clearNoteProvenance();
          jotPlayer.clearCue();
          this.transcribeStatus = { phase: 'idle' };
        });
      } catch (err) {
        const message =
          err instanceof Error
            ? `Could not convert ${file.name}: ${err.message}`
            : String(err);
        runInAction(() => {
          this.transcribeStatus = { phase: 'error', message };
        });
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
        runInAction(() => {
          this.transcribeStatus = {
            phase: 'error',
            message: `Could not load ${file.name}: ${message}`,
          };
        });
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
        this.currentJot = new RenderedJot(jot, this.viewConfig);
        this.currentExampleId = undefined;
        this.clearNoteProvenance();
        jotPlayer.clearCue();
        this.transcribeStatus = { phase: 'idle' };
      });

      // Audio tracks are best-effort: a chart with the score loaded is
      // still useful even if one is absent or fails to decode.
      // loadAudioTrack already reports its own failures on the status pill.
      // Drum tracks load too but start muted — you're playing the drums,
      // so the backing music should be the only thing you hear by default.
      for (const track of map.audioTracks) {
        const id = await this.loadAudioTrack(track.file);
        if (id && track.defaultMuted) {
          runInAction(() => {
            this.mutedAudioTracks.add(id);
          });
        }
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
        runInAction(() => {
          this.transcribeStatus = {
            phase: 'error',
            message: `Could not load ${file.name}: ${message}`,
          };
        });
        return;
      }
      const ok = await this.applyDebugBundle(bundle, file.name);
      // The shared apply path doesn't touch `transcribeStatus` (callers
      // own that pill — `transcribeAudio` keeps its success state visible
      // after auto-loading the bundle). For the explicit "load a bundle"
      // path the user expects the pill to clear on success or carry the
      // bundle-specific error on failure.
      runInAction(() => {
        this.transcribeStatus = ok
          ? { phase: 'idle' }
          : {
              phase: 'error',
              message: `Could not parse score from ${file.name}.`,
            };
      });
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
    fallbackName: string,
  ): Promise<boolean> {
    runInAction(() => {
      this.clearAllAudioTracks();
      this.resetPitchMixer();
      this.lastDebugBundle = bundle.manifest;
      // Replace (or clear) the per-note debug provenance whenever a
      // new bundle loads. Older bundles may not carry one (e.g. a
      // hand-built or legacy zip); the absent-case clears the previous
      // bundle's provenance so it doesn't leak onto the new score.
      this.noteProvenance = bundle.noteProvenance ?? undefined;
      // Reset the visibility toggle so a freshly loaded bundle reads
      // as just "the score" — operator opts into the ghost overlays.
      this.showFilteredOnsets = false;
      this.debugPanelOpen = true;
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
        // The downbeat detector recorded the global beat-alignment offset
        // it applied in the provenance sidecar. Convert it into the same
        // quarter-note-beat coordinates the Beat control uses (negate
        // because adding `offset_sec` to every beat.time moves notes by
        // `-offset_sec * bpm/60` on the beat grid) and seed it as both
        // the control value AND the baseline — net applied shift is 0,
        // so notes stay at the MIDI positions while the operator can
        // see the alignment value and reset it to expose the pre-
        // alignment positions.
        const alignmentSec = bundle.noteProvenance?.beat_alignment_offset_sec;
        const bpm = jot.globalMetadata.bpm;
        const alignmentBeats =
          typeof alignmentSec === 'number' &&
          Number.isFinite(alignmentSec) &&
          typeof bpm === 'number' &&
          bpm > 0
            ? (-alignmentSec * bpm) / 60
            : 0;
        runInAction(() => {
          const rendered = new RenderedJot(jot, this.viewConfig);
          if (alignmentBeats !== 0) {
            rendered.setDrumOffsetBaseline(alignmentBeats);
            rendered.setDrumOffset(alignmentBeats);
          }
          this.currentJot = rendered;
          this.currentExampleId = undefined;
          jotPlayer.clearCue();
        });
        scoreLoaded = true;
      } catch (err) {
        const message =
          err instanceof Error
            ? `Could not convert prediction.mid: ${err.message}`
            : String(err);
        runInAction(() => {
          this.transcribeStatus = { phase: 'error', message };
        });
      }
    }

    // Decode every audio track in parallel — `decodeAudioData` runs on
    // browser-side codec threads, so concurrent calls overlap well and
    // turn what used to be a one-by-one wait into a single combined
    // wait. `Promise.all` preserves input order so the resolved array
    // still matches `bundle.audioTracks` (which is already in manifest
    // order — `no_drums` first, then pitch letters), keeping the
    // post-load pair-with-instrument-row logic stable.
    const resolved = await Promise.all(
      bundle.audioTracks.map((track) =>
        this.loadAudioTrack(
          track.file,
          track.key !== NO_DRUMS_KEY ? track.key : undefined,
        ).then((id) => ({ key: track.key, id })),
      ),
    );
    const loadedByKey = new Map<string, AudioTrackId>();
    const toMute: AudioTrackId[] = [];
    for (const { key, id } of resolved) {
      if (!id) continue;
      loadedByKey.set(key, id);
      // Mute the per-pitch stems by default so the (audible) drums come
      // from the smplr score scheduler; the drumless backing stays unmuted.
      if (key !== NO_DRUMS_KEY) toMute.push(id);
    }

    // Batch the mute updates and the reorder into a single observable
    // mutation so the mixer renders once at the end instead of once
    // per loaded track.
    runInAction(() => {
      for (const id of toMute) this.mutedAudioTracks.add(id);
      this.applyDebugBundleTrackOrder(loadedByKey);
    });
    return scoreLoaded;
  }

  /**
   * Re-order the mixer after a debug bundle is loaded so each per-pitch
   * audio track sits immediately above its instrument row, with any
   * unmatched audio (e.g. the `no_drums` backing) at the top.
   *
   * Layout (top → bottom):
   *
   *   audio: <unmatched-key>   ← e.g. no_drums, or audio for a pitch the
   *   ...                       loaded jot doesn't actually contain
   *   ┌ audio: <pitch-1>       ┐
   *   └ pitch: <pitch-1>       ┘ paired, share groupId `pair:<pitch>`
   *   ┌ audio: <pitch-2>       ┐
   *   └ pitch: <pitch-2>       ┘ paired, share groupId `pair:<pitch>`
   *   ...
   *
   * Each paired (audio, pitch) gets a fresh `groupId` so the mixer
   * draws them flush together with a small gap to the next pair —
   * KickAudio + KickInstrument visually distinct from SnareAudio +
   * SnareInstrument even though they're all in one flat list.
   *
   * Pitches in the jot that the bundle didn't provide audio for still
   * appear as their normal pitch row (no audio above them, no group).
   * The `syncTrackOrder` reaction won't reshuffle this — it only ever
   * drops stale entries and appends new ones, both of which are no-ops
   * after a fresh bundle load.
   */
  private applyDebugBundleTrackOrder(
    loadedByKey: ReadonlyMap<string, AudioTrackId>,
  ): void {
    const pitches = collectJotPitches(this.currentJot);
    const pitchesWithAudio = new Set(pitches.filter((p) => loadedByKey.has(p)));
    const next: TrackKey[] = [];

    // 1) Audio tracks that don't correspond to any pitch in the loaded
    //    jot (no_drums always; also any per-pitch stem the score didn't
    //    end up using) sit at the top, in the manifest's mapping order.
    //    These stay ungrouped — they're standalone backing tracks, not
    //    half of an audio↔instrument pair.
    for (const [key, id] of loadedByKey) {
      if (!pitchesWithAudio.has(key)) {
        next.push({ kind: 'audio', id });
      }
    }

    // 2) For each pitch in the jot, slot its audio (if any) directly
    //    above the instrument row. Paired entries share a fresh
    //    `groupId` so they render as a single visual cluster.
    for (const pitch of pitches) {
      const id = loadedByKey.get(pitch);
      if (id !== undefined) {
        const groupId = `pair:${pitch}`;
        next.push({ kind: 'audio', id, groupId });
        next.push({ kind: 'pitch', pitch, groupId });
      } else {
        next.push({ kind: 'pitch', pitch });
      }
    }

    this.trackOrder = next;
  }

  /** Toggle the {@link DebugPanel}'s open state without forgetting the bundle. */
  toggleDebugPanel(): void {
    this.debugPanelOpen = !this.debugPanelOpen;
  }

  /** Resize the {@link DebugPanel}. Clamped so it can't shrink past the
   * header or grow past the viewport (with headroom for the toolbar). */
  setDebugPanelHeight(px: number): void {
    const max = Math.max(120, window.innerHeight - 160);
    this.debugPanelHeight = Math.min(max, Math.max(80, px));
  }

  /** Resize the sticky gutter column. Clamped to a sensible range so a
   * runaway drag can't collapse the controls or push the bars row off
   * screen. */
  setGutterWidth(px: number): void {
    if (!Number.isFinite(px)) return;
    this.gutterWidth = Math.min(MAX_GUTTER_WIDTH, Math.max(MIN_GUTTER_WIDTH, px));
  }

  async playCurrent(): Promise<void> {
    const jot = this.currentJot;
    if (!jot) return;
    // Pass the laid-out RenderedJot (not its source) so the player's
    // timeline reads live bar widths — the playhead then tracks correctly
    // across zoom changes.
    await jotPlayer.play(jot);
  }

  stopPlayback(): void {
    jotPlayer.stop();
  }

  /** Current beat-grid offset (quarter-note beats) on the loaded jot. */
  get drumOffsetBeats(): number {
    return this.currentJot?.drumOffsetBeats ?? 0;
  }

  /**
   * Slide every drum note across the bar grid by `beats` quarter-note
   * beats to realign a consistently mis-detected groove (see
   * {@link RenderedJot.drumOffsetBeats}). Reflows the score reactively and
   * reschedules in-flight playback so the change is heard immediately.
   */
  setDrumOffset(beats: number): void {
    const jot = this.currentJot;
    if (!jot) return;
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
    const jot = this.currentJot;
    if (!jot) return;
    const timeline =
      jotPlayer.timeline.bars.length > 0 ? jotPlayer.timeline : buildTimeline(jot);
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
        await this.playCurrent();
        break;
      case 'playing':
        await jotPlayer.pause();
        break;
      case 'paused':
        await jotPlayer.resume();
        break;
    }
  }
}
