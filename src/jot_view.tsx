import classNames from 'classnames';
import { makeAutoObservable, reaction, runInAction } from 'mobx';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Instrument, Modifier, Sticking } from 'src/dsl';
import { ExampleJot } from 'src/fakes';
import { Point } from 'src/geom';
import {
  RenderedJot,
  ResolvedJot,
  ResolvedNote,
  StructuralBar,
  StructuralNote,
  StructuralPatternSpan,
  StructuralTupletSpan,
  ViewConfig,
  px,
} from 'src/jot';
import {
  DebugBundleManifest,
  loadDebugZip,
  NO_DRUMS_KEY,
  NoteProvenanceEntry,
  NoteProvenanceFile,
} from 'src/debug_zip';
import { fromMidi } from 'src/midi';
import { parse, ParseError } from 'src/parser';
import { loadParadbZip } from 'src/rlrr';
import {
  buildTimeline,
  computeWaveformPeaksForJot,
  isAudibleUnder,
  isAudioTrackAudibleUnder,
  jotPlayer,
  PlayerState,
  SampleLoadProgress,
  KitInfo,
  AudioTrackId,
  AudioTrack,
  timeToX,
  xToTime,
} from 'src/playback';
import { SelectionStore } from 'src/selection';
import { RefinementLog, stemUrl, titleFromFilename, transcriber } from 'src/transcriber';
import styles from './jot_view.module.css';

/**
 * Routes the active {@link SelectionStore} to deep score chrome (today:
 * `NoteView`) without threading props through `JotView → MixerView →
 * PitchRow → BarView`. `null` outside the view so a `NoteView` rendered
 * in isolation just no-ops the click-to-select interaction.
 */
const SelectionContext = React.createContext<SelectionStore | null>(null);

/**
 * Routes the loaded debug bundle's per-note provenance to two deep
 * consumers: `NoteView` (looks up its own entry via `byTick` to render
 * the `Debug details` collapsible in the selection label) and `PitchRow`
 * (reads `rejectedByPitch` + `leadBars` + `showFiltered` to render
 * filtered onsets as ghost overlays). `null` outside the View, or when
 * no bundle is loaded — both consumers no-op in that case.
 */
type NoteProvenanceContextValue = {
  /** Keyed by `${pitch}:${tick}` — exact-match lookup from NoteView. */
  byTick: Map<string, NoteProvenanceEntry>;
  /**
   * Per-pitch rejected onsets used by PitchRow to render the dashed
   * ghost overlays. Out-of-range entries are pre-filtered out (they
   * have no anchored bar to render against).
   */
  rejectedByPitch: Map<string, NoteProvenanceEntry[]>;
  /**
   * The `lead_bars` field from the provenance file. The MIDI lays
   * `lead_bars` empty bar-0-sized blocks before bar 0 to absorb the
   * audio lead-in, so a struct bar `b` maps to the rendered jot's
   * `bars[lead_bars + b]`.
   */
  leadBars: number;
  /** Toolbar checkbox state — true when the user opted into rendering
   * the rejected-onset overlays. NoteView's Debug details remain
   * available regardless (they are per-kept-note, not gated). */
  showFiltered: boolean;
};

const NoteProvenanceContext = React.createContext<NoteProvenanceContextValue | null>(null);

export type TranscribeStatus =
  | { phase: 'idle' }
  | { phase: 'uploading'; filename: string }
  | { phase: 'error'; message: string }
  | {
      phase: 'success';
      filename: string;
      tempo: number;
      hasTempoChanges: boolean;
      hasTimeSigChanges: boolean;
      barCount: number;
      refinement?: RefinementLog | null;
      debugDir?: string | null;
      /** Resolved URL of the debug bundle for this run (see
       * `TranscribeResponse.debug_zip_url`). The success pill renders a
       * download link to this; the user can then drop the zip back into
       * "Load debug bundle" to inspect logs in-app. */
      debugZipUrl?: string | null;
    };

export type TranscribeOptions = {
  refine: boolean;
  lint: boolean;
  bestOfK: number;
  debug: boolean;
};

/**
 * Pixels-per-bar at zoom = 1. Same numeric value as `ViewConfig.barWidth`'s
 * own default so existing layouts are unchanged for users who never touch
 * the slider.
 */
const BASE_BAR_WIDTH = 448;
const MIN_ZOOM = 0.3;
const MAX_ZOOM = 3.0;
// Row volume faders are pure attenuation (0 = silent, 1 = unscaled).
// The kit's overall loudness is handled by the drum master gain.
const VOLUME_STEP = 0.05;
function clampVolume(v: number): number {
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

function trackKeyEq(a: TrackKey, b: TrackKey): boolean {
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
 */
function collectJotPitches(jot: RenderedJot | undefined): string[] {
  if (!jot) return [];
  const out: string[] = [];
  for (const voice of jot.resolved.voices) {
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
  /** UI-controlled options for the next transcribe call. */
  transcribeOptions: TranscribeOptions = {
    refine: true,
    lint: true,
    bestOfK: 1,
    debug: false,
  };
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
    // transcribed lead-in (`globalMetadata.startOffset`). Tracking
    // `currentJot` (an observable reference) re-fires whenever a new jot
    // is loaded, resetting the offset to that recording's value; manual
    // nudges via the Offset control persist until the next load. We read
    // `globalMetadata` (the raw source) rather than `resolved` so seeding
    // doesn't force a layout pass.
    reaction(
      () => {
        const raw = this.currentJot?.globalMetadata.startOffset;
        return typeof raw === 'number' && raw > 0 ? raw : 0;
      },
      (offsetSec) => jotPlayer.setStartOffset(offsetSec),
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
  async loadAudioTrack(file: File): Promise<AudioTrackId | undefined> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      try {
        return await jotPlayer.loadAudioTrack(file);
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

  setRefine(enabled: boolean) {
    this.transcribeOptions.refine = enabled;
  }

  setLint(enabled: boolean) {
    this.transcribeOptions.lint = enabled;
  }

  setBestOfK(n: number) {
    this.transcribeOptions.bestOfK = Math.max(1, Math.min(5, n));
  }

  setDebug(enabled: boolean) {
    this.transcribeOptions.debug = enabled;
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
        refine: this.transcribeOptions.refine,
        lint: this.transcribeOptions.lint,
        bestOfK: this.transcribeOptions.bestOfK,
        debug: this.transcribeOptions.debug,
        signal: controller.signal,
      });
      const jot = parse(response.jot_dsl);
      // The transcriber service no longer injects a title (the regex
      // pass over the DSL lived in Python and was the only string-level
      // mutation we did to LLM output). Set it here from the upload
      // filename so the rendered jot shows a useful heading.
      const derivedTitle = titleFromFilename(file.name);
      if (derivedTitle) jot.title = derivedTitle;
      runInAction(() => {
        this.currentJot = new RenderedJot(jot, this.viewConfig);
        this.currentExampleId = undefined;
        this.clearNoteProvenance();
        jotPlayer.clearCue();
        this.transcribeStatus = {
          phase: 'success',
          filename: file.name,
          tempo: response.metadata.initial_tempo,
          hasTempoChanges: response.metadata.has_tempo_changes,
          hasTimeSigChanges: response.metadata.has_time_sig_changes,
          barCount: response.metadata.bars.length,
          refinement: response.refinement ?? null,
          debugDir: response.debug_dir ?? null,
          debugZipUrl: stemUrl(response.debug_zip_url ?? null),
        };
      });
    } catch (err) {
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
      } else {
        const message =
          err instanceof ParseError
            ? `Transcriber returned invalid DSL: ${err.message}`
            : err instanceof Error
              ? err.message
              : String(err);
        runInAction(() => {
          this.transcribeStatus = { phase: 'error', message };
        });
      }
    } finally {
      if (this.transcribeController === controller) {
        this.transcribeController = undefined;
      }
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

      runInAction(() => {
        this.clearAllAudioTracks();
        this.resetPitchMixer();
        this.lastDebugBundle = bundle.manifest;
        // Replace (or clear) the per-note debug provenance whenever a
        // new bundle loads. Filter-mode bundles carry one; DSL-mode
        // bundles don't, in which case the previous bundle's
        // provenance must not leak onto the new score.
        this.noteProvenance = bundle.noteProvenance ?? undefined;
        // Reset the visibility toggle so a freshly loaded bundle reads
        // as just "the score" — operator opts into the ghost overlays.
        this.showFilteredOnsets = false;
        this.debugPanelOpen = true;
      });

      // Prefer `final.jot` (DSL-mode bundle) over `prediction.mid` (filter-
      // mode bundle) — DSL is the richer artifact when both are present.
      // Filter mode emits no `final.jot`, so the MIDI fallback is the score
      // for that pathway.
      if (bundle.jotDsl.trim()) {
        try {
          const jot = parse(bundle.jotDsl);
          if (!jot.title) {
            const derivedTitle = titleFromFilename(file.name);
            if (derivedTitle) jot.title = derivedTitle;
          }
          runInAction(() => {
            this.currentJot = new RenderedJot(jot, this.viewConfig);
            this.currentExampleId = undefined;
            jotPlayer.clearCue();
            this.transcribeStatus = { phase: 'idle' };
          });
        } catch (err) {
          const message =
            err instanceof ParseError
              ? `Could not parse final.jot: ${err.message}`
              : err instanceof Error
                ? err.message
                : String(err);
          runInAction(() => {
            this.transcribeStatus = { phase: 'error', message };
          });
        }
      } else if (bundle.predictionMidi) {
        try {
          const jot = fromMidi(bundle.predictionMidi);
          if (!jot.title) {
            const derivedTitle = titleFromFilename(file.name);
            if (derivedTitle) jot.title = derivedTitle;
          }
          runInAction(() => {
            this.currentJot = new RenderedJot(jot, this.viewConfig);
            this.currentExampleId = undefined;
            jotPlayer.clearCue();
            this.transcribeStatus = { phase: 'idle' };
          });
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
          this.loadAudioTrack(track.file).then((id) => ({ key: track.key, id })),
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
    });
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

type CreateJotViewOptions = {
  examples?: readonly ExampleJot[];
};

type CreateJotViewResult = {
  store: JotViewStore;
  View: React.FC;
};

export function createJotView(options: CreateJotViewOptions = {}): CreateJotViewResult {
  const store = new JotViewStore();
  if (options.examples) store.setExamples(options.examples);
  const selection = new SelectionStore(store);

  // The marquee div is `position: absolute` inside `.jotContainer`
  // (the scroll surface, `position: relative`), so its `top`/`left`
  // need to be in that container's content coordinate space — viewport
  // `clientX`/`Y` would offset the rectangle by everything between the
  // viewport edge and the container's content origin (toolbar + the
  // Audio/Drums master rows + whatever's scrolled out of view above).
  const containerPoint = (e: React.MouseEvent<HTMLDivElement>): Point => {
    const el = e.currentTarget;
    const rect = el.getBoundingClientRect();
    return new Point(
      e.clientX - rect.left + el.scrollLeft,
      e.clientY - rect.top + el.scrollTop,
    );
  };
  const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    selection.beginSelection(containerPoint(e));
  };
  const onMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    selection.moveSelection(containerPoint(e));
  };

  const View: React.FC = observer(() => {
    const jot = store.currentJot;

    // Spacebar = play / pause / resume, from anywhere on the page. Skip
    // only when a text-entry control has focus (the user is typing) or
    // a SELECT is focused (let space/arrows drive the native picker).
    // A focused BUTTON deliberately falls through: preventDefault both
    // stops the browser's space-to-scroll and suppresses the button's
    // space-activation, so spacebar *always* toggles transport. A
    // focused range slider (e.g. Zoom) also falls through — space has
    // no native slider function, so swallowing it here would silently
    // break play/pause until the user clicked elsewhere.
    React.useEffect(() => {
      // INPUT types where space is meaningful text/native input and the
      // shortcut must yield. A range/checkbox/etc. input is not listed,
      // so spacebar still toggles transport while it has focus.
      const TEXT_ENTRY_INPUT_TYPES = new Set([
        'text',
        'search',
        'email',
        'url',
        'tel',
        'password',
        'number',
      ]);
      const onKeyDown = (e: KeyboardEvent) => {
        if (e.code !== 'Space' && e.key !== ' ') return;
        const el = e.target as HTMLElement | null;
        const tag = el?.tagName;
        const isTextEntryInput =
          tag === 'INPUT' &&
          TEXT_ENTRY_INPUT_TYPES.has((el as HTMLInputElement).type);
        if (
          isTextEntryInput ||
          tag === 'TEXTAREA' ||
          tag === 'SELECT' ||
          el?.isContentEditable
        ) {
          return;
        }
        e.preventDefault();
        void store.togglePlayPause();
      };
      window.addEventListener('keydown', onKeyDown);
      return () => window.removeEventListener('keydown', onKeyDown);
    }, []);

    const provenanceContextValue: NoteProvenanceContextValue | null = store.noteProvenance
      ? {
          byTick: store.noteProvenanceByTick,
          rejectedByPitch: store.filteredOnsetsByPitch,
          leadBars: store.noteProvenance.lead_bars ?? 0,
          showFiltered: store.showFilteredOnsets,
        }
      : null;

    return (
      <SelectionContext.Provider value={selection}>
      <NoteProvenanceContext.Provider value={provenanceContextValue}>
      <div className={styles.appContainer}>
        <Toolbar
          examples={store.examples}
          currentId={store.currentExampleId}
          onSelect={(id) => store.loadExample(id)}
          transcribeStatus={store.transcribeStatus}
          transcribeOptions={store.transcribeOptions}
          onTranscribe={(file) => store.transcribeAudio(file)}
          onLoadJot={(file) => store.loadJotFile(file)}
          onLoadMidi={(file) => store.loadMidiFile(file)}
          onLoadParadb={(file) => store.loadParadbMap(file)}
          onLoadDebugBundle={(file) => store.loadDebugBundleFile(file)}
          onLoadAudioTrack={(file) => store.loadAudioTrack(file)}
          onCancelTranscribe={() => store.cancelTranscribe()}
          onClearTranscribeStatus={() => store.clearTranscribeStatus()}
          onSetRefine={(v) => store.setRefine(v)}
          onSetLint={(v) => store.setLint(v)}
          onSetBestOfK={(n) => store.setBestOfK(n)}
          onSetDebug={(v) => store.setDebug(v)}
          zoom={store.zoom}
          onSetZoom={(z) => store.setZoom(z)}
          hasNoteProvenance={store.noteProvenance !== undefined}
          showFilteredOnsets={store.showFilteredOnsets}
          onSetShowFilteredOnsets={(v) => store.setShowFilteredOnsets(v)}
        />
        {jot ? (
          <JotView
            jot={jot}
            highlightedPattern={selection.selectedPattern}
            onPatternClick={(name) => selection.togglePattern(name)}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={selection.endSelection}
            onSeek={(x) => store.seekToX(x)}
            onZoomBy={(factor) => store.setZoom(store.zoom * factor)}
            trackOrder={store.trackOrder}
            onMoveTrack={(from, to) => store.moveTrack(from, to)}
            voiceControls={{
              mutedPitches: store.mutedPitches,
              soloedPitches: store.soloedPitches,
              isPitchAudible: (pitch) => store.isPitchAudible(pitch),
              volumeFor: (pitch) => store.pitchVolume(pitch),
              onSetVolume: (pitch, v) => store.setPitchVolume(pitch, v),
              onToggleMute: (pitch) => store.toggleMute(pitch),
              onToggleSolo: (pitch) => store.toggleSolo(pitch),
            }}
            audioTrackControls={{
              mutedAudioTracks: store.mutedAudioTracks,
              soloedAudioTracks: store.soloedAudioTracks,
              isAudioTrackAudible: (id) => store.isAudioTrackAudible(id),
              volumeFor: (id) => store.audioTrackVolume(id),
              onSetVolume: (id, v) => store.setAudioTrackVolume(id, v),
              onToggleMute: (id) => store.toggleAudioTrackMute(id),
              onToggleSolo: (id) => store.toggleAudioTrackSolo(id),
              onClear: (id) => store.clearAudioTrack(id),
            }}
          />
        ) : (
          <div className={styles.empty}>No jot loaded</div>
        )}
        <PlaybackBar store={store} />
        <DebugPanel store={store} />
        <LoadingOverlay store={store} />
      </div>
      </NoteProvenanceContext.Provider>
      </SelectionContext.Provider>
    );
  });

  return { store, View };
}

/**
 * Native `<select>` that releases focus once a value is committed.
 *
 * The global spacebar play/pause shortcut deliberately ignores
 * keystrokes while a SELECT is focused (so arrow/space can drive the
 * open list), which means a dropdown that keeps focus after the user
 * picks a value would silently swallow the shortcut until they click
 * elsewhere. Routing every dropdown through this wrapper makes the
 * correct behaviour the default — new dropdowns can't reintroduce the
 * regression. All native `<select>` props pass straight through.
 */
const Select = ({
  onChange,
  ...rest
}: React.ComponentPropsWithoutRef<'select'>) => (
  <select
    {...rest}
    onChange={(e) => {
      onChange?.(e);
      e.currentTarget.blur();
    }}
  />
);

/**
 * A toolbar button that toggles a floating panel of related controls,
 * used to collapse the formerly-flat header into grouped menus ("Load",
 * "Transcribe"). Closes on outside click or Escape. `children` is a
 * render prop receiving a `close` callback so menu items that complete
 * an action (open a file picker, fire a request) can dismiss the panel,
 * while sticky controls (option checkboxes) can leave it open.
 */
const DropdownButton = ({
  label,
  title,
  className,
  panelClassName,
  children,
}: {
  label: React.ReactNode;
  title?: string;
  className?: string;
  panelClassName?: string;
  children: (close: () => void) => React.ReactNode;
}) => {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div className={styles.dropdown} ref={ref}>
      <button
        type="button"
        className={className ?? styles.playButton}
        title={title}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        {label} <span aria-hidden="true">▾</span>
      </button>
      {open && (
        <div className={classNames(styles.dropdownPanel, panelClassName)} role="menu">
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
};

const Toolbar = observer(
  ({
    examples,
    currentId,
    onSelect,
    transcribeStatus,
    transcribeOptions,
    onTranscribe,
    onLoadJot,
    onLoadMidi,
    onLoadParadb,
    onLoadDebugBundle,
    onLoadAudioTrack,
    onCancelTranscribe,
    onClearTranscribeStatus,
    onSetRefine,
    onSetLint,
    onSetBestOfK,
    onSetDebug,
    zoom,
    onSetZoom,
    hasNoteProvenance,
    showFilteredOnsets,
    onSetShowFilteredOnsets,
  }: {
    examples: readonly ExampleJot[];
    currentId: string | undefined;
    onSelect: (id: string) => void;
    transcribeStatus: TranscribeStatus;
    transcribeOptions: TranscribeOptions;
    onTranscribe: (file: File) => void;
    onLoadJot: (file: File) => void;
    onLoadMidi: (file: File) => void;
    onLoadParadb: (file: File) => void;
    onLoadDebugBundle: (file: File) => void;
    onLoadAudioTrack: (file: File) => void;
    onCancelTranscribe: () => void;
    onClearTranscribeStatus: () => void;
    onSetRefine: (enabled: boolean) => void;
    onSetLint: (enabled: boolean) => void;
    onSetBestOfK: (n: number) => void;
    onSetDebug: (enabled: boolean) => void;
    zoom: number;
    onSetZoom: (z: number) => void;
    /** True iff a filter-mode debug bundle is loaded — gates the
     * `Show filtered` checkbox so it's only present when there's
     * actually filtered-onset data to render. */
    hasNoteProvenance: boolean;
    showFilteredOnsets: boolean;
    onSetShowFilteredOnsets: (show: boolean) => void;
  }) => {
    const fileInputRef = React.useRef<HTMLInputElement>(null);
    const jotInputRef = React.useRef<HTMLInputElement>(null);
    const midiInputRef = React.useRef<HTMLInputElement>(null);
    const paradbInputRef = React.useRef<HTMLInputElement>(null);
    const debugBundleInputRef = React.useRef<HTMLInputElement>(null);
    const audioTrackInputRef = React.useRef<HTMLInputElement>(null);
    const uploading = transcribeStatus.phase === 'uploading';

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onTranscribe(file);
      // Reset so picking the same file twice in a row still fires onChange.
      e.target.value = '';
    };

    const handleJotFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onLoadJot(file);
      e.target.value = '';
    };

    const handleMidiFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onLoadMidi(file);
      e.target.value = '';
    };

    const handleParadbChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onLoadParadb(file);
      e.target.value = '';
    };

    const handleDebugBundleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onLoadDebugBundle(file);
      e.target.value = '';
    };

    const handleAudioTrackChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      // Multiple-select: load every chosen file as its own track.
      for (const file of Array.from(e.target.files ?? [])) onLoadAudioTrack(file);
      e.target.value = '';
    };

    return (
      <div className={styles.toolbar}>
        {examples.length > 0 && (
          <>
            <label htmlFor="drumjot-example-select" className={styles.toolbarLabel}>
              Example
            </label>
            <Select
              id="drumjot-example-select"
              className={styles.exampleSelect}
              value={currentId ?? ''}
              onChange={(e) => onSelect(e.target.value)}
            >
              {currentId === undefined && (
                <option value="" disabled>
                  Select an example...
                </option>
              )}
              {examples.map((ex) => (
                <option key={ex.id} value={ex.id}>
                  {ex.label}
                </option>
              ))}
            </Select>
            <span className={styles.toolbarDivider} aria-hidden="true" />
          </>
        )}
        <DropdownButton label="Load" title="Load a score or audio tracks from disk">
          {(close) => (
            <>
              <button
                type="button"
                className={styles.dropdownItem}
                onClick={() => {
                  jotInputRef.current?.click();
                  close();
                }}
                title="Load a Drumjot DSL file (`.jot`) from disk and render it. Parser runs entirely client-side; no transcriber service required."
              >
                Load .jot file
              </button>
              <button
                type="button"
                className={styles.dropdownItem}
                onClick={() => {
                  midiInputRef.current?.click();
                  close();
                }}
                title="Load a Standard MIDI File (`.mid`) from disk. Drum-channel notes are quantized to a 16th grid and converted to a score. Runs entirely client-side; no transcriber service required."
              >
                Load midi
              </button>
              <button
                type="button"
                className={styles.dropdownItem}
                onClick={() => {
                  paradbInputRef.current?.click();
                  close();
                }}
                title="Load a ParaDB / Paradiddle map pack (`.zip`). The chart is converted to a score and its audio tracks are loaded automatically for play-along practice. Runs entirely client-side."
              >
                Load ParaDB map (.zip)
              </button>
              <button
                type="button"
                className={styles.dropdownItem}
                onClick={() => {
                  debugBundleInputRef.current?.click();
                  close();
                }}
                title="Load a transcriber debug bundle (`.zip`) — the same artifact `Transcribe audio` produces server-side. Restores the score, every per-stem audio track (MP3), and surfaces the captured logs + per-stage timings in the debug panel for inspection. Runs entirely client-side."
              >
                Load debug bundle (.zip)
              </button>
              <button
                type="button"
                className={styles.dropdownItem}
                onClick={() => {
                  audioTrackInputRef.current?.click();
                  close();
                }}
                title="Load one or more audio files (FLAC / WAV / MP3 / ...) as backing tracks. Each plays alongside the MIDI drums and shows a waveform aligned to the score; mute/solo/volume each from its track gutter. Select multiple files to load them all at once."
              >
                Load audio track(s)
              </button>
            </>
          )}
        </DropdownButton>
        <DropdownButton
          label={uploading ? 'Transcribing…' : 'Transcribe'}
          className={styles.transcribeButton}
          panelClassName={styles.dropdownPanelWide}
          title="Transcribe an audio file to a Jot, with refinement options"
        >
          {(close) => (
            <>
              <label
                className={styles.toolbarCheckbox}
                title="Run the deterministic Jot linter and ask the LLM to fix any instrument-tier (invalid modifier) or performance-tier (impossible sticking, too many hands, ...) issues it flags. Cheap relative to the F1 refinement; per-segment LLM calls scoped to the affected bars only."
              >
                <input
                  type="checkbox"
                  checked={transcribeOptions.lint}
                  disabled={uploading}
                  onChange={(e) => onSetLint(e.target.checked)}
                />
                Lint
              </label>
              <label
                className={styles.toolbarCheckbox}
                title="Run the multi-level convergence loop after the initial transcription. Adds ~30-60s but typically lifts accuracy by 5-10 F1 points."
              >
                <input
                  type="checkbox"
                  checked={transcribeOptions.refine}
                  disabled={uploading}
                  onChange={(e) => onSetRefine(e.target.checked)}
                />
                Refine accuracy
              </label>
              <label
                className={styles.toolbarCheckbox}
                title="Generate K candidate initial transcriptions at different temperatures and pick the highest-scoring one."
              >
                <span>Samples</span>
                <Select
                  className={styles.samplesSelect}
                  value={transcribeOptions.bestOfK}
                  disabled={uploading}
                  onChange={(e) => onSetBestOfK(Number(e.target.value))}
                >
                  <option value={1}>1</option>
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                </Select>
              </label>
              <label
                className={styles.toolbarCheckbox}
                title="Persist intermediate audio (drum stems, per-instrument stems), beat tracking, onsets, and LLM input/output to the transcriber's debug directory so you can listen back and inspect issues."
              >
                <input
                  type="checkbox"
                  checked={transcribeOptions.debug}
                  disabled={uploading}
                  onChange={(e) => onSetDebug(e.target.checked)}
                />
                Save debug files
              </label>
              <span className={styles.dropdownDivider} aria-hidden="true" />
              <button
                type="button"
                className={styles.transcribeButton}
                onClick={() => {
                  fileInputRef.current?.click();
                  close();
                }}
                disabled={uploading}
                title="Upload an audio file; the transcriber service will return a Jot. The Python backend decodes anything ffmpeg understands."
              >
                {uploading ? 'Transcribing…' : 'Transcribe audio…'}
              </button>
              {uploading && (
                <button
                  type="button"
                  className={styles.cancelButton}
                  onClick={() => {
                    onCancelTranscribe();
                    close();
                  }}
                  title="Abort the in-flight transcription request."
                >
                  Stop transcription
                </button>
              )}
            </>
          )}
        </DropdownButton>
        {/* Hidden file inputs live outside the dropdown panels so the
            refs stay mounted whether or not a menu is open. */}
        <input
          ref={jotInputRef}
          type="file"
          accept=".jot,.txt,text/plain"
          className={styles.hiddenInput}
          onChange={handleJotFileChange}
        />
        <input
          ref={midiInputRef}
          type="file"
          accept=".mid,.midi,audio/midi,audio/x-midi"
          className={styles.hiddenInput}
          onChange={handleMidiFileChange}
        />
        <input
          ref={paradbInputRef}
          type="file"
          accept=".zip,application/zip"
          className={styles.hiddenInput}
          onChange={handleParadbChange}
        />
        <input
          ref={debugBundleInputRef}
          type="file"
          accept=".zip,application/zip"
          className={styles.hiddenInput}
          onChange={handleDebugBundleChange}
        />
        <input
          ref={audioTrackInputRef}
          type="file"
          accept="audio/*,.flac"
          multiple
          className={styles.hiddenInput}
          onChange={handleAudioTrackChange}
        />
        <input
          ref={fileInputRef}
          type="file"
          // The Python backend decodes audio via librosa + soundfile +
          // ffmpeg, so anything ffmpeg understands works. Trust the
          // browser's `audio/*` filter; if the OS hasn't tagged a file
          // (rare on macOS / Linux, occasionally on Windows for .opus),
          // the user can switch the picker to "All files" and pick it
          // manually.
          accept="audio/*"
          className={styles.hiddenInput}
          onChange={handleFileChange}
        />
        <span className={styles.toolbarDivider} aria-hidden="true" />
        <label
          className={styles.toolbarCheckbox}
          title="Compress or expand the score horizontally. Has no effect on audio playback, only on how the notation is laid out."
        >
          <span>Zoom</span>
          <input
            type="range"
            min={0.3}
            max={3.0}
            step={0.05}
            value={zoom}
            onChange={(e) => onSetZoom(Number(e.target.value))}
            className={styles.zoomSlider}
          />
          <span className={styles.zoomValue}>{Math.round(zoom * 100)}%</span>
        </label>
        {hasNoteProvenance && (
          <label
            className={styles.toolbarCheckbox}
            title="Render the onsets the filter LLM rejected as dashed ghost overlays at their detected (bar, beat) position. Click one to see why it was filtered out. Only available when a filter-mode debug bundle is loaded."
          >
            <input
              type="checkbox"
              checked={showFilteredOnsets}
              onChange={(e) => onSetShowFilteredOnsets(e.target.checked)}
            />
            Show filtered
          </label>
        )}
        <DrumLoadingIndicator />
        <TranscribeStatusPill status={transcribeStatus} onClear={onClearTranscribeStatus} />
      </div>
    );
  }
);

const PLAYBACK_SPEEDS: readonly number[] = [0.25, 0.5, 0.75, 1.0, 1.25];

function samplePct(p: SampleLoadProgress): number {
  return Math.min(100, Math.round((p.loaded / p.total) * 100));
}

/** Bar fill width: real percentage when the server sent a size,
 * otherwise a fixed sliver so an unknown-total download still reads as
 * "working" rather than empty. */
function sampleProgressWidth(p: SampleLoadProgress | undefined): string {
  if (!p) return '8%';
  if (p.fromCache) return '100%';
  return p.total > 0 ? `${samplePct(p)}%` : '40%';
}

function sampleProgressLabel(p: SampleLoadProgress | undefined): string {
  if (!p) return 'Drums…';
  if (p.fromCache) return 'Drums (cached)';
  return p.total > 0 ? `Drums ${samplePct(p)}%` : 'Drums…';
}

/**
 * Top-right drum-sample download indicator. Reads `jotPlayer` directly so
 * the toolbar around it doesn't re-render on every progress tick.
 */
const DrumLoadingIndicator = observer(() => {
  if (jotPlayer.state !== 'loading') return null;
  const progress = jotPlayer.sampleLoadProgress;
  return (
    <span
      className={styles.sampleProgress}
      title="Downloading the GeneralUser GS SoundFont (~30 MB, one time). Cached in the browser after the first load — instant next time."
    >
      <span className={styles.sampleProgressTrack}>
        <span
          className={styles.sampleProgressFill}
          style={{ width: sampleProgressWidth(progress) }}
        />
      </span>
      <span>{sampleProgressLabel(progress)}</span>
    </span>
  );
});

/**
 * Full-app modal overlay shown while a file is loading (jot, midi, paradb
 * map, debug bundle, audio track). Lightly transparent so the user can
 * still see the underlying UI freeze in place, and `pointer-events: auto`
 * blocks all clicks underneath until the load resolves — protects against
 * double-clicks racing a long debug-bundle import. Driven by the store's
 * `withLoading` counter, so nested loads (debug bundle → many audio
 * tracks) read as one continuous spinner.
 */
const LoadingOverlay = observer(({ store }: { store: JotViewStore }) => {
  if (!store.isLoading) return null;
  return (
    <div
      className={styles.loadingOverlay}
      role="status"
      aria-live="polite"
      onMouseDown={(e) => e.stopPropagation()}
    >
      <div className={styles.loadingSpinner} aria-hidden="true" />
      {store.loadingLabel && (
        <div className={styles.loadingLabel}>{store.loadingLabel}</div>
      )}
    </div>
  );
});

/**
 * Numeric up/down for a playback offset. Editing commits live — every
 * keystroke and spinner click pushes the new value through `onChange`,
 * which the caller applies immediately (including mid-playback). A local
 * text buffer lets the user clear/retype the field freely; it re-syncs to
 * the incoming value whenever the input isn't focused (e.g. when loading a
 * new jot reseeds the offset).
 *
 * Used for two distinct offsets: the audio-track offset (seconds, the
 * recording's lead-in) and the drum beat-grid offset (beats, realigning a
 * mis-detected groove).
 */
const OffsetControl = ({
  label,
  unit,
  value,
  step,
  min,
  title,
  ariaLabel,
  onChange,
}: {
  label: string;
  unit: string;
  value: number;
  step: number;
  min?: number;
  title: string;
  ariaLabel: string;
  onChange: (v: number) => void;
}) => {
  const [text, setText] = React.useState(value.toFixed(2));
  const [editing, setEditing] = React.useState(false);
  React.useEffect(() => {
    if (!editing) setText(value.toFixed(2));
  }, [value, editing]);
  const commit = (raw: string) => {
    const n = parseFloat(raw);
    if (Number.isFinite(n)) onChange(n);
  };
  return (
    <label className={styles.toolbarCheckbox} title={title}>
      <span>{label}</span>
      <input
        type="number"
        className={styles.offsetInput}
        min={min}
        step={step}
        value={text}
        onFocus={() => setEditing(true)}
        onBlur={(e) => {
          setEditing(false);
          commit(e.target.value);
        }}
        onChange={(e) => {
          setText(e.target.value);
          commit(e.target.value);
        }}
        aria-label={ariaLabel}
      />
      <span>{unit}</span>
    </label>
  );
};

const PlaybackControls = observer(
  ({
    hasJot,
    playerState,
    playerError,
    playbackSpeed,
    drumKits,
    drumPreset,
    hasAudioTracks,
    audioOffsetSec,
    drumOffsetBeats,
    onTogglePlayPause,
    onStop,
    onSetPlaybackSpeed,
    onSetDrumPreset,
    onSetAudioOffset,
    onSetDrumOffset,
  }: {
    hasJot: boolean;
    playerState: PlayerState;
    playerError: string | undefined;
    playbackSpeed: number;
    drumKits: KitInfo[];
    drumPreset: number;
    hasAudioTracks: boolean;
    audioOffsetSec: number;
    drumOffsetBeats: number;
    onTogglePlayPause: () => void;
    onStop: () => void;
    onSetPlaybackSpeed: (speed: number) => void;
    onSetDrumPreset: (preset: number) => void;
    onSetAudioOffset: (sec: number) => void;
    onSetDrumOffset: (beats: number) => void;
  }) => {
    const loading = playerState === 'loading';
    const playing = playerState === 'playing';
    const paused = playerState === 'paused';
    // Playback is "active" (Stop is meaningful, playhead is on screen)
    // while either playing or paused.
    const active = playing || paused;
    const hasError = !!playerError && !loading && !active;
    // Icon-only, like a media player. Glyphs: ▶ play/resume, ⏸ pause,
    // ■ stop, ⚠ error, ⏳ loading.
    const transportIcon = loading ? '⏳' : playing ? '⏸' : hasError ? '⚠' : '▶';
    const transportAria = loading
      ? 'Loading'
      : playing
        ? 'Pause'
        : paused
          ? 'Resume'
          : 'Play';
    return (
      <>
        {/* Empty left cell balances the right-hand aux controls so the
            transport group stays optically centred in the bar. */}
        <div className={styles.transportSpacer} aria-hidden="true" />
        <div className={styles.transportCenter}>
          <button
            type="button"
            className={classNames(
              styles.transportButton,
              hasError && styles.transportButtonError
            )}
            onClick={onTogglePlayPause}
            disabled={!hasJot || loading}
            aria-label={transportAria}
            title={
              playerError
                ? `Playback error: ${playerError}`
                : playing
                  ? 'Pause playback (spacebar). The playhead and audio freeze in place; press again to resume.'
                  : paused
                    ? 'Resume playback (spacebar).'
                    : 'Play the current jot through an acoustic General MIDI drum kit (GeneralUser GS, spacebar also toggles play/pause). The first play downloads a ~30 MB SoundFont; it is then cached in the browser for instant loads on later sessions.'
            }
          >
            {transportIcon}
          </button>
          <button
            type="button"
            className={classNames(
              styles.transportButton,
              styles.transportButtonStop,
              styles.transportStop
            )}
            onClick={onStop}
            disabled={!active}
            aria-label="Stop"
            title={
              active
                ? 'Stop playback and reset to the start.'
                : 'Stop (available once playback has started).'
            }
          >
            ■
          </button>
        </div>
        <div className={styles.transportAux}>
          <MasterVolumes />
          {drumKits.length > 0 && (
            <label
              className={styles.toolbarCheckbox}
              title="Drum kit (a preset of the GeneralUser GS SoundFont). Switching is instant — the SoundFont is already downloaded; only the active samples change. Takes effect immediately, including mid-playback."
            >
              <span>Kit</span>
              <Select
                className={styles.samplesSelect}
                value={String(drumPreset)}
                onChange={(e) => onSetDrumPreset(Number(e.target.value))}
              >
                {drumKits.map((k) => (
                  <option key={k.preset} value={String(k.preset)}>
                    {k.name}
                  </option>
                ))}
              </Select>
            </label>
          )}
          <label
            className={styles.toolbarCheckbox}
            title="Tempo multiplier applied to playback. Slowing down spaces the drum hits further apart and time-stretches the audio tracks — pitch is preserved for both, so a half-speed practice pass stays in tune."
          >
            <span>Speed</span>
            <Select
              className={styles.samplesSelect}
              value={String(playbackSpeed)}
              onChange={(e) => onSetPlaybackSpeed(Number(e.target.value))}
            >
              {PLAYBACK_SPEEDS.map((s) => (
                <option key={s} value={String(s)}>
                  {s.toFixed(2)}×
                </option>
              ))}
            </Select>
          </label>
          {hasJot && (
            <OffsetControl
              label="Beat"
              unit="beats"
              value={drumOffsetBeats}
              step={0.25}
              title="Slide every drum note across the bars by this many beats to realign a consistently mis-detected groove (e.g. a kick transcribed 1.5 beats late in every bar). Positive = later, negative = earlier. Reflows the score and reschedules playback live. Notes pushed off either end of the score are dropped."
              ariaLabel="Drum beat offset in beats"
              onChange={onSetDrumOffset}
            />
          )}
          {hasAudioTracks && (
            <OffsetControl
              label="Audio"
              unit="s"
              value={audioOffsetSec}
              step={0.01}
              min={0}
              title="Drum-to-audio-track offset (the recording's lead-in), in seconds. Raising it slides the backing audio ahead of the drums; lowering it pulls them together. Takes effect instantly, including mid-playback, so you can nudge it until the drums lock to the track."
              ariaLabel="Drum to audio track offset in seconds"
              onChange={onSetAudioOffset}
            />
          )}
          {hasError && (
            <span
              className={classNames(styles.statusPill, styles.statusPillError)}
              title={playerError}
            >
              Playback: {truncate(playerError ?? '', 60)}
            </span>
          )}
        </div>
      </>
    );
  }
);

/**
 * Bottom transport bar. Pinned below the score so the (formerly
 * header-crowding) play / pause / stop / speed controls have their own
 * dedicated strip. `observer` + reading `jotPlayer` here keeps player
 * state re-renders scoped to this bar instead of bubbling up through
 * `View` and re-rendering the score on every transport change.
 */
const PlaybackBar = observer(({ store }: { store: JotViewStore }) => (
  <div className={styles.playbackBar}>
    <PlaybackControls
      hasJot={!!store.currentJot}
      playerState={jotPlayer.state}
      playerError={jotPlayer.errorMessage}
      playbackSpeed={jotPlayer.playbackSpeed}
      drumKits={jotPlayer.drumKits}
      drumPreset={jotPlayer.drumPreset}
      hasAudioTracks={jotPlayer.audioTracks.size > 0}
      audioOffsetSec={jotPlayer.startOffsetSec}
      drumOffsetBeats={store.drumOffsetBeats}
      onTogglePlayPause={() => store.togglePlayPause()}
      onStop={() => store.stopPlayback()}
      onSetPlaybackSpeed={(s) => jotPlayer.setPlaybackSpeed(s)}
      onSetDrumPreset={(p) => jotPlayer.setDrumPreset(p)}
      onSetAudioOffset={(sec) => jotPlayer.setStartOffset(sec)}
      onSetDrumOffset={(beats) => store.setDrumOffset(beats)}
    />
  </div>
));

/**
 * Bottom-pinned drawer that surfaces a loaded transcriber debug bundle.
 *
 * Renders nothing until {@link JotViewStore.loadDebugBundleFile} has
 * populated `lastDebugBundle`. When open, shows the per-stage timings on
 * the left and the captured log stream on the right; collapses to a thin
 * toggle bar so the user can hide it without forgetting the bundle.
 *
 * No interaction beyond expand/collapse — the underlying score + audio
 * tracks are operated through the existing toolbar / gutter controls
 * exactly as if they had been loaded by hand.
 */
const DebugPanel = observer(({ store }: { store: JotViewStore }) => {
  const bundle = store.lastDebugBundle;
  if (!bundle) return null;
  if (!store.debugPanelOpen) {
    return (
      <div
        className={styles.debugPanelToggleBar}
        role="button"
        onClick={() => store.toggleDebugPanel()}
        title="Re-open the debug panel."
      >
        ▴ Debug bundle loaded — show panel
      </div>
    );
  }
  const stages = bundle.stage_timings ?? [];
  const logs = bundle.logs ?? [];
  const totalElapsed = bundle.elapsed_seconds;
  const startedAt = bundle.started_at;
  const onResizePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    const startY = e.clientY;
    const startHeight = store.debugPanelHeight;
    const target = e.currentTarget;
    target.setPointerCapture(e.pointerId);
    const onMove = (ev: PointerEvent) => {
      // Drag up = grow the panel (top edge moves up).
      store.setDebugPanelHeight(startHeight + (startY - ev.clientY));
    };
    const onUp = (ev: PointerEvent) => {
      target.releasePointerCapture(ev.pointerId);
      target.removeEventListener('pointermove', onMove);
      target.removeEventListener('pointerup', onUp);
      target.removeEventListener('pointercancel', onUp);
    };
    target.addEventListener('pointermove', onMove);
    target.addEventListener('pointerup', onUp);
    target.addEventListener('pointercancel', onUp);
  };
  return (
    <div className={styles.debugPanel} style={{ height: store.debugPanelHeight }}>
      <div
        className={styles.debugPanelResizeHandle}
        onPointerDown={onResizePointerDown}
        title="Drag to resize the debug panel."
      />
      <div
        className={styles.debugPanelHeader}
        onClick={() => store.toggleDebugPanel()}
      >
        <span className={styles.debugPanelTitle}>Debug bundle</span>
        <span className={styles.debugPanelStats}>
          {bundle.filename ? `${bundle.filename} · ` : ''}
          {stages.length} stage{stages.length === 1 ? '' : 's'} · {logs.length} log
          line{logs.length === 1 ? '' : 's'}
          {typeof totalElapsed === 'number' ? ` · ${totalElapsed.toFixed(2)}s total` : ''}
          {startedAt ? ` · ${startedAt}` : ''}
        </span>
      </div>
      <div className={styles.debugPanelBody}>
        <div className={styles.debugPanelColumn}>
          <h4>Stage timings</h4>
          {stages.length === 0 ? (
            <p style={{ color: '#888', fontSize: 11 }}>No stage timings recorded.</p>
          ) : (
            <ul className={styles.debugStageList}>
              {stages.map((s, i) => (
                <li key={i} className={styles.debugStageRow}>
                  <span className={styles.debugStageName}>{s.stage}</span>
                  <span className={styles.debugStageElapsed}>
                    {s.elapsed_seconds.toFixed(2)}s
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className={styles.debugPanelColumn}>
          <h4>Logs</h4>
          {logs.length === 0 ? (
            <p style={{ color: '#888', fontSize: 11 }}>No logs captured.</p>
          ) : (
            <ul className={styles.debugLogList}>
              {logs.map((entry, i) => (
                <li key={i} className={styles.debugLogRow}>
                  <span className={styles.debugLogTimestamp}>
                    +{entry.elapsed_seconds.toFixed(2)}s
                  </span>
                  <span
                    className={classNames(
                      styles.debugLogLevel,
                      entry.level === 'WARNING' && styles.debugLogLevelWARNING,
                      entry.level === 'ERROR' && styles.debugLogLevelERROR
                    )}
                  >
                    {entry.level}
                  </span>
                  <span className={styles.debugLogLogger}>{entry.logger}</span>
                  <span className={styles.debugLogMessage}>{entry.message}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
});

const Playhead = observer(
  ({
    showLabel = false,
    onSeek,
  }: {
    showLabel?: boolean;
    onSeek: (x: number) => void;
  }) => {
    const timeline = jotPlayer.timeline;
    const active =
      jotPlayer.state === 'playing' ||
      jotPlayer.state === 'paused' ||
      // Idle but the user clicked to position the playhead before
      // pressing Play — show it parked at the cued spot.
      jotPlayer.cued;
    if (!active || timeline.bars.length === 0) return null;
    const x = timeToX(timeline, jotPlayer.currentTime);

    // Drag-to-scrub on the line itself or its label. stopPropagation
    // blocks the page-level marquee start; data-noseek prevents the
    // bars-row onClick from firing on mouseup of the drag.
    const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();
      const parent = e.currentTarget.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      onSeek(e.clientX - rect.left);
      const onMove = (ev: MouseEvent) => {
        onSeek(ev.clientX - rect.left);
      };
      const onUp = () => {
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
      };
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    };

    return (
      <div
        className={styles.playhead}
        style={{ left: x }}
        onMouseDown={onMouseDown}
        data-noseek
      >
        {showLabel && (
          <div className={styles.playheadLabel}>
            {formatPlayheadTime(jotPlayer.currentTime)}
          </div>
        )}
      </div>
    );
  }
);

function formatPlayheadTime(seconds: number): string {
  const negative = seconds < 0;
  const abs = Math.abs(seconds);
  const totalSec = Math.floor(abs);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  const cs = Math.floor((abs - totalSec) * 100);
  return `${negative ? '-' : ''}${min}:${String(sec).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;
}

const TranscribeStatusPill = observer(
  ({ status, onClear }: { status: TranscribeStatus; onClear: () => void }) => {
    if (status.phase === 'idle') return null;
    if (status.phase === 'uploading') {
      return (
        <span className={classNames(styles.statusPill, styles.statusPillBusy)}>
          Transcribing {status.filename}...
        </span>
      );
    }
    if (status.phase === 'error') {
      return (
        <span
          className={classNames(styles.statusPill, styles.statusPillError)}
          onClick={onClear}
          role="button"
          title={status.message}
        >
          Error: {truncate(status.message, 60)} (click to dismiss)
        </span>
      );
    }
    const refinement = status.refinement;
    let detail = `@ ${status.tempo.toFixed(0)} bpm, ${status.barCount} bars`;
    if (status.hasTempoChanges) detail += ', tempo changes';
    if (status.hasTimeSigChanges) detail += ', time-sig changes';
    if (refinement) {
      const accepted = refinement.iterations.filter((i) => i.accepted).length;
      const delta = refinement.final_score - refinement.initial_score;
      const sign = delta >= 0 ? '+' : '';
      detail += `, F1 ${refinement.initial_score.toFixed(2)} \u2192 ${refinement.final_score.toFixed(2)} (${sign}${delta.toFixed(2)}, ${accepted} revisions)`;
    }
    if (status.debugDir) {
      detail += `, debug @ ${status.debugDir}`;
    }
    const titleLines: string[] = [];
    if (refinement) {
      titleLines.push(
        `Refined ${refinement.iterations.length} iterations in ${refinement.elapsed_seconds.toFixed(1)}s.`,
      );
    }
    if (status.debugDir) {
      titleLines.push(
        `Debug artifacts saved to ${status.debugDir} (under ./debug/ on the host with the default docker-compose mount).`,
      );
    }
    return (
      <span
        className={classNames(styles.statusPill, styles.statusPillSuccess)}
        title={titleLines.length > 0 ? titleLines.join('\n') : undefined}
      >
        <span onClick={onClear} role="button">
          Loaded {status.filename} {detail} (click to dismiss)
        </span>
        {status.debugZipUrl && (
          <>
            {' '}
            <a
              href={status.debugZipUrl}
              download
              data-noseek="true"
              onClick={(e) => e.stopPropagation()}
              title="Download the debug bundle (.zip) for this run — score + per-stem MP3s + JSON manifest with stage timings + the full log stream. Drop the file into `Load > Load debug bundle` to inspect it back in this UI."
            >
              [debug.zip]
            </a>
          </>
        )}
      </span>
    );
  }
);

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}\u2026`;
}

type VoiceControls = {
  mutedPitches: ReadonlySet<string>;
  soloedPitches: ReadonlySet<string>;
  /** True if the row would currently make sound; false = muted via M or solo exclusion. */
  isPitchAudible: (pitch: string) => boolean;
  /** Current row fader value, 0..1 (1 = full). */
  volumeFor: (pitch: string) => number;
  onSetVolume: (pitch: string, v: number) => void;
  onToggleMute: (pitch: string) => void;
  onToggleSolo: (pitch: string) => void;
};

type AudioTrackControls = {
  mutedAudioTracks: ReadonlySet<AudioTrackId>;
  soloedAudioTracks: ReadonlySet<AudioTrackId>;
  isAudioTrackAudible: (id: AudioTrackId) => boolean;
  volumeFor: (id: AudioTrackId) => number;
  onSetVolume: (id: AudioTrackId, v: number) => void;
  onToggleMute: (id: AudioTrackId) => void;
  onToggleSolo: (id: AudioTrackId) => void;
  /** Drop a loaded audio track (button in the gutter clears the slot). */
  onClear: (id: AudioTrackId) => void;
};

type JotViewProps = {
  jot: RenderedJot;
  highlightedPattern: string | undefined;
  onPatternClick: (name: string) => void;
  onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseMove: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseUp: () => void;
  /** Click-to-seek with a bars-row-local pixel x. */
  onSeek: (x: number) => void;
  /**
   * Multiply the current score zoom by `factor` (Cmd/Ctrl + wheel).
   * The store clamps to the slider's range.
   */
  onZoomBy: (factor: number) => void;
  /**
   * User-customizable mixer ordering — drum-instrument rows and audio
   * tracks freely interleaved. Drives both row order and which
   * drum-pitch row hosts the pattern/tuplet bracket overlay (the
   * topmost drum row in this list).
   */
  trackOrder: readonly TrackKey[];
  /** Move the row at `from` to position `to` (drag-and-drop / Alt+arrow). */
  onMoveTrack: (from: number, to: number) => void;
  voiceControls: VoiceControls;
  audioTrackControls: AudioTrackControls;
};

const JotView = observer((props: JotViewProps) => {
  const {
    jot,
    highlightedPattern,
    onPatternClick,
    onMouseDown,
    onMouseMove,
    onMouseUp,
    onSeek,
    onZoomBy,
    trackOrder,
    onMoveTrack,
    voiceControls,
    audioTrackControls,
  } = props;
  // Intentionally NOT reading `jot.resolved` here — every observable
  // touched in this body triggers a JotView re-render on zoom, and the
  // title / subtitle / Legend / mixer subtree all derive from zoom-
  // invariant data via `jot.structure` / `jot.title` /
  // `jot.globalMetadata`. JotView itself is then stable across zoom
  // (ScoreZoomVar updates the one CSS variable that propagates the
  // new scale to every descendant via calc()).
  const config = jot.config;
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Wheel zooms the score (mirrors the Zoom slider) — no modifier
  // required, and Ctrl/Cmd + wheel still works (also covers the macOS
  // trackpad pinch gesture, which Chrome/Safari deliver as a synthetic
  // Ctrl + wheel). The listener is registered natively with
  // `{ passive: false }` because React's synthetic `onWheel` is passive
  // — `preventDefault` there is a no-op, and we must cancel both the
  // native page scroll and the browser's own page zoom on Ctrl/Cmd +
  // wheel. Wheel events are coalesced per animation frame: a 120 Hz
  // trackpad fires ~8 events per frame, but only the final composite
  // zoom is visible, so summing deltas and applying once skips ~7×
  // wasted layout/render passes per frame.
  const onZoomByRef = React.useRef(onZoomBy);
  onZoomByRef.current = onZoomBy;
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    let pendingDelta = 0;
    let rafId = 0;
    const flush = () => {
      rafId = 0;
      const delta = pendingDelta;
      pendingDelta = 0;
      if (delta === 0) return;
      onZoomByRef.current(Math.exp(-delta * 0.0015));
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      // deltaMode 1 = lines (typically a notched mouse wheel); scale
      // it up so a single notch zooms a comparable amount to a
      // pixel-mode trackpad swipe. Scrolling up (deltaY < 0) zooms in.
      const unit = e.deltaMode === 1 ? 16 : 1;
      pendingDelta += e.deltaY * unit;
      if (rafId === 0) rafId = requestAnimationFrame(flush);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      el.removeEventListener('wheel', onWheel);
      if (rafId !== 0) cancelAnimationFrame(rafId);
    };
  }, []);

  // Middle-mouse + drag pans the scroller in both axes. The mousedown
  // listener is on the container so preventDefault can suppress the
  // Windows/Linux autoscroll cursor (and X11 middle-click paste); the
  // mousemove/up listeners go on window so a drag that wanders out of
  // the container still tracks and releases cleanly.
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    let panning = false;
    let lastX = 0;
    let lastY = 0;
    let prevCursor = '';
    const onMouseDown = (e: MouseEvent) => {
      if (e.button !== 1) return;
      e.preventDefault();
      panning = true;
      lastX = e.clientX;
      lastY = e.clientY;
      prevCursor = el.style.cursor;
      el.style.cursor = 'grabbing';
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!panning) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      el.scrollLeft -= dx;
      el.scrollTop -= dy;
    };
    const stop = () => {
      if (!panning) return;
      panning = false;
      el.style.cursor = prevCursor;
    };
    el.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', stop);
    window.addEventListener('blur', stop);
    return () => {
      el.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', stop);
      window.removeEventListener('blur', stop);
    };
  }, []);

  // `--note-pad-px` is the engraving inset every note's CSS `left`
  // calc() reads; it never changes at runtime. `--px-per-beat` is the
  // ONE value the zoom slider mutates — both live on the same root so
  // every descendant calc() chain reads from a single ancestor. The
  // pad var goes in inline style (set once). `--px-per-beat` is
  // updated by `ScoreZoomVar`, a side-effect-only observer that
  // writes via `setProperty` on `containerRef.current` so a zoom tick
  // doesn't re-render JotView — only mutates one DOM attribute.
  const containerStyle = {
    ['--note-pad-px' as string]: `${config.barNotePaddingLeft}px`,
  } as React.CSSProperties;
  return (
    <div
      ref={containerRef}
      className={styles.jotContainer}
      style={containerStyle}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
    >
      <ScoreZoomVar jot={jot} containerRef={containerRef} />
      <h2 className={styles.title}>{jot.title || 'Untitled jot'}</h2>
      <p className={styles.subtitle}>{formatSubtitle(jot)}</p>
      <Legend jot={jot} />
      <TimelineHeader jot={jot} onSeek={onSeek} />
      <MixerView
        jot={jot}
        config={config}
        trackOrder={trackOrder}
        highlightedPattern={highlightedPattern}
        onPatternClick={onPatternClick}
        onSeek={onSeek}
        onMoveTrack={onMoveTrack}
        voiceControls={voiceControls}
        audioTrackControls={audioTrackControls}
      />
      <PlayheadAutoScroller containerRef={containerRef} />
      <MarqueeOverlay />
    </div>
  );
});

/**
 * Side-effect-only observer that writes `--px-per-beat` onto the score
 * container whenever the zoom-derived pixel-per-beat changes. Isolated
 * so reading `jot.pxPerBeat` (a zoom-dependent observable) doesn't
 * re-render JotView — the variable update happens via DOM
 * `setProperty` on the ref instead, then CSS `calc()` propagates the
 * new value to every bar / note / bracket without React touching the
 * subtree.
 */
const ScoreZoomVar = observer(
  ({
    jot,
    containerRef,
  }: {
    jot: RenderedJot;
    containerRef: React.RefObject<HTMLDivElement>;
  }) => {
    const pxPerBeat = jot.pxPerBeat;
    React.useEffect(() => {
      const el = containerRef.current;
      if (!el) return;
      el.style.setProperty('--px-per-beat', `${pxPerBeat}px`);
    }, [pxPerBeat, containerRef]);
    return null;
  }
);

/**
 * Isolated observer for the in-flight marquee rectangle so a mousemove
 * (which fires many times per second and mutates `selection.marquee`)
 * only re-renders this 4-style div instead of the whole JotView tree —
 * `JotView`/`MixerView`/per-row waveforms etc. are expensive enough that
 * reading `marquee` in any of their ancestors made the drag visibly laggy.
 */
const MarqueeOverlay = observer(() => {
  const selection = React.useContext(SelectionContext);
  const marquee = selection?.marquee;
  if (!marquee) return null;
  return (
    <div
      className={styles.marquee}
      style={{
        top: marquee.y,
        left: marquee.x,
        width: marquee.width,
        height: marquee.height,
      }}
    />
  );
});

/**
 * Sticky-gutter header above the audio tracks / score that labels each
 * bar boundary with its 1-based bar number and the playback time at that
 * boundary (mm:ss). Tick marks sit on the same `bar.x` line as the
 * score's barlines below so the header reads as a ruler over the
 * timeline. Click-to-seek mirrors the score and audio-track rows.
 *
 * Per-bar timings come from the live playback timeline whenever it
 * matches the current jot (so tempo overrides and the lead-in offset
 * stay in sync with the playhead); otherwise we build a one-shot
 * timeline so the header still labels everything correctly before the
 * user hits Play.
 */
const TimelineHeader = observer(
  ({ jot, onSeek }: { jot: RenderedJot; onSeek: (x: number) => void }) => {
    const voice = jot.resolved.voices[0];
    if (!voice || voice.bars.length === 0) return null;

    const liveTimeline = jotPlayer.timeline;
    const timeline =
      liveTimeline.bars.length > 0 && liveTimeline.rendered === jot
        ? liveTimeline
        : buildTimeline(jot);

    return (
      <div className={styles.timelineHeader}>
        <div className={styles.timelineHeaderGutter}>
          <span className={styles.timelineHeaderLabel}>Bar / Time</span>
        </div>
        <div
          className={styles.timelineHeaderBarsRow}
          style={{ width: voice.width }}
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          {voice.bars.map((bar, i) => {
            const timing = timeline.bars[i];
            const timeSec = timing?.startSec ?? 0;
            return (
              <div
                key={i}
                className={styles.timelineHeaderTick}
                style={{ left: bar.x }}
              >
                <span className={styles.timelineHeaderBar}>{bar.index}</span>
                <span className={styles.timelineHeaderTime}>{formatTime(timeSec)}</span>
              </div>
            );
          })}
          <Playhead showLabel onSeek={onSeek} />
        </div>
      </div>
    );
  }
);

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const min = Math.floor(total / 60);
  const sec = total % 60;
  return `${min}:${String(sec).padStart(2, '0')}`;
}

/**
 * Drag-source identifier carried on the DataTransfer of a mixer-row
 * drag. A custom MIME type lets us reject foreign drops (files,
 * external pages) so the gutter never tries to swallow them.
 */
const MIXER_DRAG_MIME = 'application/x-drumjot-mixer-row';

/**
 * The unified mixer that replaced the old separate "audio tracks" and
 * "voice staves" sections. Renders the two section masters at the top,
 * then one row per entry in `trackOrder` — an audio track or a single
 * drum-instrument pitch, freely interleavable. Drag-and-drop on each
 * row's gutter handle rewrites the order via {@link
 * JotViewStore.moveTrack}; the topmost drum-pitch row hosts the
 * pattern/tuplet bracket overlay so they read as a single piece of
 * score chrome regardless of where the user has moved the rows.
 */
const MixerView = observer(
  ({
    jot,
    config,
    trackOrder,
    highlightedPattern,
    onPatternClick,
    onSeek,
    onMoveTrack,
    voiceControls,
    audioTrackControls,
  }: {
    jot: RenderedJot;
    config: ViewConfig;
    trackOrder: readonly TrackKey[];
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    onSeek: (x: number) => void;
    onMoveTrack: (from: number, to: number) => void;
    voiceControls: VoiceControls;
    audioTrackControls: AudioTrackControls;
  }) => {
    // The drop indicator is rendered above the row at `dropTargetIdx`
    // when it lies in [0, length]; `length` is the "after the last row"
    // slot. `dragFromIdx` short-circuits a hover-over-self update so the
    // indicator doesn't flash on the row the user picked up.
    const [dragFromIdx, setDragFromIdx] = React.useState<number | undefined>(undefined);
    const [dropTargetIdx, setDropTargetIdx] = React.useState<number | undefined>(undefined);
    const resetDrag = () => {
      setDragFromIdx(undefined);
      setDropTargetIdx(undefined);
    };

    // The topmost drum-pitch row in the user's mixer order hosts the
    // pattern/tuplet bracket overlay. Brackets describe score structure
    // (not a specific instrument), so anchoring them to whichever drum
    // row currently sits at the top of the drum block keeps the overlay
    // visible no matter how the user rearranges the mixer.
    const firstPitchIdx = trackOrder.findIndex((k) => k.kind === 'pitch');

    return (
      <div className={styles.mixer}>
        <GutterMasterRow
          label="Audio"
          title="Master volume for all loaded audio (backing) tracks together. Multiplies on top of each track's own fader; takes effect instantly, including mid-playback."
          value={jotPlayer.audioTrackMasterVolume}
          onChange={(v) => jotPlayer.setAudioTrackMasterVolume(v)}
          testId="audio-track-master"
        />
        <GutterMasterRow
          label="Drums"
          title="Master volume for all drum/instrument rows together. Multiplies on top of each row's own fader; takes effect instantly, including mid-playback."
          value={jotPlayer.drumMasterVolume}
          onChange={(v) => jotPlayer.setDrumMasterVolume(v)}
          testId="drum-master"
        />
        {trackOrder.map((key, idx) => {
          // Reuse a stable React key per row so dragging doesn't tear
          // down + remount expensive children (the AudioTrackWaveformCanvas
          // would otherwise re-decode peaks on every reorder).
          const reactKey = key.kind === 'audio' ? `audio:${key.id}` : `pitch:${key.pitch}`;
          // A row begins a new "group" — and so renders with a small
          // top gap — whenever its `groupId` differs from the previous
          // row's. Solo (groupId undefined) rows are each their own
          // group. The first row never gets a gap (nothing above it).
          const prevGroupId = idx > 0 ? trackOrder[idx - 1].groupId : undefined;
          const groupStart = idx > 0 && key.groupId !== prevGroupId;
          const rowProps = {
            idx,
            dragFromIdx,
            dropTargetIdx,
            onDragStartIdx: setDragFromIdx,
            onDropTargetIdx: setDropTargetIdx,
            onMoveTrack,
            onResetDrag: resetDrag,
            mixerLength: trackOrder.length,
            groupStart,
          };
          if (key.kind === 'audio') {
            const track = jotPlayer.audioTracks.get(key.id);
            // The reaction in JotViewStore drops dead audio ids on the
            // same MobX tick, so this gap is one-frame at most. Render
            // nothing rather than crash if the maps race.
            if (!track) return null;
            return (
              <AudioTrackRow
                key={reactKey}
                id={key.id}
                track={track}
                jot={jot}
                controls={audioTrackControls}
                onSeek={onSeek}
                {...rowProps}
              />
            );
          }
          return (
            <PitchRow
              key={reactKey}
              pitch={key.pitch}
              jot={jot}
              config={config}
              showBrackets={idx === firstPitchIdx}
              highlightedPattern={highlightedPattern}
              onPatternClick={onPatternClick}
              onSeek={onSeek}
              voiceControls={voiceControls}
              {...rowProps}
            />
          );
        })}
        {/* "Drop at the very end" zone — without this the user can't
            move a row past the last existing row because the indicator
            target would clamp to its own bottom edge. Kept thin so it
            barely affects layout when no drag is in flight. */}
        <MixerEndDropZone
          idx={trackOrder.length}
          dragFromIdx={dragFromIdx}
          dropTargetIdx={dropTargetIdx}
          onDropTargetIdx={setDropTargetIdx}
          onMoveTrack={onMoveTrack}
          onResetDrag={resetDrag}
        />
      </div>
    );
  }
);

/**
 * Shared drag-target behaviour for the row gutter: a drag-over either
 * marks "drop above this row" (top half) or "drop below this row"
 * (bottom half), `onDrop` commits the move. Returns the props/style
 * fragments the row should spread onto its wrapper + a boolean for
 * whether the drop indicator should render above this row.
 */
function useMixerRowDropTarget({
  idx,
  dragFromIdx,
  dropTargetIdx,
  onDropTargetIdx,
  onMoveTrack,
  onResetDrag,
}: {
  idx: number;
  dragFromIdx: number | undefined;
  dropTargetIdx: number | undefined;
  onDropTargetIdx: (i: number | undefined) => void;
  onMoveTrack: (from: number, to: number) => void;
  onResetDrag: () => void;
}) {
  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    if (dragFromIdx === undefined) return;
    if (!e.dataTransfer.types.includes(MIXER_DRAG_MIME)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    const rect = e.currentTarget.getBoundingClientRect();
    const isTopHalf = e.clientY < rect.top + rect.height / 2;
    const target = isTopHalf ? idx : idx + 1;
    if (target !== dropTargetIdx) onDropTargetIdx(target);
  };
  const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    // Don't clear when the pointer just crossed into a child element;
    // only when it actually leaves the row bounds (relatedTarget
    // outside the gutter element).
    const next = e.relatedTarget as Node | null;
    if (next && e.currentTarget.contains(next)) return;
    if (dropTargetIdx === idx || dropTargetIdx === idx + 1) onDropTargetIdx(undefined);
  };
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    if (dragFromIdx === undefined) return;
    const data = e.dataTransfer.getData(MIXER_DRAG_MIME);
    if (!data) return;
    e.preventDefault();
    const from = parseInt(data, 10);
    if (Number.isFinite(from) && dropTargetIdx !== undefined) {
      onMoveTrack(from, dropTargetIdx);
    }
    onResetDrag();
  };
  const isDropIndicatorAbove = dropTargetIdx === idx && dragFromIdx !== undefined;
  const isDropIndicatorBelow = dropTargetIdx === idx + 1 && dragFromIdx !== undefined;
  return { onDragOver, onDragLeave, onDrop, isDropIndicatorAbove, isDropIndicatorBelow };
}

/**
 * A small "drop after the last row" zone. The per-row drop logic
 * already covers "before me" and "after me", but it bottoms out at the
 * last row's "after" position; this acts as the explicit final slot so
 * the indicator renders cleanly between the last row and the bottom of
 * the mixer.
 */
const MixerEndDropZone = ({
  idx,
  dragFromIdx,
  dropTargetIdx,
  onDropTargetIdx,
  onMoveTrack,
  onResetDrag,
}: {
  idx: number;
  dragFromIdx: number | undefined;
  dropTargetIdx: number | undefined;
  onDropTargetIdx: (i: number | undefined) => void;
  onMoveTrack: (from: number, to: number) => void;
  onResetDrag: () => void;
}) => {
  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    if (dragFromIdx === undefined) return;
    if (!e.dataTransfer.types.includes(MIXER_DRAG_MIME)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (dropTargetIdx !== idx) onDropTargetIdx(idx);
  };
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    if (dragFromIdx === undefined) return;
    const data = e.dataTransfer.getData(MIXER_DRAG_MIME);
    if (!data) return;
    e.preventDefault();
    const from = parseInt(data, 10);
    if (Number.isFinite(from)) onMoveTrack(from, idx);
    onResetDrag();
  };
  const showIndicator = dropTargetIdx === idx && dragFromIdx !== undefined;
  return (
    <div
      className={classNames(styles.mixerEndDrop, showIndicator && styles.mixerDropIndicator)}
      onDragOver={onDragOver}
      onDrop={onDrop}
    />
  );
};

/**
 * Drag handle (≡) parked on the leftmost edge of every mixer row's
 * gutter. Only this element is `draggable`, so the user can still click
 * mute/solo, drag the volume slider, etc. without accidentally lifting
 * the whole row.
 */
const MixerDragHandle = ({
  idx,
  onDragStartIdx,
  onResetDrag,
  ariaLabel,
}: {
  idx: number;
  onDragStartIdx: (i: number) => void;
  onResetDrag: () => void;
  ariaLabel: string;
}) => {
  return (
    <div
      className={styles.mixerDragHandle}
      draggable={true}
      // The page-level mousedown listener (createJotView's marquee
      // selection) calls `preventDefault()`, which also cancels the
      // subsequent native dragstart — so without this stop the row
      // never lifts and the user just gets a marquee instead. The
      // handle's own mousedown still fires; only the bubbled handler
      // is suppressed.
      onMouseDown={(e) => e.stopPropagation()}
      onDragStart={(e) => {
        e.dataTransfer.setData(MIXER_DRAG_MIME, String(idx));
        // Some browsers refuse the drag with no plain-text payload.
        e.dataTransfer.setData('text/plain', String(idx));
        e.dataTransfer.effectAllowed = 'move';
        onDragStartIdx(idx);
      }}
      onDragEnd={() => {
        // dragend fires whether or not the drop took — clear the
        // ephemeral state either way so a cancelled drag (Escape, drop
        // outside) doesn't leave the indicator stuck.
        onResetDrag();
      }}
      title={`${ariaLabel} (drag to reorder)`}
      aria-label={`Reorder ${ariaLabel}`}
      role="button"
    >
      ⋮⋮
    </div>
  );
};

/** Audio-track display name: filename with its extension stripped. */
function audioTrackLabel(filename: string): string {
  return filename.replace(/\.[^./\\]+$/, '') || filename;
}
const AUDIO_TRACK_HEIGHT = 56;

/** Common drag/drop props passed to every mixer row. */
type MixerRowDragProps = {
  idx: number;
  dragFromIdx: number | undefined;
  dropTargetIdx: number | undefined;
  onDragStartIdx: (i: number) => void;
  onDropTargetIdx: (i: number | undefined) => void;
  onMoveTrack: (from: number, to: number) => void;
  onResetDrag: () => void;
  /** Length of the mixer list (used by the end-of-list drop zone). */
  mixerLength: number;
  /**
   * True when this row starts a new group (its `groupId` differs from
   * the previous row's, or it's not in a group at all and follows a
   * row that was). The row renders a small top margin so adjacent
   * groups read as distinct clusters; same-group rows render flush.
   * The first row in the mixer never receives this — nothing above it
   * to gap against.
   */
  groupStart: boolean;
};

const AudioTrackRow = observer(
  ({
    id,
    track,
    jot,
    controls,
    onSeek,
    idx,
    dragFromIdx,
    dropTargetIdx,
    onDragStartIdx,
    onDropTargetIdx,
    onMoveTrack,
    onResetDrag,
    groupStart,
  }: {
    id: AudioTrackId;
    track: AudioTrack;
    jot: RenderedJot;
    controls: AudioTrackControls;
    onSeek: (x: number) => void;
  } & MixerRowDragProps) => {
    const voice = jot.resolved.voices[0];
    const width = (voice?.width ?? 0) as number;
    const audible = controls.isAudioTrackAudible(id);
    const muted = controls.mutedAudioTracks.has(id);
    const soloed = controls.soloedAudioTracks.has(id);
    const label = audioTrackLabel(track.filename);
    const lc = `"${track.filename}"`;
    const drop = useMixerRowDropTarget({
      idx,
      dragFromIdx,
      dropTargetIdx,
      onDropTargetIdx,
      onMoveTrack,
      onResetDrag,
    });
    const isDragging = dragFromIdx === idx;
    return (
      <div
        className={classNames(
          styles.musicTrack,
          groupStart && styles.mixerRowGroupStart,
          isDragging && styles.mixerRowDragging,
          drop.isDropIndicatorAbove && styles.mixerDropIndicatorAbove,
          drop.isDropIndicatorBelow && styles.mixerDropIndicatorBelow,
        )}
        data-testid={`audio-track-row-${id}`}
        onDragOver={drop.onDragOver}
        onDragLeave={drop.onDragLeave}
        onDrop={drop.onDrop}
      >
        <div className={styles.musicTrackGutter} style={{ height: AUDIO_TRACK_HEIGHT }}>
          <MixerDragHandle
            idx={idx}
            onDragStartIdx={onDragStartIdx}
            onResetDrag={onResetDrag}
            ariaLabel={`${label} audio track`}
          />
          <div className={styles.musicTrackContent}>
            <div className={classNames(styles.musicTrackLabel, !audible && styles.musicTrackLabelDim)}>
              <span className={styles.musicTrackName}>{label}</span>
              <span className={styles.musicTrackFile} title={track.filename}>
                {track.filename}
              </span>
            </div>
            <div className={styles.musicTrackButtons}>
              <RowVolumeSlider
                value={controls.volumeFor(id)}
                onChange={(v) => controls.onSetVolume(id, v)}
                label={`${label} audio track`}
              />
              {/* Clear sits first so Mute/Solo stay flush with the gutter's
                  right edge — lining up with the M/S column on the
                  instrument rows below (both gutters share a width). */}
              <button
                type="button"
                className={styles.musicTrackClear}
                onClick={(e) => {
                  e.stopPropagation();
                  controls.onClear(id);
                }}
                onMouseDown={(e) => e.stopPropagation()}
                title={`Remove the ${lc} audio track`}
                aria-label={`Remove the ${lc} audio track`}
                data-testid={`audio-track-clear-${id}`}
              >
                ×
              </button>
              <button
                type="button"
                className={classNames(styles.muteButton, muted && styles.muteButtonActive)}
                onClick={(e) => {
                  e.stopPropagation();
                  controls.onToggleMute(id);
                }}
                onMouseDown={(e) => e.stopPropagation()}
                title={muted ? `Unmute ${lc} audio track` : `Mute ${lc} audio track`}
                aria-label={muted ? `Unmute ${lc} audio track` : `Mute ${lc} audio track`}
                aria-pressed={muted}
                data-testid={`audio-track-mute-${id}`}
              >
                M
              </button>
              <button
                type="button"
                className={classNames(styles.soloButton, soloed && styles.soloButtonActive)}
                onClick={(e) => {
                  e.stopPropagation();
                  controls.onToggleSolo(id);
                }}
                onMouseDown={(e) => e.stopPropagation()}
                title={soloed ? `Unsolo ${lc} audio track` : `Solo ${lc} audio track`}
                aria-label={soloed ? `Unsolo ${lc} audio track` : `Solo ${lc} audio track`}
                aria-pressed={soloed}
                data-testid={`audio-track-solo-${id}`}
              >
                S
              </button>
            </div>
          </div>
        </div>
        <div
          className={styles.musicTrackBarsRow}
          style={{ width, height: AUDIO_TRACK_HEIGHT }}
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          <AudioTrackWaveformCanvas
            jot={jot}
            track={track}
            width={width}
            height={AUDIO_TRACK_HEIGHT}
            dim={!audible}
            testId={`audio-track-waveform-${id}`}
          />
          <Playhead onSeek={onSeek} />
        </div>
      </div>
    );
  }
);

/**
 * One drum-instrument row in the unified mixer — exactly one DSL pitch
 * (kick, snare, hi-hat, …). Mirrors {@link AudioTrackRow}: same
 * gutter geometry, M/S/volume controls, drag handle, bars-row + barlines
 * + beat dividers; the lane content is this pitch's notes (drawn through
 * {@link BarView} with `pitches=[pitch]`). The topmost drum row in the
 * mixer (`showBrackets={true}`) also paints the pattern + tuplet
 * brackets so the score chrome stays visible regardless of where the
 * user has dragged the rows.
 *
 * Multi-voice jots: pitches can belong to any voice (e.g. kick lives in
 * the "Feet" voice). The bar geometry is taken from voice[0] (every voice
 * shares the same bar grid), and per-bar tracks are looked up across all
 * voices for this pitch — so the row works whether the pitch lives in
 * voice 0 or 1.
 */
const PitchRow = observer(
  ({
    pitch,
    jot,
    config,
    showBrackets,
    highlightedPattern,
    onPatternClick,
    onSeek,
    voiceControls,
    idx,
    dragFromIdx,
    dropTargetIdx,
    onDragStartIdx,
    onDropTargetIdx,
    onMoveTrack,
    onResetDrag,
    groupStart,
  }: {
    pitch: string;
    jot: RenderedJot;
    config: ViewConfig;
    showBrackets: boolean;
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    onSeek: (x: number) => void;
    voiceControls: VoiceControls;
  } & MixerRowDragProps) => {
    const structure = jot.structure;
    const voice0 = structure.voices[0];
    if (!voice0) return null;
    const trackHeight = config.trackHeight as number;
    // Look up the first instrument and color found for this pitch
    // across all voices, so the gutter label is correct even when the
    // pitch lives in voice[1] (e.g. kick under the "Feet" voice).
    let instrumentName: string | undefined;
    for (const v of structure.voices) {
      if (instrumentName) break;
      for (const bar of v.bars) {
        const t = bar.tracks[pitch];
        if (t?.instrument.name) {
          instrumentName = t.instrument.name;
          break;
        }
      }
    }
    // Replace voice[0]'s per-bar tracks with this pitch's track wherever
    // it appears across the jot's voices. Bar geometry (time, beats,
    // patternSpans, tupletSpans) is untouched — only `tracks` changes
    // — so BarView reads the same beat-coord layout as before. Reading
    // the structural cache (not `jot.resolved`) keeps these bar refs
    // stable across zoom changes; the surrounding container's
    // `--px-per-beat` CSS variable does the actual rescaling.
    const bars: StructuralBar[] = voice0.bars.map((b, i) => {
      let track = b.tracks[pitch];
      if (!track) {
        for (let v = 1; v < structure.voices.length; v++) {
          const t = structure.voices[v].bars[i]?.tracks[pitch];
          if (t) {
            track = t;
            break;
          }
        }
      }
      return { ...b, tracks: track ? { [pitch]: track } : {} };
    });
    // Voice-level totals for the bars-row width (in beats — the row's
    // pixel width is `voiceBeats × --px-per-beat` via CSS calc). Lead-in
    // contributes `leadInSec × bpm/60` quarter notes at the row's tempo.
    const leadInBeats = voice0.leadInSec * (voice0.leadInBpm / 60);
    let voiceBeats = leadInBeats;
    for (const b of voice0.bars) voiceBeats += b.beats;

    // Filtered-onset ghost overlays (debug bundle + checkbox gated).
    // Resolve once per row so the per-entry render below is just a map.
    const provenance = React.useContext(NoteProvenanceContext);
    const showFiltered = provenance?.showFiltered ?? false;
    const rejectedForPitch = showFiltered
      ? provenance!.rejectedByPitch.get(pitch) ?? []
      : [];
    // Cumulative beat offsets so each rejected entry can be positioned
    // absolutely in the bars row without walking back through bar
    // widths on every render. Same scale (quarter-note beats) as the
    // CSS-var positioning the kept notes use.
    const barBeatStart: number[] = [];
    {
      let acc = leadInBeats;
      for (let i = 0; i < bars.length; i++) {
        barBeatStart.push(acc);
        acc += bars[i].beats;
      }
    }
    // Pitch's lane colour for the ghost dashed outline. Best-effort
    // lookup: a pitch with no kept notes has no `tracks[pitch]` entry
    // in any bar — falls back to neutral grey then.
    let pitchColor = '#888';
    for (const b of bars) {
      const t = b.tracks[pitch];
      if (t?.color) {
        pitchColor = t.color;
        break;
      }
    }

    const audible = voiceControls.isPitchAudible(pitch);
    const muted = voiceControls.mutedPitches.has(pitch);
    const soloed = voiceControls.soloedPitches.has(pitch);
    const drop = useMixerRowDropTarget({
      idx,
      dragFromIdx,
      dropTargetIdx,
      onDropTargetIdx,
      onMoveTrack,
      onResetDrag,
    });
    const isDragging = dragFromIdx === idx;
    const labelText = instrumentName ?? `Pitch ${pitch}`;
    const stopBubble = (e: React.MouseEvent) => e.stopPropagation();
    return (
      <div
        className={classNames(
          styles.pitchRow,
          groupStart && styles.mixerRowGroupStart,
          isDragging && styles.mixerRowDragging,
          drop.isDropIndicatorAbove && styles.mixerDropIndicatorAbove,
          drop.isDropIndicatorBelow && styles.mixerDropIndicatorBelow,
        )}
        data-testid={`pitch-row-${pitch}`}
        onDragOver={drop.onDragOver}
        onDragLeave={drop.onDragLeave}
        onDrop={drop.onDrop}
      >
        <div className={styles.pitchRowGutter} style={{ height: trackHeight }}>
          <MixerDragHandle
            idx={idx}
            onDragStartIdx={onDragStartIdx}
            onResetDrag={onResetDrag}
            ariaLabel={labelText}
          />
          <div
            className={classNames(styles.pitchRowLabel, !audible && styles.musicTrackLabelDim)}
            title={instrumentName ? `${instrumentName} (pitch ${pitch})` : `Pitch ${pitch}`}
          >
            <span className={styles.gutterPitch}>{pitch}</span>
            {instrumentName && <span className={styles.pitchRowName}>{instrumentName}</span>}
          </div>
          <div className={styles.musicTrackButtons}>
            <RowVolumeSlider
              value={voiceControls.volumeFor(pitch)}
              onChange={(v) => voiceControls.onSetVolume(pitch, v)}
              label={labelText}
            />
            <button
              type="button"
              className={classNames(styles.muteButton, muted && styles.muteButtonActive)}
              onClick={(e) => {
                stopBubble(e);
                voiceControls.onToggleMute(pitch);
              }}
              onMouseDown={stopBubble}
              title={muted ? `Unmute ${pitch}` : `Mute ${pitch}`}
              aria-label={muted ? `Unmute ${pitch}` : `Mute ${pitch}`}
              aria-pressed={muted}
            >
              M
            </button>
            <button
              type="button"
              className={classNames(styles.soloButton, soloed && styles.soloButtonActive)}
              onClick={(e) => {
                stopBubble(e);
                voiceControls.onToggleSolo(pitch);
              }}
              onMouseDown={stopBubble}
              title={soloed ? `Unsolo ${pitch}` : `Solo ${pitch}`}
              aria-label={soloed ? `Unsolo ${pitch}` : `Solo ${pitch}`}
              aria-pressed={soloed}
            >
              S
            </button>
          </div>
        </div>
        <div
          className={styles.barsRow}
          style={
            {
              ['--voice-beats' as string]: voiceBeats,
              height: trackHeight,
            } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          {leadInBeats > 0 && (
            <div
              className={styles.leadIn}
              style={
                {
                  ['--lead-in-beats' as string]: leadInBeats,
                  height: trackHeight,
                } as React.CSSProperties
              }
              title={`Lead-in: ${voice0.leadInSec.toFixed(
                2,
              )}s of pre-roll before the first beat — keeps the drum notation aligned with a loaded audio-track waveform.`}
            >
              {showBrackets && (
                <span className={styles.leadInLabel}>lead-in</span>
              )}
            </div>
          )}
          {bars.map((bar, i) => (
            <BarView
              key={i}
              bar={bar}
              pitches={[pitch]}
              config={config}
              isAnacrusis={bar.index === 0}
              highlightedPattern={highlightedPattern}
              onPatternClick={onPatternClick}
              isPitchAudible={voiceControls.isPitchAudible}
              showBrackets={showBrackets}
            />
          ))}
          {rejectedForPitch.map((entry, i) => {
            // The MIDI lays `leadBars` empty bar-0-sized blocks before
            // struct bar 0, so the struct bar index maps to the
            // rendered jot's bars array as `leadBars + entry.bar`.
            // Out-of-range entries are already filtered out upstream.
            const barIdx = provenance!.leadBars + entry.bar;
            if (barIdx < 0 || barIdx >= bars.length) return null;
            // beat_in_bar is 1-indexed in the provenance (per the
            // transcriber's OnsetCandidate convention); the CSS calc
            // expects a 0-indexed beat offset within the bar.
            const beatInBar = Math.max(0, entry.beat_in_bar - 1);
            const beatOffset = barBeatStart[barIdx] + beatInBar;
            return (
              <FilteredOnsetView
                key={`f-${entry.bar}-${i}-${entry.detected_time_sec}`}
                entry={entry}
                beatOffset={beatOffset}
                color={pitchColor}
                trackHeight={trackHeight as number}
              />
            );
          })}
          <Playhead onSeek={onSeek} />
        </div>
      </div>
    );
  }
);

/**
 * Canvas-rendered waveform for one audio track, aligned to the score's
 * bar timeline. Peaks are recomputed in a `useEffect` whenever the
 * (zoom-dependent) total bar width changes or the underlying track
 * swaps — same cadence the score uses to re-flow under
 * `viewConfig.barWidth`.
 */
const AudioTrackWaveformCanvas = observer(
  ({
    jot,
    track,
    width,
    height,
    dim,
    testId,
  }: {
    jot: RenderedJot;
    track: AudioTrack;
    width: number;
    height: number;
    dim: boolean;
    testId?: string;
  }) => {
    const canvasRef = React.useRef<HTMLCanvasElement>(null);
    // The live drum↔audio offset (Offset control). Reading it here under
    // `observer` re-renders the waveform when the user nudges the offset
    // so it stays aligned with where the audio actually plays.
    const startOffsetSec = jotPlayer.startOffsetSec;

    React.useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas || width <= 0) return;
      const dpr = window.devicePixelRatio || 1;
      // Browsers cap a canvas's backing-store dimensions (and total
      // area). A long score at high zoom × dpr easily blows past that
      // and throws "Canvas exceeds max size". Clamp the backing store;
      // the element stays CSS-sized to `width`, so past the cap it just
      // renders at reduced horizontal resolution instead of crashing.
      // 16384 is the safe cross-browser per-axis limit (Safari/iOS is
      // the tightest; Chrome/Firefox allow more).
      const MAX_CANVAS_DIM = 16384;
      const backingW = Math.min(Math.max(1, Math.floor(width * dpr)), MAX_CANVAS_DIM);
      const backingH = Math.min(Math.max(1, Math.floor(height * dpr)), MAX_CANVAS_DIM);
      canvas.width = backingW;
      canvas.height = backingH;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      // Map CSS-pixel drawing coords (0..width, 0..height) onto the
      // possibly-clamped backing store. Reduces to ctx.scale(dpr, dpr)
      // when nothing was clamped.
      ctx.setTransform(backingW / width, 0, 0, backingH / height, 0, 0);

      const { peaks } = computeWaveformPeaksForJot(
        jot,
        track.buffer,
        startOffsetSec,
      );

      ctx.clearRect(0, 0, width, height);
      const mid = height / 2;
      ctx.fillStyle = dim ? '#d3c8b6' : '#5BA8E8';
      // Each pixel column is a vertical line from min*scale to max*scale.
      // A single fillRect per column is faster than building a Path2D
      // for thousands of segments and lets us keep the colour-by-column
      // option open if we ever want to tint clipped peaks differently.
      const scale = mid * 0.95;
      for (let p = 0; p < width; p++) {
        const mn = peaks[p * 2];
        const mx = peaks[p * 2 + 1];
        if (mn === 0 && mx === 0) continue;
        const y0 = mid - mx * scale;
        const y1 = mid - mn * scale;
        ctx.fillRect(p, y0, 1, Math.max(1, y1 - y0));
      }
    }, [jot, track, width, height, dim, startOffsetSec]);

    if (width <= 0) return null;
    return (
      <canvas
        ref={canvasRef}
        className={styles.musicTrackWaveform}
        style={{ width, height }}
        data-testid={testId}
      />
    );
  }
);

/**
 * Side-effect-only component: keeps the playhead pinned to the
 * horizontal centre of the viewport during playback by tracking
 * `scrollLeft` to it every frame. `scrollLeft` is auto-clamped by the
 * browser, so near the start / end of the score — where there isn't
 * enough content on one side to centre — the playhead simply rides
 * toward that edge instead of snapping. Renders nothing.
 *
 * Wrapped with `observer` so MobX reactivity drives re-renders on every
 * rAF-driven `currentTime` update; the body just reads observables and
 * runs the side effect.
 */
const PlayheadAutoScroller = observer(
  ({ containerRef }: { containerRef: React.RefObject<HTMLDivElement> }) => {
    const t = jotPlayer.currentTime;
    const state = jotPlayer.state;
    const timeline = jotPlayer.timeline;

    React.useEffect(() => {
      if (state !== 'playing' || timeline.bars.length === 0) return;
      const container = containerRef.current;
      if (!container) return;
      // Anchor x via any barsRow inside the container — they all share
      // the same left edge because voices stack vertically.
      const barsRow = container.querySelector<HTMLDivElement>(`.${styles.barsRow}`);
      if (!barsRow) return;

      const containerRect = container.getBoundingClientRect();
      const barsRect = barsRow.getBoundingClientRect();
      const playheadViewportX = barsRect.left + timeToX(timeline, t);
      // Pin the playhead to the viewport's horizontal centre. Assigning
      // an out-of-range scrollLeft is clamped by the browser, so the
      // first/last screenful (not enough content to centre) degrades
      // gracefully — the playhead rides toward that edge instead.
      const viewportCenter = containerRect.left + containerRect.width / 2;
      container.scrollLeft += playheadViewportX - viewportCenter;
    }, [t, state, timeline, containerRef]);

    return null;
  }
);

function formatSubtitle(jot: RenderedJot): string {
  const parts: string[] = [];
  const { bpm, time, vol } = jot.globalMetadata;
  if (typeof bpm === 'number') parts.push(`${bpm} bpm`);
  else if (bpm) parts.push(`${bpm.start ?? '?'}-${bpm.end} bpm`);
  if (time) parts.push(`${time.count}/${time.unit}`);
  if (typeof vol === 'string') parts.push(vol);
  else if (vol) parts.push(`${vol.start ?? '?'} -> ${vol.end}`);
  return parts.join('  -  ');
}

const Legend = observer(({ jot }: { jot: RenderedJot }) => {
  // Aggregate unique pitches across all voices, in first-seen order.
  // Reads `jot.structure` (zoom-invariant) so the legend doesn't
  // re-render every time the zoom slider moves.
  const seen = new Map<string, { color: string; name?: string }>();
  for (const voice of jot.structure.voices) {
    for (const bar of voice.bars) {
      for (const pitch of Object.keys(bar.tracks)) {
        if (!seen.has(pitch)) {
          const track = bar.tracks[pitch];
          seen.set(pitch, { color: track.color, name: track.instrument.name });
        }
      }
    }
  }
  if (seen.size === 0) return null;
  return (
    <div className={styles.legend}>
      {Array.from(seen.entries()).map(([pitch, info]) => (
        <span key={pitch} className={styles.legendChip}>
          <span className={styles.legendSwatch} style={{ background: info.color }} />
          <strong>{pitch}</strong>
          {info.name ? <span>{info.name}</span> : null}
        </span>
      ))}
    </div>
  );
});

/**
 * One labelled master fader in the transport bar. Pure attenuation
 * (0..1); the percent readout doubles as a "back to default" affordance
 * since 100% is unity.
 */
const MasterVolumeSlider = ({
  label,
  title,
  value,
  onChange,
}: {
  label: string;
  title: string;
  value: number;
  onChange: (v: number) => void;
}) => (
  <label className={styles.masterVolume} title={title}>
    <span>{label}</span>
    <input
      type="range"
      min={0}
      max={1}
      step={VOLUME_STEP}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      aria-label={`${label} master volume`}
    />
    <span className={styles.masterVolumeValue}>{Math.round(value * 100)}%</span>
  </label>
);

/**
 * The page-wide master fader, read straight off the observable
 * `jotPlayer` (no prop drilling — it's app-wide, not per-jot). Takes
 * effect instantly, including mid-playback, and persists across plays.
 * It's the last gain stage so it scales the drums and every audio track
 * together; the per-section masters live in their gutters (see
 * {@link GutterMasterRow}).
 */
const MasterVolumes = observer(() => (
  <div className={styles.masterVolumes}>
    <MasterVolumeSlider
      label="Master"
      title="Page-wide master volume — scales the drums and every audio track together. The last fader before output."
      value={jotPlayer.masterVolume}
      onChange={(v) => jotPlayer.setMasterVolume(v)}
    />
  </div>
));

/**
 * A per-section master fader that sits in the sticky lane gutter,
 * directly above the section it controls (the loaded audio tracks, or
 * the drum/instrument staff). Gutter-aligned (same 132px sticky column
 * as the per-row M/S/volume controls below it) so it reads as the
 * "header" for that column. Reads/writes the global observable
 * `jotPlayer`; all pointer events are kept from bubbling so dragging
 * the fader doesn't start the page marquee or trip seek-on-click.
 */
const GutterMasterRow = observer(
  ({
    label,
    title,
    value,
    onChange,
    testId,
  }: {
    label: string;
    title: string;
    value: number;
    onChange: (v: number) => void;
    testId?: string;
  }) => {
    const stop = (e: React.MouseEvent) => e.stopPropagation();
    const pct = Math.round(value * 100);
    return (
      <div className={styles.gutterMasterRow}>
        <div className={styles.gutterMasterGutter} title={title} data-testid={testId}>
          <span className={styles.gutterMasterLabel}>{label}</span>
          <input
            type="range"
            className={styles.gutterMasterSlider}
            min={0}
            max={1}
            step={VOLUME_STEP}
            value={value}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            onClick={stop}
            onMouseDown={stop}
            onMouseUp={stop}
            aria-label={`${label} master volume`}
            title={`${label} master volume: ${pct}%`}
          />
          <span className={styles.gutterMasterValue}>{pct}%</span>
        </div>
      </div>
    );
  }
);

/**
 * Compact horizontal volume fader shared by the pitch gutter and the
 * audio-track gutter. Range is 0..1 (pure attenuation). All mouse events are
 * kept from bubbling so dragging the fader doesn't start the page-level
 * marquee selection or trip the seek-on-click handler.
 */
const RowVolumeSlider = ({
  value,
  onChange,
  label,
}: {
  value: number;
  onChange: (v: number) => void;
  label: string;
}) => {
  const stop = (e: React.MouseEvent) => e.stopPropagation();
  return (
    <input
      type="range"
      className={styles.rowVolume}
      min={0}
      max={1}
      step={VOLUME_STEP}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      onClick={stop}
      onMouseDown={stop}
      onMouseUp={stop}
      title={`${label} volume: ${Math.round(value * 100)}%`}
      aria-label={`${label} volume`}
    />
  );
};

const BarView = observer(
  ({
    bar,
    pitches,
    config,
    isAnacrusis,
    highlightedPattern,
    onPatternClick,
    isPitchAudible,
    showBrackets = true,
  }: {
    bar: StructuralBar;
    pitches: string[];
    config: ViewConfig;
    isAnacrusis: boolean;
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    isPitchAudible: (pitch: string) => boolean;
    /**
     * Whether to draw the bar's pattern + tuplet brackets on top of the
     * lane. In the unified mixer the brackets describe the score
     * structure (not any one instrument), so only the topmost
     * drum-pitch row of each contiguous drum block renders them — every
     * other pitch row sets this `false` to avoid duplicating the same
     * overlay on every lane.
     */
    showBrackets?: boolean;
  }) => {
    const beatCount = bar.time.count;
    // Beat spacing inside the bar, in quarter notes. Each beat divider
    // is `i × beatSpacingBeats` quarter-notes into the bar, scaled to
    // pixels by the score-root's `--px-per-beat`. Stable per bar.
    const beatSpacingBeats = bar.beats / beatCount;
    // Inline style carries only zoom-invariant data so React's prop
    // diff sees no change on a zoom tick: `--bar-beats` is the bar's
    // length in quarter notes, `height` is config-derived.
    const barStyle = {
      ['--bar-beats' as string]: bar.beats,
      height: pitches.length * (config.trackHeight as number),
    } as React.CSSProperties;
    return (
      <div
        className={classNames(styles.bar, isAnacrusis && styles.barAnacrusis)}
        style={barStyle}
        title={`Bar ${bar.index} - ${bar.time.count}/${bar.time.unit}`}
      >
        {/* One dashed line directly under each beat's notehead — the
            same x the renderer places that beat's note at, computed
            in CSS from `--divider-beat × --px-per-beat`. */}
        {Array.from({ length: beatCount }, (_, i) => (
          <div
            key={`beat-${i + 1}`}
            className={styles.beatDivider}
            style={{ ['--divider-beat' as string]: i * beatSpacingBeats } as React.CSSProperties}
          />
        ))}
        {pitches.map((pitch) => {
          const track = bar.tracks[pitch];
          const dim = !isPitchAudible(pitch);
          return (
            <div
              key={pitch}
              className={classNames(styles.lane, dim && styles.laneDim)}
              style={{ height: config.trackHeight }}
            >
              {track?.notes.map((note, i) => (
                <NoteView
                  key={i}
                  note={note}
                  color={track.color}
                  config={config}
                  instrument={track.instrument}
                  // A non-straight note already inside a tuplet bracket
                  // is explained by that bracket, so only flag the
                  // strays (e.g. an off-grid note not authored as a
                  // group) individually.
                  offGrid={!note.straight && !coveredByTuplet(bar, note.beat)}
                />
              ))}
            </div>
          );
        })}
        {showBrackets &&
          bar.patternSpans.map((span, i) => (
            <PatternBracket
              key={i}
              span={span}
              highlighted={highlightedPattern === span.name}
              onClick={onPatternClick}
            />
          ))}
        {showBrackets &&
          bar.tupletSpans.map((span, i) => <TupletBracket key={i} span={span} />)}
      </div>
    );
  }
);

const PatternBracket = observer(
  ({
    span,
    highlighted,
    onClick,
  }: {
    span: StructuralPatternSpan;
    highlighted: boolean;
    onClick: (name: string) => void;
  }) => {
    return (
      <div
        className={classNames(
          styles.patternBracket,
          highlighted && styles.patternBracketHighlight
        )}
        style={
          {
            ['--span-start-beat' as string]: span.startBeat,
            ['--span-end-beat' as string]: span.endBeat,
          } as React.CSSProperties
        }
      >
        <button
          type="button"
          data-noseek="true"
          className={classNames(
            styles.patternLabel,
            highlighted && styles.patternLabelHighlight
          )}
          onClick={(e) => {
            e.stopPropagation();
            onClick(span.name);
          }}
          onMouseDown={(e) => e.stopPropagation()}
          title={`Pattern usage: ${span.name} (click to highlight other usages)`}
        >
          {span.name}
        </button>
      </div>
    );
  }
);

/**
 * Classic engraved tuplet bracket: a thin line over the grouped notes
 * with the slot count (3 = triplet, 5 = quintuplet, ...) on it. Purely
 * decorative — no interaction, unlike the pattern bracket.
 */
const TupletBracket = observer(({ span }: { span: StructuralTupletSpan }) => (
  <div
    className={styles.tupletBracket}
    style={
      {
        ['--span-start-beat' as string]: span.startBeat,
        ['--span-end-beat' as string]: span.endBeat,
      } as React.CSSProperties
    }
    title={`${span.count}-tuplet (not a straight subdivision)`}
  >
    <span className={styles.tupletNumber}>{span.count}</span>
  </div>
));

/**
 * Shared click-to-seek handler for the score bars row and the audio-track
 * waveforms. Bails on clicks that originated on a note, pattern label,
 * or anything else tagged `data-noseek` so those keep their own
 * behaviour. `e.currentTarget` is the bars-row element whose left edge
 * is x=0 in `bar.x` space, so `clientX - rect.left` is the bars-row-
 * local pixel regardless of horizontal scroll.
 */
function seekFromClick(
  e: React.MouseEvent<HTMLDivElement>,
  onSeek: (x: number) => void
): void {
  if (e.button !== 0) return;
  if ((e.target as HTMLElement).closest('[data-noseek]')) return;
  const rect = e.currentTarget.getBoundingClientRect();
  onSeek(e.clientX - rect.left);
}

/**
 * True when `beat` falls inside any tuplet bracket on this bar. The
 * upper bound is inclusive because `endBeat` is now the last slot's
 * onset (see jot.ts) — the final tuplet note sits exactly on it and is
 * still covered by the bracket.
 */
function coveredByTuplet(bar: StructuralBar, beat: number): boolean {
  const eps = 1e-6;
  return bar.tupletSpans.some(
    (s) => beat >= s.startBeat - eps && beat <= s.endBeat + eps
  );
}

const NoteView = observer(
  ({
    note,
    color,
    config,
    instrument,
    offGrid,
  }: {
    note: StructuralNote;
    color: string;
    config: ViewConfig;
    instrument: Instrument;
    offGrid: boolean;
  }) => {
    const isAccent = note.modifiers.has('a');
    const isGhost = note.modifiers.has('g');
    const isFlam = note.modifiers.has('fl');
    const isDrag = note.modifiers.has('dr');
    const isCross = note.modifiers.has('x');
    const badge = pickBadge(note);
    const selection = React.useContext(SelectionContext);
    const selected = selection?.selectedNote === note;
    const [hovered, setHovered] = React.useState(false);
    const showLabel = selected || hovered;
    const description = offGrid
      ? `${describeNote(note, instrument)} — off the straight grid (triplet/tuplet)`
      : describeNote(note, instrument);
    // Per-note debug provenance, when a filter-mode debug bundle is
    // loaded. Keyed by the original MIDI tick preserved through
    // `from_midi.ts`. Falls back to `undefined` for notes that didn't
    // round-trip through MIDI (e.g. examples, hand-loaded jots) or
    // when no bundle is loaded — the `Debug details` section is hidden
    // in those cases.
    const provenance = React.useContext(NoteProvenanceContext);
    const sourceMeta = note.source.metadata as
      | { midi?: { tick?: number } }
      | undefined;
    const tick = sourceMeta?.midi?.tick;
    const provenanceEntry =
      provenance && typeof tick === 'number'
        ? provenance.byTick.get(`${note.pitch}:${tick}`)
        : undefined;

    return (
      <div
        // Notes opt out of click-to-seek so clicking a note keeps its
        // own meaning (selection / hover label) instead of moving the
        // playhead.
        data-noseek="true"
        className={classNames(
          styles.note,
          isAccent && styles.accent,
          isGhost && styles.ghost,
          note.roll && styles.roll,
          offGrid && styles.offGrid,
          selected && styles.noteSelected,
          showLabel && styles.noteShowingLabel,
          hovered && styles.noteHovered
        )}
        style={
          {
            // Beat is stable per note (set inline); the CSS rule on
            // `.note` derives `left` from `padLeft + beat × pxPerBeat`,
            // so a zoom tick changes one root var instead of mutating
            // this element's style.
            ['--note-beat' as string]: note.beat,
            top: (config.trackHeight as number) / 2,
            width: config.noteDiameter,
            height: config.noteDiameter,
            background: isCross ? '#fff' : color,
            color,
            borderStyle: isCross ? 'solid' : undefined,
            border: isCross ? `2px solid ${color}` : undefined,
          } as React.CSSProperties
        }
        // Suppress the container's mousedown handler — it begins a
        // marquee selection that clears the existing state on every
        // press, which would wipe this note's selection before the
        // click ever fires.
        onMouseDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation();
          selection?.selectNote(note);
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {isFlam && <FlamGrace color={color} config={config} />}
        {isDrag && <DragGrace color={color} config={config} />}
        {badge && <span className={styles.modifierBadge}>{badge}</span>}
        {note.sticking && <span className={styles.stickingBadge}>{note.sticking.toUpperCase()}</span>}
        {showLabel && (
          <div className={styles.noteLabel}>
            <div className={styles.noteLabelText}>{description}</div>
            {provenanceEntry && <NoteProvenanceDetails entry={provenanceEntry} />}
          </div>
        )}
      </div>
    );
  }
);

function FlamGrace({ color, config }: { color: string; config: ViewConfig }) {
  const size = (config.noteDiameter as number) * 0.55;
  return (
    <span
      style={{
        position: 'absolute',
        left: -size - 2,
        top: '50%',
        transform: 'translateY(-50%)',
        width: size,
        height: size,
        background: color,
        borderRadius: '50%',
        opacity: 0.7,
      }}
    />
  );
}

function DragGrace({ color, config }: { color: string; config: ViewConfig }) {
  const size = (config.noteDiameter as number) * 0.45;
  return (
    <>
      {[0, 1].map((i) => (
        <span
          key={i}
          style={{
            position: 'absolute',
            left: -((size + 2) * (i + 1)),
            top: '50%',
            transform: 'translateY(-50%)',
            width: size,
            height: size,
            background: color,
            borderRadius: '50%',
            opacity: 0.6,
          }}
        />
      ))}
    </>
  );
}

/**
 * Collapsible "Debug details" block that surfaces a single onset's full
 * provenance (detected time, strength, beat-tracker placement, filter
 * decision, MIDI tick, …) inside the selection label. Shared between
 * {@link NoteView} (where it appears under the human-readable
 * description) and {@link FilteredOnsetView} (where it IS the label —
 * filtered onsets have no description to lead with).
 *
 * Toggle state is local to this component, so it resets every time the
 * label remounts. Closing the popover (de-selecting / un-hovering the
 * note) collapses the details again next time — acceptable for v1
 * since the block is short.
 */
const NoteProvenanceDetails = ({
  entry,
  startOpen = false,
}: {
  entry: NoteProvenanceEntry;
  /** Open by default (used by FilteredOnsetView so the user immediately
   * sees why the onset was rejected — for kept notes the toggle is
   * collapsed by default so it doesn't crowd the basic description). */
  startOpen?: boolean;
}) => {
  const [open, setOpen] = React.useState(startOpen);
  // Stop the container's mousedown handler so clicks on the toggle don't
  // begin a marquee selection (which would clear the surrounding note's
  // selection and immediately unmount this component).
  const stop = (e: React.MouseEvent) => e.stopPropagation();
  return (
    <div className={styles.debugDetails} onMouseDown={stop}>
      <button
        type="button"
        className={styles.debugDetailsToggle}
        onClick={(e) => {
          stop(e);
          setOpen((o) => !o);
        }}
        onMouseDown={stop}
        aria-expanded={open}
      >
        {open ? '▾' : '▸'} Debug details
      </button>
      {open && (
        <dl className={styles.debugDetailsList}>
          <dt>Detected at</dt>
          <dd>{entry.detected_time_sec.toFixed(3)}s</dd>
          <dt>Strength</dt>
          <dd>{entry.strength.toFixed(3)}</dd>
          <dt>Bar / beat</dt>
          <dd>
            {entry.bar} · {entry.beat_in_bar.toFixed(3)}
          </dd>
          <dt>Backend</dt>
          <dd>{entry.detection_backend}</dd>
          {entry.midi_note !== null && (
            <>
              <dt>MIDI note</dt>
              <dd>{entry.midi_note}</dd>
            </>
          )}
          {entry.tick !== null && (
            <>
              <dt>MIDI tick</dt>
              <dd>{entry.tick}</dd>
            </>
          )}
          <dt>Filter</dt>
          <dd>
            {entry.kept
              ? 'kept'
              : entry.rejected_by ?? (entry.out_of_range ? 'out of range' : 'rejected')}
          </dd>
        </dl>
      )}
    </div>
  );
};

/**
 * Renders one rejected onset as a dashed ghost circle at its detected
 * `(bar, beat_in_bar)` position inside a {@link PitchRow}'s bars row.
 * Absolutely positioned via the same `--note-pad-px` / `--px-per-beat`
 * CSS vars the real notes use, but with `--filtered-beat` = the
 * onset's cumulative beat offset from the start of the row (lead-in +
 * prior bars + intra-bar offset) so it lands at the right absolute x
 * without needing per-bar ResolvedBar geometry.
 *
 * Click toggles a stuck-open detail popover (independent of the
 * SelectionStore — a filtered onset is not a real note); hover shows
 * the same popover transiently.
 */
const FilteredOnsetView = ({
  entry,
  beatOffset,
  color,
  trackHeight,
}: {
  entry: NoteProvenanceEntry;
  /** Total beat offset from the start of the bars row (leadInBeats +
   * cumulative bar beats + (beat_in_bar - 1)). The CSS calc derives
   * the pixel `left` from this and the score-root's `--px-per-beat`. */
  beatOffset: number;
  /** Pitch lane colour. Mirrors what the real notes use; falls back to
   * a neutral grey for filtered-only pitches with no rendered notes. */
  color: string;
  trackHeight: number;
}) => {
  const [hovered, setHovered] = React.useState(false);
  const [clicked, setClicked] = React.useState(false);
  const show = hovered || clicked;
  const stop = (e: React.MouseEvent) => e.stopPropagation();
  return (
    <div
      // Same opt-out as real notes so a click on the ghost doesn't move
      // the playhead via the bars-row seek handler.
      data-noseek="true"
      className={classNames(
        styles.filteredOnset,
        show && styles.filteredOnsetShowingLabel,
      )}
      style={
        {
          ['--filtered-beat' as string]: beatOffset,
          top: trackHeight / 2,
          color,
        } as React.CSSProperties
      }
      onMouseDown={stop}
      onClick={(e) => {
        stop(e);
        setClicked((c) => !c);
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={`Filtered onset · pitch ${entry.pitch} · bar ${entry.bar} beat ${entry.beat_in_bar.toFixed(2)}`}
    >
      {show && (
        <div className={styles.filteredOnsetLabel}>
          <NoteProvenanceDetails entry={entry} startOpen />
        </div>
      )}
    </div>
  );
};

function pickBadge(note: StructuralNote): string | undefined {
  const m = note.modifiers;
  if (m.has('c')) return 'C';
  if (m.has('o')) return 'O';
  if (m.has('h')) return 'H';
  if (m.has('f')) return 'F';
  if (m.has('s')) return 'S';
  if (m.has('r')) return 'R';
  if (m.has('z')) return 'Z';
  if (m.has('k')) return 'K';
  if (m.has('m')) return 'M';
  if (m.has('l')) return 'L';
  if (m.has('rf')) return 'Ruff';
  return undefined;
}

/**
 * Human-readable tooltip text for a note. Combines the resolved instrument
 * name with friendly modifier / sticking / roll labels.
 *
 * Examples:
 *   `s:a`       -> "Snare (accented)"
 *   `s:fl@l`    -> "Snare (flam, left hand)"
 *   `h:c`       -> "Hi-Hat (closed)"
 *   `c~_8:o`    -> "Crash (open, roll)"
 */
function describeNote(note: StructuralNote, instrument: Instrument): string {
  const name = instrument.name ?? `Pitch ${note.pitch}`;
  const qualifiers: string[] = [];
  for (const mod of note.modifiers) {
    qualifiers.push(MODIFIER_LABELS[mod] ?? mod);
  }
  if (note.roll) qualifiers.push('roll');
  if (note.sticking) qualifiers.push(STICKING_LABELS[note.sticking]);
  return qualifiers.length > 0 ? `${name} (${qualifiers.join(', ')})` : name;
}

const MODIFIER_LABELS: Partial<Record<Modifier, string>> = {
  a: 'accented',
  g: 'ghost',
  c: 'closed',
  h: 'half-open',
  o: 'open',
  f: 'foot',
  s: 'splash',
  r: 'rim shot',
  x: 'cross-stick',
  z: 'buzz',
  k: 'choke',
  m: 'mute',
  l: 'let ring',
  fl: 'flam',
  dr: 'drag',
  rf: 'ruff',
};

const STICKING_LABELS: Record<Sticking, string> = {
  r: 'right hand',
  l: 'left hand',
  rf: 'right foot',
  lf: 'left foot',
};
