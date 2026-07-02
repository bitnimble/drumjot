/**
 * Browser playback of a Jot through an acoustic General MIDI drum kit
 * (the GeneralUser GS SoundFont, see {@link GeneralUserGsKit}).
 *
 * `JotPlayer` is the transport: it owns the observable playback state
 * (`state`, `currentTime`, `timeline`, faders), the play cursor / epoch
 * math, and the rAF playhead loop, and orchestrates four collaborators:
 *   - {@link AudioGraph}       the persistent Web Audio bus (ctx + gains).
 *   - {@link SoundfontLoader}  the ~30 MB `.sf2` fetch/parse + kit picker.
 *   - {@link DrumScheduler}    turning events into scheduled `drums.start`s.
 *   - {@link AudioTrackPlaybackController}  live audio-track playback.
 *   - {@link AvSyncEstimator}  the visual↔audio latency compensation.
 *
 * Lifecycle:
 *   - The `AudioContext` and drum kit are created on first `play()`, as
 *     browsers require a user gesture before audio output is allowed, so
 *     deferring construction lets us keep this module side-effect free at
 *     import time.
 *   - One Player instance is shared across the app (singleton at the bottom
 *     of the module). Calling `play()` while a jot is already playing
 *     cancels the in-flight schedule and starts the new one.
 *   - `state`, `currentTime`, and `timeline` are MobX-observable so the
 *     toolbar can switch button labels and the score can render a
 *     traveling playhead without prop drilling.
 *
 * Live mute / solo: the caller (`JotEditorStore`) pushes a `Filter` via
 * `applyLaneFilter()` whenever the user toggles a row's M/S buttons. While
 * playback is in flight, `applyLaneFilter()` cancels every scheduled note and
 * re-schedules those whose audio time hasn't passed yet; so unmuting a
 * row brings it back in the same play, and muting silences it
 * immediately.
 *
 * Diagnostics: scheduling failures (no events extracted, no notes mapped
 * to drum samples on the current kit, etc.) are surfaced both as
 * `errorMessage` on the singleton and via `console` so the UI can show
 * a visible indicator and the operator can inspect details.
 */
import { makeAutoObservable, runInAction } from 'mobx';
import type { MutableJot } from 'src/schema/schema';
import type { TempoPresenter } from 'src/editing/playback/tempo_presenter';
import { MixerContext } from 'src/editing/tracks/tracks';
import type { PlaybackStore } from './playback_store';
import { jotToEvents, PlaybackEvent } from './events';
import {
  decodeAudioTrackFile,
  decodeAudioTrackUrl,
  audioTrackGainUnder,
  PASSTHROUGH_AUDIO_TRACK_FILTER,
  AudioTrackFilter,
  AudioTrackId,
  AudioTrackPlaybackController,
  AudioTrack,
  AudioTrackRole,
  preloadStretch,
} from './audio_tracks';
import { stretchInitFailure } from './stretch_node';
import { buildDriftMap, DriftMap } from './drift_map';
import { BarTiming, EMPTY_TIMELINE, JotTimeline } from './timeline';
import { waveformWorker } from './waveform_worker_client';
import { AvSyncEstimator } from './av_sync';
import { AudioGraph, DRUM_MASTER_GAIN } from './audio_graph';
import { SoundfontLoader } from './soundfont_loader';
import { DrumScheduler } from './drum_scheduler';
import { isAudibleUnder, PASSTHROUGH_FILTER, PlayerFilter } from './player_filter';
import type { KitInfo } from './gm_kit';
import type { SampleLoadProgress } from './sample_storage';

// Re-exported for the historical import path: consumers outside `playback/`
// (e.g. `mixer/mixer_store.ts`) import the lane filter from here. The
// definitions live in `./player_filter` so the drum scheduler can share
// them without a cycle back through this module.
export { isAudibleUnder, PASSTHROUGH_FILTER };
export type { PlayerFilter };

export type PlayerState = 'idle' | 'loading' | 'playing' | 'paused';

// Default master volume for the primary gain.
const DEFAULT_MASTER_VOLUME = 0.5;
// Unity reference (1 = no attenuation) for the user-facing volume
// faders. Used as the initial position of the drum and audio-track
// master faders and as the fallback in `clampMasterVolume` when a bad
// value comes in. Pure attenuation in [0, 1]; the
// GM-vs-music balance trims (DRUM_MASTER_GAIN, AUDIO_TRACK_MASTER_GAIN)
// sit underneath so a fader at 1 keeps today's loudness.
const DEFAULT_VOLUME = 1;
function clampMasterVolume(v: number): number {
  if (!Number.isFinite(v)) return DEFAULT_VOLUME;
  return Math.max(0, Math.min(1, v));
}
// Small lead time so the first hit doesn't race the audio thread.
const SCHEDULE_LEAD_SECONDS = 0.05;
// Buffer added to the last event's time before flipping back to `idle`, so
// late-decaying samples (cymbals, open hats) aren't cut off visually.
const PLAYBACK_TAIL_SECONDS = 1.0;

// Playback speed bounds + step. The toolbar exposes a numeric stepper
// that nudges the value by `PLAYBACK_SPEED_STEP` at a time; the player
// clamps and quantizes anything reaching it (tests, keyboard shortcuts,
// direct programmatic calls) so the contract is consistent across
// entry points. Quarter-speed feels are the practice grid musicians
// already think in, so the step is `0.25`.
export const PLAYBACK_SPEED_MIN = 0.25;
export const PLAYBACK_SPEED_MAX = 2.0;
export const PLAYBACK_SPEED_STEP = 0.25;

// Memoise the drift map on the timeline's bars-array identity (a fresh ref per
// `buildTimeline`) so `JotPlayer.driftMap` can stay a pure computed yet rebuild
// only when the timeline or audio alignment actually changes, the rAF read
// path hits this WeakMap, not `buildDriftMap`'s per-call allocations.
const _driftMapByBars = new WeakMap<readonly BarTiming[], { songLeadIn: number; map: DriftMap }>();
function driftMapFor(bars: readonly BarTiming[], songLeadIn: number): DriftMap {
  const hit = _driftMapByBars.get(bars);
  if (hit && hit.songLeadIn === songLeadIn) return hit.map;
  const map = buildDriftMap(bars, songLeadIn);
  _driftMapByBars.set(bars, { songLeadIn, map });
  return map;
}

export class JotPlayer {
  state: PlayerState = 'idle';
  /** Last error surfaced to the UI; cleared the next time playback succeeds. */
  errorMessage: string | undefined;
  /** Seconds since the current `play()` started (in JOT time, already
   * adjusted for `playbackSpeed`, so a 60-second jot played at 0.5x reports
   * `currentTime` going from 0 → 60 over 120 real seconds). */
  currentTime: number = 0;
  /** Bar-by-bar time→pixel map for the currently-playing jot. `EMPTY_TIMELINE` when idle. */
  timeline: JotTimeline = EMPTY_TIMELINE;
  /**
   * True when the user has clicked the score / a waveform to position
   * the playhead while idle (no audio running). It makes the playhead
   * render at the cued spot before playback starts; `play()` then
   * begins from there. Cleared by `play()` and `stop()`.
   */
  cued: boolean = false;
  /**
   * Tempo multiplier applied to scheduled events and the playhead. 1.0 =
   * native tempo; 0.5 = half speed (notes still sound at the same lane
   * because we space scheduled `drums.start` times further apart rather
   * than touching sample playback rate). Persists across plays.
   */
  playbackSpeed: number = 1;

  /**
   * Page-wide master fader (0..1). Scales *everything*; drums and audio
   * tracks alike; because its GainNode is the last stage before
   * `ctx.destination`. Observable so the slider reflects the live value.
   */
  masterVolume: number = DEFAULT_MASTER_VOLUME;
  /** Master fader for all drum/instrument rows together (0..1). */
  drumMasterVolume: number = DEFAULT_VOLUME;
  /** Master fader for all audio tracks together (0..1). */
  audioTrackMasterVolume: number = DEFAULT_VOLUME;

  /**
   * Late-bound transport store. The engine PULLS its mute/solo/volume
   * filter and section-audibility from here (see {@link laneFilter} etc.
   * below) rather than having them pushed in, so the mixer state has a
   * single home. Bound once at view construction via
   * {@link attachPlayback}; undefined for a standalone engine (tests /
   * stories), where the filter getters fall back to PASSTHROUGH. Not
   * observable itself (it's a large MobX graph; the getters read its
   * observable computeds, which is what drives reactivity).
   */
  playback: PlaybackStore | undefined;

  /**
   * The persistent audio graph (ctx + page/drum/audio-bus gains + the
   * test-only output capture). Not observable: it holds Web Audio nodes,
   * which must never be wrapped in observables.
   *
   * The four collaborators below are `readonly` (public) rather than
   * `private` so they can be listed in `makeAutoObservable`'s exclude map:
   * its `AnnotationsMap` is a mapped type over `this`, which only sees public
   * keys, so a `private` field there is a type error.
   */
  readonly audioGraph = new AudioGraph();
  /** The soundfont loader + kit picker; owns the observable load state. */
  readonly loader = new SoundfontLoader(
    this.audioGraph,
    () => this.ensureAudioContext(),
    () => (this.drumMasterAudible ? DRUM_MASTER_GAIN * this.drumMasterVolume : 0),
  );
  /** Turns the event list into scheduled `drums.start` calls; owns the
   *  per-note stop callbacks. */
  readonly scheduler = new DrumScheduler(() => this.loader.drums);
  /** Visual↔audio sync compensation; owns the observable `audioLatencyMs`
   *  and the auto-detected baseline. */
  readonly avSync = new AvSyncEstimator();

  /**
   * Whether each section's bus is currently audible, master mute, master
   * solo, and per-row solo folded into one boolean per section. Pulled
   * from {@link PlaybackStore} (which delegates to the mixer's
   * `isAudioSectionAudible` / `isDrumSectionAudible`). When false the
   * corresponding bus gain is pinned at 0 regardless of the master fader.
   * True (audible) for a standalone engine with no store wired.
   */
  get drumMasterAudible(): boolean {
    return this.playback?.drumMasterAudible ?? true;
  }
  get audioMasterAudible(): boolean {
    return this.playback?.audioMasterAudible ?? true;
  }

  /**
   * Audio tracks loaded by the user, any number (a ParaDB pack's
   * song + drum tracks, a transcriber's `no_drums`/`drum_stem`, ad-hoc
   * backing tracks, …). Each gets a fresh unique id from {@link
   * allocateAudioTrackId} and decodes to an `AudioBuffer` for playback
   * plus a mono Float32Array for the waveform. Map insertion order is
   * load order, which is the order the gutter renders. Observable so
   * the audio-tracks UI re-renders when tracks are loaded / cleared.
   */
  audioTracks: Map<AudioTrackId, AudioTrack> = new Map();
  /** Monotonic counter backing {@link allocateAudioTrackId}; never reused. */
  private audioTrackIdCounter = 0;
  /**
   * Mixer-context lookup the {@link AudioTrack.color} getter needs to
   * resolve grouped-instrument inheritance. Late-bound by the UI store
   * via {@link attachMixerContext} because the store imports the player
   * (not the other way round), so the player is constructed first with
   * no context and the store wires itself in at its own construction
   * time. AudioTracks loaded before the attach still work, their colour
   * just stays on the neutral fallback until the context arrives.
   */
  mixerContext: MixerContext | undefined;
  /**
   * Latest filename per track id, displayed in the toolbar status. Kept
   * separate from `audioTracks` so the UI knows there was an in-flight load
   * even before the buffer finishes decoding.
   */
  audioTrackError: string | undefined;

  /**
   * The track the minimap picks for its waveform: prefer the first
   * non-lane backing track (a `no_drums` / song stem) and fall back to
   * the first loaded track if every track has a lane (e.g. only
   * isolated drum stems are loaded). `undefined` when no tracks are
   * loaded. Mirrors `pickWaveformTrack` in `minimap.tsx` so consumers
   * can read the choice straight off the player instead of re-deriving
   * it; observable because it's a computed over `audioTracks`.
   */
  get primaryWaveformTrack(): AudioTrack | undefined {
    let backing: AudioTrack | undefined;
    let first: AudioTrack | undefined;
    for (const t of this.audioTracks.values()) {
      if (!first) first = t;
      if (!t.lane && !backing) backing = t;
    }
    return backing ?? first;
  }

  /**
   * Audio tracks indexed by lowercased filename, for O(1) filename
   * lookups (e.g. matching a debug bundle's per-lane manifest entry to
   * the matching loaded track). Computed once per `audioTracks` change
   * so per-onset callers in the timing-visualization don't each rebuild
   * an `Array.from(…).find(…)` walk on every render.
   */
  get audioTracksByFilename(): ReadonlyMap<string, AudioTrack> {
    const out = new Map<string, AudioTrack>();
    for (const t of this.audioTracks.values()) {
      out.set(t.filename.toLowerCase(), t);
    }
    return out;
  }

  /**
   * Live per-play audio-track playback state (worklet slots + applied
   * per-track gains). Recreated on every `play()`, disposed on `stop()`.
   * Exposed so a non-React renderer can read the controller's
   * `appliedGains` observable to know exactly what's audible right now
   * without poking the audio graph. Undefined while idle.
   */
  audioTrackController: AudioTrackPlaybackController | undefined;
  /** Audio-track mute/solo/volume filter the engine runs under, pulled
   *  live from {@link PlaybackStore} (PASSTHROUGH when no store is wired).
   *  A computed: an alt renderer can read what's audible right now, and
   *  the audio path reads it directly instead of caching a snapshot. */
  get currentAudioTrackFilter(): AudioTrackFilter {
    return this.playback?.audioTrackFilter ?? PASSTHROUGH_AUDIO_TRACK_FILTER;
  }
  /**
   * The `songLeadIn` epoch: the JOT time (<= 0) at which the recorded audio
   * begins, i.e. the recording's pre-drum lead-in. Each audio track plays at
   * media time `jot - songLeadInSec` (see {@link jotToMedia}); lowering it
   * (more negative) slides the backing audio *ahead* of the drums and raising
   * it pulls them together. Drum scheduling is in jot-time and doesn't depend
   * on this, so a change only repositions the audio tracks.
   *
   * Read-only mirror: the source of truth is {@link PlaybackStore.songLeadInSec}
   * (seeded + nudged by `PlaybackPresenter`, where the full {@link Epochs}
   * record also lives). The engine reads it from the attached store and falls
   * back to 0 for a standalone engine (tests / stories with no store). After
   * a change the presenter calls {@link repositionAudioForOffset} to slide the
   * audio tracks; it stays a getter so every existing `jotPlayer.songLeadInSec`
   * read keeps reacting to the store observable.
   */
  get songLeadInSec(): number {
    return this.playback?.songLeadInSec ?? 0;
  }

  /**
   * AudioContext time of the playback anchor (updated at `play()` and
   * whenever `setPlaybackSpeed` re-anchors mid-flight); `currentJotTime`
   * is computed from this plus the elapsed real time times speed.
   * Public so a non-React renderer can map jot-time ↔ audio-time
   * without going through the rAF tick (`audioTime = startContextTime +
   * (jotTime - startJotTime) / playbackSpeed`).
   */
  startContextTime: number = 0;
  /** Jot-time value at `startContextTime`; non-zero after a mid-flight
   * speed change so the playhead doesn't snap back to 0. */
  startJotTime: number = 0;
  private endTimerId: number | undefined;
  /**
   * AudioContext time of the last scheduled note's start. Retained so
   * `resume()` can re-arm the end-of-playback fallback timer, that
   * timer is a wall-clock `setTimeout`, so `suspend()` (which only
   * freezes the audio clock) would otherwise let it fire while paused.
   */
  private tailAudioTime: number = 0;
  private rafId: number | undefined;
  /**
   * The full event list for the currently-playing jot, retained so that
   * `applyLaneFilter` can re-derive the scheduled subset on a mute/solo
   * toggle without having to re-walk the layout. Public + observable so
   * a non-React renderer can read "all scheduled notes for this jot"
   * and combine with the observable {@link currentTime} to know which
   * are upcoming. Empty while idle.
   */
  events: PlaybackEvent[] = [];
  /** Lane-side mute/solo/volume filter the scheduler runs under, pulled
   *  live from {@link PlaybackStore} (PASSTHROUGH when no store is wired).
   *  Same computed/pull rationale as {@link currentAudioTrackFilter}. */
  get currentFilter(): PlayerFilter {
    return this.playback?.trackFilter ?? PASSTHROUGH_FILTER;
  }
  /**
   * Jot-time (seconds) the next `play()` should start from, set by a
   * click-to-seek while idle. `undefined` means "start from the
   * beginning" (honouring `songLeadIn` lead-in as before).
   */
  private pendingStartSec: number | undefined;

  constructor() {
    // `mixerContext` is the UI store itself (large MobX-observable graph);
    // wrapping it in another observable shell would have no benefit and
    // could create observation cycles. The AudioTrack getter reads it
    // through the late-bound callback at compute time, which is the
    // path the picker UI depends on, not through MobX tracking of the
    // field itself. The four collaborators are stable non-observable
    // refs (they carry their own observability where needed); the getters
    // that surface their observable fields read straight through.
    makeAutoObservable(this, {
      mixerContext: false,
      playback: false,
      audioGraph: false,
      loader: false,
      scheduler: false,
      avSync: false,
    });
  }

  // --- Delegated observable surface (kept stable for React consumers) ---

  /** @see AvSyncEstimator.audioLatencyMs */
  get audioLatencyMs(): number {
    return this.avSync.audioLatencyMs;
  }
  /** @see AvSyncEstimator.setAudioLatencyMs */
  setAudioLatencyMs(ms: number): void {
    this.avSync.setAudioLatencyMs(ms);
  }
  /** @see SoundfontLoader.drumKits */
  get drumKits(): KitInfo[] {
    return this.loader.drumKits;
  }
  /** @see SoundfontLoader.drumPreset */
  get drumPreset(): number {
    return this.loader.drumPreset;
  }
  /** @see SoundfontLoader.sampleLoadProgress */
  get sampleLoadProgress(): SampleLoadProgress | undefined {
    return this.loader.sampleLoadProgress;
  }
  /** @see SoundfontLoader.sampleLoadPhase */
  get sampleLoadPhase(): 'connecting' | 'downloading' | 'decoding' | undefined {
    return this.loader.sampleLoadPhase;
  }

  /**
   * Re-apply the (pulled) lane mute/solo filter to the live schedule. If
   * playback is in flight OR paused, every scheduled note is cancelled and
   * the remaining events (those whose audio time hasn't elapsed) are
   * re-scheduled against the current filter; so toggling M or S takes
   * effect immediately, including bringing previously-muted rows back
   * in mid-song. The filter itself is the {@link currentFilter} computed
   * (pulled from {@link PlaybackStore}); a `PlaybackPresenter` reaction
   * calls this whenever it changes. Idle does nothing, `play()` reads
   * the filter when it next schedules.
   *
   * The paused case matters: pause → toggle a row's M → resume is a
   * natural practice workflow, and `resume()` only reschedules audio tracks,
   * not drum events. Without re-filtering here the pre-mute drum
   * schedule survives the pause and the muted row keeps sounding on
   * resume. `ctx.currentTime` is frozen while paused, which is exactly
   * the anchor we want there, the rescheduled notes line up relative
   * to it and come alive when the context resumes (same approach as
   * `seek`).
   */
  applyLaneFilter(): void {
    const ctx = this.audioGraph.ctx;
    if ((this.state !== 'playing' && this.state !== 'paused') || !ctx) return;

    const now = ctx.currentTime;
    const jotOffset = this.currentJotTime(now);
    // `cancelScheduledStops()` alone is not enough: the per-note stopFns
    // are no-ops for layers smplr hasn't instantiated yet (it only
    // builds the BufferSourceNode when the scheduled time arrives), so
    // the still-pending notes (including the row the user just muted)
    // would keep firing alongside the rescheduled set and the toggle
    // would appear to do nothing. `drums.stop()` flushes the whole
    // pending queue; same reason `setPlaybackSpeed` calls it.
    this.scheduler.cancelScheduledStops();
    this.scheduler.stopDrums();
    const lastTime = this.scheduleEvents(jotOffset, now);
    // Paused: resume() re-arms the end-of-playback timer from
    // tailAudioTime, so keep it current with the re-filtered schedule
    // (an unmute can extend the tail; a mute can shorten it).
    if (this.state === 'paused') this.tailAudioTime = lastTime;
  }

  /**
   * Update the audio-track mute/solo filter. Tracks already playing get their
   * `GainNode.gain` toggled to 0 / 1 immediately, no source recreation
   * needed, so the change is sample-accurate without the click-risk of
   * stopping and restarting a `BufferSourceNode` mid-decay.
   */
  applyAudioTrackFilter(): void {
    if (this.audioTrackController) {
      const filter = this.currentAudioTrackFilter;
      this.audioTrackController.applyAudibility((id) => audioTrackGainUnder(id, filter));
    }
  }

  /**
   * Start recording the page-master output as a windowed-RMS time series.
   * Returns false if there's no AudioContext yet (call after `play()` has
   * begun). Idempotent stop via {@link stopOutputCapture}.
   */
  startOutputCapture(): boolean {
    return this.audioGraph.startOutputCapture();
  }

  /** Stop capture and return the recorded {t, rms} series. */
  stopOutputCapture(): { t: number; rms: number }[] {
    return this.audioGraph.stopOutputCapture();
  }

  /**
   * Move the whole-page master fader. Takes effect instantly (it's a
   * single GainNode at the end of the graph) and persists across plays;
   * works before any audio exists, the value is stored and applied when
   * the graph is built.
   */
  setMasterVolume(v: number): void {
    const clamped = clampMasterVolume(v);
    runInAction(() => {
      this.masterVolume = clamped;
    });
    this.audioGraph.setPageGain(clamped);
  }

  /** Move the all-drums master fader. Same instant/persistent semantics. */
  setDrumMasterVolume(v: number): void {
    const clamped = clampMasterVolume(v);
    runInAction(() => {
      this.drumMasterVolume = clamped;
    });
    this.applyDrumBusGain();
  }

  /** Move the all-audio-tracks master fader. Same instant/persistent semantics. */
  setAudioTrackMasterVolume(v: number): void {
    const clamped = clampMasterVolume(v);
    runInAction(() => {
      this.audioTrackMasterVolume = clamped;
    });
    this.applyAudioBusGain();
  }

  /**
   * Re-apply the drum bus gain from the pulled {@link drumMasterAudible}.
   * False pins `drumGain` at 0 regardless of the fader (so master mute /
   * cross-domain solo silences every drum row at once at the bus, not by
   * editing the per-lane mute/solo state); true restores the fader value.
   * Public so a `PlaybackPresenter` reaction can call it when the pulled
   * audibility changes; also called internally by the fader setter.
   */
  applyDrumBusGain(): void {
    this.audioGraph.setDrumBusGain(
      this.drumMasterAudible ? DRUM_MASTER_GAIN * this.drumMasterVolume : 0,
    );
  }

  /** Mirror of {@link applyDrumBusGain} for the audio-track bus. */
  applyAudioBusGain(): void {
    this.audioGraph.setAudioBusGain(this.audioMasterAudible ? this.audioTrackMasterVolume : 0);
  }

  /** Fresh, never-reused audio-track id. Load order ⇒ ascending ids. */
  private allocateAudioTrackId(): AudioTrackId {
    return `track-${++this.audioTrackIdCounter}`;
  }

  /** Wire the UI store in as the mixer-context source for freshly-
   *  constructed {@link AudioTrack}s. Called once from the store's
   *  constructor. Safe to re-call (the new context replaces the old). */
  attachMixerContext(ctx: MixerContext): void {
    this.mixerContext = ctx;
  }

  /** Wire the transport store in as the source the engine pulls its
   *  mute/solo/volume filter + section-audibility from. Called once at
   *  view construction (the store imports the player, not vice-versa, so
   *  the player is constructed context-free and the view binds it). */
  attachPlayback(store: PlaybackStore): void {
    this.playback = store;
  }

  /**
   * Load an audio file (a ParaDB pack track, a transcriber FLAC, any
   * audio the user drops in) as a new track and return its allocated id.
   * Every call appends a track, there is no replace-by-name slot any
   * more, so loading N files yields N independent tracks. Decoding
   * shares the `AudioContext` with the drum machine, so this method
   * constructs the context even before playback starts, meaning the
   * call must happen inside a user gesture on some browsers (the click
   * that triggered the file picker inherits the gesture grant).
   */
  async loadAudioTrack(
    file: File,
    lane?: string,
    role?: AudioTrackRole,
    extraLanes?: readonly string[],
  ): Promise<AudioTrackId> {
    runInAction(() => {
      this.audioTrackError = undefined;
    });
    try {
      const ctx = this.ensureAudioContext();
      // Start the soundfont download in the background now; it overlaps
      // with the file decode below and leaves the kit warm for the user's
      // next `play()`. See `preloadDrums()`. Warm up the audio-track
      // stretch worklet in parallel so the first speed change doesn't
      // wait on a fresh WASM/worklet install.
      this.preloadDrums();
      preloadStretch(ctx);
      const { buffer, sourceBlob } = await decodeAudioTrackFile(ctx, file);
      const id = this.allocateAudioTrackId();
      this.installAudioTrack(id, file.name, buffer, sourceBlob, lane, role, extraLanes);
      return id;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      runInAction(() => {
        this.audioTrackError = `Could not load ${file.name}: ${message}`;
      });
      throw err;
    }
  }

  /** Same as {@link loadAudioTrack} but fetches from a URL (transcriber output). */
  async loadAudioTrackFromUrl(
    url: string,
    filename: string,
    lane?: string,
    role?: AudioTrackRole,
  ): Promise<AudioTrackId> {
    runInAction(() => {
      this.audioTrackError = undefined;
    });
    try {
      const ctx = this.ensureAudioContext();
      // See `loadAudioTrack`; overlap soundfont download and stretch
      // worklet load with the network fetch + decode of this track.
      this.preloadDrums();
      preloadStretch(ctx);
      const { buffer, sourceBlob } = await decodeAudioTrackUrl(ctx, url);
      const id = this.allocateAudioTrackId();
      this.installAudioTrack(id, filename, buffer, sourceBlob, lane, role);
      return id;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      runInAction(() => {
        this.audioTrackError = `Could not load ${filename}: ${message}`;
      });
      throw err;
    }
  }

  /** Drop a loaded audio track. If playback is in flight, the track's
   * source is stopped so it falls silent immediately. */
  clearAudioTrack(id: AudioTrackId): void {
    const prev = this.audioTracks.get(id);
    if (!prev) return;
    runInAction(() => {
      this.audioTracks.delete(id);
    });
    // Tear down the removed track's worklet/gain so it leaves no
    // dangling nodes; remaining tracks keep playing untouched (each
    // owns its own slot, so no reschedule is needed).
    this.audioTrackController?.dropAudioTrack(id);
    // Free the worker-side PCM copy too; otherwise the worker would
    // accumulate dead tracks across the session (the PCM there is a
    // separate copy from the AudioBuffer the player owns).
    waveformWorker.dropTrack(id);
  }

  private installAudioTrack(
    id: AudioTrackId,
    filename: string,
    buffer: AudioBuffer,
    sourceBlob: Blob,
    lane?: string,
    role?: AudioTrackRole,
    extraLanes?: readonly string[],
  ): void {
    const prev = this.audioTracks.get(id);
    const track = new AudioTrack(
      {
        id,
        filename,
        buffer,
        sourceBlob,
        durationSec: buffer.duration,
        lane,
        extraLanes,
        role,
      },
      () => this.mixerContext,
    );
    runInAction(() => {
      this.audioTracks.set(id, track);
    });
    // Hand a copy of the decoded PCM to the waveform worker so future
    // peak recomputes (zoom, offset change, per-onset timing viz)
    // don't run on the main thread. Done immediately so the worker
    // has it before the first React effect fires a peak request; the
    // worker's message queue is FIFO, so a `register` posted here is
    // always processed before a later `peaks` for the same id.
    waveformWorker.registerTrack(id, buffer);
    // If playback is in flight, start the new track in sync with the
    // existing schedule so the user hears it appear at the current
    // playhead without having to stop+restart.
    const ctx = this.audioGraph.ctx;
    if (this.state === 'playing' && ctx && this.audioTrackController) {
      const now = ctx.currentTime;
      const jotOffset = this.currentJotTime(now);
      this.audioTrackController.scheduleAll(
        [track],
        now,
        jotOffset,
        this.playbackSpeed,
        this.driftMap,
        (sid) => audioTrackGainUnder(sid, this.currentAudioTrackFilter)
      );
    }
    // No URL.revoke needed; the source bytes live on the track as a
    // Blob now, so a replaced track is GC'd when nothing references it.
    void prev;
  }

  /**
   * Switch the active drum kit to another preset in the percussion bank.
   *
   * The preset is always recorded so the *next* load uses it. If the kit
   * is already loaded, the swap happens immediately (no refetch, the
   * parsed SoundFont is retained); it takes effect on every subsequently
   * scheduled note, so changing kit mid-play is fine. Failures surface
   * via `errorMessage` rather than throwing into the UI handler.
   */
  async setDrumPreset(preset: number): Promise<void> {
    await this.loader.setDrumPreset(preset, (message) => {
      runInAction(() => {
        this.errorMessage = message;
      });
    });
  }

  /**
   * Set the tempo multiplier. Takes effect immediately, including during
   * playback: we re-anchor `startContextTime` / `startJotTime` to "now",
   * cancel every scheduled note, and reschedule remaining notes at their
   * new audio times under the new spacing.
   *
   * Sample lane is unchanged, drum samples still play at native rate.
   * Slowing down just spaces successive `drums.start` calls further
   * apart, which is exactly what you want for practicing along to a
   * complex fill at half speed.
   */
  setPlaybackSpeed(speed: number): void {
    // Quantize to the 0.25 step grid then clamp into [min, max]. Both
    // belt and suspenders for entry points other than the stepper UI
    // (keyboard, tests, future shortcuts), which can hand us anything.
    const snapped = Math.round(speed / PLAYBACK_SPEED_STEP) * PLAYBACK_SPEED_STEP;
    const clamped = Math.max(PLAYBACK_SPEED_MIN, Math.min(PLAYBACK_SPEED_MAX, snapped));
    const ctx = this.audioGraph.ctx;
    if (this.state !== 'playing' || !ctx) {
      runInAction(() => {
        this.playbackSpeed = clamped;
      });
      return;
    }
    const now = ctx.currentTime;
    // Snapshot current jot time AT THE OLD SPEED before mutating, so the
    // playhead doesn't jump when the user changes speed mid-song.
    const jotOffset = this.currentJotTime(now);

    // Kill the old DRUM schedule before laying down the new one. The
    // per-layer stopFns returned by `drums.start({ time: <future> })`
    // don't actually cancel a layer that hasn't begun yet; smplr only
    // instantiates the BufferSourceNode when the scheduled time
    // arrives, so calling stopFn early is a no-op. The global
    // `drums.stop()` clears the entire pending queue, which is what we
    // need here. Without this, the old (still-pending) notes play
    // alongside the rescheduled ones and the speed change appears to
    // have no effect.
    //
    // Audio tracks DON'T need a cancel + reschedule: setting
    // `playbackRate` on each live MediaElement changes speed seamlessly
    // (with `preservesPitch` already on, pitch is preserved too). The
    // old code's cancel + reschedule path forced each track through a
    // pause + retimed play, introducing a ~20 ms audible gap on every
    // speed change.
    this.scheduler.cancelScheduledStops();
    this.scheduler.stopDrums();

    runInAction(() => {
      this.playbackSpeed = clamped;
    });
    this.startContextTime = now;
    this.startJotTime = jotOffset;
    const lastTime = this.scheduleEvents(jotOffset, now);
    // `setPlaybackRate` may need a full reschedule under the hybrid
    // BufferSource (1.0×) / MediaElement (non-1.0×) split; pass the
    // anchor state so it can reschedule from `(now, jotOffset)` when
    // the path crosses the 1.0× boundary.
    this.audioTrackController?.setPlaybackRate(
      clamped,
      now,
      jotOffset,
      this.driftMap,
      (id) => audioTrackGainUnder(id, this.currentAudioTrackFilter),
    );
    this.scheduleTailTimer(lastTime);
  }

  /**
   * Reposition the audio tracks to the current {@link songLeadInSec} (the
   * drum↔audio offset / lead-in). Called by `PlaybackPresenter` after it
   * writes the new offset into {@link PlaybackStore.songLeadInSec}. Takes
   * effect immediately, including mid-playback and while paused: the drums
   * and playhead are anchored in jot-time and don't move, so we only reseek
   * the audio tracks to their new media position (`currentJotTime + offset`).
   * While paused the AudioContext clock is frozen, so the rescheduled
   * elements stay silent (their `play()` no-ops against a suspended context)
   * and realign on resume, same approach as `seek`. No-op unless playing or
   * paused with a live context + audio controller.
   *
   * (The value itself is clamped to <= 0 by the presenter: the audio can't
   * start *after* bar 1, so a positive lead-in has no meaning.)
   */
  repositionAudioForOffset(): void {
    const ctx = this.audioGraph.ctx;
    if ((this.state !== 'playing' && this.state !== 'paused') || !ctx) return;
    if (!this.audioTrackController) return;
    const now = ctx.currentTime;
    const jotOffset = this.currentJotTime(now);
    // Only the audio tracks depend on the offset; reposition them to the
    // new media time without disturbing the drum schedule or playhead.
    this.audioTrackController.cancelSources();
    this.audioTrackController.scheduleAll(
      this.audioTracks.values(),
      now,
      jotOffset,
      this.playbackSpeed,
      this.driftMap,
      (id) => audioTrackGainUnder(id, this.currentAudioTrackFilter)
    );
    // The auto-stop fires when the latest of (last drum event, last
    // audio-track sample) is reached. `computeAudioTracksEndTime` reads
    // `songLeadIn`, so a mid-flight offset change shifts the audio end
    // time and the tail must be re-armed; otherwise raising the offset
    // can cause auto-stop to fire before the audio finishes.
    if (this.state === 'playing') {
      this.scheduleTailTimer(this.scheduler.lastScheduledDrumTime);
    } else {
      this.tailAudioTime = this.computeAudioTracksEndTime();
    }
  }

  /**
   * Re-derive the drum events from `rendered` and reschedule them live.
   * Called when the jot's beat-grid offset (the "Beat offset" control,
   * which slides drum notes across bars to fix a transcription beat
   * error) changes mid-flight, the notes have moved, so the scheduled
   * hits must follow. No-op unless playing or paused.
   *
   * Mirrors {@link applyLaneFilter}'s reschedule: the bar grid (and thus the
   * timeline) is unchanged, so only the drum schedule is rebuilt; audio
   * tracks and the playhead stay put. While paused the AudioContext clock
   * is frozen, so the rescheduled notes line up against it and come alive
   * on resume (which re-arms the tail timer from `tailAudioTime`).
   */
  refreshDrumSchedule(jot: MutableJot, precomputedEvents?: PlaybackEvent[]): void {
    const ctx = this.audioGraph.ctx;
    if ((this.state !== 'playing' && this.state !== 'paused') || !ctx) return;
    // Reuse the event list the caller already derived: the live-edit reaction
    // walks the whole jot to detect the change under structural equality, so
    // re-walking here would double the per-edit cost. Fall back to deriving it.
    this.events = precomputedEvents ?? jotToEvents(jot);
    const now = ctx.currentTime;
    const jotOffset = this.currentJotTime(now);
    this.scheduler.cancelScheduledStops();
    this.scheduler.stopDrums();
    const lastTime = this.scheduleEvents(jotOffset, now);
    if (this.state === 'paused') this.tailAudioTime = lastTime;
    else this.scheduleTailTimer(lastTime);
  }

  /**
   * Map an absolute AudioContext time to its jot-time position, taking
   * the current `playbackSpeed` (and any prior speed-change anchor) into
   * account.
   */
  currentJotTime(audioTime: number): number {
    // Media (recorded-audio) time advances linearly with the AudioContext
    // clock at `playbackSpeed`; jot time does NOT when the recording drifts,
    // so map media → jot through the drift map. With no drift the two closures
    // cancel and this is exactly `startJotTime + elapsed * speed` as before.
    const map = this.driftMap;
    const elapsedMedia = (audioTime - this.startContextTime) * this.playbackSpeed;
    return map.mediaToJot(map.jotToMedia(this.startJotTime) + elapsedMedia);
  }

  /**
   * Drift-aware jot ↔ media conversion for the current timeline + live audio
   * alignment. A pure computed (no observable writes); the rebuild is memoised
   * by {@link driftMapFor} on the bars-array ref (fresh per `buildTimeline`) +
   * `songLeadInSec`, so `currentJotTime` reading it every rAF stays
   * allocation-free even though, off-reaction, MobX doesn't cache the computed.
   */
  private get driftMap(): DriftMap {
    return driftMapFor(this.timeline.bars, this.songLeadInSec);
  }

  /**
   * Move the playhead (and, if audio is running, the playback position)
   * to `seconds` of jot time.
   *
   *  - **idle**: build the timeline so the playhead can be drawn, park
   *    it at the cued position, and remember it so the next `play()`
   *    starts there. No audio is touched (no CDN sample fetch).
   *  - **playing**: re-anchor and reschedule everything from the new
   *    position, a live scrub. Mirrors `setPlaybackSpeed`'s re-anchor.
   *  - **paused**: same re-anchor, but against the frozen AudioContext
   *    clock; the rescheduled notes stay silent until `resume()`, which
   *    re-arms the tail timer from `tailAudioTime`.
   */
  seek(tempo: TempoPresenter, seconds: number): void {
    // Lower-bound the seek at the left edge of the RENDERED timeline (the
    // first bar's `startSec`), so the user can scrub all the way to the
    // start of the lead-in. That left edge is the view's virtual lead-in
    // bar (always at least one bar), which can sit further left than the
    // audio pre-roll alone (`songLeadIn`), clamping at the latter would
    // strand the playhead short of the rendered left edge. The media
    // mapping still clamps audio to >= 0, so the silent pre-bar scrubs in
    // silence.
    if (this.state === 'idle') {
      const timeline = tempo.timeline;
      if (timeline.bars.length === 0) return;
      const start = timeline.bars[0].startSec;
      const target = Math.min(Math.max(seconds, start), timeline.totalDurationSec);
      this.pendingStartSec = target;
      this.startJotTime = target;
      runInAction(() => {
        this.timeline = timeline;
        this.currentTime = target;
        this.cued = true;
      });
      return;
    }

    const ctx = this.audioGraph.ctx;
    if (!ctx) return;
    const start = this.timeline.bars[0]?.startSec ?? this.songLeadInSec;
    let target = Math.max(seconds, start);
    const dur = this.timeline.totalDurationSec;
    if (dur > 0) target = Math.min(target, dur);

    // ctx.currentTime is frozen while paused, which is exactly the
    // anchor we want there: scheduled notes line up relative to it and
    // come alive when the context resumes.
    const now = ctx.currentTime;
    this.scheduler.cancelScheduledStops();
    this.scheduler.stopDrums();
    this.audioTrackController?.cancelSources();

    this.startContextTime = now;
    this.startJotTime = target;
    const lastTime = this.scheduleEvents(target, now);
    if (this.audioTrackController) {
      this.audioTrackController.scheduleAll(
        this.audioTracks.values(),
        now,
        target,
        this.playbackSpeed,
        this.driftMap,
        (id) => audioTrackGainUnder(id, this.currentAudioTrackFilter)
      );
    }
    if (this.state === 'playing') {
      this.scheduleTailTimer(lastTime);
    } else {
      // Paused: resume() re-arms the tail timer from tailAudioTime.
      this.tailAudioTime = lastTime;
    }
    runInAction(() => {
      this.currentTime = target;
    });
  }

  async play(jot: MutableJot, tempo: TempoPresenter): Promise<void> {
    // Capture the click-to-seek cue before stop() clears it.
    const cueSec = this.pendingStartSec;
    this.stop();
    runInAction(() => {
      this.state = 'loading';
      this.errorMessage = undefined;
    });

    try {
      const { ctx } = await this.loader.ensureLoaded();

      // Chrome / Safari sometimes start a freshly-constructed AudioContext
      // suspended even when the constructor ran inside a user gesture, so
      // we always re-check before scheduling. The resume() promise inherits
      // the user-gesture grant from the click that called play().
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }

      // Measure the device + frame-rate latency once per session and
      // bake it into `internalLatencyMs`. Fires once on first play
      // (after resume, when `outputLatency` is meaningful) and is
      // latched so subsequent plays skip the work. Runs in the
      // background; the ~200 ms frame measurement does not block
      // scheduling and the result self-applies in the next rAF tick.
      this.avSync.estimateOnce(ctx);

      this.events = jotToEvents(jot);
      if (this.events.length === 0) {
        throw new Error('No playable notes in this jot.');
      }

      const timeline = tempo.timeline;
      const audioStartTime = ctx.currentTime + SCHEDULE_LEAD_SECONDS;
      // Start from the click-to-seek cue if one is pending, otherwise from
      // the rendered left edge (`fullLeadIn` = the first bar's startSec,
      // which includes the view's virtual lead-in) so a no-cue play begins
      // at the very top of the lead-in / count-in. A cue (including a
      // negative one parked in the lead-in) is honoured, clamped into
      // [fullLeadIn, total]. The live `songLeadInSec` (seeded from the jot's
      // metadata, tunable via the Offset control) drives audio-track timing.
      const fullLeadIn = timeline.bars[0]?.startSec ?? this.songLeadInSec;
      const anchorJot =
        cueSec !== undefined
          ? Math.min(Math.max(cueSec, fullLeadIn), timeline.totalDurationSec)
          : fullLeadIn;
      this.startContextTime = audioStartTime;
      this.startJotTime = anchorJot;
      const lastTime = this.scheduleEvents(anchorJot, audioStartTime);

      // Audio tracks play through the same AudioContext so they share the
      // clock with the drum scheduler. The controller is recreated on
      // every play() to drop any residual nodes from the previous run.
      this.audioTrackController?.dispose();
      this.audioTrackController = new AudioTrackPlaybackController(
        ctx,
        this.audioGraph.audioBusGain ?? ctx.destination
      );
      this.audioTrackController.scheduleAll(
        this.audioTracks.values(),
        audioStartTime,
        anchorJot,
        this.playbackSpeed,
        this.driftMap,
        (id) => audioTrackGainUnder(id, this.currentAudioTrackFilter)
      );
      // The stretch worklet preloads on first audio-track load; if it
      // failed (CSP, network, missing AudioWorklet API) the controller
      // will silently fail per-track. Surface the failure once here so
      // the user understands why their music track is silent.
      const stretchErr = stretchInitFailure();
      if (stretchErr && this.audioTracks.size > 0) {
        runInAction(() => {
          this.audioTrackError = `Audio-track playback unavailable: ${stretchErr.message}`;
        });
      }

      runInAction(() => {
        this.state = 'playing';
        this.timeline = timeline;
        // Negative when starting from the lead-in (no cue); timeToX
        // maps it into the reserved pre-roll pixels.
        this.currentTime = anchorJot;
        this.cued = false;
      });
      this.startRaf();
      this.scheduleTailTimer(lastTime);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('[jotPlayer] play failed:', err);
      runInAction(() => {
        this.state = 'idle';
        this.errorMessage = message;
        this.timeline = EMPTY_TIMELINE;
        this.currentTime = 0;
      });
    }
  }

  /**
   * Freeze playback in place. Suspending the `AudioContext` halts its
   * clock, which transitively pauses every scheduled drum note and
   * (because `currentJotTime` derives from `ctx.currentTime`) the
   * playhead. Audio tracks are the exception now that they play through media
   * elements: a suspended context silences them but their media clock
   * keeps advancing, so they're paused explicitly here and realigned
   * in `resume()`. The wall-clock tail timer is also cleared here and
   * re-armed in `resume()`. No-op unless currently playing.
   */
  async pause(): Promise<void> {
    const ctx = this.audioGraph.ctx;
    if (this.state !== 'playing' || !ctx) return;
    if (this.endTimerId !== undefined) {
      window.clearTimeout(this.endTimerId);
      this.endTimerId = undefined;
    }
    this.stopRaf();
    // Pause the audio-track elements (no graph teardown) before freezing the
    // clock so they don't run on past the playhead while suspended.
    this.audioTrackController?.cancelSources();
    await ctx.suspend();
    runInAction(() => {
      this.state = 'paused';
    });
  }

  /**
   * Continue from a {@link pause}. `ctx.currentTime` froze while
   * suspended, so the drum schedule and cached tail time are still
   * correct relative to it, resuming the context is enough for those.
   * The audio-track elements were paused in `pause()`, so they're rescheduled
   * here at the current jot-time (anchored to the same frozen clock the
   * drums use, so the two stay together). No-op unless currently paused.
   */
  async resume(): Promise<void> {
    const ctx = this.audioGraph.ctx;
    if (this.state !== 'paused' || !ctx) return;
    await ctx.resume();
    runInAction(() => {
      this.state = 'playing';
    });
    if (this.audioTrackController) {
      const now = ctx.currentTime;
      const jotOffset = this.currentJotTime(now);
      this.audioTrackController.scheduleAll(
        this.audioTracks.values(),
        now,
        jotOffset,
        this.playbackSpeed,
        this.driftMap,
        (id) => audioTrackGainUnder(id, this.currentAudioTrackFilter)
      );
    }
    this.startRaf();
    this.scheduleTailTimer(this.tailAudioTime);
  }

  stop(): void {
    if (this.endTimerId !== undefined) {
      window.clearTimeout(this.endTimerId);
      this.endTimerId = undefined;
    }
    this.stopRaf();
    // Tear down any output capture so its 10ms interval + the analyser wired
    // into the master bus don't outlive playback. stopOutputCapture keeps the
    // recorded series, so a pending reader still gets it; it's idempotent when
    // no capture is active (the production default).
    this.stopOutputCapture();
    this.scheduler.cancelScheduledStops();
    this.scheduler.stopDrums();
    this.audioTrackController?.dispose();
    this.audioTrackController = undefined;
    this.events = [];
    this.startJotTime = 0;
    // `songLeadIn` needs no reset here: it lives on PlaybackStore (the
    // loaded jot's offset, seeded + live-tuned by PlaybackPresenter), not on
    // the engine, so it naturally survives stop()/replay. The store re-seeds
    // it when a different jot is loaded.
    this.pendingStartSec = undefined;
    runInAction(() => {
      if (this.state !== 'idle') this.state = 'idle';
      this.timeline = EMPTY_TIMELINE;
      this.currentTime = 0;
      this.cued = false;
    });
  }

  /**
   * Replace the end-of-playback fallback timer with one that fires
   * `PLAYBACK_TAIL_SECONDS` after the last scheduled note's AUDIO time
   * (which already accounts for speed because scheduleEvents bakes
   * 1/speed into each event's audio time). Always call after rescheduling:
   * the existing timer is cleared first so back-to-back speed changes
   * don't accumulate stale callbacks.
   */
  private scheduleTailTimer(drumsLastAudioTime: number): void {
    if (this.endTimerId !== undefined) {
      window.clearTimeout(this.endTimerId);
      this.endTimerId = undefined;
    }
    const ctx = this.audioGraph.ctx;
    if (!ctx) return;
    // The drum scheduler's last note isn't the only thing keeping
    // playback alive; loaded audio tracks play through `BufferSource`
    // nodes that run independently to their buffer ends. If the user
    // mutes / solos all drums (e.g. solos an audio track to ear-check
    // a stem), `drumsLastAudioTime` collapses to the play anchor and
    // we'd otherwise call `stop()` after `PLAYBACK_TAIL_SECONDS` while
    // the audio tracks are still mid-stream. Take the max with the
    // longest currently-playing audio track's end time so the tail
    // timer outlives whichever side is still producing sound.
    const lastAudioTime = Math.max(drumsLastAudioTime, this.computeAudioTracksEndTime());
    this.tailAudioTime = lastAudioTime;
    const tailMs = Math.max(
      0,
      (lastAudioTime - ctx.currentTime + PLAYBACK_TAIL_SECONDS) * 1000
    );
    this.endTimerId = window.setTimeout(() => {
      this.stop();
    }, tailMs);
  }

  /** Audio-context time at which the longest currently-scheduled audio
   * track will run out of buffer, computed from the most recent
   * play / resume / `setPlaybackSpeed` anchor (`startContextTime` +
   * `startJotTime`). Returns 0 when no audio tracks are loaded; the
   * caller takes a `Math.max` with the drum scheduler's last note time,
   * so 0 means "audio doesn't constrain the tail". */
  private computeAudioTracksEndTime(): number {
    if (!this.audioGraph.ctx || this.audioTracks.size === 0) return 0;
    // Mirror `AudioTrackPlaybackController.scheduleOne`'s anchoring so the end
    // time here matches when the underlying buffer actually stops: the buffer
    // clamps to t=0 and the output is delayed by `leadInDelaySec` when the
    // start sits before the audio's own t=0 (playhead in the pre-roll / virtual
    // lead-in). Omitting the delay would under-count the tail and auto-stop
    // could fire before the audio finishes.
    const rawInput = this.driftMap.jotToMedia(this.startJotTime);
    const mediaOffset = Math.max(0, rawInput);
    const speed = this.playbackSpeed;
    const leadInDelaySec = rawInput < 0 ? -rawInput / speed : 0;
    let maxEnd = 0;
    for (const track of this.audioTracks.values()) {
      const remaining = track.durationSec - mediaOffset;
      if (remaining <= 0) continue;
      const end = this.startContextTime + leadInDelaySec + remaining / speed;
      if (end > maxEnd) maxEnd = end;
    }
    return maxEnd;
  }

  /**
   * Schedule every event whose source time is >= `fromOffset` (in jot
   * seconds) onto the drum kit via {@link DrumScheduler}. Thin wrapper
   * that feeds the scheduler the transport's live event list, drift map,
   * speed, and pulled mute/solo filter, and reports whether the transport
   * is already playing (which governs the all-notes-dropped guard).
   *
   * Returns the latest audio context time at which a note was scheduled
   * (or `audioStartTime` if nothing scheduled) so the caller can compute
   * when to drop back to idle.
   */
  private scheduleEvents(fromOffset: number, audioStartTime: number): number {
    return this.scheduler.scheduleEvents(
      this.events,
      fromOffset,
      audioStartTime,
      this.driftMap,
      this.playbackSpeed,
      this.currentFilter,
      this.state === 'playing',
    );
  }

  private startRaf(): void {
    const tick = () => {
      const ctx = this.audioGraph.ctx;
      if (this.state !== 'playing' || !ctx) {
        this.rafId = undefined;
        return;
      }
      // Audio tracks now play through the Signalsmith Stretch worklet,
      // which consumes samples on the audio thread in lockstep with
      // the AudioContext clock the drum scheduler uses; no drift
      // subsystem needed at any speed.
      this.advancePlayhead(this.currentJotTime(ctx.currentTime));
      this.rafId = window.requestAnimationFrame(tick);
    };
    this.rafId = window.requestAnimationFrame(tick);
  }

  /** Advance the visual playhead to `jotTime`. An @action (via makeAutoObservable)
   *  so the 120fps RAF loop reuses one action wrapper instead of allocating a
   *  fresh `runInAction` closure every frame. */
  private advancePlayhead(jotTime: number): void {
    // Allow negative jot time during the lead-in so the playhead travels the
    // reserved pre-roll space (timeToX maps it into the lead-in pixels). Clamp
    // at the rendered left edge (`fullLeadIn` = the first bar's startSec, incl.
    // the view's virtual lead-in) so it can't run off the left of the lead-in.
    // The user's `audioLatencyMs` fine-tune and the auto-detected
    // `internalLatencyMs` baseline are summed; both shift the visual ahead of
    // the audio clock to compensate for perceived audio/visual sync drift.
    const latencyShiftSec = this.avSync.latencyShiftSec;
    const fullLeadIn = this.timeline.bars[0]?.startSec ?? this.songLeadInSec;
    this.currentTime = Math.max(jotTime + latencyShiftSec, fullLeadIn);
  }

  private stopRaf(): void {
    if (this.rafId !== undefined) {
      window.cancelAnimationFrame(this.rafId);
      this.rafId = undefined;
    }
  }

  /**
   * Construct (or return the existing) AudioContext without triggering
   * the smplr sample download. `loadAudioTrack` needs this, it has to
   * decode audio into the same context that will eventually play the
   * score, but shouldn't pay the ~150KB drum-samples fetch just to
   * attach a track.
   */
  private ensureAudioContext(): AudioContext {
    return this.audioGraph.ensureContext(
      this.masterVolume,
      this.audioMasterAudible ? this.audioTrackMasterVolume : 0,
    );
  }

  /**
   * Kick off the soundfont load in the background. See
   * {@link SoundfontLoader.preload}. Called from the audio-track loaders so
   * the ~30 MB cache read + SF2 parse can overlap with the user's file
   * decode; by the time they hit Play, `ensureLoaded` short-circuits.
   */
  private preloadDrums(): void {
    this.loader.preload();
  }
}

export const jotPlayer = new JotPlayer();
