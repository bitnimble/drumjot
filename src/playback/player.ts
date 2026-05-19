/**
 * Browser playback of a Jot through an acoustic General MIDI drum kit
 * (the GeneralUser GS SoundFont — see {@link GeneralUserGsKit}).
 *
 * The ~30 MB `.sf2` is fetched through `ProgressCacheStorage`, so the
 * first play of a session streams it with a visible byte-progress bar
 * and it's then cached in the browser Cache API (instant on later
 * sessions).
 *
 * Lifecycle:
 *   - The `AudioContext` and drum kit are created on first `play()` —
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
 * Live mute / solo: the caller (`JotViewStore`) pushes a `Filter` via
 * `setFilter()` whenever the user toggles a row's M/S buttons. While
 * playback is in flight, `setFilter()` cancels every scheduled note and
 * re-schedules those whose audio time hasn't passed yet — so unmuting a
 * row brings it back in the same play, and muting silences it
 * immediately.
 *
 * Diagnostics: scheduling failures (no events extracted, no notes mapped
 * to drum samples on the current kit, etc.) are surfaced both as
 * `errorMessage` on the singleton and via `console` so the UI can show
 * a visible indicator and the operator can inspect details.
 */
import { makeAutoObservable, runInAction } from 'mobx';
import { RenderedJot } from 'src/jot';
import { jotToEvents, PlaybackEvent } from './events';
import { GeneralUserGsKit, KitInfo } from './gm_kit';
import { SampleLoadProgress } from './sample_storage';
import {
  decodeAudioTrackFile,
  decodeAudioTrackUrl,
  audioTrackGainUnder,
  PASSTHROUGH_AUDIO_TRACK_FILTER,
  AudioTrackFilter,
  AudioTrackId,
  AudioTrackPlaybackController,
  AudioTrack,
} from './audio_tracks';
import { buildTimeline, EMPTY_TIMELINE, JotTimeline } from './timeline';

export type PlayerState = 'idle' | 'loading' | 'playing' | 'paused';

export type PlayerFilter = {
  mutedPitches: ReadonlySet<string>;
  /**
   * When solo is active, ONLY these pitches are audible (others behave
   * as if muted). Soloed-AND-muted = muted; explicit mute always wins so
   * the user can keep solo on while temporarily silencing a soloed row.
   */
  soloedPitches: ReadonlySet<string>;
  /**
   * True when a solo is engaged *anywhere* — on a pitch row OR an
   * audio track. Solo is a single global mode shared across both
   * domains: as soon as the user solos any row, every non-soloed row
   * (drums *and* music) drops out. Computed by the store, which is the
   * only place that sees both the pitch and audio-track solo sets.
   */
  soloActive: boolean;
  /** Per-pitch volume multiplier in [0, 1]; missing = full (1). */
  volumes: ReadonlyMap<string, number>;
};

export const PASSTHROUGH_FILTER: PlayerFilter = {
  mutedPitches: new Set(),
  soloedPitches: new Set(),
  soloActive: false,
  volumes: new Map(),
};

export function isAudibleUnder(pitch: string, filter: PlayerFilter): boolean {
  if (filter.mutedPitches.has(pitch)) return false;
  if (filter.soloActive && !filter.soloedPitches.has(pitch)) return false;
  if ((filter.volumes.get(pitch) ?? 1) <= 0) return false;
  return true;
}

// GeneralUser GS GM SoundFont, pulled straight from its GitHub repo
// (raw.githubusercontent.com is CORS-open and sends Content-Length, so
// the byte-progress bar works). `main` is a moving ref — fine for now;
// pin to a commit if reproducibility matters later.
const GM_SOUNDFONT_URL =
  'https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/GeneralUser-GS.sf2';
// Cache API entry for the downloaded .sf2. Bump the suffix if the URL
// (or the chosen kit) changes so stale bytes aren't served from an old
// cache entry.
const GM_SOUNDFONT_CACHE = 'drumjot-generaluser-gs-v1';
// GM percussion lives in bank 128; each preset there is a different kit
// (GeneralUser GS: 0 = Standard, 8 = Room, 16 = Power, …). The exact set
// is discovered from the SoundFont and exposed via `drumKits`; the user
// picks one with `setDrumPreset`. 0 (Standard) is the initial choice.
const GM_DRUM_BANK = 128;
const DEFAULT_DRUM_PRESET = 0;
// SF2 percussion samples are recorded near full scale, so unlike the old
// synthesised DrumMachine path the kit already sits well against a
// full-scale audio track. Keep a dedicated routing node at unity (one
// place to trim/boost later) rather than the old +12 dB lift, which
// would clip these samples.
const DRUM_MASTER_GAIN = 1;
// Per-row loudness trim applied on top of the user's volume fader,
// keyed by DSL pitch letter ('k' = kick, 'h' = hi-hat, …). The GM
// SoundFont's hats are hot and the kick is weak relative to a real
// kit / backing track, so we duck the hats and lift the kick by
// default. Rows not listed play at their native velocity (1.0).
// Scaling velocity (not a GainNode) keeps accents/ghosts' relative
// dynamics intact and matches how the user volume fader already works.
const DEFAULT_PITCH_GAIN: Record<string, number> = {
  h: 0.6,
  k: 1.5,
};
// Small lead time so the first hit doesn't race the audio thread.
const SCHEDULE_LEAD_SECONDS = 0.05;
// Buffer added to the last event's time before flipping back to `idle`, so
// late-decaying samples (cymbals, open hats) aren't cut off visually.
const PLAYBACK_TAIL_SECONDS = 1.0;
// How often the rAF loop re-locks the audio-track media elements to the
// AudioContext clock. Sub-second so audible slip never accumulates,
// but far coarser than the frame rate — drift correction is a slow
// control loop, not a per-frame job.
const DRIFT_CHECK_INTERVAL_SECONDS = 0.5;
// If the SoundFont can't be fetched within this window we give up so the
// UI doesn't sit on "Loading…" forever — a typical local network failure
// mode that's otherwise invisible. Generous because it's a ~30 MB
// one-time download on a slow link (cached loads are instant); a cache
// hit resolves long before this.
const LOAD_TIMEOUT_SECONDS = 120;

type Drums = ReturnType<typeof GeneralUserGsKit>;
type StopFn = (time?: number) => void;

export class JotPlayer {
  state: PlayerState = 'idle';
  /** Last error surfaced to the UI; cleared the next time playback succeeds. */
  errorMessage: string | undefined;
  /**
   * Drum-sample download progress during the one-time soundfont fetch
   * (first play of a session that isn't cache-hot). `undefined` once
   * loaded or before any load has started. Observable so the transport
   * bar can render a small progress bar while `state === 'loading'`.
   */
  sampleLoadProgress: SampleLoadProgress | undefined;
  /**
   * Drum kits available in the SoundFont's percussion bank, for the kit
   * picker. Empty until the kit has loaded once (we only know the list
   * after the ~30 MB SoundFont is downloaded + parsed), so the UI hides
   * the dropdown until then.
   */
  drumKits: KitInfo[] = [];
  /**
   * Currently-selected drum preset. Used by `ensureLoaded` for the
   * initial load and updated by {@link setDrumPreset}; observable so the
   * picker reflects the active kit.
   */
  drumPreset: number = DEFAULT_DRUM_PRESET;
  /** Seconds since the current `play()` started (in JOT time — already
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
   * native tempo; 0.5 = half speed (notes still sound at the same pitch
   * because we space scheduled `drums.start` times further apart rather
   * than touching sample playback rate). Persists across plays.
   */
  playbackSpeed: number = 1;

  /**
   * Audio tracks loaded by the user — any number (a ParaDB pack's
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
   * Latest filename per track id, displayed in the toolbar status. Kept
   * separate from `audioTracks` so the UI knows there was an in-flight load
   * even before the buffer finishes decoding.
   */
  audioTrackError: string | undefined;

  private ctx: AudioContext | undefined;
  private drums: Drums | undefined;
  /**
   * Master gain the drum kit is routed through (see
   * {@link DRUM_MASTER_GAIN}). Created once alongside `drums` and lives
   * for the AudioContext's lifetime; the context is never closed mid-
   * session so there's no teardown to do here.
   */
  private drumGain: GainNode | undefined;
  private audioTrackController: AudioTrackPlaybackController | undefined;
  private currentAudioTrackFilter: AudioTrackFilter = PASSTHROUGH_AUDIO_TRACK_FILTER;
  /** Cached start offset (seconds) of the currently-playing jot, so
   * `setPlaybackSpeed` can re-anchor audio tracks to the same audio position
   * the new rate started from. */
  private startOffsetSec: number = 0;
  /**
   * AudioContext time of the playback anchor (updated at `play()` and
   * whenever `setPlaybackSpeed` re-anchors mid-flight) — `currentJotTime`
   * is computed from this plus the elapsed real time times speed.
   */
  private startContextTime: number = 0;
  /** Jot-time value at `startContextTime`; non-zero after a mid-flight
   * speed change so the playhead doesn't snap back to 0. */
  private startJotTime: number = 0;
  private endTimerId: number | undefined;
  /**
   * AudioContext time of the last scheduled note's start. Retained so
   * `resume()` can re-arm the end-of-playback fallback timer — that
   * timer is a wall-clock `setTimeout`, so `suspend()` (which only
   * freezes the audio clock) would otherwise let it fire while paused.
   */
  private tailAudioTime: number = 0;
  private rafId: number | undefined;
  /**
   * AudioContext time of the last drift check, so the rAF loop can
   * throttle {@link AudioTrackPlaybackController.correctDrift} to
   * {@link DRIFT_CHECK_INTERVAL_SECONDS} instead of running it every frame.
   */
  private lastDriftCheckTime: number = 0;
  /**
   * Per-note stop callbacks returned by `drums.start()`. `drums.stop()`
   * on its own only halts notes that have already begun sounding — notes
   * scheduled for future audio-context times keep firing until they
   * reach their start, so we have to invoke each scheduled note's stop
   * function explicitly to make Stop actually stop playback.
   */
  private scheduledStops: StopFn[] = [];
  /**
   * The full event list for the currently-playing jot, retained so that
   * `setFilter` can re-derive the scheduled subset on a mute/solo
   * toggle without having to re-walk the layout.
   */
  private events: PlaybackEvent[] = [];
  private currentFilter: PlayerFilter = PASSTHROUGH_FILTER;
  /**
   * Jot-time (seconds) the next `play()` should start from, set by a
   * click-to-seek while idle. `undefined` means "start from the
   * beginning" (honouring `startOffset` lead-in as before).
   */
  private pendingStartSec: number | undefined;

  constructor() {
    makeAutoObservable(this);
  }

  /**
   * Update the mute/solo filter. Stored unconditionally; if playback is
   * in flight OR paused, every scheduled note is cancelled and the
   * remaining events (those whose audio time hasn't elapsed) are
   * re-scheduled against the new filter — so toggling M or S takes
   * effect immediately, including bringing previously-muted rows back
   * in mid-song.
   *
   * The paused case matters: pause → toggle a row's M → resume is a
   * natural practice workflow, and `resume()` only reschedules audio tracks,
   * not drum events. Without re-filtering here the pre-mute drum
   * schedule survives the pause and the muted row keeps sounding on
   * resume. `ctx.currentTime` is frozen while paused, which is exactly
   * the anchor we want there — the rescheduled notes line up relative
   * to it and come alive when the context resumes (same approach as
   * `seek`).
   */
  setFilter(filter: PlayerFilter): void {
    this.currentFilter = filter;
    if ((this.state !== 'playing' && this.state !== 'paused') || !this.ctx) return;

    const now = this.ctx.currentTime;
    const jotOffset = this.currentJotTime(now);
    // `cancelScheduledStops()` alone is not enough: the per-note stopFns
    // are no-ops for voices smplr hasn't instantiated yet (it only
    // builds the BufferSourceNode when the scheduled time arrives), so
    // the still-pending notes — including the row the user just muted —
    // would keep firing alongside the rescheduled set and the toggle
    // would appear to do nothing. `drums.stop()` flushes the whole
    // pending queue; same reason `setPlaybackSpeed` calls it.
    this.cancelScheduledStops();
    this.drums?.stop();
    const lastTime = this.scheduleEvents(jotOffset, now);
    // Paused: resume() re-arms the end-of-playback timer from
    // tailAudioTime, so keep it current with the re-filtered schedule
    // (an unmute can extend the tail; a mute can shorten it).
    if (this.state === 'paused') this.tailAudioTime = lastTime;
  }

  /**
   * Update the audio-track mute/solo filter. Tracks already playing get their
   * `GainNode.gain` toggled to 0 / 1 immediately — no source recreation
   * needed, so the change is sample-accurate without the click-risk of
   * stopping and restarting a `BufferSourceNode` mid-decay.
   */
  setAudioTrackFilter(filter: AudioTrackFilter): void {
    this.currentAudioTrackFilter = filter;
    if (this.audioTrackController) {
      this.audioTrackController.applyAudibility((id) => audioTrackGainUnder(id, filter));
    }
  }

  /** Fresh, never-reused audio-track id. Load order ⇒ ascending ids. */
  private allocateAudioTrackId(): AudioTrackId {
    return `track-${++this.audioTrackIdCounter}`;
  }

  /**
   * Load an audio file (a ParaDB pack track, a transcriber FLAC, any
   * audio the user drops in) as a new track and return its allocated id.
   * Every call appends a track — there is no replace-by-name slot any
   * more, so loading N files yields N independent tracks. Decoding
   * shares the `AudioContext` with the drum machine, so this method
   * constructs the context even before playback starts — meaning the
   * call must happen inside a user gesture on some browsers (the click
   * that triggered the file picker inherits the gesture grant).
   */
  async loadAudioTrack(file: File): Promise<AudioTrackId> {
    runInAction(() => {
      this.audioTrackError = undefined;
    });
    try {
      const ctx = this.ensureAudioContext();
      const { buffer, mono, objectUrl } = await decodeAudioTrackFile(ctx, file);
      const id = this.allocateAudioTrackId();
      this.installAudioTrack(id, file.name, buffer, mono, objectUrl);
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
  async loadAudioTrackFromUrl(url: string, filename: string): Promise<AudioTrackId> {
    runInAction(() => {
      this.audioTrackError = undefined;
    });
    try {
      const ctx = this.ensureAudioContext();
      const { buffer, mono, objectUrl } = await decodeAudioTrackUrl(ctx, url);
      const id = this.allocateAudioTrackId();
      this.installAudioTrack(id, filename, buffer, mono, objectUrl);
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
    // Tear down the removed track's element/node so it leaves no
    // dangling media; the remaining tracks keep playing untouched
    // (each owns its own element, so no reschedule is needed).
    this.audioTrackController?.dropAudioTrack(id);
    URL.revokeObjectURL(prev.objectUrl);
  }

  private installAudioTrack(
    id: AudioTrackId,
    filename: string,
    buffer: AudioBuffer,
    mono: Float32Array,
    objectUrl: string,
  ): void {
    const prev = this.audioTracks.get(id);
    const track: AudioTrack = {
      id,
      filename,
      buffer,
      mono,
      objectUrl,
      durationSec: buffer.duration,
    };
    runInAction(() => {
      this.audioTracks.set(id, track);
    });
    // If playback is in flight, start the new track in sync with the
    // existing schedule so the user hears it appear at the current
    // playhead without having to stop+restart.
    if (this.state === 'playing' && this.ctx && this.audioTrackController) {
      const now = this.ctx.currentTime;
      const jotOffset = this.currentJotTime(now);
      this.audioTrackController.scheduleAll(
        [track],
        now,
        jotOffset,
        this.playbackSpeed,
        this.startOffsetSec,
        (sid) => audioTrackGainUnder(sid, this.currentAudioTrackFilter),
      );
    }
    // The controller has now repointed its element at the new blob
    // (ensureSlot rebuilds on a URL change), so the old one is safe to
    // release. Replacing the same id with the same bytes can't happen
    // (each decode mints a fresh URL), so the guard is just defensive.
    if (prev && prev.objectUrl !== objectUrl) {
      URL.revokeObjectURL(prev.objectUrl);
    }
  }

  /**
   * Switch the active drum kit to another preset in the percussion bank.
   *
   * The preset is always recorded so the *next* load uses it. If the kit
   * is already loaded, the swap happens immediately (no refetch — the
   * parsed SoundFont is retained); it takes effect on every subsequently
   * scheduled note, so changing kit mid-play is fine. Failures surface
   * via `errorMessage` rather than throwing into the UI handler.
   */
  async setDrumPreset(preset: number): Promise<void> {
    runInAction(() => {
      this.drumPreset = preset;
    });
    if (!this.drums) return; // not loaded yet — ensureLoaded will use it
    try {
      await this.drums.loadPreset(preset);
      console.log(`[jotPlayer] switched to drum preset ${preset}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('[jotPlayer] drum preset switch failed:', err);
      runInAction(() => {
        this.errorMessage = `Could not switch drum kit: ${message}`;
      });
    }
  }

  /**
   * Set the tempo multiplier. Takes effect immediately, including during
   * playback: we re-anchor `startContextTime` / `startJotTime` to "now",
   * cancel every scheduled note, and reschedule remaining notes at their
   * new audio times under the new spacing.
   *
   * Sample pitch is unchanged — drum samples still play at native rate.
   * Slowing down just spaces successive `drums.start` calls further
   * apart, which is exactly what you want for practicing along to a
   * complex fill at half speed.
   */
  setPlaybackSpeed(speed: number): void {
    const clamped = Math.max(0.1, Math.min(2.0, speed));
    if (this.state !== 'playing' || !this.ctx) {
      runInAction(() => {
        this.playbackSpeed = clamped;
      });
      return;
    }
    const now = this.ctx.currentTime;
    // Snapshot current jot time AT THE OLD SPEED before mutating, so the
    // playhead doesn't jump when the user changes speed mid-song.
    const jotOffset = this.currentJotTime(now);

    // Kill the old schedule before laying down the new one. The
    // per-voice stopFns returned by `drums.start({ time: <future> })`
    // don't actually cancel a voice that hasn't begun yet — smplr only
    // instantiates the BufferSourceNode when the scheduled time
    // arrives, so calling stopFn early is a no-op. The global
    // `drums.stop()` clears the entire pending queue, which is what we
    // need here. Without this, the old (still-pending) notes play
    // alongside the rescheduled ones and the speed change appears to
    // have no effect.
    this.cancelScheduledStops();
    this.drums?.stop();
    this.audioTrackController?.cancelSources();

    runInAction(() => {
      this.playbackSpeed = clamped;
    });
    this.startContextTime = now;
    this.startJotTime = jotOffset;
    const lastTime = this.scheduleEvents(jotOffset, now);
    if (this.audioTrackController) {
      this.audioTrackController.scheduleAll(
        this.audioTracks.values(),
        now,
        jotOffset,
        clamped,
        this.startOffsetSec,
        (id) => audioTrackGainUnder(id, this.currentAudioTrackFilter),
      );
    }
    this.scheduleTailTimer(lastTime);
  }

  /**
   * Map an absolute AudioContext time to its jot-time position, taking
   * the current `playbackSpeed` (and any prior speed-change anchor) into
   * account.
   */
  private currentJotTime(audioTime: number): number {
    return this.startJotTime + (audioTime - this.startContextTime) * this.playbackSpeed;
  }

  /**
   * Move the playhead (and, if audio is running, the playback position)
   * to `seconds` of jot time.
   *
   *  - **idle**: build the timeline so the playhead can be drawn, park
   *    it at the cued position, and remember it so the next `play()`
   *    starts there. No audio is touched (no CDN sample fetch).
   *  - **playing**: re-anchor and reschedule everything from the new
   *    position — a live scrub. Mirrors `setPlaybackSpeed`'s re-anchor.
   *  - **paused**: same re-anchor, but against the frozen AudioContext
   *    clock; the rescheduled notes stay silent until `resume()`, which
   *    re-arms the tail timer from `tailAudioTime`.
   */
  seek(rendered: RenderedJot, seconds: number): void {
    // The recording's drumless lead-in is jot time [-startOffsetSec, 0).
    // Allow scrubbing back into it instead of clamping at 0 so the user
    // can play the intro. `this.startOffsetSec` is only populated while
    // playing, so derive the bound from the jot's metadata (same source
    // play() uses) to keep idle and live seeks consistent.
    const rawOffset = rendered.resolved.globalMetadata.startOffset;
    const leadInSec =
      typeof rawOffset === 'number' && rawOffset > 0 ? rawOffset : 0;
    let target = Math.max(seconds, -leadInSec);

    if (this.state === 'idle') {
      const timeline = buildTimeline(rendered);
      if (timeline.bars.length === 0) return;
      target = Math.min(target, timeline.totalDurationSec);
      this.pendingStartSec = target;
      this.startJotTime = target;
      runInAction(() => {
        this.timeline = timeline;
        this.currentTime = target;
        this.cued = true;
      });
      return;
    }

    if (!this.ctx) return;
    const dur = this.timeline.totalDurationSec;
    if (dur > 0) target = Math.min(target, dur);

    // ctx.currentTime is frozen while paused, which is exactly the
    // anchor we want there: scheduled notes line up relative to it and
    // come alive when the context resumes.
    const now = this.ctx.currentTime;
    this.cancelScheduledStops();
    this.drums?.stop();
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
        this.startOffsetSec,
        (id) => audioTrackGainUnder(id, this.currentAudioTrackFilter),
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

  /**
   * Drop an idle click-to-seek cue (playhead parked before pressing
   * Play). No-op while playing / paused, so loading a new score
   * mid-playback doesn't disturb the transport. Called when the
   * current jot is replaced so a stale cued playhead doesn't linger.
   */
  clearCue(): void {
    if (this.state !== 'idle' || !this.cued) return;
    this.pendingStartSec = undefined;
    runInAction(() => {
      this.cued = false;
      this.timeline = EMPTY_TIMELINE;
      this.currentTime = 0;
    });
  }

  async play(rendered: RenderedJot): Promise<void> {
    // Capture the click-to-seek cue before stop() clears it.
    const cueSec = this.pendingStartSec;
    this.stop();
    runInAction(() => {
      this.state = 'loading';
      this.errorMessage = undefined;
    });

    try {
      const { drums, ctx } = await this.ensureLoaded();

      // Chrome / Safari sometimes start a freshly-constructed AudioContext
      // suspended even when the constructor ran inside a user gesture, so
      // we always re-check before scheduling. The resume() promise inherits
      // the user-gesture grant from the click that called play().
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }

      this.events = jotToEvents(rendered);
      if (this.events.length === 0) {
        throw new Error('No playable notes in this jot.');
      }

      const timeline = buildTimeline(rendered);
      const audioStartTime = ctx.currentTime + SCHEDULE_LEAD_SECONDS;
      // `startOffset` on globalMetadata is the audio-time of jot-time 0 in
      // the original recording — i.e. how much silence / non-drum intro
      // preceded the first detected beat. Anchor jot time at -offset so
      // the rAF loop's `jotTime > 0 ? jotTime : 0` clamp parks the
      // playhead at position 0 during the lead-in, then it advances
      // naturally once the first hit fires. Scheduling from -offset
      // pushes every event's audio time forward by offset/speed so the
      // drums come in delayed to match the original.
      const rawOffset = rendered.resolved.globalMetadata.startOffset;
      const startOffsetSec =
        typeof rawOffset === 'number' && rawOffset > 0 ? rawOffset : 0;
      // Start from the click-to-seek cue if one is pending, otherwise
      // from -startOffsetSec so the rAF clamp parks the playhead at 0
      // through the recording's lead-in (unchanged default behaviour).
      // A cue (including a negative one parked in the lead-in) is
      // honoured, clamped into [-startOffsetSec, total]. No cue still
      // means "start from the top of the lead-in" (-startOffsetSec).
      const anchorJot =
        cueSec !== undefined
          ? Math.min(Math.max(cueSec, -startOffsetSec), timeline.totalDurationSec)
          : -startOffsetSec;
      this.startContextTime = audioStartTime;
      this.startJotTime = anchorJot;
      this.startOffsetSec = startOffsetSec;
      const lastTime = this.scheduleEvents(anchorJot, audioStartTime);

      // Audio tracks play through the same AudioContext so they share the
      // clock with the drum scheduler. The controller is recreated on
      // every play() to drop any residual nodes from the previous run.
      this.audioTrackController?.dispose();
      this.audioTrackController = new AudioTrackPlaybackController(ctx);
      this.audioTrackController.scheduleAll(
        this.audioTracks.values(),
        audioStartTime,
        anchorJot,
        this.playbackSpeed,
        startOffsetSec,
        (id) => audioTrackGainUnder(id, this.currentAudioTrackFilter),
      );

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
    if (this.state !== 'playing' || !this.ctx) return;
    if (this.endTimerId !== undefined) {
      window.clearTimeout(this.endTimerId);
      this.endTimerId = undefined;
    }
    this.stopRaf();
    // Pause the audio-track elements (no graph teardown) before freezing the
    // clock so they don't run on past the playhead while suspended.
    this.audioTrackController?.cancelSources();
    await this.ctx.suspend();
    runInAction(() => {
      this.state = 'paused';
    });
  }

  /**
   * Continue from a {@link pause}. `ctx.currentTime` froze while
   * suspended, so the drum schedule and cached tail time are still
   * correct relative to it — resuming the context is enough for those.
   * The audio-track elements were paused in `pause()`, so they're rescheduled
   * here at the current jot-time (anchored to the same frozen clock the
   * drums use, so the two stay together). No-op unless currently paused.
   */
  async resume(): Promise<void> {
    if (this.state !== 'paused' || !this.ctx) return;
    await this.ctx.resume();
    runInAction(() => {
      this.state = 'playing';
    });
    if (this.audioTrackController) {
      const now = this.ctx.currentTime;
      const jotOffset = this.currentJotTime(now);
      this.audioTrackController.scheduleAll(
        this.audioTracks.values(),
        now,
        jotOffset,
        this.playbackSpeed,
        this.startOffsetSec,
        (id) => audioTrackGainUnder(id, this.currentAudioTrackFilter),
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
    this.cancelScheduledStops();
    this.drums?.stop();
    this.audioTrackController?.dispose();
    this.audioTrackController = undefined;
    this.events = [];
    this.startJotTime = 0;
    this.startOffsetSec = 0;
    this.lastDriftCheckTime = 0;
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
   * 1/speed into each event's audio time). Always call after rescheduling
   * — the existing timer is cleared first so back-to-back speed changes
   * don't accumulate stale callbacks.
   */
  private scheduleTailTimer(lastAudioTime: number): void {
    if (this.endTimerId !== undefined) {
      window.clearTimeout(this.endTimerId);
      this.endTimerId = undefined;
    }
    if (!this.ctx) return;
    this.tailAudioTime = lastAudioTime;
    const tailMs = Math.max(
      0,
      (lastAudioTime - this.ctx.currentTime + PLAYBACK_TAIL_SECONDS) * 1000,
    );
    this.endTimerId = window.setTimeout(() => {
      this.stop();
    }, tailMs);
  }

  /**
   * Schedule every event whose source time is >= `fromOffset` (in jot
   * seconds) to play at `audioStartTime + (event.time - fromOffset) /
   * playbackSpeed` on the audio context. The speed division spaces
   * successive hits further apart in real time at sub-1x speeds without
   * touching sample pitch — drums still sound like drums at half speed,
   * they just play more slowly.
   *
   * Events filtered out by `currentFilter` are skipped.
   *
   * Returns the latest audio context time at which a note was scheduled
   * (or `audioStartTime` if nothing scheduled) so the caller can
   * compute when to drop back to idle.
   */
  private scheduleEvents(fromOffset: number, audioStartTime: number): number {
    const drums = this.drums;
    if (!drums) return audioStartTime;

    let lastTime = audioStartTime;
    let scheduled = 0;
    let mutedFiltered = 0;
    const speed = this.playbackSpeed;

    for (const ev of this.events) {
      if (ev.time < fromOffset) continue;
      if (!isAudibleUnder(ev.pitch, this.currentFilter)) {
        mutedFiltered++;
        continue;
      }
      // The GeneralUser GS kit is keyed by GM percussion MIDI note
      // number (36 = kick, 38 = snare, …) — exactly what `jotToEvents`
      // emits — so trigger the note directly; the SF2 zones map each
      // note to its own sample, no kit-group resolution needed.
      //
      // Per-row volume scales the note's velocity. smplr maps velocity
      // to gain, so a 0.5 fader roughly halves the row's loudness while
      // accents/ghosts (already baked into ev.velocity) keep their
      // relative dynamics. isAudibleUnder already rejected vol <= 0.
      // The DEFAULT_PITCH_GAIN trim stacks on top so hats/kick sit
      // right out of the box even before the user touches a fader.
      const vol = this.currentFilter.volumes.get(ev.pitch) ?? 1;
      const defaultGain = DEFAULT_PITCH_GAIN[ev.pitch] ?? 1;
      const velocity = Math.max(
        1,
        Math.min(127, Math.round(ev.velocity * vol * defaultGain)),
      );
      const t = audioStartTime + (ev.time - fromOffset) / speed;
      const stopFn = drums.start({ note: ev.midiNote, time: t, velocity });
      this.scheduledStops.push(stopFn);
      scheduled++;
      if (t > lastTime) lastTime = t;
    }

    console.log(
      `[jotPlayer] scheduled ${scheduled}/${this.events.length} events ` +
        `(filtered by mute/solo: ${mutedFiltered})`,
    );

    if (
      scheduled === 0 &&
      this.events.length - mutedFiltered > 0 &&
      this.state !== 'playing'
    ) {
      // Nothing scheduled but notes survived the mute/solo filter — a
      // genuine, otherwise-invisible failure (e.g. the kit loaded with
      // no usable zones). Notes dropped purely by an active mute/solo
      // are instead a valid silent-start state, handled by the caller
      // exactly like a live reschedule.
      throw new Error(
        `None of ${this.events.length - mutedFiltered} audible notes could be ` +
          `scheduled on the GeneralUser GS kit. See console for the breakdown.`,
      );
    }
    return lastTime;
  }

  private cancelScheduledStops(): void {
    for (const fn of this.scheduledStops) {
      try {
        fn();
      } catch (err) {
        // A stop fn for a note that already finished may throw; ignore.
        console.debug('[jotPlayer] stopFn threw:', err);
      }
    }
    this.scheduledStops = [];
  }

  private startRaf(): void {
    const tick = () => {
      if (this.state !== 'playing' || !this.ctx) {
        this.rafId = undefined;
        return;
      }
      const now = this.ctx.currentTime;
      const jotTime = this.currentJotTime(now);
      // Re-lock the audio tracks to the AudioContext clock (the clock the
      // drums are scheduled on) so a media element's independent clock can't
      // slew the backing track away from the score over a long take.
      // `expectedMediaSec` mirrors audio_tracks.ts's mediaOffset = max(0,
      // jot + startOffset) so the target matches where the track started.
      if (this.audioTrackController && now - this.lastDriftCheckTime >= DRIFT_CHECK_INTERVAL_SECONDS) {
        this.lastDriftCheckTime = now;
        const expectedMediaSec = Math.max(0, jotTime + this.startOffsetSec);
        this.audioTrackController.correctDrift(expectedMediaSec, this.playbackSpeed);
      }
      runInAction(() => {
        // Allow negative jot time during the recording's lead-in so the
        // playhead travels the reserved pre-roll space (timeToX maps
        // [-startOffset, 0) into the lead-in pixels). Clamp at
        // -startOffset so it can't run off the left of the lead-in.
        this.currentTime = Math.max(jotTime, -this.startOffsetSec);
      });
      this.rafId = window.requestAnimationFrame(tick);
    };
    this.rafId = window.requestAnimationFrame(tick);
  }

  private stopRaf(): void {
    if (this.rafId !== undefined) {
      window.cancelAnimationFrame(this.rafId);
      this.rafId = undefined;
    }
  }

  /**
   * Construct (or return the existing) AudioContext without triggering
   * the smplr sample download. `loadAudioTrack` needs this — it has to
   * decode audio into the same context that will eventually play the
   * score — but shouldn't pay the ~150KB drum-samples fetch just to
   * attach a track.
   */
  private ensureAudioContext(): AudioContext {
    if (this.ctx) return this.ctx;
    const Ctx: typeof AudioContext | undefined =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) {
      throw new Error('Web Audio is not available in this browser.');
    }
    this.ctx = new Ctx();
    return this.ctx;
  }

  private async ensureLoaded(): Promise<{ drums: Drums; ctx: AudioContext }> {
    if (this.drums && this.ctx) return { drums: this.drums, ctx: this.ctx };

    const ctx = this.ensureAudioContext();
    const drumGain = ctx.createGain();
    drumGain.gain.value = DRUM_MASTER_GAIN;
    drumGain.connect(ctx.destination);
    this.drumGain = drumGain;
    const drums = GeneralUserGsKit(ctx, {
      url: GM_SOUNDFONT_URL,
      cacheName: GM_SOUNDFONT_CACHE,
      bank: GM_DRUM_BANK,
      preset: this.drumPreset,
      destination: drumGain,
      // Byte progress for the (large, one-time) .sf2 download; the
      // storage layer also serves it from the Cache API on later
      // sessions, in which case this fires once with fromCache = true.
      onProgress: (p) => {
        runInAction(() => {
          this.sampleLoadProgress = p;
        });
      },
    });

    // Race the load against a timeout so a stuck download surfaces as an
    // error instead of an infinite "Loading…" state.
    try {
      await Promise.race([
        drums.load,
        new Promise<never>((_, reject) =>
          window.setTimeout(
            () =>
              reject(
                new Error(
                  `Drum kit failed to load within ${LOAD_TIMEOUT_SECONDS}s — ` +
                    `check network access to raw.githubusercontent.com from the browser.`,
                ),
              ),
            LOAD_TIMEOUT_SECONDS * 1000,
          ),
        ),
      ]);
    } finally {
      // Clear the progress readout whether we finished or timed out so a
      // stale bar doesn't linger; the UI also gates on `state` anyway.
      runInAction(() => {
        this.sampleLoadProgress = undefined;
      });
    }

    this.drums = drums;
    runInAction(() => {
      this.drumKits = drums.availableKits;
    });
    console.log('[jotPlayer] GeneralUser GS kit loaded');
    return { drums, ctx };
  }
}

export const jotPlayer = new JotPlayer();
