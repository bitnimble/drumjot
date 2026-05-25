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
// Brief settle window after `drums.load` resolves on the cold path. The
// SoundFont is parsed but smplr's per-note pipeline (zone lookup, voice
// allocation) needs a moment before its first scheduled hit lands
// reliably; without this the very first note of a fresh-session play
// occasionally drops. Only paid on the one-time load (`ensureLoaded`
// short-circuits on subsequent plays), so normal play latency is
// unchanged.
const POST_LOAD_SETTLE_SECONDS = 0.2;

// Per CLEANROOM_SPEC §7.4: allowed playback speeds. The transport-bar
// dropdown picks from this list (see `PLAYBACK_SPEEDS` in
// jot_view/playback.tsx); the player snaps any value reaching it
// programmatically to the nearest in-spec speed.
const SUPPORTED_PLAYBACK_SPEEDS: readonly number[] = [0.25, 0.5, 0.75, 1.0, 1.25];

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
   * Which sub-phase of the soundfont load we're in:
   *   - `connecting`:  request issued, waiting for the first byte.
   *   - `downloading`: bytes streaming in (or being read from cache).
   *   - `decoding`:    bytes done; smplr is parsing the .sf2.
   * Lets the toolbar tell the user *what* is happening, not just *how
   * much*. `undefined` outside of `state === 'loading'`.
   */
  sampleLoadPhase: 'connecting' | 'downloading' | 'decoding' | undefined;
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
   * Page-wide master fader (0..1). Scales *everything* — drums and audio
   * tracks alike — because its GainNode is the last stage before
   * `ctx.destination`. Observable so the slider reflects the live value.
   */
  masterVolume: number = DEFAULT_MASTER_VOLUME;
  /** Master fader for all drum/instrument rows together (0..1). */
  drumMasterVolume: number = DEFAULT_VOLUME;
  /** Master fader for all audio tracks together (0..1). */
  audioTrackMasterVolume: number = DEFAULT_VOLUME;

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
  /**
   * Audio-graph bus, built once with the AudioContext and living for its
   * lifetime:
   *
   *   smplr drum voices ─▶ drumGain ──┐
   *   per audio-track GainNode ─▶ audioBusGain ──┤
   *                                              ├─▶ pageGain ─▶ destination
   *
   * `drumGain` carries the all-drums master fader, `audioBusGain` the
   * all-audio-tracks fader, and `pageGain` the whole-page fader (last
   * stage, so it scales both). They exist as soon as the context does
   * (audio tracks can be loaded before the first play), and their
   * `gain.value` tracks the observable `*Volume` fields.
   */
  private pageGain: GainNode | undefined;
  private audioBusGain: GainNode | undefined;
  private audioTrackController: AudioTrackPlaybackController | undefined;
  private currentAudioTrackFilter: AudioTrackFilter = PASSTHROUGH_AUDIO_TRACK_FILTER;
  /**
   * Drum-track ↔ audio-track offset, in seconds — the recording's
   * lead-in. It's the audio-time that lines up with jot-time 0: each
   * audio track plays at media time `jotTime + drumsT0Sec`, so
   * raising it slides the backing audio *ahead* of the drums (the drums
   * come in later relative to the song) and lowering it pulls them
   * together. Drum scheduling is in jot-time and doesn't depend on this,
   * so a change only has to reposition the audio tracks.
   *
   * Observable and user-adjustable live via {@link setDrumsT0Sec} (the
   * transport bar's Offset control); the store seeds it from each loaded
   * jot's `globalMetadata.drumsT0Sec`, after which manual nudges persist
   * until a different jot is loaded. `setPlaybackSpeed` / `seek` also read
   * it to re-anchor audio tracks to the right audio position.
   */
  drumsT0Sec: number = 0;
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
  // Audio-context time of the last drum event scheduled by the most
  // recent `scheduleEvents` call. Tracked separately from
  // `tailAudioTime` (which already takes the max with audio-track
  // endings) so callers that don't reschedule drums; `setDrumsT0Sec`
  // is the only one today; can recompute the tail when only the
  // audio side moves.
  private lastScheduledDrumTime: number = 0;
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
   * beginning" (honouring `drumsT0Sec` lead-in as before).
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

  /**
   * Move the whole-page master fader. Takes effect instantly (it's a
   * single GainNode at the end of the graph) and persists across plays;
   * works before any audio exists — the value is stored and applied when
   * the graph is built.
   */
  setMasterVolume(v: number): void {
    const clamped = clampMasterVolume(v);
    runInAction(() => {
      this.masterVolume = clamped;
    });
    if (this.pageGain) this.pageGain.gain.value = clamped;
  }

  /** Move the all-drums master fader. Same instant/persistent semantics. */
  setDrumMasterVolume(v: number): void {
    const clamped = clampMasterVolume(v);
    runInAction(() => {
      this.drumMasterVolume = clamped;
    });
    if (this.drumGain) this.drumGain.gain.value = DRUM_MASTER_GAIN * clamped;
  }

  /** Move the all-audio-tracks master fader. Same instant/persistent semantics. */
  setAudioTrackMasterVolume(v: number): void {
    const clamped = clampMasterVolume(v);
    runInAction(() => {
      this.audioTrackMasterVolume = clamped;
    });
    if (this.audioBusGain) this.audioBusGain.gain.value = clamped;
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
  async loadAudioTrack(file: File, pitch?: string): Promise<AudioTrackId> {
    runInAction(() => {
      this.audioTrackError = undefined;
    });
    try {
      const ctx = this.ensureAudioContext();
      const { buffer, objectUrl } = await decodeAudioTrackFile(ctx, file);
      const id = this.allocateAudioTrackId();
      this.installAudioTrack(id, file.name, buffer, objectUrl, pitch);
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
      const { buffer, objectUrl } = await decodeAudioTrackUrl(ctx, url);
      const id = this.allocateAudioTrackId();
      this.installAudioTrack(id, filename, buffer, objectUrl);
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
    objectUrl: string,
    pitch?: string
  ): void {
    const prev = this.audioTracks.get(id);
    const track: AudioTrack = {
      id,
      filename,
      buffer,
      objectUrl,
      durationSec: buffer.duration,
      pitch,
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
        this.drumsT0Sec,
        (sid) => audioTrackGainUnder(sid, this.currentAudioTrackFilter)
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
    // Per CLEANROOM_SPEC §7.4, allowed speeds are a fixed set; clamp
    // anything reaching the player programmatically (tests, future
    // shortcuts) to the nearest in-spec value rather than the broad
    // [0.1, 2.0] range we used to accept.
    const clamped = SUPPORTED_PLAYBACK_SPEEDS.reduce(
      (best, s) => (Math.abs(s - speed) < Math.abs(best - speed) ? s : best),
      SUPPORTED_PLAYBACK_SPEEDS[0]
    );
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

    // Kill the old DRUM schedule before laying down the new one. The
    // per-voice stopFns returned by `drums.start({ time: <future> })`
    // don't actually cancel a voice that hasn't begun yet; smplr only
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
    this.cancelScheduledStops();
    this.drums?.stop();

    runInAction(() => {
      this.playbackSpeed = clamped;
    });
    this.startContextTime = now;
    this.startJotTime = jotOffset;
    const lastTime = this.scheduleEvents(jotOffset, now);
    this.audioTrackController?.setPlaybackRate(clamped);
    this.scheduleTailTimer(lastTime);
  }

  /**
   * Set the drum↔audio offset (lead-in) in seconds. Takes effect
   * immediately, including mid-playback and while paused: the drums and
   * playhead are anchored in jot-time and don't move, so we only reseek
   * the audio tracks to their new media position (`currentJotTime +
   * offset`). While paused the AudioContext clock is frozen, so the
   * rescheduled elements stay silent (their `play()` no-ops against a
   * suspended context) and realign on resume — same approach as `seek`.
   *
   * Clamped at 0: media time can't precede the start of the recording,
   * so a negative offset has no meaning (the audio would just clamp to
   * its own t=0).
   */
  setDrumsT0Sec(sec: number): void {
    const clamped = Number.isFinite(sec) ? Math.max(0, sec) : 0;
    runInAction(() => {
      this.drumsT0Sec = clamped;
    });
    if ((this.state !== 'playing' && this.state !== 'paused') || !this.ctx) return;
    if (!this.audioTrackController) return;
    const now = this.ctx.currentTime;
    const jotOffset = this.currentJotTime(now);
    // Only the audio tracks depend on the offset; reposition them to the
    // new media time without disturbing the drum schedule or playhead.
    this.audioTrackController.cancelSources();
    this.audioTrackController.scheduleAll(
      this.audioTracks.values(),
      now,
      jotOffset,
      this.playbackSpeed,
      clamped,
      (id) => audioTrackGainUnder(id, this.currentAudioTrackFilter)
    );
    // The auto-stop fires when the latest of (last drum event, last
    // audio-track sample) is reached. `computeAudioTracksEndTime` reads
    // `drumsT0Sec`, so a mid-flight offset change shifts the audio end
    // time and the tail must be re-armed; otherwise raising the offset
    // can cause auto-stop to fire before the audio finishes.
    if (this.state === 'playing') {
      this.scheduleTailTimer(this.lastScheduledDrumTime);
    } else {
      this.tailAudioTime = this.computeAudioTracksEndTime();
    }
  }

  /**
   * Re-derive the drum events from `rendered` and reschedule them live.
   * Called when the jot's beat-grid offset (the "Beat offset" control,
   * which slides drum notes across bars to fix a transcription beat
   * error) changes mid-flight — the notes have moved, so the scheduled
   * hits must follow. No-op unless playing or paused.
   *
   * Mirrors {@link setFilter}'s reschedule: the bar grid (and thus the
   * timeline) is unchanged, so only the drum schedule is rebuilt; audio
   * tracks and the playhead stay put. While paused the AudioContext clock
   * is frozen, so the rescheduled notes line up against it and come alive
   * on resume (which re-arms the tail timer from `tailAudioTime`).
   */
  refreshDrumSchedule(rendered: RenderedJot): void {
    if ((this.state !== 'playing' && this.state !== 'paused') || !this.ctx) return;
    this.events = jotToEvents(rendered);
    const now = this.ctx.currentTime;
    const jotOffset = this.currentJotTime(now);
    this.cancelScheduledStops();
    this.drums?.stop();
    const lastTime = this.scheduleEvents(jotOffset, now);
    if (this.state === 'paused') this.tailAudioTime = lastTime;
    else this.scheduleTailTimer(lastTime);
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
    // The recording's drumless lead-in is jot time [-drumsT0Sec, 0).
    // Allow scrubbing back into it instead of clamping at 0 so the user
    // can play the intro. `this.drumsT0Sec` is the live, user-tunable
    // offset (seeded from the jot's metadata by the store), so it's the
    // right bound for both idle and live seeks.
    const drumsLeadInSec = this.drumsT0Sec;
    let target = Math.max(seconds, -drumsLeadInSec);

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
        this.drumsT0Sec,
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
      // `drumsT0Sec` on globalMetadata is the audio-time of jot-time 0 in
      // the original recording — i.e. how much silence / non-drum intro
      // preceded the first detected beat. Anchor jot time at -offset so
      // the rAF loop's `jotTime > 0 ? jotTime : 0` clamp parks the
      // playhead at position 0 during the lead-in, then it advances
      // naturally once the first hit fires. Scheduling from -offset
      // pushes every event's audio time forward by offset/speed so the
      // drums come in delayed to match the original.
      // The live offset (seeded from the jot's metadata by the store,
      // tunable via the Offset control) is the source of truth — not a
      // re-read of the metadata — so a user nudge survives stop/replay.
      const drumsT0Sec = this.drumsT0Sec;
      // Start from the click-to-seek cue if one is pending, otherwise
      // from -drumsT0Sec so the rAF clamp parks the playhead at 0
      // through the recording's lead-in (unchanged default behaviour).
      // A cue (including a negative one parked in the lead-in) is
      // honoured, clamped into [-drumsT0Sec, total]. No cue still
      // means "start from the top of the lead-in" (-drumsT0Sec).
      const anchorJot =
        cueSec !== undefined
          ? Math.min(Math.max(cueSec, -drumsT0Sec), timeline.totalDurationSec)
          : -drumsT0Sec;
      this.startContextTime = audioStartTime;
      this.startJotTime = anchorJot;
      const lastTime = this.scheduleEvents(anchorJot, audioStartTime);

      // Audio tracks play through the same AudioContext so they share the
      // clock with the drum scheduler. The controller is recreated on
      // every play() to drop any residual nodes from the previous run.
      this.audioTrackController?.dispose();
      this.audioTrackController = new AudioTrackPlaybackController(
        ctx,
        this.audioBusGain ?? ctx.destination
      );
      this.audioTrackController.scheduleAll(
        this.audioTracks.values(),
        audioStartTime,
        anchorJot,
        this.playbackSpeed,
        drumsT0Sec,
        (id) => audioTrackGainUnder(id, this.currentAudioTrackFilter)
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
        this.drumsT0Sec,
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
    this.cancelScheduledStops();
    this.drums?.stop();
    this.audioTrackController?.dispose();
    this.audioTrackController = undefined;
    this.events = [];
    this.startJotTime = 0;
    // `drumsT0Sec` is deliberately NOT reset here: it's the loaded
    // jot's offset (seeded by the store, live-tunable via setDrumsT0Sec),
    // so it must survive stop()/replay. The store re-seeds it when a
    // different jot is loaded.
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
  private scheduleTailTimer(drumsLastAudioTime: number): void {
    if (this.endTimerId !== undefined) {
      window.clearTimeout(this.endTimerId);
      this.endTimerId = undefined;
    }
    if (!this.ctx) return;
    // The drum scheduler's last note isn't the only thing keeping
    // playback alive; loaded audio tracks play through `BufferSource`
    // nodes that run independently to their buffer ends. If the user
    // mutes / solos all drums (e.g. solos an audio track to ear-check
    // a stem), `drumsLastAudioTime` collapses to the play anchor and
    // we'd otherwise call `stop()` after `PLAYBACK_TAIL_SECONDS` while
    // the audio tracks are still mid-stream. Take the max with the
    // longest currently-playing audio track's end time so the tail
    // timer outlives whichever side is still producing sound.
    const lastAudioTime = Math.max(
      drumsLastAudioTime,
      this.computeAudioTracksEndTime(),
    );
    this.tailAudioTime = lastAudioTime;
    const tailMs = Math.max(
      0,
      (lastAudioTime - this.ctx.currentTime + PLAYBACK_TAIL_SECONDS) * 1000
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
    if (!this.ctx || this.audioTracks.size === 0) return 0;
    // Same `mediaOffset = max(0, jot + drumsT0Sec)` formula
    // `AudioTrackPlaybackController.scheduleOne` uses, so the end time
    // here matches when the underlying `BufferSource` actually stops.
    const mediaOffset = Math.max(0, this.startJotTime + this.drumsT0Sec);
    const speed = this.playbackSpeed;
    let maxEnd = 0;
    for (const track of this.audioTracks.values()) {
      const remaining = track.durationSec - mediaOffset;
      if (remaining <= 0) continue;
      const end = this.startContextTime + remaining / speed;
      if (end > maxEnd) maxEnd = end;
    }
    return maxEnd;
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
    // Events whose `time` falls before the play cursor are silently
    // skipped (they're already in the past for this play call). Track
    // them separately so the "no audible notes scheduled" guard below
    // doesn't conflate them with "audible notes the kit failed to
    // schedule"; that wrong attribution turned a clean soloed-audio
    // playback (where the cymbal lane has a couple of cued events
    // sitting at the playhead's exact start time and skipping by ≤µs
    // float drift) into a hard error abort.
    let silentlySkipped = 0;
    const speed = this.playbackSpeed;

    for (const ev of this.events) {
      if (ev.time < fromOffset) {
        silentlySkipped++;
        continue;
      }
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
      const velocity = Math.max(1, Math.min(127, Math.round(ev.velocity * vol * defaultGain)));
      const t = audioStartTime + (ev.time - fromOffset) / speed;
      const stopFn = drums.start({ note: ev.midiNote, time: t, velocity });
      this.scheduledStops.push(stopFn);
      scheduled++;
      if (t > lastTime) lastTime = t;
    }
    this.lastScheduledDrumTime = lastTime;

    console.log(
      `[jotPlayer] scheduled ${scheduled}/${this.events.length} events ` +
        `(filtered by mute/solo: ${mutedFiltered}, ` +
        `skipped pre-cursor: ${silentlySkipped})`
    );

    // "Audible" here = passed both the pre-cursor time check AND the
    // mute/solo filter. Notes that were silently skipped for being
    // before `fromOffset` aren't candidates this call ever tried to
    // schedule, so they don't count toward "the kit failed us".
    const audible = this.events.length - mutedFiltered - silentlySkipped;
    if (scheduled === 0 && audible > 0 && this.state !== 'playing') {
      // Nothing scheduled but notes survived BOTH the time check and
      // the mute/solo filter; a genuine, otherwise-invisible failure
      // (e.g. the kit loaded with no usable zones). Notes dropped
      // purely by an active mute/solo (audible=0) are instead a valid
      // silent-start state, handled by the caller exactly like a live
      // reschedule.
      throw new Error(
        `None of ${audible} audible notes could be ` +
          `scheduled on the GeneralUser GS kit. See console for the breakdown.`
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
      // jot + drumsT0Sec) so the target matches where the track started.
      if (
        this.audioTrackController &&
        now - this.lastDriftCheckTime >= DRIFT_CHECK_INTERVAL_SECONDS
      ) {
        this.lastDriftCheckTime = now;
        const expectedMediaSec = Math.max(0, jotTime + this.drumsT0Sec);
        this.audioTrackController.correctDrift(expectedMediaSec, this.playbackSpeed);
      }
      runInAction(() => {
        // Allow negative jot time during the recording's lead-in so the
        // playhead travels the reserved pre-roll space (timeToX maps
        // [-drumsT0Sec, 0) into the lead-in pixels). Clamp at
        // -drumsT0Sec so it can't run off the left of the lead-in.
        this.currentTime = Math.max(jotTime, -this.drumsT0Sec);
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
    // Build the master bus now (not in ensureLoaded) so audio tracks
    // loaded before the first play() route through the same faders.
    const pageGain = this.ctx.createGain();
    pageGain.gain.value = this.masterVolume;
    pageGain.connect(this.ctx.destination);
    const audioBusGain = this.ctx.createGain();
    audioBusGain.gain.value = this.audioTrackMasterVolume;
    audioBusGain.connect(pageGain);
    this.pageGain = pageGain;
    this.audioBusGain = audioBusGain;
    return this.ctx;
  }

  private async ensureLoaded(): Promise<{ drums: Drums; ctx: AudioContext }> {
    if (this.drums && this.ctx) return { drums: this.drums, ctx: this.ctx };

    const ctx = this.ensureAudioContext();
    const drumGain = ctx.createGain();
    drumGain.gain.value = DRUM_MASTER_GAIN * this.drumMasterVolume;
    // Into the page master (built by ensureAudioContext above), not
    // straight to destination, so the page fader scales drums too.
    drumGain.connect(this.pageGain ?? ctx.destination);
    this.drumGain = drumGain;
    // Phase before the first byte arrives — the toolbar shows "waiting
    // for server…" until the storage layer reports progress.
    runInAction(() => {
      this.sampleLoadPhase = 'connecting';
    });
    const drums = GeneralUserGsKit(ctx, {
      url: GM_SOUNDFONT_URL,
      cacheName: GM_SOUNDFONT_CACHE,
      bank: GM_DRUM_BANK,
      preset: this.drumPreset,
      destination: drumGain,
      // Byte progress for the (large, one-time) .sf2 download; the
      // storage layer also serves it from the Cache API on later
      // sessions, in which case this fires once with fromCache = true.
      // Storage emits a final tick with `loaded === total` once bytes
      // are in, which we treat as the start of the decode phase.
      onProgress: (p) => {
        const downloadComplete = p.total > 0 && p.loaded >= p.total;
        runInAction(() => {
          this.sampleLoadProgress = p;
          this.sampleLoadPhase = downloadComplete ? 'decoding' : 'downloading';
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
                    `check network access to raw.githubusercontent.com from the browser.`
                )
              ),
            LOAD_TIMEOUT_SECONDS * 1000
          )
        ),
      ]);
    } finally {
      // Clear the progress readout whether we finished or timed out so a
      // stale bar doesn't linger; the UI also gates on `state` anyway.
      runInAction(() => {
        this.sampleLoadProgress = undefined;
        this.sampleLoadPhase = undefined;
      });
    }

    // Give smplr a moment to settle before the first scheduled hit; see
    // POST_LOAD_SETTLE_SECONDS. Only reached on the cold load — the
    // early return at the top of this method skips it on every later play.
    await new Promise((resolve) => window.setTimeout(resolve, POST_LOAD_SETTLE_SECONDS * 1000));

    this.drums = drums;
    runInAction(() => {
      this.drumKits = drums.availableKits;
    });
    console.log('[jotPlayer] GeneralUser GS kit loaded');
    return { drums, ctx };
  }
}

export const jotPlayer = new JotPlayer();
