/**
 * Audio-track playback alongside the MIDI score.
 *
 * Loaded audio tracks (a ParaDB pack's song/drum tracks; the transcriber's
 * `no_drums`/`drum_stem` FLACs; ad-hoc backing tracks) play through the
 * same `AudioContext` as the smplr drum machine; so they share the audio
 * clock and start / stop / speed changes stay in lockstep with MIDI
 * playback.
 *
 * Each track is dispatched along one of two paths chosen by playback
 * speed; both feeding the same per-track `GainNode`:
 *
 *  1. **`AudioBufferSourceNode` (1.0× only).** Sample-accurate against
 *     the AudioContext clock the drum scheduler uses — `start(t; off)`
 *     schedules playback at an exact audio-clock time and a starting
 *     buffer offset; so drums and music *cannot* drift no matter how
 *     long the main thread stalls. This is the default path and the
 *     reason a heavy zoom no longer pulls the backing track off the
 *     score. One-shot semantics: pause / seek / speed each create a
 *     fresh `BufferSource`; the previous one is `.stop(0)`-ed and
 *     disconnected.
 *
 *  2. **`HTMLAudioElement` (non-1.0× only).** A media element's
 *     `playbackRate` decouples time from pitch via `preservesPitch`
 *     (the browser's built-in time-stretcher) — `BufferSource` can't.
 *     Used at 0.25× / 0.5× / 0.75× / 1.25× so practising at half speed
 *     keeps the original pitch. The element has its own media clock,
 *     so the periodic drift-correction subsystem (`correctDrift`) is
 *     still required *for this path only*. See `TODO.md` for the
 *     Options A / B work that would let us drop this path entirely.
 *
 * Mute/solo/volume mutate `gain.value` directly so the response is
 * immediate and shared between both paths. Switching speed across the
 * 1.0× boundary forces a path swap (cancel + reschedule); which costs
 * a sub-frame audible gap on speed change — accepted in exchange for
 * sample-accurate sync at 1.0×.
 *
 * The decoded `AudioBuffer` lives on {@link AudioTrack} and is used by
 * BOTH paths now: the `BufferSource` path feeds it directly; the
 * waveform renderer still streams through `buffer.getChannelData(ch)`
 * for peak extraction (we deliberately do *not* pre-collapse to mono
 * at load time because the peaks compute touches every sample anyway
 * and can fold channels in the same pass). The MediaElement path uses
 * the same `objectUrl` as before.
 */

/**
 * Opaque per-track id. Every loaded audio track gets a fresh unique id
 * (the player allocates them); there is no fixed set and no music-vs-drums
 * distinction — an audio track is just audio that plays alongside the
 * score. Iteration order of the player's track map is load order.
 */
export type AudioTrackId = string;

// Drift correction (see {@link AudioTrackPlaybackController.correctDrift}).
// Only applied on the HTMLAudioElement (non-1.0×) path; the
// `AudioBufferSourceNode` path runs on the same clock as the drum
// scheduler so it can't drift by construction.
// Below this offset the track is considered in sync; small enough that
// drums-vs-music slip is imperceptible, large enough to ignore the
// few-ms jitter of reading `el.currentTime`.
const DRIFT_SOFT_TOLERANCE_SEC = 0.04;
// Above this we hard-seek instead of trimming the rate: a gap this big
// only comes from rAF stalling (backgrounded tab) or a decode/GC pause,
// where easing back at a fraction of a percent would take far too long.
const DRIFT_HARD_RESEEK_SEC = 0.5;
// Rate trim applied while easing a small drift away. 0.6% is well under
// the audible-tempo-change threshold (and pitch is preserved anyway),
// yet removes ~3 ms of error per 500 ms correction tick.
const DRIFT_RATE_TRIM = 0.006;
// Speed at which the BufferSource (sample-accurate, no pitch-preserve)
// path is used. Anything outside this tolerance falls through to the
// MediaElement path so pitch is preserved by the browser's built-in
// time-stretcher. Strict 1.0× equality also works against the fixed
// `SUPPORTED_PLAYBACK_SPEEDS` set but the epsilon keeps it robust to
// any future float-formatted setter.
const UNITY_SPEED_EPSILON = 1e-4;
function isUnityRate(speed: number): boolean {
  return Math.abs(speed - 1) < UNITY_SPEED_EPSILON;
}

/**
 * What the audio in this track is, as far as the loader could tell. Drives
 * the mixer-row overflow menu: a `full-mix` row can be re-split into
 * drums + backing; a `drums` row can be split into per-instrument pieces;
 * the other roles can't usefully feed either separation stage. `unknown`
 * is the safe default for ad-hoc user-loaded files; the menu enables
 * both items and lets the user pick.
 *
 * Source → role assignment lives at the loader sites in
 * `src/jot_view/store.ts` (ParaDB pack, debug bundle / transcriber,
 * ad-hoc drop).
 */
export type AudioTrackRole =
  | 'full-mix'    // contains drums + everything else (or could)
  | 'no-drums'    // drumless backing, produced by our own Demucs run
  | 'drums'       // isolated drum kit, all pieces together
  | 'drum-piece'  // a single instrument (kick / snare / hi-hat / cymbals)
  | 'unknown';    // ad-hoc / can't tell

export type AudioTrack = {
  id: AudioTrackId;
  /** Original filename for display in the gutter and tooltips. */
  filename: string;
  /**
   * Decoded PCM. The BufferSource (1.0×) playback path feeds this
   * directly to an `AudioBufferSourceNode` for sample-accurate sync
   * with the drum scheduler; the waveform renderer also reads channel
   * data off it for peak extraction. Kept on the immutable track so
   * mid-playback reloads with the same id can swap PCM without
   * tearing the slot down.
   */
  buffer: AudioBuffer;
  /**
   * Object URL of the original encoded bytes, used by the
   * `HTMLAudioElement` (non-1.0×) playback path. Owned by the player,
   * which revokes it when the track is replaced or cleared so blobs
   * don't leak across reloads. The BufferSource path doesn't consume
   * this; it plays from the decoded `buffer` directly.
   */
  objectUrl: string;
  durationSec: number;
  /**
   * DSL pitch letter (e.g. `k`, `s`, `h`) this audio track is the isolated
   * stem of, when known. Set from a debug bundle's `mapping` entry; the
   * waveform canvas tints itself with that pitch's lane color so the
   * audio track reads as visually paired with its instrument row.
   * Undefined for ad-hoc / drumless tracks.
   */
  pitch?: string;
  /**
   * What the loader believes the audio is. Drives the per-row overflow
   * menu's enable matrix. Undefined is treated as `unknown`.
   */
  role?: AudioTrackRole;
};

/**
 * Live playback state for an audio track during a single `play()` call.
 * Held inside the {@link AudioTrackPlaybackController} (created and
 * destroyed by the player) rather than on the immutable {@link
 * AudioTrack} so reloading a track mid-flight doesn't leak old nodes.
 *
 * Two source paths share the same {@link gainNode}; exactly one of
 * `bufferSource` / `media` is "active" at a time (the other is either
 * stopped or not yet built). See the module header for the path-choice
 * rules.
 */
type ActiveAudioTrack = {
  id: AudioTrackId;
  /** Per-track mute/solo/volume gain, persists across (re)schedules. */
  gainNode: GainNode;
  /**
   * Decoded PCM. Held on the slot so the BufferSource path can construct
   * fresh `AudioBufferSourceNode`s without going back through the
   * controller's `tracks` argument; reassigned by {@link
   * AudioTrackPlaybackController.ensureSlot} when a track is reloaded
   * mid-playback with the same id.
   */
  buffer: AudioBuffer;
  /**
   * The current source on the BufferSource (1.0×) path, if active.
   * One-shot: `.start(t, off)`-scheduled then `.stop(0)`-ed and
   * disconnected on cancel/reschedule. Recreated fresh every time.
   */
  bufferSource: AudioBufferSourceNode | undefined;
  /**
   * MediaElement (non-1.0×) path. Built lazily on first use because the
   * BufferSource path covers the common case and most sessions never
   * touch it; once built it persists for the slot's lifetime (a
   * `MediaElementAudioSourceNode` can only be created once per
   * element).
   *
   * `objectUrl` is the url currently wired into `el`; a mismatch with
   * the `track.objectUrl` passed to {@link ensureSlot} signals a
   * mid-playback track reload and triggers a media-side rebuild.
   */
  media:
    | {
        el: HTMLAudioElement;
        node: MediaElementAudioSourceNode;
        objectUrl: string;
        /** Pending aligned-start timer, cleared on cancel/reschedule. */
        startTimer: number | undefined;
      }
    | undefined;
  /**
   * objectUrl this slot was last scheduled with; used by {@link
   * ensureSlot} to detect a mid-playback track reload regardless of
   * whether the media path has been built yet.
   */
  objectUrl: string;
  /**
   * Bumped on every (re)schedule and cancel. A deferred MediaElement
   * start (waiting on a timer or `loadedmetadata`) checks this before
   * firing so a superseded schedule can't start the element late.
   */
  gen: number;
};

/** Filter for audio-track mute/solo, parallel to {@link PlayerFilter} for pitches. */
export type AudioTrackFilter = {
  mutedAudioTracks: ReadonlySet<AudioTrackId>;
  soloedAudioTracks: ReadonlySet<AudioTrackId>;
  /**
   * True when a solo is engaged anywhere; on an audio track OR a pitch
   * row. Solo is one global mode shared across both domains, so soloing
   * a drum instrument silences the audio tracks too (and vice versa).
   * The store computes this since it owns both solo sets.
   */
  soloActive: boolean;
  /** True when the audio section's master mute is engaged; silences every
   * track regardless of per-row state (mirrors the bus gain pin to 0). */
  sectionMasterMuted: boolean;
  /** True when the audio section's master solo is engaged; treats every
   * track as if it were soloed for the purpose of the solo-exclusion
   * rule (otherwise master-solo-with-no-row-solos = silent section). */
  sectionMasterSoloed: boolean;
  /** Per-track volume multiplier in [0, 1]; missing = full (1). */
  volumes: ReadonlyMap<AudioTrackId, number>;
};

export const PASSTHROUGH_AUDIO_TRACK_FILTER: AudioTrackFilter = {
  mutedAudioTracks: new Set<AudioTrackId>(),
  soloedAudioTracks: new Set<AudioTrackId>(),
  soloActive: false,
  sectionMasterMuted: false,
  sectionMasterSoloed: false,
  volumes: new Map<AudioTrackId, number>(),
};

export function isAudioTrackAudibleUnder(
  id: AudioTrackId,
  filter: AudioTrackFilter,
): boolean {
  if (filter.sectionMasterMuted) return false;
  if (filter.mutedAudioTracks.has(id)) return false;
  if (
    filter.soloActive &&
    !filter.sectionMasterSoloed &&
    !filter.soloedAudioTracks.has(id)
  ) {
    return false;
  }
  return true;
}

/**
 * Master trim applied to every audio (music) track on top of its
 * per-track volume fader. The GM drum kit sits hot relative to typical
 * backing recordings, so the music is halved by default — a 100% fader
 * now sounds like the old 50%. Adjust here to retune the drums-to-music
 * balance globally.
 */
const AUDIO_TRACK_MASTER_GAIN = 0.5;

/**
 * Resolved playback gain for an audio track: 0 when filtered out,
 * otherwise the per-track volume fader (default 1) scaled by
 * {@link AUDIO_TRACK_MASTER_GAIN}. This is what the controller writes
 * straight onto the track's `GainNode`.
 */
export function audioTrackGainUnder(
  id: AudioTrackId,
  filter: AudioTrackFilter,
): number {
  if (!isAudioTrackAudibleUnder(id, filter)) return 0;
  return (filter.volumes.get(id) ?? 1) * AUDIO_TRACK_MASTER_GAIN;
}

/**
 * Decode an audio file's bytes into an {@link AudioBuffer} and produce
 * an object URL for the playback `HTMLAudioElement`. The waveform
 * renderer reads channels off the buffer directly — there's no mono
 * pre-collapse here (see the module header).
 *
 * `decodeAudioData` is delegated to the browser; FLAC works in modern
 * Chromium / Firefox / Safari. WAV, MP3, and (most) AAC all work too.
 */
export async function decodeAudioTrackFile(
  ctx: AudioContext,
  file: File,
): Promise<{ buffer: AudioBuffer; objectUrl: string }> {
  // `file.arrayBuffer()` returns a fresh ArrayBuffer that we own; pass
  // it straight to decodeAudioData (which neuters the input) instead of
  // copying defensively. The playback element gets its own copy via the
  // `File` object URL below, so neutering here has no observable effect.
  const bytes = await file.arrayBuffer();
  const buffer = await ctx.decodeAudioData(bytes);
  const objectUrl = URL.createObjectURL(file);
  return { buffer, objectUrl };
}

/**
 * Fetch an audio track from a URL (typically the transcriber's
 * `/outputs/...` route) and decode it. Equivalent to {@link
 * decodeAudioTrackFile} for File inputs — kept as a separate entry
 * point so the auto-load-on-transcribe path doesn't have to round-trip
 * through a synthetic File.
 */
export async function decodeAudioTrackUrl(
  ctx: AudioContext,
  url: string,
): Promise<{ buffer: AudioBuffer; objectUrl: string }> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch audio track (${res.status} ${res.statusText})`);
  }
  const bytes = await res.arrayBuffer();
  // The Blob constructor copies its input bytes per spec, so the object
  // URL stays valid after `decodeAudioData` neuters our `bytes` view.
  // Avoids the extra ArrayBuffer copy a defensive `bytes.slice(0)` would
  // otherwise add for a 3-min FLAC.
  const objectUrl = URL.createObjectURL(new Blob([bytes]));
  const buffer = await ctx.decodeAudioData(bytes);
  return { buffer, objectUrl };
}

/**
 * Manages live audio-track playback for one `play()` cycle. Created
 * lazily by the player when it first needs to schedule tracks;
 * destroyed (via `dispose`) on every player `stop()` so a fresh play()
 * starts with no residual nodes.
 *
 * Mute is a gain mutation on the per-track `GainNode` (no
 * reschedule). Speed changes either keep the existing source (live
 * `playbackRate` mutation inside the MediaElement path) or force a
 * cancel + reschedule (when the BufferSource ↔ MediaElement boundary
 * is crossed). See the module header for the path-choice rules.
 */
export class AudioTrackPlaybackController {
  private active: Map<AudioTrackId, ActiveAudioTrack> = new Map();

  /**
   * `destination` is the node every track's `GainNode` feeds into —
   * the player passes its all-audio-tracks master bus so that fader
   * (and, downstream, the page fader) scales the music. Defaults to
   * `ctx.destination` so a bare controller still makes sound.
   */
  constructor(
    private readonly ctx: AudioContext,
    private readonly destination: AudioNode = ctx.destination,
  ) {}

  /**
   * Start every loaded audio track at `audioStartTime`, seeking its
   * buffer to the position corresponding to `jotOffsetSec` (negative
   * during lead-in — the buffer plays from t=0 in that case and the
   * audible lead-in is just the recording's intro).
   *
   * `gainFor` controls each track's initial gain: muted tracks start at
   * 0 so the user can pre-mute before pressing Play and the change is
   * honoured immediately rather than fading in once playback starts.
   * Non-zero values are the per-track volume fader.
   */
  scheduleAll(
    tracks: Iterable<AudioTrack>,
    audioStartTime: number,
    jotOffsetSec: number,
    speed: number,
    drumsT0Sec: number,
    gainFor: (id: AudioTrackId) => number,
  ): void {
    this.scheduleAllInternal(
      tracks,
      audioStartTime,
      jotOffsetSec,
      speed,
      drumsT0Sec,
      gainFor,
    );
  }

  /**
   * Apply a new playback rate to every currently-active track. When the
   * switch crosses the 1.0× boundary (e.g. 1.0× → 0.5× or 0.5× → 1.0×)
   * the path itself must change. BufferSource → MediaElement or back; * which forces a cancel + reschedule with a sub-frame audible gap.
   * Inside the MediaElement path (e.g. 0.5× → 0.75×) we keep the
   * existing live rate change so the seamless / pitch-preserved
   * transition is preserved there.
   *
   * The caller (the player's `setPlaybackSpeed`) hands us the anchor
   * state (`audioStartTime`, `jotOffsetSec`, `drumsT0Sec`, `gainFor`)
   * because a path switch needs to call back into `scheduleAll` and
   * those are exactly the args it would pass.
   */
  setPlaybackRate(
    speed: number,
    audioStartTime: number,
    jotOffsetSec: number,
    drumsT0Sec: number,
    gainFor: (id: AudioTrackId) => number,
  ): void {
    const wantUnity = isUnityRate(speed);
    let pathSwitchNeeded = false;
    for (const slot of this.active.values()) {
      const onBufferSourcePath = slot.bufferSource !== undefined;
      if (onBufferSourcePath !== wantUnity) {
        pathSwitchNeeded = true;
        break;
      }
    }
    if (pathSwitchNeeded) {
      // Cancel both paths on every slot, then schedule fresh from the
      // new anchor; this is the only time mid-playback that a cancel
      // + reschedule is unavoidable for audio tracks.
      this.cancelSources();
      this.scheduleAllInternal(
        this.activeTracksSnapshot(),
        audioStartTime,
        jotOffsetSec,
        speed,
        drumsT0Sec,
        gainFor,
      );
      return;
    }
    // Same path on every slot. BufferSource path: nothing to do
    // (playbackRate at 1.0× is a no-op and we don't otherwise touch
    // the source). MediaElement path: live rate change, no reschedule.
    if (!wantUnity) {
      for (const slot of this.active.values()) {
        if (slot.media) slot.media.el.playbackRate = speed;
      }
    }
  }

  /**
   * Snapshot of the bare {@link AudioTrack} objects for every slot
   * currently in `this.active`, in slot iteration (= load) order. Used
   * by {@link setPlaybackRate}'s path-switch branch to feed
   * `scheduleAllInternal` without forcing the caller to thread the
   * track list back in; the slot already owns a reference to the
   * decoded buffer.
   */
  private activeTracksSnapshot(): AudioTrack[] {
    const out: AudioTrack[] = [];
    for (const slot of this.active.values()) {
      out.push({
        id: slot.id,
        // filename/durationSec aren't used by schedule paths; fill
        // them from the slot fields we have.
        filename: '',
        buffer: slot.buffer,
        objectUrl: slot.objectUrl,
        durationSec: slot.buffer.duration,
      });
    }
    return out;
  }

  /**
   * Internal entry point for both {@link scheduleAll} (called by the
   * player) and {@link setPlaybackRate}'s path-switch branch. Identical
   * loop body; the public `scheduleAll` just forwards through.
   */
  private scheduleAllInternal(
    tracks: Iterable<AudioTrack>,
    audioStartTime: number,
    jotOffsetSec: number,
    speed: number,
    drumsT0Sec: number,
    gainFor: (id: AudioTrackId) => number,
  ): void {
    for (const track of tracks) {
      this.scheduleOne(
        track,
        audioStartTime,
        jotOffsetSec,
        speed,
        drumsT0Sec,
        gainFor(track.id),
      );
    }
  }

  private scheduleOne(
    track: AudioTrack,
    audioStartTime: number,
    jotOffsetSec: number,
    speed: number,
    drumsT0Sec: number,
    gain: number,
  ): void {
    // Media time = jot time + drumsT0Sec. A negative jot offset
    // (lead-in) clamps to 0 so the recording's own intro is what plays
    // during the lead-in, exactly as the old buffer path did.
    const mediaOffset = Math.max(0, jotOffsetSec + drumsT0Sec);
    const slot = this.ensureSlot(track);
    slot.gainNode.gain.value = gain;

    // Supersede any deferred MediaElement start from a previous
    // (re)schedule.
    const gen = ++slot.gen;
    this.stopActiveSources(slot);

    if (isUnityRate(speed)) {
      this.scheduleBufferSource(slot, audioStartTime, mediaOffset);
      return;
    }
    this.scheduleMediaElement(slot, gen, track.objectUrl, audioStartTime, mediaOffset, speed);
  }

  /**
   * Start the slot's `AudioBufferSourceNode` at `audioStartTime`, with
   * `mediaOffset` seconds into the decoded buffer. Sample-accurate
   * against the AudioContext clock; the same clock the drum scheduler
   * is on, so no drift can accumulate. One-shot: a future cancel /
   * reschedule will `.stop(0)` + disconnect this node and create a
   * fresh one.
   *
   * If `mediaOffset` is past the buffer's end the source naturally
   * never produces audio; we still arm it so the rest of the
   * controller treats this slot like any other (mute/solo gain
   * mutation, drift no-op, etc.).
   */
  private scheduleBufferSource(
    slot: ActiveAudioTrack,
    audioStartTime: number,
    mediaOffset: number,
  ): void {
    const source = this.ctx.createBufferSource();
    source.buffer = slot.buffer;
    source.connect(slot.gainNode);
    // `start()` accepts negative absolute times (it clamps to "now"),
    // and a `mediaOffset` past the buffer end is also tolerated; in
    // both cases the source produces silence, which is what we want.
    const offset = Math.max(0, Math.min(mediaOffset, slot.buffer.duration));
    source.start(audioStartTime, offset);
    slot.bufferSource = source;
  }

  /**
   * MediaElement (non-1.0×) start path. Lazily builds the media subslot
   * the first time this slot needs it; arms `preservesPitch`; seeks to
   * `mediaOffset` and `play()`s at `audioStartTime` (via wall-clock
   * `setTimeout`, the same approximation the legacy single-path code
   * used; a media element can't be told to begin at an exact
   * AudioContext time).
   */
  private scheduleMediaElement(
    slot: ActiveAudioTrack,
    gen: number,
    objectUrl: string,
    audioStartTime: number,
    mediaOffset: number,
    speed: number,
  ): void {
    const media = this.ensureMediaSubslot(slot, objectUrl);
    const el = media.el;
    el.playbackRate = speed;
    setPreservesPitch(el, true);

    const fire = () => {
      media.startTimer = undefined;
      if (slot.gen !== gen) return; // a newer schedule won the slot
      const seekThenPlay = () => {
        if (slot.gen !== gen) return;
        const dur = Number.isFinite(el.duration) ? el.duration : mediaOffset;
        try {
          el.currentTime = Math.max(0, Math.min(mediaOffset, dur));
        } catch {
          // Setting currentTime can throw if the element isn't
          // seekable yet; the loadedmetadata path below retries.
        }
        // play() rejects if a pause races it (reschedule); benign.
        void el.play().catch(() => {});
      };
      if (el.readyState >= 1 /* HAVE_METADATA: duration/seek known */) {
        seekThenPlay();
      } else {
        el.addEventListener('loadedmetadata', seekThenPlay, { once: true });
      }
    };

    // The drum scheduler hands us a context time slightly in the
    // future (SCHEDULE_LEAD_SECONDS). A media element can't be told to
    // begin at an exact AudioContext time, so approximate it with a
    // wall-clock delay; sub-frame deltas just start now.
    const delaySec = audioStartTime - this.ctx.currentTime;
    if (delaySec <= 0.02) fire();
    else media.startTimer = window.setTimeout(fire, delaySec * 1000);
  }

  /**
   * Stop both source paths on the slot; disconnects the
   * `bufferSource` (BufferSource is one-shot, so we throw it away)
   * and pauses the media element if built. Idempotent.
   */
  private stopActiveSources(slot: ActiveAudioTrack): void {
    if (slot.bufferSource) {
      try {
        slot.bufferSource.stop(0);
      } catch {
        // `stop()` throws if the source has never been started or is
        // already stopped; either way it's safe to disconnect.
      }
      try {
        slot.bufferSource.disconnect();
      } catch (err) {
        console.debug('[audio-tracks] bufferSource.disconnect threw', err);
      }
      slot.bufferSource = undefined;
    }
    if (slot.media) {
      if (slot.media.startTimer !== undefined) {
        window.clearTimeout(slot.media.startTimer);
        slot.media.startTimer = undefined;
      }
      try {
        slot.media.el.pause();
      } catch (err) {
        console.debug('[audio-tracks] el.pause threw', err);
      }
    }
  }

  private ensureSlot(track: AudioTrack): ActiveAudioTrack {
    const existing = this.active.get(track.id);
    if (existing) {
      if (existing.objectUrl === track.objectUrl && existing.buffer === track.buffer) {
        return existing;
      }
      // Same slot, different audio (track reloaded mid-playback). Stop
      // both source paths cleanly, then repoint the buffer (used by the
      // BufferSource path) and tear down the media subslot if it
      // exists; a `MediaElementAudioSourceNode` is bound to its
      // element for life, so the element must be rebuilt to play new
      // bytes.
      existing.gen++;
      this.stopActiveSources(existing);
      if (existing.media) {
        try {
          existing.media.node.disconnect();
          existing.media.el.removeAttribute('src');
          existing.media.el.load();
        } catch (err) {
          console.debug('[audio-tracks] slot rebuild teardown threw', err);
        }
        existing.media = undefined;
      }
      existing.buffer = track.buffer;
      existing.objectUrl = track.objectUrl;
      return existing;
    }
    const gainNode = this.ctx.createGain();
    gainNode.gain.value = 1;
    gainNode.connect(this.destination);

    const slot: ActiveAudioTrack = {
      id: track.id,
      gainNode,
      buffer: track.buffer,
      bufferSource: undefined,
      media: undefined,
      objectUrl: track.objectUrl,
      gen: 0,
    };
    this.active.set(track.id, slot);
    return slot;
  }

  /**
   * Lazily build (or rebuild after a track reload) the MediaElement
   * subslot. `objectUrl` is the current url for the slot's track; a
   * mismatch with `slot.media.objectUrl` triggers a rebuild because a
   * `MediaElementAudioSourceNode` is bound to its element for life.
   */
  private ensureMediaSubslot(
    slot: ActiveAudioTrack,
    objectUrl: string,
  ): NonNullable<ActiveAudioTrack['media']> {
    if (slot.media && slot.media.objectUrl === objectUrl) return slot.media;
    if (slot.media) {
      try {
        slot.media.node.disconnect();
        slot.media.el.pause();
        slot.media.el.removeAttribute('src');
        slot.media.el.load();
      } catch (err) {
        console.debug('[audio-tracks] media rebuild teardown threw', err);
      }
    }
    const el = makeAudioTrackElement(objectUrl);
    const node = this.ctx.createMediaElementSource(el);
    node.connect(slot.gainNode);
    const next = { el, node, objectUrl, startTimer: undefined };
    slot.media = next;
    return next;
  }

  /**
   * Fully tear down one track's slot (sources, media element, gain).
   * Used by the player's `clearAudioTrack` so a removed track leaves
   * no dangling nodes.
   */
  dropAudioTrack(id: AudioTrackId): void {
    const slot = this.active.get(id);
    if (!slot) return;
    slot.gen++;
    this.stopActiveSources(slot);
    try {
      if (slot.media) {
        slot.media.node.disconnect();
        slot.media.el.removeAttribute('src');
        slot.media.el.load();
      }
      slot.gainNode.disconnect();
    } catch (err) {
      console.debug('[audio-tracks] dropAudioTrack threw', err);
    }
    this.active.delete(id);
  }

  /** Apply a live gain decision (mute/solo + volume) to every active track. */
  applyAudibility(gainFor: (id: AudioTrackId) => number): void {
    for (const slot of this.active.values()) {
      slot.gainNode.gain.value = gainFor(slot.id);
    }
  }

  /**
   * Re-lock every playing track to the AudioContext clock.
   *
   * A media element runs on its own clock, not the `AudioContext` sample
   * clock that the drum scheduler and playhead use, so the two slew
   * apart over a long track even at `playbackRate === 1` (see the module
   * header's sync caveat). Called periodically by the player with
   * `expectedMediaSec` derived from the *same* `currentJotTime` math the
   * drums are scheduled against, so correcting toward it keeps drums,
   * playhead, and music together.
   *
   *  - Small error: trim `playbackRate` by a fraction of a percent so
   *    the element catches up / eases off. With `preservesPitch` on this
   *    time-stretch is inaudible, and it converges over a few ticks
   *    without the click of a seek.
   *  - Large error (tab was backgrounded so rAF stalled, a GC pause, a
   *    decode hiccup): a gentle trim would take too long, so hard-seek.
   *  - Within tolerance: restore the exact user-selected `speed` so the
   *    correction doesn't leave a residual tempo offset.
   */
  correctDrift(expectedMediaSec: number, speed: number): void {
    for (const slot of this.active.values()) {
      // BufferSource path is sample-accurate against the AudioContext
      // clock by construction; nothing to correct, and there's no
      // readable media-clock anyway. Skip.
      if (slot.bufferSource) continue;
      const media = slot.media;
      if (!media) continue;
      const el = media.el;
      // Skip tracks that haven't started yet (deferred timer still
      // pending → paused) or whose media clock isn't readable yet.
      if (el.paused || el.readyState < 1) continue;
      const drift = el.currentTime - expectedMediaSec; // +ve ⇒ element is ahead
      const abs = Math.abs(drift);
      if (abs > DRIFT_HARD_RESEEK_SEC) {
        try {
          const dur = Number.isFinite(el.duration) ? el.duration : expectedMediaSec;
          el.currentTime = Math.max(0, Math.min(expectedMediaSec, dur));
        } catch {
          // Not seekable this instant; the next tick retries.
        }
        el.playbackRate = speed;
      } else if (abs > DRIFT_SOFT_TOLERANCE_SEC) {
        // Ahead → play a touch slower; behind → a touch faster.
        const correction = drift > 0 ? 1 - DRIFT_RATE_TRIM : 1 + DRIFT_RATE_TRIM;
        el.playbackRate = speed * correction;
      } else if (el.playbackRate !== speed) {
        el.playbackRate = speed;
      }
    }
  }

  /**
   * Stop playback of every track. For the BufferSource path the source
   * is one-shot, so it's `.stop(0)`-ed and discarded; a subsequent
   * `scheduleAll` will create a fresh one. For the MediaElement path
   * the element is paused and any pending deferred-start timer
   * cleared, so the element / gain graph can be reused. Used by
   * `setPlaybackSpeed`, `seek`, and (via the player) pause.
   *
   * Named `cancelSources` for parity with the player's drum-side
   * vocabulary.
   */
  cancelSources(): void {
    for (const slot of this.active.values()) {
      slot.gen++;
      this.stopActiveSources(slot);
    }
  }

  /** Teardown; invoked when playback ends so the graph doesn't leak. */
  dispose(): void {
    this.cancelSources();
    for (const slot of this.active.values()) {
      try {
        if (slot.media) {
          slot.media.node.disconnect();
          // Detach the source so the browser can release the decoded
          // media; the Blob URL itself is revoked by the player.
          slot.media.el.removeAttribute('src');
          slot.media.el.load();
        }
        slot.gainNode.disconnect();
      } catch (err) {
        console.debug('[audio-tracks] dispose threw', err);
      }
    }
    this.active.clear();
  }
}

/** A preloading `<audio>` element pointed at an audio track's blob URL. */
function makeAudioTrackElement(objectUrl: string): HTMLAudioElement {
  const el = new Audio();
  el.preload = 'auto';
  el.src = objectUrl;
  el.load();
  return el;
}

/**
 * Cross-browser `preservesPitch`. The unprefixed property is standard
 * in current Chromium / Firefox / Safari; the vendor-prefixed names
 * cover older engines. Assigning an unknown property is harmless, so
 * we just set whichever exist.
 */
function setPreservesPitch(el: HTMLAudioElement, value: boolean): void {
  const anyEl = el as HTMLAudioElement & {
    preservesPitch?: boolean;
    mozPreservesPitch?: boolean;
    webkitPreservesPitch?: boolean;
  };
  if ('preservesPitch' in anyEl) anyEl.preservesPitch = value;
  if ('mozPreservesPitch' in anyEl) anyEl.mozPreservesPitch = value;
  if ('webkitPreservesPitch' in anyEl) anyEl.webkitPreservesPitch = value;
}
