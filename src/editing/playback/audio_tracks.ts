/**
 * Audio-track playback alongside the MIDI score.
 *
 * Loaded audio tracks (a ParaDB pack's song/drum tracks, the transcriber's
 * `no_drums`/`drum_stem` FLACs, ad-hoc backing tracks) play through the
 * same `AudioContext` as the smplr drum machine, so they share the audio
 * clock with the drum scheduler and stay in lockstep with MIDI playback
 * at every speed.
 *
 * Each track plays through one path:
 *
 *   PCM (Float32Array per channel)
 *      └─▶ Signalsmith Stretch `AudioWorkletNode`
 *             ├─▶ owns the PCM ring (fed via `addBuffers`)
 *             ├─▶ pulls samples per audio-thread block
 *             └─▶ output ─▶ per-track GainNode ─▶ audio bus
 *
 * Speed change is a single message: `node.schedule({ rate, output })`.
 * Pitch is preserved by the worklet's internal time-stretcher
 * (Signalsmith Stretch, MIT). Seek is `node.schedule({ input, output })`.
 * Pause is `node.stop()`. No source `AudioBufferSourceNode`, no media
 * element, no separate clock to drift against; the worklet runs on
 * the audio thread and consumes / emits samples in lockstep with the
 * AudioContext.
 *
 * Mute/solo/volume mutate `gainNode.gain` directly so the response is
 * immediate without disturbing the stretch state.
 *
 * The decoded `AudioBuffer` lives on {@link AudioTrack} and is read
 * once to seed the worklet's PCM ring (`createStretchNode`); the
 * waveform renderer still streams through `buffer.getChannelData(ch)`
 * for peak extraction (we deliberately do *not* pre-collapse to mono
 * at load time because the peaks compute touches every sample anyway
 * and can fold channels in the same pass).
 */

import { makeAutoObservable, runInAction } from 'mobx';
import {
  AUDIO_FALLBACK_COLOR,
  groupInstrumentLanes,
  MixerContext,
  resolveAudioInheritedColor,
  Track,
} from 'src/editing/tracks/tracks';
import { createStretchNode, preloadStretch, StretchNode } from './stretch_node';
import { backendFetch } from 'src/net/backend_fetch';

/**
 * Opaque per-track id. Every loaded audio track gets a fresh unique id
 * (the player allocates them); there is no fixed set and no music-vs-drums
 * distinction; an audio track is just audio that plays alongside the
 * score. Iteration order of the player's track map is load order.
 */
export type AudioTrackId = string;

/**
 * How far ahead the worklet's `schedule()` events are placed past
 * `ctx.currentTime`. The library compensates for its own latency but
 * wants the request slightly in the future so it can transition
 * smoothly across the parameter change; the SCHEDULE_LEAD_SECONDS the
 * player uses upstream is the same idea on the drum side. Stays below
 * a single rAF frame so a live speed change isn't audibly deferred.
 */
const SCHEDULE_PAD_SEC = 0.02;

/**
 * What the audio in this track is, as far as the loader could tell. Drives
 * the mixer-row overflow menu: a `full-mix` row can be re-split into
 * drums + backing; a `drums` row can be split into per-instrument pieces;
 * the other roles can't usefully feed either separation stage. `unknown`
 * is the safe default for ad-hoc user-loaded files; the menu enables
 * both items and lets the user pick.
 *
 * Source → role assignment lives at the loader sites in
 * `src/editing/store.ts` (ParaDB pack, debug bundle / transcriber,
 * ad-hoc drop).
 */
export type AudioTrackRole =
  | 'full-mix'    // contains drums + everything else (or could)
  | 'no-drums'    // drumless backing, produced by our own Demucs run
  | 'drums'       // isolated drum kit, all pieces together
  | 'drum-piece'  // a single instrument (kick / snare / hi-hat / cymbals)
  | 'unknown';    // ad-hoc / can't tell

/**
 * One loaded audio (backing) track. Observable so the mixer can react
 * to in-place colour overrides without reloading the track. Identity
 * (id, filename, buffer, sourceBlob, durationSec, lane, role) is
 * fixed at construction; the only mutable field is the colour override
 * the row's overflow menu writes into.
 *
 * Implements the unified {@link Track} interface so the picker UI can
 * read/write `track.color` without branching on track kind. The colour
 * computation falls back through the {@link resolveAudioInheritedColor}
 * chain so an unset override inherits from the grouped instrument row.
 */
export class AudioTrack implements Track {
  readonly kind = 'audio' as const;
  readonly id: AudioTrackId;
  /** Original filename for display in the gutter and tooltips. */
  readonly filename: string;
  /**
   * Decoded PCM. Fed once to the stretch worklet via `addBuffers` to
   * seed its input ring; the waveform renderer also reads channel data
   * off it for peak extraction. Kept on the immutable track so mid-
   * playback reloads with the same id can swap PCM without tearing the
   * slot down.
   */
  readonly buffer: AudioBuffer;
  /**
   * Original encoded bytes, retained for upload paths (the lyrics-
   * alignment flow re-uploads the source file to the transcriber).
   * Playback doesn't read this; the worklet plays from the decoded
   * `buffer` directly; but holding the blob lets us avoid re-fetching
   * the original from network on a re-upload.
   */
  readonly sourceBlob: Blob;
  readonly durationSec: number;
  /**
   * Own-state fallback for {@link lane}. Holds the load-time mapping
   * (a debug bundle's `mapping` entry) and the value baked in by
   * {@link detachLane} when the row is dragged out of a group. The
   * effective lane is normally *derived* from the mixer group (see
   * {@link lane}); this only takes over when the track is solo.
   * Undefined for ad-hoc / drumless tracks that were never mapped.
   */
  private _laneOverride: string | undefined = undefined;
  /**
   * What the loader believes the audio is. Drives the per-row overflow
   * menu's enable matrix. Undefined is treated as `unknown`.
   */
  readonly role?: AudioTrackRole;

  /** Per-track waveform-colour override (`#rrggbb`). `undefined` means
   *  "inherit from the grouped instrument track via the mixer context". */
  _color: string | undefined = undefined;

  /**
   * @param fields    Immutable identity + audio bits.
   * @param getCtx    Late-bound mixer-context getter. Called on every
   *                  read of {@link color} so a context attached after
   *                  the track was constructed still applies.
   */
  constructor(
    fields: {
      id: AudioTrackId;
      filename: string;
      buffer: AudioBuffer;
      sourceBlob: Blob;
      durationSec: number;
      lane?: string;
      role?: AudioTrackRole;
    },
    private readonly getCtx: () => MixerContext | undefined,
  ) {
    this.id = fields.id;
    this.filename = fields.filename;
    this.buffer = fields.buffer;
    this.sourceBlob = fields.sourceBlob;
    this.durationSec = fields.durationSec;
    this._laneOverride = fields.lane;
    this.role = fields.role;
    // `_color` + `_laneOverride` are the mutable observable fields and
    // `lane` is a computed (derives from the mixer group); the buffer /
    // blob are large immutables that don't need MobX wrappers.
    makeAutoObservable<
      this,
      'getCtx' | 'id' | 'filename' | 'buffer' | 'sourceBlob' | 'durationSec' | 'role'
    >(this, {
      getCtx: false,
      id: false,
      filename: false,
      buffer: false,
      sourceBlob: false,
      durationSec: false,
      role: false,
    });
  }

  get color(): string {
    if (this._color !== undefined) return this._color;
    const ctx = this.getCtx();
    if (!ctx) return AUDIO_FALLBACK_COLOR;
    return resolveAudioInheritedColor(this.id, this.lane, ctx) ?? AUDIO_FALLBACK_COLOR;
  }

  set color(c: string) {
    this._color = c;
  }

  /** Drop the user override so the colour reverts to inheritance. */
  clearColor(): void {
    this._color = undefined;
  }

  /**
   * Instrument lanes sharing this audio track's mixer group, in row
   * order. Empty when solo. Delegates to the shared free function (NOT a
   * method) so the read of `trackOrder` stays tracked when called from
   * the `lane` / `color` computeds, see its note in `tracks.ts`.
   */
  private get inGroupLanes(): string[] {
    const ctx = this.getCtx();
    return ctx ? groupInstrumentLanes(this.id, ctx) : [];
  }

  /**
   * DSL lane letter (e.g. `k`, `s`, `h`) this audio track is the
   * isolated stem of. **Derived from the mixer group** (the user's own
   * drag-and-drop grouping is the source of truth): when the track shares
   * a group with one or more instrument rows it reports that lane, so
   * regrouping the row to a different instrument retints it correctly
   * instead of clinging to the load-time mapping. The load-time mapping
   * (kept in {@link _laneOverride}) only acts as the tiebreaker when one
   * audio file maps to several lanes in the same group, and as the
   * fallback when the track is solo. Undefined for ad-hoc / drumless
   * tracks that were never mapped or grouped.
   */
  get lane(): string | undefined {
    const inGroup = this.inGroupLanes;
    if (inGroup.length > 0) {
      if (this._laneOverride !== undefined && inGroup.includes(this._laneOverride)) {
        return this._laneOverride;
      }
      return inGroup[0];
    }
    return this._laneOverride;
  }

  /**
   * Bake the current group-derived lane into {@link _laneOverride} so a
   * row dragged out of its group keeps its instrument association as its
   * own state. Called by the mixer the moment a reorder clears this row's
   * group — while the old group is still live — so the association isn't
   * lost. No-op for a track that wasn't grouped.
   */
  detachLane(): void {
    const inGroup = this.inGroupLanes;
    if (inGroup.length === 0) return;
    this._laneOverride =
      this._laneOverride !== undefined && inGroup.includes(this._laneOverride)
        ? this._laneOverride
        : inGroup[0];
  }

  /** Whether the user has set an explicit override on this track.
   *  Drives the picker popover's Reset enable state. */
  get hasOverride(): boolean {
    return this._color !== undefined;
  }
}

/**
 * Live playback state for an audio track during a single `play()` call.
 * Held inside the {@link AudioTrackPlaybackController} (created and
 * destroyed by the player) rather than on the immutable {@link
 * AudioTrack} so reloading a track mid-flight doesn't leak old nodes.
 *
 * The slot is built lazily: `node` is a Promise resolved once the
 * stretch worklet has finished loading PCM into its input ring (the
 * worklet's `addBuffers` is async because it round-trips a postMessage
 * to the audio thread). Scheduling calls await the node first, then
 * issue the `schedule` message; that means a play() may race a still-
 * loading slot, which is handled by the `gen` supersession check.
 */
type ActiveAudioTrack = {
  id: AudioTrackId;
  /** Per-track mute/solo/volume gain, persists across (re)schedules. */
  gainNode: GainNode;
  /**
   * The decoded PCM held on the slot so a mid-flight track reload can
   * detect a change vs the new track's buffer and rebuild only when
   * the underlying audio actually changed.
   */
  buffer: AudioBuffer;
  /**
   * The track's stretch worklet, or a Promise that resolves to it
   * during the first-load window. Once resolved the same node is reused
   * for the slot's lifetime: speed change / seek / pause / resume are
   * all `schedule()` messages, no node teardown.
   */
  node: Promise<StretchNode>;
  /**
   * Bumped on every (re)schedule and cancel. A still-loading slot
   * checks this against the gen captured at request time before sending
   * its own `schedule` message, so a superseded request can't fire late.
   */
  gen: number;
};

/** Filter for audio-track mute/solo, parallel to {@link PlayerFilter} for lanes. */
export type AudioTrackFilter = {
  mutedAudioTracks: ReadonlySet<AudioTrackId>;
  soloedAudioTracks: ReadonlySet<AudioTrackId>;
  /**
   * True when a solo is engaged anywhere; on an audio track OR a lane
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
 * backing recordings, so the music is halved by default; a 100% fader
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
 * Decode an audio file's bytes into an {@link AudioBuffer} for playback,
 * and return the original Blob so an upload path (e.g. lyrics alignment)
 * can re-submit the source file without re-fetching.
 *
 * `decodeAudioData` is delegated to the browser; FLAC works in modern
 * Chromium / Firefox / Safari. WAV, MP3, and (most) AAC all work too.
 */
export async function decodeAudioTrackFile(
  ctx: AudioContext,
  file: File,
): Promise<{ buffer: AudioBuffer; sourceBlob: Blob }> {
  // `file.arrayBuffer()` returns a fresh ArrayBuffer that we own; pass
  // it straight to decodeAudioData (which neuters the input) instead of
  // copying defensively. The File itself still owns the original bytes
  // and is returned to the caller as `sourceBlob`.
  const bytes = await file.arrayBuffer();
  const buffer = await ctx.decodeAudioData(bytes);
  return { buffer, sourceBlob: file };
}

/**
 * Fetch an audio track from a URL (typically the transcriber's
 * `/outputs/...` route) and decode it. Equivalent to {@link
 * decodeAudioTrackFile} for File inputs; kept as a separate entry
 * point so the auto-load-on-transcribe path doesn't have to round-trip
 * through a synthetic File.
 */
export async function decodeAudioTrackUrl(
  ctx: AudioContext,
  url: string,
): Promise<{ buffer: AudioBuffer; sourceBlob: Blob }> {
  const res = await backendFetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch audio track (${res.status} ${res.statusText})`);
  }
  const bytes = await res.arrayBuffer();
  // The Blob constructor copies its input bytes per spec, so the blob
  // we hand back stays valid after `decodeAudioData` neuters our
  // `bytes` view.
  const sourceBlob = new Blob([bytes]);
  const buffer = await ctx.decodeAudioData(bytes);
  return { buffer, sourceBlob };
}

/**
 * Manages live audio-track playback for one `play()` cycle. Created
 * lazily by the player when it first needs to schedule tracks;
 * destroyed (via `dispose`) on every player `stop()` so a fresh play()
 * starts with no residual nodes.
 *
 * Mute is a gain mutation on the per-track `GainNode` (no reschedule).
 * Speed changes and seeks are `schedule()` messages on the stretch
 * worklet; no node teardown, no path swap, no audible gap at any speed.
 */
export class AudioTrackPlaybackController {
  private active: Map<AudioTrackId, ActiveAudioTrack> = new Map();
  /**
   * Per-track resolved gain that's CURRENTLY written to the slot's
   * `GainNode.gain`. The Web Audio node holds the canonical value; this
   * map mirrors it so a non-React renderer (or any TS consumer reading
   * the player) can observe what's actually audible without poking into
   * the audio graph. Entry exists from the slot's first
   * `gainNode.gain.value = ...` write through to teardown; absence
   * means the track isn't currently scheduled.
   */
  appliedGains: Map<AudioTrackId, number> = new Map();

  /**
   * `destination` is the node every track's `GainNode` feeds into; the
   * player passes its all-audio-tracks master bus so that fader (and,
   * downstream, the page fader) scales the music. Defaults to
   * `ctx.destination` so a bare controller still makes sound.
   */
  constructor(
    private readonly ctx: AudioContext,
    private readonly destination: AudioNode = ctx.destination,
  ) {
    makeAutoObservable<this, 'active' | 'ctx' | 'destination' | 'scheduleOne' | 'ensureSlot' | 'buildStretchSlot'>(this, {
      active: false,
      ctx: false,
      destination: false,
      scheduleOne: false,
      ensureSlot: false,
      buildStretchSlot: false,
    });
  }

  /**
   * Internal helper: write `gain` onto the slot's `GainNode` AND mirror
   * it into the observable {@link appliedGains}. All gain mutations
   * funnel through here so the mirror can't drift from the live value.
   */
  private setSlotGain(slot: ActiveAudioTrack, gain: number): void {
    slot.gainNode.gain.value = gain;
    runInAction(() => {
      this.appliedGains.set(slot.id, gain);
    });
  }

  /**
   * Start every loaded audio track at `audioStartTime`, with the input
   * position corresponding to `jotOffsetSec` (negative during lead-in
   *; the buffer plays from t=0 in that case and the audible lead-in
   * is just the recording's intro).
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
    songLeadInSec: number,
    gainFor: (id: AudioTrackId) => number,
  ): void {
    for (const track of tracks) {
      this.scheduleOne(
        track,
        audioStartTime,
        jotOffsetSec,
        speed,
        songLeadInSec,
        gainFor(track.id),
      );
    }
  }

  /**
   * Apply a new playback rate to every currently-active track. The
   * stretch worklet handles every speed in [0.25, 1.25] uniformly so
   * this is just a `schedule({rate, output})` per slot; no path
   * switch, no source rebuild. The caller still passes the anchor
   * state (`audioStartTime`, `jotOffsetSec`, `songLeadInSec`, `gainFor`)
   * for parity with the drum-side reschedule, but they're unused in
   * the worklet path.
   */
  setPlaybackRate(
    speed: number,
    audioStartTime: number,
    _jotOffsetSec: number,
    _songLeadInSec: number,
    _gainFor: (id: AudioTrackId) => number,
  ): void {
    const when = Math.max(audioStartTime, this.ctx.currentTime + SCHEDULE_PAD_SEC);
    for (const slot of this.active.values()) {
      const gen = slot.gen;
      void slot.node
        .then((node) => {
          if (slot.gen !== gen) return;
          return node.schedule({ rate: speed, output: when });
        })
        .catch((err) => {
          console.warn('[audio-tracks] setPlaybackRate threw', err);
        });
    }
  }

  private scheduleOne(
    track: AudioTrack,
    audioStartTime: number,
    jotOffsetSec: number,
    speed: number,
    songLeadInSec: number,
    gain: number,
  ): void {
    // Input time = jot time - songLeadIn. A negative jot offset
    // (lead-in) clamps to 0 so the recording's own intro is what plays
    // during the lead-in, exactly as the legacy path did.
    const inputTime = Math.max(0, jotOffsetSec - songLeadInSec);
    const slot = this.ensureSlot(track);
    this.setSlotGain(slot, gain);

    // Supersede any in-flight schedule message that hasn't been posted
    // yet (e.g. one waiting on the initial node load).
    const gen = ++slot.gen;
    const when = Math.max(audioStartTime, this.ctx.currentTime + SCHEDULE_PAD_SEC);
    void slot.node
      .then((node) => {
        if (slot.gen !== gen) return;
        return node.schedule({
          active: true,
          input: inputTime,
          rate: speed,
          output: when,
        });
      })
      .catch((err) => {
        console.warn('[audio-tracks] scheduleOne threw', err);
      });
  }

  private ensureSlot(track: AudioTrack): ActiveAudioTrack {
    const existing = this.active.get(track.id);
    if (existing) {
      if (existing.buffer === track.buffer) return existing;
      // Same slot, different audio (track reloaded mid-playback). The
      // cheapest path is a fresh node; the old one's PCM ring is
      // bound to the previous buffer, and dropBuffers + addBuffers
      // would still need a round-trip per channel anyway. Stop the old
      // slot's schedule, disconnect its node, and replace.
      existing.gen++;
      const oldNode = existing.node;
      void oldNode.then((n) => {
        try {
          void n.stop();
          n.disconnect();
        } catch (err) {
          console.debug('[audio-tracks] old node teardown threw', err);
        }
      });
      existing.buffer = track.buffer;
      existing.node = this.buildStretchSlot(existing.gainNode, track.buffer);
      return existing;
    }
    const gainNode = this.ctx.createGain();
    gainNode.gain.value = 1;
    gainNode.connect(this.destination);

    const slot: ActiveAudioTrack = {
      id: track.id,
      gainNode,
      buffer: track.buffer,
      node: this.buildStretchSlot(gainNode, track.buffer),
      gen: 0,
    };
    this.active.set(track.id, slot);
    runInAction(() => {
      this.appliedGains.set(track.id, 1);
    });
    return slot;
  }

  /**
   * Construct a stretch worklet for the slot, load its PCM, wire its
   * output through `gainNode`, and return the node. Errors propagate
   * back through the slot's `node` Promise; `scheduleOne` /
   * `setPlaybackRate` log and skip on rejection, so a single bad track
   * load doesn't kill the rest of the schedule.
   */
  private buildStretchSlot(gainNode: GainNode, buffer: AudioBuffer): Promise<StretchNode> {
    return createStretchNode(this.ctx, buffer).then((node) => {
      node.connect(gainNode);
      return node;
    });
  }

  /**
   * Fully tear down one track's slot (node, gain). Used by the
   * player's `clearAudioTrack` so a removed track leaves no dangling
   * nodes.
   */
  dropAudioTrack(id: AudioTrackId): void {
    const slot = this.active.get(id);
    if (!slot) return;
    slot.gen++;
    void slot.node
      .then((node) => {
        try {
          void node.stop();
          node.disconnect();
        } catch (err) {
          console.debug('[audio-tracks] dropAudioTrack node teardown threw', err);
        }
      })
      .catch(() => {});
    try {
      slot.gainNode.disconnect();
    } catch (err) {
      console.debug('[audio-tracks] dropAudioTrack gain.disconnect threw', err);
    }
    this.active.delete(id);
    runInAction(() => {
      this.appliedGains.delete(id);
    });
  }

  /** Apply a live gain decision (mute/solo + volume) to every active track. */
  applyAudibility(gainFor: (id: AudioTrackId) => number): void {
    for (const slot of this.active.values()) {
      this.setSlotGain(slot, gainFor(slot.id));
    }
  }

  /**
   * Stop playback of every track without tearing down the worklet
   * graph: each slot's stretch node is `.stop()`-ed (an `active: false`
   * schedule message), so a subsequent `scheduleAll` can resume from
   * a fresh input position. Used by `setPlaybackSpeed`, `seek`, and
   * (via the player) pause.
   *
   * Named `cancelSources` for parity with the player's drum-side
   * vocabulary.
   */
  cancelSources(): void {
    const when = this.ctx.currentTime + SCHEDULE_PAD_SEC;
    for (const slot of this.active.values()) {
      slot.gen++;
      void slot.node
        .then((node) => node.stop(when))
        .catch((err) => {
          console.debug('[audio-tracks] cancelSources stop threw', err);
        });
    }
  }

  /** Teardown; invoked when playback ends so the graph doesn't leak. */
  dispose(): void {
    for (const slot of this.active.values()) {
      slot.gen++;
      void slot.node
        .then((node) => {
          try {
            void node.stop();
            node.disconnect();
          } catch (err) {
            console.debug('[audio-tracks] dispose node teardown threw', err);
          }
        })
        .catch(() => {});
      try {
        slot.gainNode.disconnect();
      } catch (err) {
        console.debug('[audio-tracks] dispose gain.disconnect threw', err);
      }
    }
    this.active.clear();
    runInAction(() => {
      this.appliedGains.clear();
    });
  }
}

/** Re-export the warmup entry point so `JotPlayer` can preload the
 * stretch worklet alongside the drum soundfont without a second import. */
export { preloadStretch };
