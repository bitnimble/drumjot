/**
 * Audio-track playback alongside the MIDI score.
 *
 * Loaded audio tracks (a ParaDB pack's song/drum tracks, the transcriber's
 * `no_drums`/`drum_stem` FLACs, ad-hoc backing tracks) play through the
 * same `AudioContext` as the smplr drum machine, so they share the audio
 * clock and start / stop / speed changes stay in lockstep with MIDI
 * playback.
 *
 * Each track plays through an `HTMLAudioElement` wired into the graph
 * via a `MediaElementAudioSourceNode` → per-track `GainNode` →
 * `ctx.destination`. Mute/solo/volume mutate `gain.value` directly so
 * the response is immediate. Speed changes set the element's
 * `playbackRate`; the element's `preservesPitch` keeps the recording
 * at its original pitch when slowed/sped, so practising a song at half
 * speed no longer detunes the backing track (the drum scheduler
 * already preserves pitch by spacing hits, so the two now agree).
 *
 * Why a media element rather than an `AudioBufferSourceNode`: a buffer
 * source's `playbackRate` rescales time *and* pitch together with no
 * way to separate them, so pitch preservation would need a granular /
 * phase-vocoder library. `HTMLMediaElement.preservesPitch` is the
 * browser's built-in time-stretch — no dependency, good quality.
 *
 * The decoded `AudioBuffer` is still kept (see {@link AudioTrack}) — but
 * only for waveform rendering, not playback. The waveform peaks are
 * computed by streaming through `buffer.getChannelData(ch)` directly;
 * we deliberately do *not* pre-collapse to a mono `Float32Array` at
 * load time because that upfront pass dominated per-track load time
 * (full-buffer scan on the main thread) for no real benefit — the
 * peaks compute touches every sample anyway and can fold channels in
 * the same pass.
 *
 * Sync caveat: a media element is driven by its own media clock, not
 * the `AudioContext` clock, so start alignment with the drum scheduler
 * is approximate (a few ms of jitter) rather than sample-accurate, and
 * `ctx.suspend()` no longer transitively freezes tracks — the player
 * pauses/realigns the elements explicitly around pause/resume.
 */
import { Pixels, RenderedJot, px } from 'src/jot';
import { buildTimeline, JotTimeline } from './timeline';

/**
 * Opaque per-track id. Every loaded audio track gets a fresh unique id
 * (the player allocates them); there is no fixed set and no music-vs-drums
 * distinction — an audio track is just audio that plays alongside the
 * score. Iteration order of the player's track map is load order.
 */
export type AudioTrackId = string;

// Drift correction (see {@link AudioTrackPlaybackController.correctDrift}).
// Below this offset the track is considered in sync — small enough that
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

export type AudioTrack = {
  id: AudioTrackId;
  /** Original filename for display in the gutter and tooltips. */
  filename: string;
  /** Decoded PCM. Used only for waveform rendering (playback is via the element). */
  buffer: AudioBuffer;
  /**
   * Object URL of the original encoded bytes, fed to the playback
   * `HTMLAudioElement`. Owned by the player, which revokes it when the
   * track is replaced or cleared so blobs don't leak across reloads.
   */
  objectUrl: string;
  durationSec: number;
};

/**
 * Live playback state for an audio track during a single `play()` call.
 * Held inside the {@link AudioTrackPlaybackController} (created and
 * destroyed by the player) rather than on the immutable {@link
 * AudioTrack} so reloading a track mid-flight doesn't leak old nodes.
 */
type ActiveAudioTrack = {
  id: AudioTrackId;
  /** Per-track mute/solo/volume gain, persists across (re)schedules. */
  gainNode: GainNode;
  /** Playback element. One per slot; reused across reschedules. */
  el: HTMLAudioElement;
  /**
   * The element's graph tap. `MediaElementAudioSourceNode` may only be
   * created once per element, so it (and the element) live for the
   * controller's lifetime rather than being rebuilt per schedule.
   */
  node: MediaElementAudioSourceNode;
  /**
   * The object URL currently wired into `el`. Lets {@link
   * AudioTrackPlaybackController.ensureSlot} detect a mid-playback track
   * replacement (same id, new audio) and rebuild the element.
   */
  objectUrl: string;
  /** Pending aligned-start timer, cleared on cancel/reschedule. */
  startTimer: number | undefined;
  /**
   * Bumped on every (re)schedule and cancel. A deferred start (waiting
   * on a timer or `loadedmetadata`) checks this before firing so a
   * superseded schedule can't start the element late.
   */
  gen: number;
};

/** Filter for audio-track mute/solo, parallel to {@link PlayerFilter} for pitches. */
export type AudioTrackFilter = {
  mutedAudioTracks: ReadonlySet<AudioTrackId>;
  soloedAudioTracks: ReadonlySet<AudioTrackId>;
  /**
   * True when a solo is engaged anywhere — on an audio track OR a pitch
   * row. Solo is one global mode shared across both domains, so soloing
   * a drum instrument silences the audio tracks too (and vice versa).
   * The store computes this since it owns both solo sets.
   */
  soloActive: boolean;
  /** Per-track volume multiplier in [0, 1]; missing = full (1). */
  volumes: ReadonlyMap<AudioTrackId, number>;
};

export const PASSTHROUGH_AUDIO_TRACK_FILTER: AudioTrackFilter = {
  mutedAudioTracks: new Set<AudioTrackId>(),
  soloedAudioTracks: new Set<AudioTrackId>(),
  soloActive: false,
  volumes: new Map<AudioTrackId, number>(),
};

export function isAudioTrackAudibleUnder(
  id: AudioTrackId,
  filter: AudioTrackFilter,
): boolean {
  if (filter.mutedAudioTracks.has(id)) return false;
  if (filter.soloActive && !filter.soloedAudioTracks.has(id)) return false;
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
 * Bar-by-bar waveform peak extraction. For each pixel column the
 * timeline maps to, scan the buffer's samples in that time slice and
 * return the [min, max] envelope of the channels collapsed to mono.
 * The result is a flat array of length `2 * pixels` (interleaved min,
 * max) so the consumer can stream it straight into Canvas / SVG
 * without re-allocating per row.
 *
 * Channels are folded inline (averaged) inside the per-pixel loop so
 * there is no upfront full-buffer pass — that used to be the dominant
 * blocking cost when loading several tracks back to back.
 *
 * `startOffsetSec` shifts buffer-time relative to jot-time: at jot
 * t=0 the audio is at t=startOffsetSec (the recording's lead-in
 * before the first drum hit). Samples outside the buffer collapse
 * to 0 / 0, leaving blank space on either side of the waveform.
 */
export function computeWaveformPeaks(
  buffer: AudioBuffer,
  timeline: JotTimeline,
  totalWidthPx: number,
  startOffsetSec: number,
): Float32Array {
  const peaks = new Float32Array(totalWidthPx * 2);
  if (totalWidthPx <= 0 || timeline.bars.length === 0) return peaks;
  const sampleRate = buffer.sampleRate;
  const numChannels = buffer.numberOfChannels;
  const sampleLen = buffer.length;
  // Snapshot the channel views once — `getChannelData` is allowed to
  // (and on some engines does) re-validate on every call.
  const channels: Float32Array[] = new Array(numChannels);
  for (let ch = 0; ch < numChannels; ch++) channels[ch] = buffer.getChannelData(ch);
  const channelScale = 1 / numChannels;
  // Pre-build a quick (pixel -> jot-time) lookup by walking the
  // rendered bars once. timeToX is rendered->jot; we want the inverse,
  // so build a piecewise-linear table over the bar's pixel range.
  const voice = timeline.rendered?.resolved.voices[0];
  const renderedBars = voice?.bars ?? [];
  // The note grid (and thus the playhead — see timeToX) sits `pad` px
  // right of the bar boxes. Shift the waveform by the same amount so a
  // transient renders directly under the note it belongs to instead of
  // a constant `pad` px to its left.
  const pad = (voice?.notePadPx as number) ?? 0;

  // Lead-in: the pixels before bar 1 (reserved for the recording's
  // pre-roll) show audio seconds [0, startOffset). Bar 1's left edge is
  // at `firstX`, which the layout scaled so `firstX` px == `startOffset`
  // s at the same px/s the bars use — so audio is continuous across the
  // bar-1 boundary and the drum notes line up with where the drums
  // actually enter in the waveform.
  const firstX = renderedBars.length ? (renderedBars[0].x as number) : 0;
  if (startOffsetSec > 0 && firstX > 0) {
    // Pre-roll occupies [pad, firstX + pad) once the note-grid shift is
    // applied; pixels left of `pad` are before the recording starts.
    const pxStart = Math.max(0, Math.floor(pad));
    const pxEnd = Math.min(totalWidthPx, Math.ceil(firstX + pad));
    for (let p = pxStart; p < pxEnd; p++) {
      const tAudio0 = ((p - pad) / firstX) * startOffsetSec;
      const tAudio1 = ((p + 1 - pad) / firstX) * startOffsetSec;
      const s0 = Math.max(0, Math.floor(tAudio0 * sampleRate));
      const s1 = Math.min(sampleLen, Math.ceil(tAudio1 * sampleRate));
      writePixelPeak(channels, numChannels, channelScale, s0, s1, peaks, p * 2);
    }
  }

  for (let bi = 0; bi < renderedBars.length; bi++) {
    const rb = renderedBars[bi];
    const timing = timeline.bars[bi];
    if (!timing) continue;
    const x0 = (rb.x as number) + pad;
    const w = rb.width as number;
    const pxStart = Math.max(0, Math.floor(x0));
    const pxEnd = Math.min(totalWidthPx, Math.ceil(x0 + w));
    for (let p = pxStart; p < pxEnd; p++) {
      const frac0 = (p - x0) / w;
      const frac1 = (p + 1 - x0) / w;
      const tJot0 = timing.startSec + frac0 * timing.durationSec;
      const tJot1 = timing.startSec + frac1 * timing.durationSec;
      const tAudio0 = tJot0 + startOffsetSec;
      const tAudio1 = tJot1 + startOffsetSec;
      const s0 = Math.max(0, Math.floor(tAudio0 * sampleRate));
      const s1 = Math.min(sampleLen, Math.ceil(tAudio1 * sampleRate));
      writePixelPeak(channels, numChannels, channelScale, s0, s1, peaks, p * 2);
    }
  }
  return peaks;
}

/**
 * Scan one pixel column's worth of samples across every channel, fold
 * them to mono on the fly, and write the [min, max] envelope into
 * `peaks[pIdx]` / `peaks[pIdx + 1]`. Mono / stereo get specialised
 * inner loops (the common cases — saves a tight inner-loop branch and
 * a multiplication per sample); >2 channels fall through to a generic
 * sum. Empty ranges write zeros so silent regions render flat.
 */
function writePixelPeak(
  channels: Float32Array[],
  numChannels: number,
  channelScale: number,
  s0: number,
  s1: number,
  peaks: Float32Array,
  pIdx: number,
): void {
  if (s1 <= s0) {
    peaks[pIdx] = 0;
    peaks[pIdx + 1] = 0;
    return;
  }
  let mn = Infinity;
  let mx = -Infinity;
  if (numChannels === 1) {
    const data = channels[0];
    for (let s = s0; s < s1; s++) {
      const v = data[s];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  } else if (numChannels === 2) {
    const c0 = channels[0];
    const c1 = channels[1];
    for (let s = s0; s < s1; s++) {
      const v = (c0[s] + c1[s]) * 0.5;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  } else {
    for (let s = s0; s < s1; s++) {
      let v = 0;
      for (let ch = 0; ch < numChannels; ch++) v += channels[ch][s];
      v *= channelScale;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  if (mn === Infinity) {
    peaks[pIdx] = 0;
    peaks[pIdx + 1] = 0;
  } else {
    peaks[pIdx] = mn;
    peaks[pIdx + 1] = mx;
  }
}

/**
 * Convenience wrapper: takes a {@link RenderedJot} and an audio track,
 * returns the waveform peaks at the score's current pixel width.
 * Re-call when zoom changes (the rendered bars' `width` changes
 * reactively, the peaks need to follow).
 */
export function computeWaveformPeaksForJot(
  rendered: RenderedJot,
  buffer: AudioBuffer,
  startOffsetSec: number,
): { peaks: Float32Array; widthPx: Pixels } {
  const timeline = buildTimeline(rendered);
  const voice = rendered.resolved.voices[0];
  const widthPx = (voice?.width ?? px(0)) as Pixels;
  const peaks = computeWaveformPeaks(
    buffer,
    timeline,
    widthPx as number,
    startOffsetSec,
  );
  return { peaks, widthPx };
}

/**
 * Manages live audio-track playback for one `play()` cycle. Created
 * lazily by the player when it first needs to schedule tracks;
 * destroyed (via `stopAll`) on every player `stop()` so a fresh play()
 * starts with no residual nodes.
 *
 * Speed and mute changes are handled like the drum scheduler does —
 * mute is a gain mutation (no re-schedule), speed re-creates the
 * source so the new `playbackRate` takes effect immediately at the
 * current jot-time anchor.
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
    startOffsetSec: number,
    gainFor: (id: AudioTrackId) => number,
  ): void {
    for (const track of tracks) {
      this.scheduleOne(
        track,
        audioStartTime,
        jotOffsetSec,
        speed,
        startOffsetSec,
        gainFor(track.id),
      );
    }
  }

  private scheduleOne(
    track: AudioTrack,
    audioStartTime: number,
    jotOffsetSec: number,
    speed: number,
    startOffsetSec: number,
    gain: number,
  ): void {
    // Media time = jot time + startOffset. A negative jot offset
    // (lead-in) clamps to 0 so the recording's own intro is what plays
    // during the lead-in, exactly as the old buffer path did.
    const mediaOffset = Math.max(0, jotOffsetSec + startOffsetSec);
    const slot = this.ensureSlot(track);
    slot.gainNode.gain.value = gain;

    // Supersede any deferred start from a previous (re)schedule.
    const gen = ++slot.gen;
    if (slot.startTimer !== undefined) {
      window.clearTimeout(slot.startTimer);
      slot.startTimer = undefined;
    }
    slot.el.pause();

    const el = slot.el;
    el.playbackRate = speed;
    // Built-in time-stretch: keep the original pitch when the rate
    // changes. `preservesPitch` is the standard property and defaults
    // to true in current browsers; set it explicitly (plus the legacy
    // vendor names) so older engines behave the same.
    setPreservesPitch(el, true);

    const fire = () => {
      slot.startTimer = undefined;
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
        // play() rejects if a pause races it (reschedule) — benign.
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
    else slot.startTimer = window.setTimeout(fire, delaySec * 1000);
  }

  private ensureSlot(track: AudioTrack): ActiveAudioTrack {
    const existing = this.active.get(track.id);
    if (existing) {
      if (existing.objectUrl === track.objectUrl) return existing;
      // Same slot, different audio (track reloaded mid-playback). A
      // MediaElementAudioSourceNode is bound to its element for life,
      // so swap in a fresh element + node, keeping the gain node (and
      // thus the mute/solo/volume state) intact.
      existing.gen++;
      if (existing.startTimer !== undefined) {
        window.clearTimeout(existing.startTimer);
        existing.startTimer = undefined;
      }
      try {
        existing.el.pause();
        existing.node.disconnect();
        existing.el.removeAttribute('src');
        existing.el.load();
      } catch (err) {
        console.debug('[audio-tracks] slot rebuild teardown threw', err);
      }
      const el = makeAudioTrackElement(track.objectUrl);
      const node = this.ctx.createMediaElementSource(el);
      node.connect(existing.gainNode);
      existing.el = el;
      existing.node = node;
      existing.objectUrl = track.objectUrl;
      return existing;
    }
    const gainNode = this.ctx.createGain();
    gainNode.gain.value = 1;
    gainNode.connect(this.destination);

    const el = makeAudioTrackElement(track.objectUrl);
    const node = this.ctx.createMediaElementSource(el);
    node.connect(gainNode);

    const slot: ActiveAudioTrack = {
      id: track.id,
      gainNode,
      el,
      node,
      objectUrl: track.objectUrl,
      startTimer: undefined,
      gen: 0,
    };
    this.active.set(track.id, slot);
    return slot;
  }

  /**
   * Fully tear down one track's slot (element, node, gain). Used by the
   * player's `clearAudioTrack` so a removed track leaves no dangling
   * element.
   */
  dropAudioTrack(id: AudioTrackId): void {
    const slot = this.active.get(id);
    if (!slot) return;
    slot.gen++;
    if (slot.startTimer !== undefined) window.clearTimeout(slot.startTimer);
    try {
      slot.el.pause();
      slot.node.disconnect();
      slot.gainNode.disconnect();
      slot.el.removeAttribute('src');
      slot.el.load();
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
      const el = slot.el;
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
   * Stop playback of every track (pause the element + drop any pending
   * deferred start) without tearing down the graph nodes, so a
   * subsequent `scheduleAll` can reuse the same element / gain. Used by
   * `setPlaybackSpeed`, `seek`, and (via the player) pause.
   *
   * Named `cancelSources` for parity with the player's drum-side
   * vocabulary even though there are no longer per-schedule sources.
   */
  cancelSources(): void {
    for (const slot of this.active.values()) {
      slot.gen++;
      if (slot.startTimer !== undefined) {
        window.clearTimeout(slot.startTimer);
        slot.startTimer = undefined;
      }
      try {
        slot.el.pause();
      } catch (err) {
        console.debug('[audio-tracks] el.pause threw', err);
      }
    }
  }

  /** Teardown — invoked when playback ends so the graph doesn't leak. */
  dispose(): void {
    this.cancelSources();
    for (const slot of this.active.values()) {
      try {
        slot.node.disconnect();
        slot.gainNode.disconnect();
        // Detach the source so the browser can release the decoded
        // media; the Blob URL itself is revoked by the player.
        slot.el.removeAttribute('src');
        slot.el.load();
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
