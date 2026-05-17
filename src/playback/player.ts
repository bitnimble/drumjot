/**
 * Browser playback of a Jot through smplr's `DrumMachine`.
 *
 * Lifecycle:
 *   - The `AudioContext` and `DrumMachine` are created on first `play()` —
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
import { DrumMachine } from 'smplr';
import { RenderedJot } from 'src/jot';
import { jotToEvents, PlaybackEvent } from './events';
import { midiNoteToRole, resolveGroupForRole } from './drums';
import { buildTimeline, EMPTY_TIMELINE, JotTimeline } from './timeline';

export type PlayerState = 'idle' | 'loading' | 'playing';

export type PlayerFilter = {
  mutedPitches: ReadonlySet<string>;
  /**
   * When non-empty, ONLY these pitches are audible (others behave as if
   * muted). Soloed-AND-muted = muted; explicit mute always wins so the
   * user can keep solo on while temporarily silencing a soloed row.
   */
  soloedPitches: ReadonlySet<string>;
};

export const PASSTHROUGH_FILTER: PlayerFilter = {
  mutedPitches: new Set(),
  soloedPitches: new Set(),
};

export function isAudibleUnder(pitch: string, filter: PlayerFilter): boolean {
  if (filter.mutedPitches.has(pitch)) return false;
  if (filter.soloedPitches.size > 0 && !filter.soloedPitches.has(pitch)) return false;
  return true;
}

// smplr 0.21 ships five kits: TR-808, Casio-RZ1, LM-2, MFB-512, Roland CR-8000.
// TR-808 is the most familiar default; swap by changing this constant or by
// exposing a kit picker in the toolbar later.
const DEFAULT_INSTRUMENT = 'TR-808';
// Small lead time so the first hit doesn't race the audio thread.
const SCHEDULE_LEAD_SECONDS = 0.05;
// Buffer added to the last event's time before flipping back to `idle`, so
// late-decaying samples (cymbals, open hats) aren't cut off visually.
const PLAYBACK_TAIL_SECONDS = 1.0;
// If smplr can't fetch its CDN samples within this window we give up so
// the UI doesn't sit on "Loading…" forever — typical local network failure
// mode that's otherwise invisible.
const LOAD_TIMEOUT_SECONDS = 30;

type Drums = ReturnType<typeof DrumMachine>;
type StopFn = (time?: number) => void;

export class JotPlayer {
  state: PlayerState = 'idle';
  /** Last error surfaced to the UI; cleared the next time playback succeeds. */
  errorMessage: string | undefined;
  /** Seconds since the current `play()` started (in JOT time — already
   * adjusted for `playbackSpeed`, so a 60-second jot played at 0.5x reports
   * `currentTime` going from 0 → 60 over 120 real seconds). */
  currentTime: number = 0;
  /** Bar-by-bar time→pixel map for the currently-playing jot. `EMPTY_TIMELINE` when idle. */
  timeline: JotTimeline = EMPTY_TIMELINE;
  /**
   * Tempo multiplier applied to scheduled events and the playhead. 1.0 =
   * native tempo; 0.5 = half speed (notes still sound at the same pitch
   * because we space scheduled `drums.start` times further apart rather
   * than touching sample playback rate). Persists across plays.
   */
  playbackSpeed: number = 1;

  private ctx: AudioContext | undefined;
  private drums: Drums | undefined;
  private groupNames: readonly string[] = [];
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
  private rafId: number | undefined;
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

  constructor() {
    makeAutoObservable(this);
  }

  /**
   * Update the mute/solo filter. Stored unconditionally; if playback is
   * in flight, every scheduled note is cancelled and the remaining
   * events (those whose audio time hasn't elapsed) are re-scheduled
   * against the new filter — so toggling M or S during playback takes
   * effect immediately, including bringing previously-muted rows back
   * in mid-song.
   */
  setFilter(filter: PlayerFilter): void {
    this.currentFilter = filter;
    if (this.state !== 'playing' || !this.ctx) return;

    const now = this.ctx.currentTime;
    const jotOffset = this.currentJotTime(now);
    this.cancelScheduledStops();
    this.scheduleEvents(jotOffset, now);
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

    runInAction(() => {
      this.playbackSpeed = clamped;
    });
    this.startContextTime = now;
    this.startJotTime = jotOffset;
    const lastTime = this.scheduleEvents(jotOffset, now);
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

  async play(rendered: RenderedJot): Promise<void> {
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
      this.startContextTime = audioStartTime;
      this.startJotTime = 0;
      const lastTime = this.scheduleEvents(0, audioStartTime);

      runInAction(() => {
        this.state = 'playing';
        this.timeline = timeline;
        this.currentTime = 0;
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

  stop(): void {
    if (this.endTimerId !== undefined) {
      window.clearTimeout(this.endTimerId);
      this.endTimerId = undefined;
    }
    this.stopRaf();
    this.cancelScheduledStops();
    this.drums?.stop();
    this.events = [];
    this.startJotTime = 0;
    runInAction(() => {
      if (this.state !== 'idle') this.state = 'idle';
      this.timeline = EMPTY_TIMELINE;
      this.currentTime = 0;
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
   * Events filtered out by `currentFilter`, or whose MIDI note doesn't
   * map to a kit sample, are skipped.
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
    let unresolvedRole = 0;
    let unresolvedGroup = 0;
    const speed = this.playbackSpeed;

    for (const ev of this.events) {
      if (ev.time < fromOffset) continue;
      if (!isAudibleUnder(ev.pitch, this.currentFilter)) {
        mutedFiltered++;
        continue;
      }
      const role = midiNoteToRole(ev.midiNote);
      if (role === undefined) {
        unresolvedRole++;
        continue;
      }
      const group = resolveGroupForRole(role, this.groupNames);
      if (group === undefined) {
        unresolvedGroup++;
        continue;
      }
      const t = audioStartTime + (ev.time - fromOffset) / speed;
      const stopFn = drums.start({ note: group, time: t, velocity: ev.velocity });
      this.scheduledStops.push(stopFn);
      scheduled++;
      if (t > lastTime) lastTime = t;
    }

    console.log(
      `[jotPlayer] scheduled ${scheduled}/${this.events.length} events ` +
        `(filtered: ${mutedFiltered}, unmapped MIDI: ${unresolvedRole}, ` +
        `no kit-group match: ${unresolvedGroup})`,
    );

    if (scheduled === 0 && this.events.length > 0 && this.state !== 'playing') {
      // Only throw on the initial schedule — during live reschedule it's
      // valid to end up with 0 audible notes (e.g. user soloed nothing).
      throw new Error(
        `None of ${this.events.length} notes mapped to a drum sample on the ${DEFAULT_INSTRUMENT} ` +
          `kit. Available groups: [${this.groupNames.join(', ')}]. ` +
          `See console for breakdown.`,
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
      const jotTime = this.currentJotTime(this.ctx.currentTime);
      runInAction(() => {
        this.currentTime = jotTime > 0 ? jotTime : 0;
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

  private async ensureLoaded(): Promise<{ drums: Drums; ctx: AudioContext }> {
    if (this.drums && this.ctx) return { drums: this.drums, ctx: this.ctx };

    const Ctx: typeof AudioContext | undefined =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) {
      throw new Error('Web Audio is not available in this browser.');
    }
    const ctx = new Ctx();
    const drums = DrumMachine(ctx, { instrument: DEFAULT_INSTRUMENT });

    // Race the sample-load promise against a timeout so a stuck CDN
    // surfaces as an error instead of an infinite "Loading…" state.
    await Promise.race([
      drums.load,
      new Promise<never>((_, reject) =>
        window.setTimeout(
          () =>
            reject(
              new Error(
                `Drum samples failed to load within ${LOAD_TIMEOUT_SECONDS}s — ` +
                  `check network access to smpldsnds.github.io from the browser.`,
              ),
            ),
          LOAD_TIMEOUT_SECONDS * 1000,
        ),
      ),
    ]);

    this.ctx = ctx;
    this.drums = drums;
    this.groupNames = drums.getGroupNames();
    console.log(`[jotPlayer] ${DEFAULT_INSTRUMENT} loaded; groups: ${this.groupNames.join(', ')}`);
    return { drums, ctx };
  }
}

export const jotPlayer = new JotPlayer();
