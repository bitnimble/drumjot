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
 *   - `state` is MobX-observable so the toolbar can switch button labels
 *     between Play / Loading / Stop without prop drilling.
 */
import { makeAutoObservable, runInAction } from 'mobx';
import { DrumMachine } from 'smplr';
import { Jot } from 'src/dsl';
import { jotToEvents } from './events';
import { midiNoteToRole, resolveGroupForRole } from './drums';

export type PlayerState = 'idle' | 'loading' | 'playing';

const DEFAULT_INSTRUMENT = 'TR-909';
// Small lead time so the first hit doesn't race the audio thread.
const SCHEDULE_LEAD_SECONDS = 0.05;
// Buffer added to the last event's time before flipping back to `idle`, so
// late-decaying samples (cymbals, open hats) aren't cut off visually.
const PLAYBACK_TAIL_SECONDS = 1.0;

type Drums = ReturnType<typeof DrumMachine>;

export class JotPlayer {
  state: PlayerState = 'idle';
  /** Last error surfaced to the UI; cleared the next time playback succeeds. */
  errorMessage: string | undefined;

  private ctx: AudioContext | undefined;
  private drums: Drums | undefined;
  private groupNames: readonly string[] = [];
  private endTimerId: number | undefined;

  constructor() {
    makeAutoObservable(this);
  }

  async play(jot: Jot): Promise<void> {
    this.stop();
    runInAction(() => {
      this.state = 'loading';
      this.errorMessage = undefined;
    });

    let drums: Drums;
    let ctx: AudioContext;
    try {
      ({ drums, ctx } = await this.ensureLoaded());
    } catch (err) {
      runInAction(() => {
        this.state = 'idle';
        this.errorMessage = err instanceof Error ? err.message : String(err);
      });
      return;
    }

    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    const events = jotToEvents(jot);
    if (events.length === 0) {
      runInAction(() => {
        this.state = 'idle';
      });
      return;
    }

    const startTime = ctx.currentTime + SCHEDULE_LEAD_SECONDS;
    let lastTime = startTime;
    for (const ev of events) {
      const role = midiNoteToRole(ev.midiNote);
      if (role === undefined) continue;
      const group = resolveGroupForRole(role, this.groupNames);
      if (group === undefined) continue;
      const t = startTime + ev.time;
      drums.start({ note: group, time: t, velocity: ev.velocity });
      if (t > lastTime) lastTime = t;
    }

    const tailMs = Math.max(0, (lastTime - ctx.currentTime + PLAYBACK_TAIL_SECONDS) * 1000);
    runInAction(() => {
      this.state = 'playing';
    });
    this.endTimerId = window.setTimeout(() => {
      runInAction(() => {
        if (this.state === 'playing') this.state = 'idle';
        this.endTimerId = undefined;
      });
    }, tailMs);
  }

  stop(): void {
    if (this.endTimerId !== undefined) {
      window.clearTimeout(this.endTimerId);
      this.endTimerId = undefined;
    }
    this.drums?.stop();
    runInAction(() => {
      if (this.state !== 'idle') this.state = 'idle';
    });
  }

  private async ensureLoaded(): Promise<{ drums: Drums; ctx: AudioContext }> {
    if (this.drums && this.ctx) return { drums: this.drums, ctx: this.ctx };

    const Ctx: typeof AudioContext | undefined =
      window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) {
      throw new Error('Web Audio is not available in this browser.');
    }
    const ctx = new Ctx();
    const drums = DrumMachine(ctx, { instrument: DEFAULT_INSTRUMENT });
    await drums.load;
    this.ctx = ctx;
    this.drums = drums;
    this.groupNames = drums.getGroupNames();
    return { drums, ctx };
  }
}

export const jotPlayer = new JotPlayer();
