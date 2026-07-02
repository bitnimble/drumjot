/**
 * Audio/visual sync compensation for the playhead.
 *
 * Two contributions are summed and applied to the visual playhead in the
 * rAF tick (see {@link latencyShiftSec}):
 *   - {@link audioLatencyMs}: the user's manual fine-tune, surfaced as the
 *     "Audio latency" stepper and persisted to localStorage.
 *   - {@link internalLatencyMs}: an auto-detected baseline derived once per
 *     session from `AudioContext.{baseLatency, outputLatency}` and the
 *     measured rAF frame interval. Not surfaced in any UI.
 *
 * Split out of {@link JotPlayer} so the transport owns only its
 * scheduling/timeline state; the estimator stays observable so the
 * toolbar's stepper reads {@link audioLatencyMs} reactively.
 */
import { makeAutoObservable } from 'mobx';

// `audioLatencyMs` bounds. Narrow enough that a stray keypress cannot
// park the playhead seconds away from the audio.
const AUDIO_LATENCY_MAX_MS = 500;
// localStorage key for the user's manual `audioLatencyMs` value, so
// the fine-tune survives reloads.
const AUDIO_LATENCY_STORAGE_KEY = 'drumjot:audioLatencyMs';
// Visual presentation latency, in frames, between an rAF callback and
// the screen actually showing what was drawn. A heuristic: a typical
// browser pipeline (rAF -> compositor -> GPU -> display) sits in the
// 1..2 frame range; 1.5 is the conventional midpoint. The unmeasurable
// pieces (monitor input lag, GPU queue depth) are absorbed into the
// user's manual fine-tune.
const VISUAL_LATENCY_FRAMES = 1.5;
// Wall-clock duration over which to sample rAF intervals when
// estimating the display refresh period. Fixed in time (not in frame
// count) so the measurement doesn't drag on high refresh rates:
// 200 ms => ~12 frames at 60 Hz, ~33 frames at 165 Hz, both plenty
// for a robust median.
const FRAME_MEASURE_DURATION_MS = 200;

function clampAudioLatencyMs(ms: number): number {
  if (!Number.isFinite(ms)) return 0;
  return Math.max(-AUDIO_LATENCY_MAX_MS, Math.min(AUDIO_LATENCY_MAX_MS, ms));
}

/**
 * Resolve with the median rAF interval (ms) sampled over
 * `durationMs` of wall-clock time. Used by the internal-latency
 * estimate to derive an approximate frame rate, from which visual
 * presentation latency is extrapolated (see `VISUAL_LATENCY_FRAMES`).
 * Falls back to a 60 Hz assumption in non-browser test environments.
 */
async function measureFrameIntervalMs(durationMs: number): Promise<number> {
  if (typeof window === 'undefined') return 1000 / 60;
  return new Promise((resolve) => {
    const intervals: number[] = [];
    let prev: number | undefined;
    let start: number | undefined;
    const tick = (t: number) => {
      if (start === undefined) start = t;
      if (prev !== undefined) intervals.push(t - prev);
      prev = t;
      if (t - start < durationMs) {
        window.requestAnimationFrame(tick);
      } else {
        intervals.sort((a, b) => a - b);
        resolve(intervals[Math.floor(intervals.length / 2)] ?? 1000 / 60);
      }
    };
    window.requestAnimationFrame(tick);
  });
}

export class AvSyncEstimator {
  /**
   * User-controlled fine-tune of the visual-vs-audio sync, in
   * milliseconds. Conceptually a delay applied to the audio engine:
   * positive values shift the visual playhead ahead of the audio
   * clock (audio appears delayed); negative values do the opposite.
   * Surfaced as the "Audio latency" stepper in the Playback menu and
   * persisted to localStorage so the user's manual tune survives
   * reloads. Default 0 (no fine-tune); the auto-detected baseline
   * lives separately on `internalLatencyMs` and is added underneath.
   */
  audioLatencyMs: number = 0;
  /**
   * Auto-detected baseline shift, derived once per session on first
   * play from `AudioContext.{baseLatency, outputLatency}` and the
   * measured rAF frame interval. Added to `audioLatencyMs` to form
   * the total shift applied in the rAF tick. Not surfaced in any UI;
   * the user only ever sees / edits their own fine-tune.
   */
  private internalLatencyMs: number = 0;
  /**
   * True once `estimateInternalLatency` has been kicked off for this
   * session. Latching prevents the kick-off from firing on every
   * play; `outputLatency` is a device property and doesn't change
   * after the AudioContext starts running.
   */
  private internalLatencyEstimated: boolean = false;

  constructor() {
    makeAutoObservable(this);
    this.hydrateFromStorage();
  }

  /** Total sync shift (seconds) applied to the visual playhead: the user's
   *  fine-tune plus the auto-detected baseline. Both shift the visual ahead
   *  of the audio clock to compensate for perceived audio/visual drift. */
  get latencyShiftSec(): number {
    return (this.audioLatencyMs + this.internalLatencyMs) * 0.001;
  }

  /**
   * Set the visual-vs-audio sync trim in milliseconds. Takes effect on
   * the next rAF tick (so within ~8 ms on a 120 Hz vsync during
   * playback, instantly when paused on the next state update). Clamped
   * to a generous range so a stray keypress can't park the playhead
   * seconds away from the audio.
   */
  setAudioLatencyMs(ms: number): void {
    this.audioLatencyMs = clampAudioLatencyMs(ms);
    try {
      window.localStorage.setItem(AUDIO_LATENCY_STORAGE_KEY, String(this.audioLatencyMs));
    } catch {
      // localStorage may throw in private mode or with quota errors;
      // don't crash on a user nudge.
    }
  }

  /**
   * Measure the device + browser audio/visual pipeline latency once
   * per session and bake it into `internalLatencyMs`, which is added
   * on top of the user's fine-tune in the rAF tick. Fired in the
   * background from the first `play()` after the AudioContext has
   * resumed (when `outputLatency` is meaningful). Latched so subsequent
   * plays skip the work. Best-effort: errors leave `internalLatencyMs`
   * at 0, falling back to the user fine-tune alone.
   */
  estimateOnce(ctx: AudioContext): void {
    if (this.internalLatencyEstimated) return;
    this.internalLatencyEstimated = true;
    void this.estimateInternalLatency(ctx);
  }

  private async estimateInternalLatency(ctx: AudioContext): Promise<void> {
    try {
      if (typeof window === 'undefined') return;
      const frameMs = await measureFrameIntervalMs(FRAME_MEASURE_DURATION_MS);
      const baseLatency = ctx.baseLatency ?? 0;
      const outputLatency = ctx.outputLatency ?? 0;
      const audioLagMs = (baseLatency + outputLatency) * 1000;
      const visualLagMs = VISUAL_LATENCY_FRAMES * frameMs;
      const computed = Math.round(visualLagMs - audioLagMs);
      this.internalLatencyMs = clampAudioLatencyMs(computed);
      console.log(
        `[jotPlayer] internal audio latency baked in: ${this.internalLatencyMs}ms ` +
          `(visualLag ~${visualLagMs.toFixed(1)}ms, audioLag ~${audioLagMs.toFixed(1)}ms, ` +
          `frame ${frameMs.toFixed(2)}ms)`
      );
    } catch (err) {
      console.warn('[jotPlayer] internal audio latency estimate failed:', err);
    }
  }

  /**
   * Restore the user's manually-saved `audioLatencyMs` from
   * localStorage if any. Runs once at construction; absence of the
   * key just means "no saved fine-tune" and leaves the default 0.
   */
  private hydrateFromStorage(): void {
    try {
      if (typeof window === 'undefined') return;
      const stored = window.localStorage.getItem(AUDIO_LATENCY_STORAGE_KEY);
      if (stored === null) return;
      const n = parseFloat(stored);
      if (!Number.isFinite(n)) return;
      this.audioLatencyMs = clampAudioLatencyMs(n);
    } catch {
      // localStorage unavailable (private mode etc.); silently skip.
    }
  }
}
