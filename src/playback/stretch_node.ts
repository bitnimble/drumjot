/**
 * Lazy loader for the Signalsmith Stretch AudioWorklet (the audio
 * time-stretcher used by audio tracks at every playback speed).
 *
 * Signalsmith ships a WASM + worklet bundle (~230 KB unpacked) that
 * registers an `AudioWorkletProcessor` and returns a factory; calling
 * the factory once per `AudioContext` arms the worklet on that context
 * and yields a builder that can mint per-track stretch nodes.
 *
 * Loading is gated on first audio-track load (the audio-track loaders
 * call {@link preloadStretch} alongside `preloadDrums()`) so the WASM
 * cost is paid only in sessions that load music, overlapping with the
 * track's decode wait. Drum-only sessions never pay it. Failures are
 * sticky: once `_failure` is set we surface the error from every later
 * loader call so the player can disable non-1.0× speeds and toast the
 * user, instead of repeatedly retrying a broken module load.
 */
import SignalsmithStretch from 'signalsmith-stretch';

/**
 * Per-track stretch node. Surface mirrors the subset of the upstream
 * library we use; see the README for `addBuffers` / `schedule` / etc.
 *
 * The worklet owns its own PCM ring (fed via `addBuffers`) and its own
 * read position, so it is the *only* node we wire; no
 * `AudioBufferSourceNode` upstream, no source-side seeks. Pause = stop
 * with `active: false`; play = schedule with `active: true, input:
 * <seconds>, rate: <speed>`.
 */
export type StretchNode = AudioWorkletNode & {
  /** Current playback position inside the input PCM, in seconds. */
  readonly inputTime: number;
  /** Append PCM frames (one Float32Array per channel) to the worklet's input ring. */
  addBuffers(channels: Float32Array[]): Promise<number>;
  /** Drop the entire input ring; used on track replace. */
  dropBuffers(): Promise<void>;
  /**
   * Schedule a state change at AudioContext time `output`. The library
   * compensates for its own latency; passing `output` slightly in the
   * future hides the transition window. See the upstream README for the
   * full field list; we only use `active`, `input`, `rate`, `output`.
   */
  schedule(opts: {
    output?: number;
    active?: boolean;
    input?: number;
    rate?: number;
    semitones?: number;
  }): Promise<void>;
  /** Convenience for `schedule({active: true, output: when})`. */
  start(when?: number): Promise<void>;
  /** Convenience for `schedule({active: false, output: when})`. */
  stop(when?: number): Promise<void>;
};

/**
 * Per-AudioContext factory. The Signalsmith module is keyed off the
 * AudioContext (the worklet has to be registered once per context); the
 * factory caches that registration and gives us a fresh `StretchNode`
 * per call. The app only ever uses one AudioContext, so the outer map
 * is effectively a single-entry cache, but keying by context is the
 * upstream contract and costs nothing.
 */
const factories: WeakMap<AudioContext, Promise<() => Promise<StretchNode>>> = new WeakMap();
let _failure: Error | undefined;

/**
 * Last init failure, sticky across retries. Surfaced by the player so
 * non-1.0× speeds can be disabled and the user toasted once instead of
 * silently failing every speed change.
 */
export function stretchInitFailure(): Error | undefined {
  return _failure;
}

/**
 * Resolve a builder that mints `StretchNode`s for `ctx`. First call on a
 * given context downloads the WASM + registers the worklet (one-shot,
 * cached); subsequent calls return the already-resolved builder. The
 * promise rejects once and is then replaced by `_failure`; callers
 * should treat a thrown error as terminal for non-1.0× playback.
 */
function getBuilder(ctx: AudioContext): Promise<() => Promise<StretchNode>> {
  if (_failure) return Promise.reject(_failure);
  // `BaseAudioContext.audioWorklet` is gated behind a secure context.
  // localhost / 127.0.0.1 / HTTPS are secure; a LAN IP over plain HTTP
  // is not, and Firefox/Chrome both surface that by leaving the
  // property undefined. Distinguish the secure-context case (fixable
  // by switching URL) from a generic browser-doesn't-support-it case
  // so the surfaced error tells the user what to do; mirrors the
  // boot-time check in `src/index.tsx`.
  if (!ctx.audioWorklet) {
    const insecure = typeof window !== 'undefined' && !window.isSecureContext;
    _failure = new Error(
      insecure
        ? 'AudioWorklet is unavailable because this page is not in a secure context. Open it via localhost / 127.0.0.1 / HTTPS instead of a LAN IP over plain HTTP.'
        : 'AudioWorklet is not available in this browser; audio-track playback will not work.',
    );
    return Promise.reject(_failure);
  }
  let pending = factories.get(ctx);
  if (!pending) {
    // The library's defaults are `{ numberOfInputs: 1, numberOfOutputs: 1,
    // outputChannelCount: [2] }`. We're driving the node from its own
    // input buffer (via `addBuffers`), not from a connected source, but
    // an unused input is harmless; the worklet ignores the silent
    // signal once its buffer is loaded.
    pending = Promise.resolve(async () => {
      try {
        const node = (await SignalsmithStretch(ctx)) as StretchNode;
        return node;
      } catch (err) {
        _failure = err instanceof Error ? err : new Error(String(err));
        throw _failure;
      }
    });
    factories.set(ctx, pending);
  }
  return pending;
}

/**
 * Build a stretch node for `ctx`, loaded with `buffer`'s PCM and ready
 * to play. Output is always stereo (mono inputs are duplicated to both
 * channels here); the caller wires its single output into a `GainNode`.
 */
export async function createStretchNode(
  ctx: AudioContext,
  buffer: AudioBuffer,
): Promise<StretchNode> {
  const build = await getBuilder(ctx);
  const node = await build();
  const ch0 = buffer.getChannelData(0);
  const ch1 = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : ch0;
  await node.addBuffers([ch0, ch1]);
  return node;
}

/**
 * Fire-and-forget warmup: kick the WASM/worklet load in the background
 * so the first call to {@link createStretchNode} is instant. Errors are
 * swallowed; they're captured in `_failure` and re-surfaced the next
 * time a real build is requested, so a foreground play with a broken
 * module still reports the failure exactly once.
 */
export function preloadStretch(ctx: AudioContext): void {
  getBuilder(ctx).catch((err) => {
    console.warn('[stretch] preload failed (will retry on demand):', err);
  });
}
