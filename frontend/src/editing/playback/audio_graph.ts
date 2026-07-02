/**
 * The persistent Web Audio graph shared across every `play()`.
 *
 *   smplr drum layers ─▶ drumGain ──┐
 *   per audio-track GainNode ─▶ audioBusGain ──┤
 *                                              ├─▶ pageGain ─▶ destination
 *
 * `drumGain` carries the all-drums master fader, `audioBusGain` the
 * all-audio-tracks fader, and `pageGain` the whole-page fader (last stage,
 * so it scales both). The `AudioContext` and buses are created on first use
 * (a user gesture is required before audio output is allowed) and live for
 * the context's lifetime; the context is never closed mid-session so there's
 * no teardown to do.
 *
 * Deliberately NOT MobX-observable: it holds only Web Audio nodes, which
 * must never be wrapped in observables. The user-facing fader *values* are
 * observable fields on {@link JotPlayer}; this class just writes them onto
 * the nodes.
 */

// SF2 percussion samples are recorded near full scale, so unlike the old
// synthesised DrumMachine path the kit already sits well against a
// full-scale audio track. Keep a dedicated routing node at unity (one
// place to trim/boost later) rather than the old +12 dB lift, which
// would clip these samples.
export const DRUM_MASTER_GAIN = 1;

export class AudioGraph {
  private ctxNode: AudioContext | undefined;
  /**
   * Master gain the drum kit is routed through (see {@link DRUM_MASTER_GAIN}).
   * Created once alongside `drums` and lives for the AudioContext's lifetime.
   */
  private drumGainNode: GainNode | undefined;
  private pageGainNode: GainNode | undefined;
  private audioBusGainNode: GainNode | undefined;

  // --- TEST-ONLY output capture ----------------------------------------
  // Used by `audio_capture.e2e.ts` to verify that playback produces real
  // audio and that mute/solo actually change the level. We tap `pageGain`
  // (the final node feeding `ctx.destination`) with an AnalyserNode, which is
  // a pure observer: it's connected FROM the master but not onward to the
  // destination, so it reads the exact signal the user hears without altering
  // it. Capture is a time series of {AudioContext time, windowed RMS}.
  private captureNode: AnalyserNode | undefined;
  private captureSamples: { t: number; rms: number }[] = [];
  private captureTimer = 0;

  get ctx(): AudioContext | undefined {
    return this.ctxNode;
  }

  get audioBusGain(): GainNode | undefined {
    return this.audioBusGainNode;
  }

  /**
   * Construct (or return the existing) AudioContext without triggering
   * the smplr sample download. `loadAudioTrack` needs this, it has to
   * decode audio into the same context that will eventually play the
   * score, but shouldn't pay the ~150KB drum-samples fetch just to
   * attach a track.
   *
   * @param masterVolume  initial `pageGain` value.
   * @param audioBusGain  initial `audioBusGain` value (0 when the audio
   *                      section starts muted).
   */
  ensureContext(masterVolume: number, audioBusGain: number): AudioContext {
    if (this.ctxNode) return this.ctxNode;
    const Ctx: typeof AudioContext | undefined =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) {
      throw new Error('Web Audio is not available in this browser.');
    }
    // `latencyHint: 'playback'` asks the browser for a larger output
    // buffer than the default 'interactive' mode, the audio thread can
    // ride through longer main-thread stalls (heavy relayout on zoom; a
    // GC pause) without buffer underruns / glitches. Drum practice
    // doesn't involve live-input feedback; so the few-tens-of-ms extra
    // scheduling latency is inaudible. All `currentTime` / scheduled
    // event times remain in the same time frame so no scheduler math
    // needs to change.
    const ctx = new Ctx({ latencyHint: 'playback' });
    // Build the master bus now (not in ensureLoaded) so audio tracks
    // loaded before the first play() route through the same faders.
    const pageGain = ctx.createGain();
    pageGain.gain.value = masterVolume;
    pageGain.connect(ctx.destination);
    const audioBus = ctx.createGain();
    audioBus.gain.value = audioBusGain;
    audioBus.connect(pageGain);
    this.ctxNode = ctx;
    this.pageGainNode = pageGain;
    this.audioBusGainNode = audioBus;
    return ctx;
  }

  /**
   * Create the drum master gain node, wire it into the page master, and
   * return it (to route the drum kit through). Called once on the cold
   * soundfont load. Routes into the page master (not straight to
   * destination) so the page fader scales drums too.
   */
  createDrumGain(gain: number): GainNode {
    const ctx = this.ctxNode;
    if (!ctx) throw new Error('AudioGraph.createDrumGain called before ensureContext');
    const drumGain = ctx.createGain();
    drumGain.gain.value = gain;
    drumGain.connect(this.pageGainNode ?? ctx.destination);
    this.drumGainNode = drumGain;
    return drumGain;
  }

  /** Move the whole-page master fader (no-op until the graph exists). */
  setPageGain(v: number): void {
    if (this.pageGainNode) this.pageGainNode.gain.value = v;
  }

  /** Set the drum bus gain (no-op until the graph exists). */
  setDrumBusGain(v: number): void {
    if (this.drumGainNode) this.drumGainNode.gain.value = v;
  }

  /** Set the audio-track bus gain (no-op until the graph exists). */
  setAudioBusGain(v: number): void {
    if (this.audioBusGainNode) this.audioBusGainNode.gain.value = v;
  }

  /**
   * Start recording the page-master output as a windowed-RMS time series.
   * Returns false if there's no AudioContext yet (call after `play()` has
   * begun). Idempotent stop via {@link stopOutputCapture}.
   */
  startOutputCapture(): boolean {
    const ctx = this.ctxNode;
    const master = this.pageGainNode;
    if (!ctx || !master) return false;
    this.stopOutputCapture();
    const node = ctx.createAnalyser();
    node.fftSize = 2048;
    master.connect(node);
    this.captureNode = node;
    this.captureSamples = [];
    const buf = new Float32Array(node.fftSize);
    this.captureTimer = self.setInterval(() => {
      node.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      this.captureSamples.push({ t: ctx.currentTime, rms: Math.sqrt(sum / buf.length) });
    }, 10);
    return true;
  }

  /** Stop capture and return the recorded {t, rms} series. */
  stopOutputCapture(): { t: number; rms: number }[] {
    if (this.captureTimer) self.clearInterval(this.captureTimer);
    this.captureTimer = 0;
    if (this.captureNode) {
      // Remove the `pageGain → captureNode` edge. It's an OUTGOING edge of
      // `pageGain` (the analyser has no onward connection of its own), so
      // `captureNode.disconnect()` wouldn't touch it, disconnect from the
      // source, or each capture cycle leaks an analyser still wired to master.
      this.pageGainNode?.disconnect(this.captureNode);
      this.captureNode = undefined;
    }
    return this.captureSamples;
  }
}
