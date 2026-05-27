declare module 'signalsmith-stretch' {
  /**
   * Build a stretch `AudioWorkletNode` on `ctx`. First call on a given
   * context registers the worklet (downloads WASM, evaluates the
   * processor); later calls reuse the registration. Returned node has
   * the extension methods documented in the library README; see
   * `playback/stretch_node.ts` for the surface we actually use.
   */
  export default function SignalsmithStretch(
    ctx: AudioContext,
    channelOptions?: AudioWorkletNodeOptions,
  ): Promise<AudioWorkletNode>;
}
