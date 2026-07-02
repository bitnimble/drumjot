/**
 * Loads and holds the GeneralUser GS drum SoundFont.
 *
 * The ~30 MB `.sf2` is fetched through `ProgressCacheStorage`, so the
 * first play of a session streams it with a visible byte-progress bar and
 * it's then cached in the browser Cache API (instant on later sessions).
 * The load is deferred to first `play()` (or a background `preloadDrums`)
 * so this stays side-effect free at import time.
 *
 * Observable so the toolbar can render the load progress bar and the kit
 * picker reactively; the loaded `drums` handle is consumed by the drum
 * scheduler.
 */
import { makeAutoObservable, runInAction } from 'mobx';
import { GeneralUserGsKit, KitInfo } from './gm_kit';
import { SampleLoadProgress } from './sample_storage';
import { AudioGraph } from './audio_graph';

// GeneralUser GS GM SoundFont, pulled straight from its GitHub repo
// (raw.githubusercontent.com is CORS-open and sends Content-Length, so
// the byte-progress bar works). `main` is a moving ref, fine for now;
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
// If the SoundFont can't be fetched within this window we give up so the
// UI doesn't sit on "Loading…" forever, a typical local network failure
// mode that's otherwise invisible. Generous because it's a ~30 MB
// one-time download on a slow link (cached loads are instant); a cache
// hit resolves long before this.
const LOAD_TIMEOUT_SECONDS = 120;
// Brief settle window after `drums.load` resolves on the cold path. The
// SoundFont is parsed but smplr's per-note pipeline (zone lookup, layer
// allocation) needs a moment before its first scheduled hit lands
// reliably; without this the very first note of a fresh-session play
// occasionally drops. Only paid on the one-time load (`ensureLoaded`
// short-circuits on subsequent plays), so normal play latency is
// unchanged.
const POST_LOAD_SETTLE_SECONDS = 0.2;

type Drums = ReturnType<typeof GeneralUserGsKit>;

/** The loaded drum kit paired with the AudioContext it plays through. */
export type LoadedKit = { drums: Drums; ctx: AudioContext };

export class SoundfontLoader {
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

  private drumsHandle: Drums | undefined;
  /**
   * In-flight soundfont load, set while `ensureLoaded` is downloading +
   * parsing the .sf2 and cleared once `drums` is populated. Lets a
   * background `preloadDrums()` and a foreground `play()` share the same
   * load (and the same `sampleLoadProgress` ticks) instead of racing two
   * parallel 30 MB cache reads / parses.
   */
  private loadingPromise: Promise<LoadedKit> | undefined;

  /**
   * @param graph  the shared audio graph (the loader creates + wires the
   *               drum gain node through it on the cold load).
   * @param ensureContext  builds-or-returns the AudioContext (the caller
   *               owns the master-fader values the graph needs to init).
   * @param drumBusGain  resolves the initial drum-gain value (pins to 0
   *               when the drum section starts muted); read on cold load.
   */
  constructor(
    private readonly graph: AudioGraph,
    private readonly ensureContext: () => AudioContext,
    private readonly drumBusGain: () => number,
  ) {
    // `drums` stays a plain (non-computed) getter: it's read imperatively
    // by the scheduler, never inside a reaction, and its backing field is a
    // large non-observable handle, so a computed would only invite a
    // stale-cache footgun.
    makeAutoObservable<
      this,
      'graph' | 'ensureContext' | 'drumBusGain' | 'drumsHandle' | 'loadingPromise'
    >(this, {
      graph: false,
      ensureContext: false,
      drumBusGain: false,
      drumsHandle: false,
      loadingPromise: false,
      drums: false,
    });
  }

  /** The loaded drum kit, or undefined before the soundfont has loaded. */
  get drums(): Drums | undefined {
    return this.drumsHandle;
  }

  /**
   * Switch the active drum kit to another preset in the percussion bank.
   *
   * The preset is always recorded so the *next* load uses it. If the kit
   * is already loaded, the swap happens immediately (no refetch, the
   * parsed SoundFont is retained); it takes effect on every subsequently
   * scheduled note, so changing kit mid-play is fine. Failures are handed
   * to `onError` rather than thrown into the UI handler.
   */
  async setDrumPreset(preset: number, onError: (message: string) => void): Promise<void> {
    runInAction(() => {
      this.drumPreset = preset;
    });
    if (!this.drumsHandle) return; // not loaded yet; ensureLoaded will use it
    try {
      await this.drumsHandle.loadPreset(preset);
      console.log(`[jotPlayer] switched to drum preset ${preset}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('[jotPlayer] drum preset switch failed:', err);
      onError(`Could not switch drum kit: ${message}`);
    }
  }

  /**
   * Kick off the soundfont load in the background without blocking the
   * caller and without surfacing the visible loading indicator (which is
   * gated on `state === 'loading'`). Called from the audio-track loaders
   * so the ~30 MB cache read + SF2 parse can overlap with the user's
   * file-decoding wait; by the time they hit Play, `ensureLoaded`
   * short-circuits and playback starts immediately. No-op if drums are
   * already loaded or already loading; errors are swallowed (a real
   * `play()` will re-attempt and surface them through `errorMessage`).
   */
  preload(): void {
    if (this.drumsHandle || this.loadingPromise) return;
    this.ensureLoaded().catch((err) => {
      console.warn('[jotPlayer] drum preload failed (will retry on play):', err);
    });
  }

  async ensureLoaded(): Promise<LoadedKit> {
    const ctx = this.graph.ctx;
    if (this.drumsHandle && ctx) return { drums: this.drumsHandle, ctx };
    if (this.loadingPromise) return this.loadingPromise;
    this.loadingPromise = this.doLoad();
    try {
      return await this.loadingPromise;
    } finally {
      this.loadingPromise = undefined;
    }
  }

  private async doLoad(): Promise<LoadedKit> {
    const ctx = this.ensureContext();
    const drumGain = this.graph.createDrumGain(this.drumBusGain());
    // Phase before the cache layer reports its first tick. On a cache
    // hit `ProgressCacheStorage` fires `fromCache: true` as soon as
    // `cache.match` resolves (well before `arrayBuffer()` finishes), so
    // this state is brief; on a cold load it lingers until the first
    // network byte arrives.
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
                  `Drum kit failed to load within ${LOAD_TIMEOUT_SECONDS}s, ` +
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
    // POST_LOAD_SETTLE_SECONDS. Only reached on the cold load, the
    // early return at the top of `ensureLoaded` skips it on every later play.
    await new Promise((resolve) => window.setTimeout(resolve, POST_LOAD_SETTLE_SECONDS * 1000));

    this.drumsHandle = drums;
    runInAction(() => {
      this.drumKits = drums.availableKits;
    });
    console.log('[jotPlayer] GeneralUser GS kit loaded');
    return { drums, ctx };
  }
}
