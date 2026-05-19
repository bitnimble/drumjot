/**
 * Acoustic drum kit from the GeneralUser GS General MIDI SoundFont.
 *
 * smplr's bundled `Soundfont` is melodic-GM only (no percussion bank) and
 * its `Soundfont2Sampler` fetches the `.sf2` with a bare `fetch()` — no
 * caching, no progress. GeneralUser-GS.sf2 is ~30 MB, so neither is
 * acceptable. We wire smplr's lower-level pieces ourselves:
 *
 *   1. fetch the `.sf2` through {@link ProgressCacheStorage} — byte
 *      progress for the UI + a one-time Cache API store (instant on
 *      every later session),
 *   2. parse it with the `soundfont2` package,
 *   3. assemble the GM drum kit and hand it to smplr via `loadInstrument`.
 *
 * Why we don't use smplr's `sf2InstrumentToSmplrJson`: that converts a
 * single SF2 *instrument*, which works for a one-instrument melodic
 * SoundFont. GeneralUser GS instead exposes hundreds of individual
 * sample-instruments ("Standard Kick 1", "Hi-Hats", …); the playable
 * kit only exists at the *preset* level (bank 128). So we resolve each
 * GM percussion key through the bank/preset with `getKeyData(key, bank,
 * preset)` and build one region per key pointing at its sample.
 *
 * Each region is `keyRange:[n,n]` with `pitch:n`, i.e. the sample plays
 * at its recorded rate with no key-tracking or looping — correct for
 * drum one-shots (SF2 fine/coarse-tune generators are intentionally
 * ignored; the recorded sound is what matters for practice).
 *
 * The resulting instrument is keyed by GM percussion MIDI note number
 * (36 = kick, 38 = snare, …) — exactly what `jotToEvents` emits — so the
 * caller triggers notes directly, no kit-group mapping.
 */
import { Instrument } from 'smplr';
import type { SmplrJson } from 'smplr';
import { SoundFont2 } from 'soundfont2';
import { ProgressCacheStorage, SampleLoadProgress } from './sample_storage';

/** One selectable drum kit (a preset within the percussion bank). */
export type KitInfo = { preset: number; name: string };

/**
 * Instance methods added on top of the base smplr instrument so the UI
 * can list the kits and switch between them. `loadPreset` rebuilds from
 * the already-downloaded, already-parsed SoundFont — no refetch.
 */
export type GmKitExtras = {
  /** Kits found in the configured bank; empty until `ready` resolves. */
  readonly availableKits: KitInfo[];
  /** Swap the active kit to another preset in the configured bank. */
  loadPreset(preset: number): Promise<void>;
};

export type GmKitOptions = {
  /** URL of the `.sf2` file. */
  url: string;
  /** Cache API entry name (see {@link ProgressCacheStorage}). */
  cacheName: string;
  /** GM bank of the drum kit. Percussion is bank 128. */
  bank: number;
  /**
   * Preset within the bank. GeneralUser GS: 0 = Standard, 8 = Room,
   * 16 = Power, 24 = Electronic, 25 = TR-808, 32 = Jazz, 40 = Brush,
   * 48 = Orchestra.
   */
  preset: number;
  /** Download progress for the (large, one-time) `.sf2` fetch. */
  onProgress: (p: SampleLoadProgress) => void;
};

/**
 * Walk every MIDI key and resolve it through the chosen bank/preset,
 * building a smplr instrument JSON + decoded buffers. Throws (rather
 * than loading a silent kit) if the preset resolves no keys at all.
 */
function buildKit(
  sf2: SoundFont2,
  ctx: BaseAudioContext,
  bank: number,
  preset: number,
): { json: SmplrJson; buffers: Map<string, AudioBuffer> } {
  const buffers = new Map<string, AudioBuffer>();
  const regions: SmplrJson['groups'][number]['regions'] = [];
  const mappedKeys: number[] = [];
  let presetName: string | undefined;

  for (let key = 0; key <= 127; key++) {
    const data = sf2.getKeyData(key, bank, preset);
    if (!data) continue;
    presetName = data.preset.header.name;
    const { header, data: pcm } = data.sample;

    if (!buffers.has(header.name)) {
      // SF2 sample data is signed 16-bit PCM; smplr wants a Float32
      // AudioBuffer (same conversion smplr's own sf2 helper does).
      const f32 = new Float32Array(pcm.length);
      for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768;
      const buf = ctx.createBuffer(1, f32.length || 1, header.sampleRate);
      buf.getChannelData(0).set(f32);
      buffers.set(header.name, buf);
    }

    // pitch === keyRange note ⇒ smplr's resample ratio is 1, so the
    // drum plays at its recorded rate regardless of which key fired it.
    regions.push({ sample: header.name, keyRange: [key, key], pitch: key });
    mappedKeys.push(key);
  }

  if (mappedKeys.length === 0) {
    throw new Error(
      `No drum keys found at bank ${bank} preset ${preset} in the ` +
        `GeneralUser GS SoundFont.`,
    );
  }
  console.log(
    `[gmKit] "${presetName ?? '?'}" (bank ${bank} preset ${preset}): ` +
      `${mappedKeys.length} keys [${mappedKeys.join(', ')}]`,
  );
  return {
    json: { samples: { baseUrl: '', formats: [] }, groups: [{ regions }] },
    buffers,
  };
}

/** Enumerate the presets (kits) defined in a bank, ordered by number. */
function discoverKits(sf2: SoundFont2, bank: number): KitInfo[] {
  const out: KitInfo[] = [];
  // `banks[bank].presets` is a sparse array indexed by preset number.
  sf2.banks[bank]?.presets.forEach((p) => {
    if (p) out.push({ preset: p.header.preset, name: p.header.name });
  });
  return out.sort((a, b) => a.preset - b.preset);
}

/**
 * smplr instrument factory for the GeneralUser GS drum kit. Construct
 * like any smplr instrument: `GeneralUserGsKit(ctx, { url, … , destination })`.
 * `destination` / `volume` are handled by smplr's option splitter.
 *
 * The parsed SoundFont is retained so {@link GmKitExtras.loadPreset} can
 * switch kits without re-downloading or re-parsing the ~30 MB file.
 */
export const GeneralUserGsKit = Instrument<GmKitOptions, GmKitExtras>(
  (ctx, options, smplr) => {
    let sf2: SoundFont2 | undefined;
    let kits: KitInfo[] = [];
    const ready = (async () => {
      const storage = new ProgressCacheStorage(
        options.cacheName,
        options.onProgress,
      );
      const res = await storage.fetch(options.url);
      if (res.status !== 200) {
        throw new Error(
          `Could not fetch the GeneralUser GS SoundFont (HTTP ${res.status}).`,
        );
      }
      sf2 = new SoundFont2(new Uint8Array(await res.arrayBuffer()));
      kits = discoverKits(sf2, options.bank);
      const { json, buffers } = buildKit(sf2, ctx, options.bank, options.preset);
      await smplr.loadInstrument(json, buffers);
    })();
    const extras: GmKitExtras = {
      get availableKits() {
        return kits;
      },
      async loadPreset(preset: number) {
        await ready;
        if (!sf2) throw new Error('GeneralUser GS SoundFont is not loaded.');
        const { json, buffers } = buildKit(sf2, ctx, options.bank, preset);
        await smplr.loadInstrument(json, buffers);
      },
    };
    return { extras, ready };
  },
);
