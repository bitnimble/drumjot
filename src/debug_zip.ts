/**
 * Load a transcriber debug bundle (`.zip`) directly in the browser.
 *
 * The bundle is what `transcriber/app/debug_bundle.py` produces — its
 * URL is surfaced as `TranscribeResponse.debug_zip_url`, and the operator
 * can also download it and feed it back here later to reconstitute the
 * score + every audio track + the per-stage timings + the full captured
 * log stream offline.
 *
 * Expected layout:
 *
 *    <name>.zip
 *    ├── final.jot
 *    ├── no_drums.mp3
 *    ├── stem_<pitch>.mp3 ...
 *    └── debug.json            # see DebugBundleManifest below
 *
 * The audio tracks come out of here as typed `File`s so the existing
 * `jotPlayer.loadAudioTrack(file)` path absorbs them with no special
 * casing.
 */
import {
  inflateEntry,
  readCentralDirectory,
  ZipEntry,
  zipEntryBasename,
} from 'src/zip';

/**
 * The manifest's `mapping` field. Maps drum pitch letter (e.g. `k`,
 * `s`, `h`, …) — and the synthetic `no_drums` key for the drumless
 * backing audio — to the filename of the corresponding MP3 inside the
 * zip. The keys are also the DSL pitch letters used by the renderer.
 */
export const NO_DRUMS_KEY = 'no_drums';

export type DebugBundleLogEntry = {
  timestamp: string;
  /** Monotonic seconds since the run started. */
  elapsed_seconds: number;
  level: string;
  logger: string;
  message: string;
};

export type DebugBundleStageTiming = {
  stage: string;
  start: string;
  end: string;
  elapsed_seconds: number;
};

export type DebugBundleManifest = {
  filename?: string | null;
  started_at?: string;
  elapsed_seconds?: number;
  options?: Record<string, unknown>;
  /** Pitch letter (or `no_drums`) -> MP3 filename inside the zip. */
  mapping: Record<string, string>;
  /**
   * Set in filter-transcribe mode (which emits MIDI instead of DSL):
   * the filename inside the zip of the predicted-onsets MIDI file. The
   * UI rehydrates the score from this MIDI when `final.jot` is absent.
   */
  prediction_midi?: string | null;
  /**
   * Set in filter-transcribe mode: filename of the per-note debug
   * provenance JSON sidecar. Lists every detected onset (kept or
   * rejected) so the UI can annotate rendered notes + render rejected
   * onsets as ghost overlays. Absent for DSL-mode runs.
   */
  note_provenance?: string | null;
  metadata?: Record<string, unknown>;
  scores?: Record<string, unknown>;
  stage_timings?: DebugBundleStageTiming[];
  logs?: DebugBundleLogEntry[];
};

/**
 * One detected onset's full provenance record. Mirrors the per-entry
 * shape `transcriber/app/pipeline/note_provenance.py` emits — keep
 * these in lockstep when the schema evolves (the `format` field on
 * {@link NoteProvenanceFile} guards against silent drift).
 */
export type NoteProvenanceEntry = {
  pitch: string;
  midi_note: number | null;
  /**
   * Unique identifier. The MIDI tick this kept onset was emitted at;
   * matched against `note.metadata.midi.tick` (preserved through
   * `from_midi.ts`) to attach this provenance to its rendered note.
   * `null` for rejected onsets — those never made it into the MIDI
   * and are rendered separately as ghost overlays.
   */
  tick: number | null;
  detected_time_sec: number;
  detection_backend: string;
  strength: number;
  /** 0-indexed bar in the transcriber's BeatStructure (NOT the rendered
   * jot's bar index — see {@link NoteProvenanceFile.lead_bars}). */
  bar: number;
  beat_in_bar: number;
  out_of_range: boolean;
  kept: boolean;
  rejected_by: string | null;
};

export type NoteProvenanceFile = {
  format: number;
  generated_at?: string;
  onset_backend?: string;
  beat_alignment_offset_sec?: number | null;
  /**
   * Count of bar-0-sized empty blocks the MIDI lays down before bar 0
   * to absorb the audio lead-in. The rendered jot (from `from_midi.ts`)
   * carries one bar per leading block; struct bar `b` maps to the
   * rendered jot's `bars[lead_bars + b]` (or equivalently bar index
   * `lead_bars + b + 1`, since jot bar indices are 1-based with no
   * anacrusis for MIDI imports).
   */
  lead_bars: number;
  per_pitch: Record<string, NoteProvenanceEntry[]>;
};

/** A single audio track extracted from the bundle, in the order the
 * caller should load them (drumless backing first if present, then the
 * per-pitch stems by pitch letter). */
export type DebugBundleAudioTrack = {
  /** Pitch letter or `no_drums`. */
  key: string;
  file: File;
};

export type DebugBundle = {
  /** Drumjot DSL text from `final.jot`. Empty string if the bundle
   * didn't include one (e.g. filter-mode transcribe). */
  jotDsl: string;
  /**
   * Raw MIDI bytes of `prediction.mid` if the bundle contained one
   * (filter-transcribe mode produces MIDI rather than DSL). The caller
   * runs this through `fromMidi` to rehydrate the score when `jotDsl`
   * is empty.
   */
  predictionMidi: ArrayBuffer | null;
  /**
   * Parsed `note_provenance.json` if the bundle contained one
   * (filter-transcribe mode only). Drives per-note debug details in
   * the selection label and the rendered ghost overlays for rejected
   * onsets. `null` when the bundle has no provenance.
   */
  noteProvenance: NoteProvenanceFile | null;
  audioTracks: DebugBundleAudioTrack[];
  manifest: DebugBundleManifest;
};

/**
 * Parse a debug `.zip` File into its three constituents: the Jot DSL,
 * the audio tracks (already typed as `audio/mpeg`), and the manifest.
 *
 * Throws with a human-readable message on a malformed bundle so callers
 * can surface the error on the status pill exactly as ParaDB does.
 */
export async function loadDebugZip(file: File): Promise<DebugBundle> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const entries = readCentralDirectory(bytes);
  if (entries.length === 0) {
    throw new Error('Not a valid zip archive (no central directory found).');
  }

  const byBasename = new Map<string, ZipEntry>();
  for (const entry of entries) {
    byBasename.set(zipEntryBasename(entry.name).toLowerCase(), entry);
  }

  const manifestEntry = byBasename.get('debug.json');
  if (!manifestEntry) {
    throw new Error('debug.json is missing — is this a transcriber debug bundle?');
  }
  const manifestText = new TextDecoder('utf-8').decode(
    await inflateEntry(bytes, manifestEntry),
  );
  let manifest: DebugBundleManifest;
  try {
    manifest = JSON.parse(manifestText) as DebugBundleManifest;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`Could not parse debug.json: ${message}`);
  }
  if (!manifest.mapping || typeof manifest.mapping !== 'object') {
    throw new Error('debug.json is missing the required `mapping` field.');
  }

  let jotDsl = '';
  const jotEntry = byBasename.get('final.jot');
  if (jotEntry) {
    jotDsl = new TextDecoder('utf-8').decode(await inflateEntry(bytes, jotEntry));
  }

  // Predicted MIDI (filter-transcribe mode). Resolved via the manifest's
  // `prediction_midi` field with a fallback to the canonical filename so
  // older bundles without the field still rehydrate.
  let predictionMidi: ArrayBuffer | null = null;
  const midiFilename =
    (typeof manifest.prediction_midi === 'string' && manifest.prediction_midi) ||
    'prediction.mid';
  const midiEntry = byBasename.get(midiFilename.toLowerCase());
  if (midiEntry) {
    const data = await inflateEntry(bytes, midiEntry);
    // `inflateEntry` may return a view backed by a larger underlying
    // buffer; slice so `fromMidi`'s DataView sees exactly the file bytes.
    predictionMidi = data.buffer.slice(
      data.byteOffset,
      data.byteOffset + data.byteLength,
    ) as ArrayBuffer;
  }

  // Per-note debug provenance (filter-transcribe mode). Same resolution
  // pattern as the MIDI: prefer the manifest field, fall back to the
  // canonical filename so older bundles produced before the manifest
  // gained the field still load if the sidecar was present by name.
  // Parse failures are swallowed — provenance is purely diagnostic, so
  // the score should still load even if the JSON is malformed.
  let noteProvenance: NoteProvenanceFile | null = null;
  const provenanceFilename =
    (typeof manifest.note_provenance === 'string' && manifest.note_provenance) ||
    'note_provenance.json';
  const provenanceEntry = byBasename.get(provenanceFilename.toLowerCase());
  if (provenanceEntry) {
    try {
      const text = new TextDecoder('utf-8').decode(
        await inflateEntry(bytes, provenanceEntry),
      );
      const parsed = JSON.parse(text) as NoteProvenanceFile;
      if (parsed && typeof parsed === 'object' && parsed.per_pitch) {
        noteProvenance = parsed;
      }
    } catch (err) {
      // Don't fail the whole bundle load; the score still works.
      // eslint-disable-next-line no-console
      console.warn('Could not parse note_provenance.json:', err);
    }
  }

  // Inflate every mapped entry in parallel — `DecompressionStream` is
  // async (browser-side workers), so concurrent inflations overlap well
  // and the user-perceived bundle-open time drops linearly with the
  // number of tracks. Manifest order is restored afterwards so the
  // audio-track gutter in the UI still lays them out the same way the
  // transcriber emitted them (no_drums first if present, then drums in
  // pitch-letter order).
  const pending: Promise<DebugBundleAudioTrack | null>[] = [];
  for (const [key, filename] of Object.entries(manifest.mapping)) {
    if (typeof filename !== 'string') {
      pending.push(Promise.resolve(null));
      continue;
    }
    const entry = byBasename.get(filename.toLowerCase());
    if (!entry) {
      pending.push(Promise.resolve(null)); // mapping out of sync; skip
      continue;
    }
    pending.push(
      inflateEntry(bytes, entry).then((data) => ({
        key,
        file: new File([data as BlobPart], filename, { type: 'audio/mpeg' }),
      })),
    );
  }
  const audioTracks: DebugBundleAudioTrack[] = [];
  for (const result of await Promise.all(pending)) {
    if (result) audioTracks.push(result);
  }

  return { jotDsl, predictionMidi, noteProvenance, audioTracks, manifest };
}
