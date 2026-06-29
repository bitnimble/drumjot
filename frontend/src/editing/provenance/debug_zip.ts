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
 *    ├── prediction.mid
 *    ├── note_provenance.json
 *    ├── no_drums.mp3
 *    ├── stem_<lane>.mp3 ...
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
} from 'src/utils/zip';

/**
 * The manifest's `mapping` field. Maps drum lane letter (e.g. `k`,
 * `s`, `h`, …) — and the synthetic `no_drums` key for the drumless
 * backing audio — to the filename of the corresponding MP3 inside the
 * zip. The keys are also the DSL lane letters used by the renderer.
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
  /** Lane letter (or `no_drums`) -> MP3 filename inside the zip. */
  mapping: Record<string, string>;
  /** Filename inside the zip of the predicted-onsets MIDI file. The UI
   *  rehydrates the score from this MIDI via `src/midi/from_midi.ts`. */
  prediction_midi?: string | null;
  /** Filename of the per-note debug provenance JSON sidecar. Lists every
   *  detected onset (kept or rejected) so the UI can annotate rendered
   *  notes + render rejected onsets as ghost overlays. */
  note_provenance?: string | null;
  metadata?: Record<string, unknown>;
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
  lane: string;
  midi_note: number | null;
  /**
   * Unique identifier. The MIDI tick this kept onset was emitted at;
   * matched against `note.metadata.midi.tick` (preserved through
   * `from_midi.ts`) to attach this provenance to its rendered note.
   * `null` for rejected onsets, those never made it into the MIDI
   * and are rendered separately as ghost overlays.
   */
  tick: number | null;
  detected_time_sec: number;
  /** Pre-envelope-refine ADTOF model peak time (the raw `peak_frame /
   * fps` before `_refine_peak_times_audio` snapped it to the audio's
   * onset-strength envelope local-max). `null` for non-ADTOF detection
   * paths and for bundles produced before provenance format v3. The
   * popup uses it to display the envelope refinement as its own
   * per-onset stage in the detected → final chain. */
  raw_model_time_sec?: number | null;
  /** Absolute audio time after the backend `quantise` stage's
   * joint-snap + LLM residual shift. `null` when that stage didn't run
   * or didn't move this onset; in that case the rendered MIDI tick falls
   * back to `detected_time_sec`. Mirrors
   * `OnsetCandidate.quantised_time` in `transcriber/app/models.py`. */
  quantised_time_sec?: number | null;
  /** Total signed integer 1/48-slot shift the backend `quantise` stage
   * applied to this onset (sum of all four passes). `null` when no shift
   * was applied. */
  quantised_shift_slots?: number | null;
  /** Per-pass quantise contributions in slot units. `null` when the pass
   * didn't run for this onset (off-grid for any pass after geometric;
   * envelope pass skipped because no envelope was available; grid/LLM
   * pass turned off; LLM cancelled/errored; or the bundle predates
   * provenance format v3). `0` means the pass ran but didn't shift (or
   * its shift was rejected by the monotonic-injective guard). The sum
   * of the four equals `quantised_shift_slots` for any onset that ran
   * the full chain. */
  geometric_shift_slots?: number | null;
  envelope_shift_slots?: number | null;
  grid_shift_slots?: number | null;
  llm_shift_slots?: number | null;
  /** Signed sub-slot residual from the geometric pass: how far the raw
   * natural slot position sat from its nearest integer slot, range
   * (-0.5, +0.5]. + = late of slot, − = early. `null` for off-grid
   * onsets and for bundles predating v3. */
  quantised_residual_slots?: number | null;
  /** Explicit off-grid flag from the geometric snap (no free slot
   * within the match band). Older bundles didn't surface this and
   * leave it `undefined`; consumers fall back to inferring off-grid
   * from `quantised_time_sec === null`. */
  off_grid?: boolean | null;
  /** ADTOF model confidence at the peak frame, in [0, 1]. Surfaced as
   * "Onset confidence" in the per-note debug popup. Distinct from
   * {@link amplitude}; see that field for the split. */
  strength: number;
  /** Raw audio amplitude (|sample| in [0, 1]) in a ±20ms window around
   * the onset, on the source stem. Drives the per-lane
   * percentile-normalised MIDI velocity mapping (so a quieter hit gets
   * a lower velocity even if the model is confidently identifying the
   * lane). `null` for non-ADTOF detection paths and re-loaded bundles
   * produced before this field existed; consumers fall back to
   * {@link strength} in that case. */
  amplitude?: number | null;
  /** Time (s) for post-onset RMS to fall 20dB below its local peak.
   * Sparse / sustained hits (a crash) measure long; dense / articulate
   * streams (a ride) measure short. Populated only by `cymbal_split`;
   * `null` everywhere else. */
  decay_s?: number | null;
  /** Spectral flatness (Wiener entropy) of the onset's early
   * attack/decay window. High = noise-like (cymbal, snare wires,
   * shaker); low = tonal (clean stick on bell, tom). Populated by the
   * cymbal + hi-hat splits. */
  flatness?: number | null;
  /** Spectral centroid (Hz) of the onset's early attack/decay window, * a "brightness" proxy. Crash > ride; open hat > closed hat.
   * Populated by the cymbal + hi-hat splits. */
  centroid_hz?: number | null;
  /** Time (s) to the nearest neighbouring onset in the same lane.
   * Dense streams (a ride pattern, a hi-hat groove) have small gaps;
   * isolated accents (a crash, an open-hat punctuation) have large
   * ones. Populated by the cymbal + hi-hat splits. */
  gap_s?: number | null;
  /** 10-90% rise time (s) of the early post-onset envelope. A fresh
   * strike has a sharp attack; a sizzle bump on top of a ringing tail
   * has a soft / non-existent one. Populated only by `hihat_split`. */
  attack_s?: number | null;
  /** Mean RMS in the [+200, +500] ms window after the onset, normalised
   * by the local peak. High = still ringing 200-500ms after the strike
   * (open hat). Populated only by `hihat_split`. */
  late_rms?: number | null;
  /** Mean RMS in the [-300, -50] ms window before the onset, normalised
   * by the local peak. High = riding on existing ring energy
   * (in-passage open-hat sizzle-train signature). Populated only by
   * `hihat_split`. */
  pre_rms?: number | null;
  /** Seconds from the onset to where its ring is considered over (per
   * `_TAIL_END_FRAC` / `_TAIL_MIN_S`). Used by the open-tail
   * post-filter and surfaced for visibility. Populated only by
   * `hihat_split`. */
  tail_end_s?: number | null;
  /** 0-indexed bar in the transcriber's BeatStructure (NOT the rendered
   * jot's bar index, see {@link NoteProvenanceFile.lead_bars}). */
  bar: number;
  beat_in_bar: number;
  out_of_range: boolean;
  kept: boolean;
  rejected_by: string | null;
  /** Filter-LLM reason code for the rejection: `bleed`,
   * `double_trigger`, `noise`, or `custom`. `null` when the rejection
   * didn't come from the filter LLM (upstream-vetted `h`/`H`/`c`/`d`
   * lanes, out-of-range padding, or kept onsets) or the bundle predates
   * the field (provenance `format` < 2). */
  reason_code?: string | null;
  /** Free-text detail accompanying `reason_code`. Always populated when
   * `reason_code === 'custom'`; optional otherwise. */
  reason_text?: string | null;
};

export type NoteProvenanceFile = {
  format: number;
  generated_at?: string;
  /** Combined beat-grid alignment shift (coarse + fine). Kept as a sum
   * for backwards compatibility; new bundles also carry the
   * per-pass split below. */
  beat_alignment_offset_sec?: number | null;
  /** Coarse envelope-phase alignment shift (`align_beats_to_envelope`
   * in `beats.py`; up to ±2 quarter-notes search). Surfaced
   * separately so the popup can attribute the grid correction to its
   * specific pass. `null` when the pass didn't apply a shift or the
   * bundle predates provenance format v3. */
  beat_align_coarse_offset_sec?: number | null;
  /** Fine median onset-snap alignment shift (`align_beats_to_onsets`;
   * ±50 ms window, median over kept matches, coverage-gated). `null`
   * when the pass didn't apply a shift or the bundle predates v3. */
  beat_align_fine_offset_sec?: number | null;
  /**
   * Count of empty bar-1-sized blocks the MIDI lays down before bar 1
   * to absorb the audio lead-in. The rendered jot (from `from_midi.ts`)
   * carries one bar per leading block; transcriber bar `b` (0-indexed
   * in the BeatStructure) maps to the rendered jot's
   * `bars[lead_bars + b]`, which has `bar.index === b + 1` under the
   * drums-t0-anchored 1-based numbering. Pre-drum bars in the rendered
   * jot have `bar.index` in `[-lead_bars, -1]`.
   */
  lead_bars: number;
  per_lane: Record<string, NoteProvenanceEntry[]>;
};

/** A single audio track extracted from the bundle, in the order the
 * caller should load them (drumless backing first if present, then the
 * per-lane stems by lane letter).
 *
 * `keys` is plural because the manifest's `mapping` can point multiple
 * lane letters at the same stem file (e.g. when the cymbal split
 * keeps both `c` crash and `d` ride against the single combined
 * cymbals stem, the manifest emits both `c → stem_c.mp3` and
 * `d → stem_c.mp3`). The bundle loader dedupes by filename and emits
 * one track per unique file, so the consumer loads the file once and
 * binds every key in `keys` to the same resulting `AudioTrackId`. */
export type DebugBundleAudioTrack = {
  /** Lane letters (or `no_drums`) that point at this file in the
   * manifest's `mapping`, in first-mentioned order. Always non-empty. */
  keys: string[];
  file: File;
};

export type DebugBundle = {
  /**
   * Raw MIDI bytes of `prediction.mid`. The caller runs this through
   * `fromMidi` to rehydrate the score.
   */
  predictionMidi: ArrayBuffer | null;
  /**
   * Parsed `note_provenance.json` if the bundle contained one. Drives
   * per-note debug details in the selection label and the rendered
   * ghost overlays for rejected onsets. `null` when the bundle has no
   * provenance (legacy / hand-built bundles).
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

  // Predicted MIDI. Resolved via the manifest's `prediction_midi` field
  // with a fallback to the canonical filename so older bundles without
  // the field still rehydrate.
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

  // Per-note debug provenance. Same resolution pattern as the MIDI:
  // prefer the manifest field, fall back to the canonical filename so
  // older bundles produced before the manifest gained the field still
  // load if the sidecar was present by name. Parse failures are
  // swallowed — provenance is purely diagnostic, so the score should
  // still load even if the JSON is malformed.
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
      if (parsed && typeof parsed === 'object' && parsed.per_lane) {
        noteProvenance = parsed;
      }
    } catch (err) {
      // Don't fail the whole bundle load; the score still works.
      // eslint-disable-next-line no-console
      console.warn('Could not parse note_provenance.json:', err);
    }
  }

  // Dedupe by filename first: when the manifest maps several lane
  // letters at the same stem file (e.g. crash `c` + ride `d` both at
  // `stem_c.mp3` after the cymbal split), we only want to inflate and
  // decode that file ONCE, the consumer will then bind every key in
  // `keys` to the resulting `AudioTrackId`. We preserve first-mention
  // order on `keys` so the audio row in the mixer ends up under the
  // first-mentioned lane (the natural "primary").
  const keysByFilename = new Map<string, string[]>();
  const filenameOrder: string[] = [];
  for (const [key, filename] of Object.entries(manifest.mapping)) {
    if (typeof filename !== 'string') continue;
    const lower = filename.toLowerCase();
    if (!byBasename.has(lower)) continue; // mapping out of sync; skip
    if (!keysByFilename.has(filename)) {
      keysByFilename.set(filename, []);
      filenameOrder.push(filename);
    }
    keysByFilename.get(filename)!.push(key);
  }

  // Inflate the unique files in parallel; `DecompressionStream` is
  // async (browser-side workers), so concurrent inflations overlap
  // well and the user-perceived bundle-open time drops linearly with
  // the number of tracks. Manifest order is restored afterwards so the
  // audio-track gutter in the UI still lays them out the way the
  // transcriber emitted them (no_drums first if present, then drums
  // in lane-letter order).
  const pending = filenameOrder.map(async (filename) => {
    const entry = byBasename.get(filename.toLowerCase())!;
    const data = await inflateEntry(bytes, entry);
    return {
      keys: keysByFilename.get(filename)!,
      file: new File([data as BlobPart], filename, { type: 'audio/mpeg' }),
    } satisfies DebugBundleAudioTrack;
  });
  const audioTracks = await Promise.all(pending);

  return { predictionMidi, noteProvenance, audioTracks, manifest };
}
