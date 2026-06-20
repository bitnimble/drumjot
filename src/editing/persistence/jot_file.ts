/**
 * The mutable `.jot` save format: encode / decode + format sniffing.
 *
 * A `.jot` file is now one of two things, sharing the extension and told
 * apart by their leading bytes:
 *
 *   - **DSL text** (the original `.jot`): the compact, hand-authorable drum-
 *     notation language. It is the *lossy subset* of what the editor holds,
 *     it can express the notation, but carries none of the editor's extended
 *     state (mixer faders, display settings, palette, the loaded audio
 *     tracks/stems, and, in time, bookmarks / loops / song metadata /
 *     connected devices). See `SPEC.md`.
 *   - **Mutable container** (this module): a `DJOT` magic + version header in
 *     front of a **zip** archive. It is the *lossless superset*, a snapshot of
 *     the live {@link JotState} document, the {@link JotEditorMetadata} the DSL
 *     can't represent, AND the loaded audio tracks/stems embedded as zip
 *     entries. This is the only format that round-trips a session, edits +
 *     mixer + backing audio (DSL export, via `writeDsl`, reflects only the
 *     originally-loaded source, not subsequent edits, and has no audio).
 *
 * Layout of the mutable file:
 *
 *     "DJOT" (4 bytes) · version uint16 LE (2 bytes) · zip archive
 *       ├── session.json          # the MutableJotFile envelope (deflated)
 *       └── audio/<i>.<ext>       # each loaded audio track's source bytes
 *
 * The leading 4-byte ASCII magic `DJOT` is what {@link isMutableJotFile}
 * sniffs; human-authored DSL text never starts with it, so a single "Load
 * .jot" entry handles both, sniff, then either {@link decodeMutableJotFile}
 * or parse the bytes as DSL text. The magic also lets us tell our container
 * apart from a bare debug-bundle / ParaDB zip. The inner archive is read back
 * with the shared zip reader (`src/utils/zip.ts`); audio entries are stored
 * (already-compressed), `session.json` is deflated.
 */
import type { Jot } from 'src/schema/dsl/dsl';
import type { JotState } from 'src/schema/schema';
import type { AudioTrackRole } from 'src/editing/playback/audio_tracks';
import type { TrackMixerState } from 'src/editing/mixer/mixer_presenter';
import type { SettingsState } from 'src/settings/settings_presenter';
import {
  inflateEntry,
  readCentralDirectory,
  writeZip,
  type ZipWriteEntry,
} from 'src/utils/zip';

/** ASCII magic at the head of every mutable `.jot` file (`DJOT`). Chosen so
 *  it can't collide with hand-authored DSL text (which starts with a title /
 *  metadata line) nor a bare zip (`PK`). */
export const JOT_FILE_MAGIC = 'DJOT';
const MAGIC_BYTES = new Uint8Array([0x44, 0x4a, 0x4f, 0x54]); // "DJOT"
/** Envelope schema version, bumped on a breaking change to {@link
 *  MutableJotFile}. Written into both the binary header (for a cheap
 *  pre-parse check) and the JSON body (authoritative). */
export const JOT_FILE_VERSION = 1;
/** Magic (4 bytes) + version (2 bytes, little-endian) = 6-byte header, in
 *  front of the zip archive. */
const HEADER_LEN = MAGIC_BYTES.length + 2;
/** Zip entry holding the JSON envelope. */
const SESSION_ENTRY = 'session.json';
/** Prefix for the embedded audio-track entries. */
export const AUDIO_ENTRY_PREFIX = 'audio/';

/**
 * One loaded audio track / stem as persisted in a mutable `.jot`: its mixer
 * state + a pointer to the zip entry holding its encoded source bytes. The
 * audio bytes themselves live in a separate zip entry (not the JSON) so the
 * archive stays a flat, inspectable set of files.
 */
export type PersistedAudioTrack = {
  /** Zip entry path holding this track's encoded source bytes (e.g.
   *  `audio/0.mp3`). */
  entry: string;
  /** Original filename (drives the gutter label + the decode MIME sniff). */
  filename: string;
  /** What the loader believed the audio was (drumless backing vs a drum
   *  stem), if known; re-applied so the per-row menu matrix matches. */
  role?: AudioTrackRole;
  /** Effective lane at save time (waveform tint / grouping hint). */
  lane?: string;
  /** Per-track mixer state, re-applied to the freshly-decoded track. */
  muted: boolean;
  soloed: boolean;
  volume: number;
};

/**
 * Editor metadata the DSL can't carry, persisted alongside the document in a
 * mutable `.jot` file. Every field is optional so the loader tolerates files
 * written by an older app version (or by a future phase that drops a field):
 * a missing field just leaves the post-load reset defaults in place.
 *
 * It's the extensible home for the rest as they land (bookmarks, A/B loop
 * regions, song metadata such as artist / album / art, connected WebMIDI /
 * WebUSB devices + their settings).
 */
export type JotEditorMetadata = {
  /** Drum-lane mixer (per-row mute/solo/volume + drum-section masters). */
  mixer?: TrackMixerState;
  /** Per-song display settings: the grid-line overlay, waveform
   *  normalisation, the visually-merge-layers toggle. */
  settings?: SettingsState;
  /** Per-lane colour palette, taken from the jot so a saved colour scheme
   *  travels with the song. */
  palette?: string[];
  /** The loaded audio tracks / stems: per-track mixer state + manifest
   *  pointers to the zip entries holding their encoded bytes. Empty / absent
   *  when no audio was loaded. */
  audioTracks?: PersistedAudioTrack[];
};

/**
 * The decoded contents of a mutable `.jot` file: the JSON envelope plus the
 * embedded audio bytes keyed by their zip-entry path (so the loader can match
 * each {@link PersistedAudioTrack} manifest entry to its bytes).
 */
export type DecodedMutableJotFile = {
  file: MutableJotFile;
  /** Audio-entry path → encoded source bytes. */
  audio: Map<string, Uint8Array>;
};

/** The JSON envelope at `session.json` inside the container. */
export type MutableJotFile = {
  /** Discriminator, guards against feeding some other zip in. */
  format: 'drumjot-mutable';
  /** Mirrors {@link JOT_FILE_VERSION} at write time. */
  version: number;
  /** ISO-8601 timestamp the file was saved at (the "last save time"). */
  savedAt: string;
  /** The edited mutable-document snapshot, the lossless heart of the file. */
  document: JotState;
  /**
   * TRANSITIONAL. The originally-loaded DSL `Jot` AST, persisted verbatim
   * purely so the editor's (frozen) `globalMetadata` readers, bpm, songLeadIn,
   * instrument mapping, keep working after a reload. Edits flow into
   * {@link document}, not here, so this is the load-time source, not a live
   * mirror. The next phase lifts `globalMetadata` into the `JotSchema` itself
   * and turns text-jot loading into a one-time conversion (the same path as
   * MIDI / RLRR), after which this field, and the need to store it, disappear.
   */
  source: Jot;
  /** Editor state outside the document; see {@link JotEditorMetadata}. */
  editor: JotEditorMetadata;
};

/** True when `bytes` begins with the mutable-format magic. A cheap,
 *  synchronous sniff over just the first {@link MAGIC_BYTES} bytes; anything
 *  else (DSL text, an empty file) is `false` and should be treated as DSL. */
export function isMutableJotFile(bytes: Uint8Array): boolean {
  if (bytes.length < MAGIC_BYTES.length) return false;
  for (let i = 0; i < MAGIC_BYTES.length; i++) {
    if (bytes[i] !== MAGIC_BYTES[i]) return false;
  }
  return true;
}

/** One audio track's bytes to embed, plus whether the container should
 *  deflate it. The caller (which knows the source codec) sets `compress`:
 *  true for uncompressed PCM sources (WAV / AIFF), false for already-
 *  compressed ones (MP3 / FLAC / OGG / …) where deflate only burns CPU. */
export type AudioEntryInput = { entry: string; bytes: Uint8Array; compress?: boolean };

/**
 * Encode a {@link MutableJotFile} + its embedded audio to on-disk bytes:
 * `DJOT` magic + little-endian version + a zip of `session.json` (the
 * deflated envelope) and one entry per audio track. The `file.editor.
 * audioTracks` manifest must reference the same entry paths the `audio`
 * array carries (the caller pairs them up). Audio is stored verbatim, the
 * source bytes in their original codec, no transcoding; `compress` only
 * controls the zip-level deflate (see {@link AudioEntryInput}).
 */
export async function encodeMutableJotFile(
  file: MutableJotFile,
  audio: ReadonlyArray<AudioEntryInput> = []
): Promise<Uint8Array> {
  const entries: ZipWriteEntry[] = [
    { name: SESSION_ENTRY, data: new TextEncoder().encode(JSON.stringify(file)), compress: true },
    ...audio.map((a) => ({ name: a.entry, data: a.bytes, compress: a.compress ?? false })),
  ];
  const zip = await writeZip(entries);
  const out = new Uint8Array(HEADER_LEN + zip.length);
  out.set(MAGIC_BYTES, 0);
  out[MAGIC_BYTES.length] = JOT_FILE_VERSION & 0xff;
  out[MAGIC_BYTES.length + 1] = (JOT_FILE_VERSION >> 8) & 0xff;
  out.set(zip, HEADER_LEN);
  return out;
}

/**
 * Decode mutable-format bytes back into the JSON envelope + the embedded
 * audio. Throws if the bytes lack the magic (caller should have sniffed with
 * {@link isMutableJotFile} first), if the zip / `session.json` is missing or
 * corrupt, or if the decoded JSON isn't a recognised envelope.
 */
export async function decodeMutableJotFile(bytes: Uint8Array): Promise<DecodedMutableJotFile> {
  if (!isMutableJotFile(bytes)) {
    throw new Error('Not a mutable .jot file (missing magic header).');
  }
  const zip = bytes.subarray(HEADER_LEN);
  const entries = readCentralDirectory(zip);
  const sessionEntry = entries.find((e) => e.name === SESSION_ENTRY);
  if (!sessionEntry) {
    throw new Error('Corrupt .jot file (no session.json in the container).');
  }
  const json = new TextDecoder().decode(await inflateEntry(zip, sessionEntry));
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch (err) {
    throw new Error(`Corrupt .jot file: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (
    typeof parsed !== 'object' ||
    parsed === null ||
    (parsed as { format?: unknown }).format !== 'drumjot-mutable'
  ) {
    throw new Error('Unrecognised .jot file (wrong format tag).');
  }
  const file = parsed as MutableJotFile;
  if (file.version > JOT_FILE_VERSION) {
    throw new Error(
      `This .jot was saved by a newer version of Drumjot (file v${file.version}, ` +
        `this app reads up to v${JOT_FILE_VERSION}). Update to open it.`
    );
  }

  const audio = new Map<string, Uint8Array>();
  for (const entry of entries) {
    if (entry.name.startsWith(AUDIO_ENTRY_PREFIX)) {
      audio.set(entry.name, await inflateEntry(zip, entry));
    }
  }
  return { file, audio };
}
