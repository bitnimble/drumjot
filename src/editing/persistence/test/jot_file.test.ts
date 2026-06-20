import { describe, expect, test } from 'bun:test';
import { rockJot } from 'src/fakes/fakes';
import { dslToMutable } from 'src/schema/dsl/from_dsl';
import { createMutableJotFromState } from 'src/schema/schema';
import {
  decodeMutableJotFile,
  encodeMutableJotFile,
  isMutableJotFile,
  JOT_FILE_VERSION,
  type MutableJotFile,
} from 'src/editing/persistence/jot_file';

/** A representative envelope built from the rock example + editor metadata,
 *  including a manifest entry for one embedded audio track. */
function sampleFile(): MutableJotFile {
  const document = dslToMutable(rockJot).snapshot();
  return {
    format: 'drumjot-mutable',
    version: JOT_FILE_VERSION,
    savedAt: '2026-06-20T12:00:00.000Z',
    document,
    source: rockJot,
    editor: {
      mixer: {
        mutedTracks: ['v0/c'],
        soloedTracks: [],
        trackVolumes: [['v0/s', 0.5]],
        drumMasterMuted: false,
        drumMasterSoloed: true,
      },
      settings: {
        gridLines: {
          mainBeat: true,
          subBeat16: false,
          subBeatQuarterTriplet: false,
          subBeatTriplet: true,
          subBeat48: false,
        },
        uniformWaveforms: false,
        mergeLayers: true,
      },
      palette: ['#112233', '#445566'],
      audioTracks: [
        {
          entry: 'audio/0.mp3',
          filename: 'backing.mp3',
          role: 'no-drums',
          lane: undefined,
          muted: false,
          soloed: false,
          volume: 0.8,
        },
      ],
    },
  };
}

/** Stand-in "audio" bytes for the manifest's one track (not real MP3; the
 *  container stores them verbatim, so any bytes round-trip). */
const AUDIO_BYTES = new Uint8Array([1, 2, 3, 4, 5, 250, 128, 0, 99]);

describe('mutable .jot container encode/decode', () => {
  test('round-trips the envelope + embedded audio bytes losslessly', async () => {
    const file = sampleFile();
    const bytes = await encodeMutableJotFile(file, [{ entry: 'audio/0.mp3', bytes: AUDIO_BYTES }]);
    const { file: decoded, audio } = await decodeMutableJotFile(bytes);

    expect(decoded.format).toBe('drumjot-mutable');
    expect(decoded.version).toBe(JOT_FILE_VERSION);
    expect(decoded.savedAt).toBe(file.savedAt);
    expect(decoded.document).toEqual(file.document);
    expect(decoded.source).toEqual(file.source);
    expect(decoded.editor).toEqual(file.editor);

    // The embedded audio entry comes back byte-for-byte under its manifest path.
    expect(audio.size).toBe(1);
    expect(Array.from(audio.get('audio/0.mp3') ?? [])).toEqual(Array.from(AUDIO_BYTES));
  });

  test('the decoded document re-seeds an identical mutable doc', async () => {
    const file = sampleFile();
    const bytes = await encodeMutableJotFile(file);
    const { file: decoded } = await decodeMutableJotFile(bytes);

    const reseeded = createMutableJotFromState(decoded.document).snapshot();
    expect(reseeded).toEqual(file.document);
  });

  test('a deflated audio entry (WAV-style PCM) round-trips byte-for-byte', async () => {
    // Repetitive bytes so deflate actually shrinks them, exercising the
    // method-8 path the container uses for uncompressed PCM sources.
    const pcm = new Uint8Array(2048).map((_, i) => (i >> 4) & 0xff);
    const file = sampleFile();
    const bytes = await encodeMutableJotFile(file, [
      { entry: 'audio/0.wav', bytes: pcm, compress: true },
    ]);
    const { audio } = await decodeMutableJotFile(bytes);
    expect(Array.from(audio.get('audio/0.wav') ?? [])).toEqual(Array.from(pcm));
  });

  test('encodes with no audio (envelope only)', async () => {
    const file = { ...sampleFile(), editor: { ...sampleFile().editor, audioTracks: undefined } };
    const bytes = await encodeMutableJotFile(file);
    const { audio } = await decodeMutableJotFile(bytes);
    expect(audio.size).toBe(0);
  });

  test('encoded bytes carry the DJOT magic header', async () => {
    const bytes = await encodeMutableJotFile(sampleFile());
    expect(isMutableJotFile(bytes)).toBe(true);
    expect(Array.from(bytes.subarray(0, 4))).toEqual([0x44, 0x4a, 0x4f, 0x54]);
  });
});

describe('format sniffing', () => {
  test('plain DSL text is not mistaken for the mutable format', () => {
    const dsl = new TextEncoder().encode('My Song @120\nc s c s\n');
    expect(isMutableJotFile(dsl)).toBe(false);
  });

  test('a bare zip (PK…) is not mistaken for the mutable format', () => {
    const zip = new Uint8Array([0x50, 0x4b, 0x03, 0x04, 0, 0]);
    expect(isMutableJotFile(zip)).toBe(false);
  });

  test('an empty buffer is not the mutable format', () => {
    expect(isMutableJotFile(new Uint8Array())).toBe(false);
  });

  test('decoding non-mutable bytes throws (caller should sniff first)', async () => {
    const dsl = new TextEncoder().encode('My Song @120\n');
    await expect(decodeMutableJotFile(dsl)).rejects.toThrow(/magic header/);
  });
});

describe('version guard', () => {
  test('refuses a file saved by a newer app version', async () => {
    const file = sampleFile();
    const bytes = await encodeMutableJotFile({ ...file, version: JOT_FILE_VERSION + 1 });
    await expect(decodeMutableJotFile(bytes)).rejects.toThrow(/newer version/);
  });
});
