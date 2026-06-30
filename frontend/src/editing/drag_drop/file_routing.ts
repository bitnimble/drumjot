/**
 * Drag-and-drop file routing: classify a dropped `File` by its name (and,
 * for `.zip`, by peeking at its central directory) and partition a whole
 * drop into a {@link DropPlan} the {@link JotEditorPresenter} can execute.
 *
 * Pure / side-effect free (no store reads, no mutation, no toasts) so the
 * partitioning logic is unit-testable in isolation; the presenter owns the
 * actual loader calls. The routing mirrors the existing loaders:
 *
 *   - `.jot`                       → load as the current jot (replaces)
 *   - audio (`.mp3` / `.wav` / …)  → add an audio track (additive)
 *   - `.mid` / `.midi`             → convert to a jot (replaces)
 *   - text (`.lrc` / `.txt`)       → add a synced-lyrics track (additive)
 *   - `.zip` → peek inside:
 *       · a `debug.json`  entry    → transcriber debug bundle (replaces)
 *       · any `.rlrr`     entry    → ParaDB / Paradiddle map (replaces)
 *       · any `.jot`      entry    → extract + load that jot (replaces)
 */
import { inflateEntry, readCentralDirectory, zipEntryBasename } from 'src/utils/zip';

/** Loads that swap the whole document (and so warrant a confirm when one
 *  is already open). All resolve to a single `File` the presenter feeds to
 *  the matching loader; a zip whose payload is a bare `.jot` is extracted
 *  in {@link planDrop} and surfaces here as a plain `'jot'`. */
export type DocumentLoadKind = 'jot' | 'midi' | 'paradb' | 'debug';

/** Loads that add a track to the current document without replacing it. */
export type AdditiveKind = 'audio' | 'lyrics';

/** Extension-level classification (synchronous; a `.zip` still needs its
 *  central directory peeked to resolve to a concrete kind). */
export type ExtensionKind = DocumentLoadKind | AdditiveKind | 'zip' | 'unknown';

export type PlannedFile<K> = { file: File; kind: K };

/** The resolved routing for a single drop. */
export type DropPlan = {
  /** The one document-replacing load to run, if any. */
  documentLoad?: PlannedFile<DocumentLoadKind>;
  /** Extra document-replacing files beyond the first: a single drop can
   *  only replace the document once, so these are surfaced to the user and
   *  skipped. */
  ignoredDocumentLoads: PlannedFile<DocumentLoadKind>[];
  /** Track-adding loads (audio / lyrics), run in drop order. */
  additive: PlannedFile<AdditiveKind>[];
  /** Files whose type couldn't be resolved (unknown extension, or a `.zip`
   *  matching none of the three recognised layouts). */
  unknown: File[];
};

/** Audio container extensions the player's `decodeAudioData` path accepts.
 *  Kept broad; an unsupported-codec file still routes here and fails with a
 *  decode error toast, which is clearer than "unknown file type". */
const AUDIO_EXTENSIONS = new Set([
  'mp3',
  'wav',
  'flac',
  'ogg',
  'oga',
  'opus',
  'm4a',
  'mp4',
  'aac',
  'aif',
  'aiff',
  'webm',
]);

/** Lowercased extension (without the dot), or `''` if the name has none. */
export function fileExtension(name: string): string {
  const dot = name.lastIndexOf('.');
  if (dot < 0 || dot === name.length - 1) return '';
  return name.slice(dot + 1).toLowerCase();
}

/**
 * Classify a file by its extension alone. `.zip` resolves only to `'zip'`
 * here; {@link classifyZip} peeks inside to pick the concrete loader.
 *
 * `.txt` routes to lyrics (the user's "text file → lyrics track" intent);
 * a Drumjot DSL file is loaded by its `.jot` extension.
 */
export function classifyByExtension(file: File): ExtensionKind {
  const ext = fileExtension(file.name);
  if (ext === 'jot') return 'jot';
  if (ext === 'mid' || ext === 'midi') return 'midi';
  if (ext === 'lrc' || ext === 'txt') return 'lyrics';
  if (ext === 'zip') return 'zip';
  if (AUDIO_EXTENSIONS.has(ext)) return 'audio';
  return 'unknown';
}

/** What a `.zip`'s contents resolve to, by central-directory layout. */
export type ZipKind = 'debug' | 'paradb' | 'jot' | 'unknown';

/**
 * Peek a `.zip`'s central directory (no inflation) to tell the three
 * recognised layouts apart. Detection order is deliberate: a debug bundle
 * is identified by its `debug.json` manifest, a ParaDB pack by carrying any
 * `.rlrr` chart, and a bare-jot archive by carrying any `.jot`. A malformed
 * or empty archive resolves to `'unknown'`.
 */
export async function classifyZip(file: File): Promise<ZipKind> {
  let names: string[];
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    names = readCentralDirectory(bytes).map((e) => zipEntryBasename(e.name).toLowerCase());
  } catch {
    return 'unknown';
  }
  if (names.length === 0) return 'unknown';
  if (names.includes('debug.json')) return 'debug';
  if (names.some((n) => n.endsWith('.rlrr'))) return 'paradb';
  if (names.some((n) => n.endsWith('.jot'))) return 'jot';
  return 'unknown';
}

/**
 * Extract the first `.jot` entry from a zip as a standalone `File` so the
 * jot loader's `file.text()` path absorbs it unchanged. Returns `null` if
 * the archive has no `.jot` entry or it can't be inflated.
 */
export async function extractJotFromZip(file: File): Promise<File | null> {
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const entry = readCentralDirectory(bytes).find((e) =>
      zipEntryBasename(e.name).toLowerCase().endsWith('.jot')
    );
    if (!entry) return null;
    const data = await inflateEntry(bytes, entry);
    return new File([data as BlobPart], zipEntryBasename(entry.name), { type: 'text/plain' });
  } catch {
    return null;
  }
}

const DOCUMENT_KINDS = new Set<ExtensionKind>(['jot', 'midi', 'paradb', 'debug']);

/**
 * Resolve a whole drop into a {@link DropPlan}. Each file is classified
 * (zips are peeked, and a bare-jot zip has its `.jot` extracted), then
 * partitioned into the single document-replacing load (extras flagged as
 * ignored), the additive track loads, and the unrecognised files.
 *
 * Pure: classifies and reshapes only; the caller ({@link JotEditorPresenter})
 * runs the loaders and shows the confirm dialog / toasts.
 */
export async function planDrop(files: File[]): Promise<DropPlan> {
  const documentLoads: PlannedFile<DocumentLoadKind>[] = [];
  const additive: PlannedFile<AdditiveKind>[] = [];
  const unknown: File[] = [];

  for (const file of files) {
    const ext = classifyByExtension(file);
    if (ext === 'zip') {
      const zk = await classifyZip(file);
      if (zk === 'debug' || zk === 'paradb') {
        documentLoads.push({ file, kind: zk });
      } else if (zk === 'jot') {
        const jotFile = await extractJotFromZip(file);
        if (jotFile) documentLoads.push({ file: jotFile, kind: 'jot' });
        else unknown.push(file);
      } else {
        unknown.push(file);
      }
    } else if (DOCUMENT_KINDS.has(ext)) {
      documentLoads.push({ file, kind: ext as DocumentLoadKind });
    } else if (ext === 'audio' || ext === 'lyrics') {
      additive.push({ file, kind: ext });
    } else {
      unknown.push(file);
    }
  }

  const [documentLoad, ...ignoredDocumentLoads] = documentLoads;
  return { documentLoad, ignoredDocumentLoads, additive, unknown };
}

/** Human-readable summary of what a plan will do, for the confirm dialog. */
export function describeDocumentLoad(item: PlannedFile<DocumentLoadKind>): string {
  switch (item.kind) {
    case 'jot':
      return `Load "${item.file.name}" as a new score`;
    case 'midi':
      return `Convert "${item.file.name}" to a new score`;
    case 'paradb':
      return `Load the ParaDB map "${item.file.name}"`;
    case 'debug':
      return `Load the debug bundle "${item.file.name}"`;
  }
}
