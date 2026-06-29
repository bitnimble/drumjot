import { type Artifact, type Op, type ResultRef } from './control_protocol';

/**
 * Audio handed to a backend op. The transport decides how it travels: HTTP
 * uploads the bytes (multipart); the local sidecar stages a blob to a temp file
 * it reads off disk, or uses a path directly. The frontend never has to know
 * which, it hands over a blob (+ filename) or a path and the adapter does the
 * right thing.
 */
export type AudioInput =
  | { kind: 'blob'; blob: Blob; filename: string }
  | { kind: 'path'; path: string };

/** Normalised progress for any backend op, regardless of transport. `frac` is a
 *  best-effort 0..1; `message` carries the stage detail for the UI pill. */
export type RunProgress = { stage: string; frac: number; message?: string };

export type RunResult = {
  /** File outputs (MIDI, stems). Resolve to bytes / media URLs via the client. */
  artifacts: Artifact[];
  /** Op-specific structured payload (e.g. alignLyrics → `{ lines }`, transcribe
   *  → the rich HTTP response). Undefined for ops that only emit files. */
  data?: unknown;
};

export type RunOptions = {
  onProgress?: (progress: RunProgress) => void;
  signal?: AbortSignal;
};

/**
 * The one seam between the frontend and the backend. Every transport-varying
 * call (transcribe / separate / alignLyrics) goes through `run`; the concrete
 * client ({@link HttpBackendClient} for a remote/dev server, the sidecar client
 * for the desktop app) owns the wire details. Mirrors the backend's own
 * transport-agnostic core (`transcriber/app/comms/core.py`). Server-only
 * features with no sidecar equivalent (recent-run list, resume, score, health)
 * stay on the HTTP client directly, they aren't part of this seam.
 */
export interface BackendClient {
  run(
    op: Op,
    audio: AudioInput,
    params: Record<string, unknown>,
    opts?: RunOptions,
  ): Promise<RunResult>;
  /** Bytes behind an artifact ref (MIDI download, etc.), per transport. */
  resolveBytes(ref: ResultRef): Promise<Uint8Array>;
  /** A loadable media URL behind an artifact ref (audio stems), per transport. */
  resolveMediaUrl(ref: ResultRef): string;
}
