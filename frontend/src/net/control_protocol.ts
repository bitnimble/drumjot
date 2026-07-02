/**
 * Backend control protocol, the wire contract the frontend uses to drive the
 * transcriber/backend over whichever transport carries it (stdio to the local
 * Tauri sidecar, or HTTP/WS to a remote backend). The core neither knows nor
 * cares which transport delivers a frame.
 *
 * Every message is one JSON object; over stdio they're newline-delimited (no
 * binary on the wire, large artifacts are passed by reference, see
 * `docs/superpowers/specs/2026-06-29-desktop-app-design.md`). These Zod schemas
 * validate BOTH directions: {@link encodeClientMessage} validates before
 * sending (catches a malformed request we built), {@link decodeServerMessage}
 * validates on receipt (no unchecked `as` casts on parsed JSON).
 */
import { z } from 'zod';

/** Bumped on any breaking change to the message shapes below. Carried as `v`
 *  on every frame so a mismatched peer fails loudly instead of silently
 *  misreading fields. */
export const PROTOCOL_VERSION = 1;

/**
 * An input audio reference. Local mode passes a filesystem path the backend
 * reads directly off disk; remote mode passes an upload id. Never bytes, the
 * frontend resolves the right kind per deployment.
 */
export const SourceRefSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('path'), path: z.string() }),
  z.object({ kind: z.literal('upload'), uploadId: z.string() }),
]);

/**
 * An output artifact reference. Local mode returns a path (frontend turns it
 * into an `asset://` URL or `readFile`s it); remote returns a URL; tiny
 * payloads may be inlined as base64.
 */
export const ResultRefSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('path'), path: z.string() }),
  z.object({ kind: z.literal('url'), url: z.string() }),
  z.object({ kind: z.literal('inline'), bytesB64: z.string() }),
]);

export const ArtifactSchema = z.object({
  role: z.enum(['midi', 'stem', 'audio']),
  ref: ResultRefSchema,
  /** Semantic label for multi-stem results ("drums", "no_drums", or a DSL pitch
   *  letter "k"/"s"/"h"/"c"/"t"); lets the frontend map a stem to a lane. */
  name: z.string().optional(),
});

/** Backend operations a `request` can invoke. (Beat tracking runs as an internal
 *  transcribe stage, not a client-driven op.) */
export const OpSchema = z.enum(['transcribe', 'separate', 'alignLyrics']);

const base = { v: z.literal(PROTOCOL_VERSION), id: z.string() };

// ---- Client -> backend ----------------------------------------------------

export const RequestMessageSchema = z.object({
  ...base,
  type: z.literal('request'),
  op: OpSchema,
  args: z.object({
    audio: SourceRefSchema,
    params: z.record(z.string(), z.unknown()),
  }),
});

/** Cooperative cancel for the request with the matching `id`; long jobs poll
 *  for it. */
export const CancelMessageSchema = z.object({
  ...base,
  type: z.literal('cancel'),
});

export const ClientMessageSchema = z.discriminatedUnion('type', [
  RequestMessageSchema,
  CancelMessageSchema,
]);

// ---- Backend -> client ----------------------------------------------------

/** A stream per request `id`, terminated by exactly one `result` or `error`. */
export const ProgressMessageSchema = z.object({
  ...base,
  type: z.literal('progress'),
  stage: z.string(),
  frac: z.number().min(0).max(1),
  message: z.string().optional(),
});

export const ResultMessageSchema = z.object({
  ...base,
  type: z.literal('result'),
  artifacts: z.array(ArtifactSchema),
  /** Op-specific structured payload for ops whose result isn't a file, e.g.
   *  `alignLyrics` → `{ lines }`. Omitted by file-only ops (transcribe/separate). */
  data: z.unknown().optional(),
});

export const ErrorMessageSchema = z.object({
  ...base,
  type: z.literal('error'),
  code: z.string(),
  message: z.string(),
  recoverable: z.boolean(),
});

export const ServerMessageSchema = z.discriminatedUnion('type', [
  ProgressMessageSchema,
  ResultMessageSchema,
  ErrorMessageSchema,
]);

export type SourceRef = z.infer<typeof SourceRefSchema>;
export type ResultRef = z.infer<typeof ResultRefSchema>;
export type Artifact = z.infer<typeof ArtifactSchema>;
export type Op = z.infer<typeof OpSchema>;
export type RequestMessage = z.infer<typeof RequestMessageSchema>;
export type CancelMessage = z.infer<typeof CancelMessageSchema>;
export type ClientMessage = z.infer<typeof ClientMessageSchema>;
export type ProgressMessage = z.infer<typeof ProgressMessageSchema>;
export type ResultMessage = z.infer<typeof ResultMessageSchema>;
export type ErrorMessage = z.infer<typeof ErrorMessageSchema>;
export type ServerMessage = z.infer<typeof ServerMessageSchema>;

/** Validate an outgoing client message and serialize it to one JSON line.
 *  Throws if `msg` doesn't satisfy the schema (a bug on our side). */
export function encodeClientMessage(msg: ClientMessage): string {
  return JSON.stringify(ClientMessageSchema.parse(msg));
}

/** Parse + validate one received backend line. Throws on malformed input,
 *  use {@link safeDecodeServerMessage} where a single bad line must not kill
 *  the stream. */
export function decodeServerMessage(line: string): ServerMessage {
  return ServerMessageSchema.parse(JSON.parse(line));
}

export type DecodeResult =
  | { ok: true; message: ServerMessage }
  | { ok: false; error: string };

/** Non-throwing decode for streaming consumers: malformed JSON or a schema
 *  miss returns `{ok: false}` so the caller can log + skip the line rather
 *  than tear down the whole stream. */
export function safeDecodeServerMessage(line: string): DecodeResult {
  let json: unknown;
  try {
    json = JSON.parse(line);
  } catch (err) {
    return { ok: false, error: `invalid JSON: ${String(err)}` };
  }
  return safeDecodeServerValue(json);
}

/** Same as {@link safeDecodeServerMessage} but for an already-parsed value, e.g.
 *  a frame delivered through a Tauri `Channel` (the Rust broker forwards each
 *  frame as a JSON object, not a string). */
export function safeDecodeServerValue(value: unknown): DecodeResult {
  const parsed = ServerMessageSchema.safeParse(value);
  return parsed.success
    ? { ok: true, message: parsed.data }
    : { ok: false, error: parsed.error.message };
}
