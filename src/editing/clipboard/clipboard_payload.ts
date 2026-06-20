/**
 * Serializable shape of a copied note cluster, shared by the in-app clipboard
 * ({@link ClipboardStore}) and the system clipboard (a custom MIME on the DOM
 * `copy`/`cut`/`paste` events). Pure data + (de)serialization + the newer-wins
 * pick, with no MobX / DOM dependency so it unit-tests in isolation.
 */

/**
 * Custom clipboard MIME for a Drumjot note cluster. Written via the synchronous
 * DOM clipboard event's `DataTransfer.setData`, deliberately WITHOUT a
 * `text/plain` companion, so a copy doesn't clobber the user's text clipboard.
 * The `+json` suffix marks the payload encoding.
 */
export const DRUMJOT_CLIPBOARD_MIME = 'application/x-drumjot-notes+json';

/** Bumped if the payload shape changes incompatibly; an unknown version is
 *  treated as an unreadable payload (ignored, never thrown). */
const PAYLOAD_VERSION = 1;

/**
 * One copied note, positioned RELATIVE to the cluster's anchor (the earliest
 * copied note, whose {@link relBeat} is 0). Carries the musical fields a paste
 * re-creates verbatim; the note's identity, owning bar, and instrument track
 * are re-resolved at paste time from the drop position, never copied.
 */
export type ClipboardNote = {
  /** Instrument lane letter; preserved across a paste. */
  lane: string;
  /** Quarter-note beats after the cluster's earliest note (anchor = 0). */
  relBeat: number;
  duration: number;
  modifiers: readonly string[];
  sticking?: string;
  roll?: boolean;
  vol?: string;
  offsetMs?: number;
  velocity?: number;
  midiNote?: number;
};

export type ClipboardPayload = {
  /** Epoch ms at copy time. On paste, the newer of the in-app + system payloads
   *  wins (see {@link newerPayload}), so a copy made in another tab supersedes a
   *  stale in-app copy and vice versa. */
  copiedAt: number;
  notes: readonly ClipboardNote[];
};

/** Encode a payload for the system clipboard. */
export function serializeClipboard(payload: ClipboardPayload): string {
  return JSON.stringify({ version: PAYLOAD_VERSION, ...payload });
}

/**
 * Decode a system-clipboard string back to a payload, or `undefined` if it's
 * absent / not ours / malformed / a future version. Never throws: a foreign
 * clipboard (no Drumjot data, or another app's text) simply yields `undefined`,
 * and paste then falls back to the in-app store.
 */
export function deserializeClipboard(text: string | null | undefined): ClipboardPayload | undefined {
  if (!text) return undefined;
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return undefined;
  }
  if (typeof raw !== 'object' || raw === null) return undefined;
  const o = raw as Record<string, unknown>;
  if (o.version !== PAYLOAD_VERSION) return undefined;
  if (typeof o.copiedAt !== 'number' || !Array.isArray(o.notes)) return undefined;
  const notes: ClipboardNote[] = [];
  for (const n of o.notes) {
    if (typeof n !== 'object' || n === null) return undefined;
    const m = n as Record<string, unknown>;
    if (typeof m.lane !== 'string' || typeof m.relBeat !== 'number' || typeof m.duration !== 'number') {
      return undefined;
    }
    notes.push({
      lane: m.lane,
      relBeat: m.relBeat,
      duration: m.duration,
      modifiers: Array.isArray(m.modifiers) ? m.modifiers.filter((x): x is string => typeof x === 'string') : [],
      ...(typeof m.sticking === 'string' ? { sticking: m.sticking } : {}),
      ...(typeof m.roll === 'boolean' ? { roll: m.roll } : {}),
      ...(typeof m.vol === 'string' ? { vol: m.vol } : {}),
      ...(typeof m.offsetMs === 'number' ? { offsetMs: m.offsetMs } : {}),
      ...(typeof m.velocity === 'number' ? { velocity: m.velocity } : {}),
      ...(typeof m.midiNote === 'number' ? { midiNote: m.midiNote } : {}),
    });
  }
  if (notes.length === 0) return undefined;
  return { copiedAt: o.copiedAt, notes };
}

/**
 * The payload to paste from, given the in-app and system candidates: whichever
 * was copied more recently. A tie (the same-tab copy writes both with one
 * stamp) favours `inApp`, which needs no clipboard read. `undefined` only when
 * both are absent.
 */
export function newerPayload(
  inApp: ClipboardPayload | undefined,
  system: ClipboardPayload | undefined
): ClipboardPayload | undefined {
  if (!inApp) return system;
  if (!system) return inApp;
  return system.copiedAt > inApp.copiedAt ? system : inApp;
}
