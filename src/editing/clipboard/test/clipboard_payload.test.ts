import { describe, expect, it } from 'bun:test';
import {
  DRUMJOT_CLIPBOARD_MIME,
  deserializeClipboard,
  newerPayload,
  serializeClipboard,
  type ClipboardPayload,
} from 'src/editing/clipboard/clipboard_payload';

const payload: ClipboardPayload = {
  copiedAt: 1000,
  notes: [
    { lane: 'k', relBeat: 0, duration: 0.25, modifiers: [] },
    { lane: 's', relBeat: 1, duration: 0.25, modifiers: ['r'], sticking: 'r', velocity: 100 },
  ],
};

describe('clipboard MIME', () => {
  it('is a custom type with no text/plain footprint', () => {
    // The point of a custom MIME: a copy doesn't clobber the user's text
    // clipboard. Guard the constant so it can't silently become text/plain.
    expect(DRUMJOT_CLIPBOARD_MIME).not.toBe('text/plain');
    expect(DRUMJOT_CLIPBOARD_MIME.startsWith('application/')).toBe(true);
  });
});

describe('serialize / deserialize', () => {
  it('round-trips a payload', () => {
    expect(deserializeClipboard(serializeClipboard(payload))).toEqual(payload);
  });

  it('drops note fields that are absent (no undefined keys)', () => {
    const back = deserializeClipboard(serializeClipboard(payload))!;
    expect(Object.keys(back.notes[0]).sort()).toEqual(['duration', 'lane', 'modifiers', 'relBeat']);
  });

  it('returns undefined for empty / non-JSON / foreign clipboard text', () => {
    expect(deserializeClipboard(undefined)).toBeUndefined();
    expect(deserializeClipboard('')).toBeUndefined();
    expect(deserializeClipboard('not json')).toBeUndefined();
    expect(deserializeClipboard('"a plain string the user copied"')).toBeUndefined();
    expect(deserializeClipboard('42')).toBeUndefined();
  });

  it('rejects a payload of an unknown version', () => {
    const future = JSON.stringify({ version: 999, copiedAt: 1, notes: payload.notes });
    expect(deserializeClipboard(future)).toBeUndefined();
  });

  it('rejects a malformed payload (missing/!typed fields, or no notes)', () => {
    expect(deserializeClipboard(JSON.stringify({ version: 1, notes: [] }))).toBeUndefined();
    expect(
      deserializeClipboard(JSON.stringify({ version: 1, copiedAt: 1, notes: [] }))
    ).toBeUndefined();
    expect(
      deserializeClipboard(JSON.stringify({ version: 1, copiedAt: 1, notes: [{ lane: 'k' }] }))
    ).toBeUndefined();
  });
});

describe('newerPayload (newer copy wins)', () => {
  const older: ClipboardPayload = { copiedAt: 100, notes: payload.notes };
  const newer: ClipboardPayload = { copiedAt: 200, notes: payload.notes };

  it('picks the later timestamp regardless of argument order', () => {
    expect(newerPayload(older, newer)).toBe(newer);
    expect(newerPayload(newer, older)).toBe(newer);
  });

  it('favours the in-app payload on a tie (same-tab copy, no clipboard read)', () => {
    const a: ClipboardPayload = { copiedAt: 100, notes: payload.notes };
    const b: ClipboardPayload = { copiedAt: 100, notes: payload.notes };
    expect(newerPayload(a, b)).toBe(a);
  });

  it('falls back to whichever is present', () => {
    expect(newerPayload(undefined, newer)).toBe(newer);
    expect(newerPayload(older, undefined)).toBe(older);
    expect(newerPayload(undefined, undefined)).toBeUndefined();
  });
});
