import { describe, expect, it } from 'bun:test';
import {
  PROTOCOL_VERSION,
  decodeServerMessage,
  encodeClientMessage,
  safeDecodeServerMessage,
  type ClientMessage,
} from './control_protocol';

describe('control protocol', () => {
  it('round-trips a request through encode + JSON', () => {
    const msg: ClientMessage = {
      v: PROTOCOL_VERSION,
      type: 'request',
      id: 'job-1',
      op: 'transcribe',
      args: { audio: { kind: 'path', path: '/tmp/song.mp3' }, params: { quantise: true } },
    };
    expect(JSON.parse(encodeClientMessage(msg))).toEqual(msg);
  });

  it('rejects an outgoing message with the wrong protocol version', () => {
    const bad = {
      v: 999,
      type: 'cancel',
      id: 'job-1',
    } as unknown as ClientMessage;
    expect(() => encodeClientMessage(bad)).toThrow();
  });

  it('decodes each backend message variant', () => {
    const progress = decodeServerMessage(
      JSON.stringify({ v: PROTOCOL_VERSION, type: 'progress', id: 'j', stage: 'onsets', frac: 0.5 }),
    );
    expect(progress).toMatchObject({ type: 'progress', stage: 'onsets', frac: 0.5 });

    const result = decodeServerMessage(
      JSON.stringify({
        v: PROTOCOL_VERSION,
        type: 'result',
        id: 'j',
        artifacts: [{ role: 'midi', ref: { kind: 'path', path: '/out/pred.mid' } }],
      }),
    );
    expect(result).toMatchObject({ type: 'result', artifacts: [{ role: 'midi' }] });
  });

  it('rejects a progress frac outside 0..1', () => {
    expect(() =>
      decodeServerMessage(
        JSON.stringify({ v: PROTOCOL_VERSION, type: 'progress', id: 'j', stage: 's', frac: 1.5 }),
      ),
    ).toThrow();
  });

  it('rejects an unknown message type', () => {
    expect(() =>
      decodeServerMessage(JSON.stringify({ v: PROTOCOL_VERSION, type: 'bogus', id: 'j' })),
    ).toThrow();
  });

  it('safeDecode tolerates malformed lines instead of throwing', () => {
    expect(safeDecodeServerMessage('not json{')).toEqual({
      ok: false,
      error: expect.stringContaining('invalid JSON'),
    });
    const good = safeDecodeServerMessage(
      JSON.stringify({
        v: PROTOCOL_VERSION,
        type: 'error',
        id: 'j',
        code: 'boom',
        message: 'kaboom',
        recoverable: false,
      }),
    );
    expect(good.ok).toBe(true);
  });
});
