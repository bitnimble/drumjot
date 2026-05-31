/**
 * Tests for `alignLyricsForced`. The client always uploads a multipart
 * form with the audio + the caller's lyrics payload; there's no longer
 * a cache-lookup probe (the realign-only flow has no cacheable output).
 *
 * `globalThis.fetch` is stubbed per test so the assertions can inspect
 * the outbound request shape.
 */

import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { alignLyricsForced } from '../forced_align';

type FetchCall = { url: string; method: string; body: BodyInit | null | undefined };

let fetchCalls: FetchCall[] = [];
let fetchHandler: (call: FetchCall) => Response | Promise<Response> = () =>
  new Response(null, { status: 500 });
const originalFetch = globalThis.fetch;

beforeEach(() => {
  fetchCalls = [];
  fetchHandler = () => new Response(null, { status: 500 });
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
    const method = (init?.method ?? 'GET').toUpperCase();
    const body = init?.body ?? null;
    const call: FetchCall = { url, method, body };
    fetchCalls.push(call);
    return fetchHandler(call);
  }) as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

function makeFile(content: string, name = 'audio.mp3'): File {
  return new File([content], name, { type: 'audio/mpeg' });
}

/** Build a streaming NDJSON response body from a list of envelopes, the
 *  wire shape the backend's /lyrics/align now emits (queued / running /
 *  result / error, one JSON object per line). */
function ndjsonResponse(envelopes: object[], status = 200): Response {
  const body = envelopes.map((e) => JSON.stringify(e)).join('\n') + '\n';
  return new Response(body, {
    status,
    headers: { 'Content-Type': 'application/x-ndjson' },
  });
}

describe('alignLyricsForced', () => {
  test('mix mode uploads the file + lyrics payload', async () => {
    const file = makeFile('mix-content');
    const aligned = [
      {
        startSec: 0,
        text: 'hello',
        words: [{ startSec: 0, endSec: 0.4, text: 'hello' }],
      },
    ];
    fetchHandler = (call) => {
      expect(call.url.endsWith('/lyrics/align')).toBe(true);
      expect(call.method).toBe('POST');
      expect(call.body).toBeInstanceOf(FormData);
      const form = call.body as FormData;
      expect(form.get('mix')).toBeInstanceOf(File);
      expect(form.get('vocals')).toBeNull();
      const payload = JSON.parse(form.get('lyrics') as string);
      expect(payload).toEqual([{ startSec: 0, text: 'hello' }]);
      return ndjsonResponse([{ type: 'running' }, { type: 'result', data: { lines: aligned } }]);
    };
    const lines = await alignLyricsForced({
      kind: 'mix',
      file,
      realign: { lines: [{ startSec: 0, text: 'hello' }] },
    });
    expect(lines).toEqual(aligned);
    expect(fetchCalls.length).toBe(1);
  });

  test('vocals mode sets the vocals form field instead of mix', async () => {
    const file = makeFile('vocals-content', 'vocals.flac');
    fetchHandler = (call) => {
      const form = call.body as FormData;
      expect(form.get('vocals')).toBeInstanceOf(File);
      expect(form.get('mix')).toBeNull();
      return ndjsonResponse([{ type: 'running' }, { type: 'result', data: { lines: [] } }]);
    };
    await alignLyricsForced({
      kind: 'vocals',
      file,
      realign: { lines: [{ startSec: 0, text: 'foo' }] },
    });
    expect(fetchCalls.length).toBe(1);
  });

  test('language hint rides on the form when present', async () => {
    const file = makeFile('lang-hint');
    fetchHandler = (call) => {
      const form = call.body as FormData;
      expect(form.get('language')).toBe('ja');
      return ndjsonResponse([{ type: 'running' }, { type: 'result', data: { lines: [] } }]);
    };
    await alignLyricsForced({
      kind: 'mix',
      file,
      realign: { lines: [{ startSec: 0, text: 'こんにちは' }], language: 'ja' },
    });
    expect(fetchCalls.length).toBe(1);
  });

  test('omits the language form field when no hint given', async () => {
    const file = makeFile('no-hint');
    fetchHandler = (call) => {
      const form = call.body as FormData;
      expect(form.get('language')).toBeNull();
      return ndjsonResponse([{ type: 'running' }, { type: 'result', data: { lines: [] } }]);
    };
    await alignLyricsForced({
      kind: 'mix',
      file,
      realign: { lines: [{ startSec: 0, text: 'hi' }] },
    });
    expect(fetchCalls.length).toBe(1);
  });

  test('non-OK status surfaces the server detail message', async () => {
    const file = makeFile('boom');
    fetchHandler = () =>
      new Response(JSON.stringify({ detail: 'no aligner for language=??' }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      });
    await expect(
      alignLyricsForced({
        kind: 'mix',
        file,
        realign: { lines: [{ startSec: 0, text: 'x' }] },
      }),
    ).rejects.toThrow(/no aligner for language/);
  });
});

describe('alignLyricsForced NDJSON stream', () => {
  const req = {
    kind: 'mix' as const,
    file: makeFile('x'),
    realign: { lines: [{ startSec: 0, text: 'hi' }] },
  };

  test('returns the lines carried by the result envelope', async () => {
    const aligned = [
      { startSec: 0, text: 'hello', words: [{ startSec: 0, endSec: 0.4, text: 'hello' }] },
    ];
    fetchHandler = () =>
      ndjsonResponse([{ type: 'running' }, { type: 'result', data: { lines: aligned } }]);
    const lines = await alignLyricsForced(req);
    expect(lines).toEqual(aligned);
  });

  test('reports queued then running progress to onProgress', async () => {
    const events: Array<{ kind: string }> = [];
    fetchHandler = () =>
      ndjsonResponse([
        { type: 'queued' },
        { type: 'running' },
        { type: 'result', data: { lines: [] } },
      ]);
    await alignLyricsForced(req, { onProgress: (e) => events.push(e) });
    expect(events).toEqual([{ kind: 'queued' }, { kind: 'running' }]);
  });

  test('omits queued from progress when the GPU was free', async () => {
    const events: Array<{ kind: string }> = [];
    fetchHandler = () =>
      ndjsonResponse([{ type: 'running' }, { type: 'result', data: { lines: [] } }]);
    await alignLyricsForced(req, { onProgress: (e) => events.push(e) });
    expect(events).toEqual([{ kind: 'running' }]);
  });

  test('throws the message from a terminal error envelope', async () => {
    fetchHandler = () =>
      ndjsonResponse([
        { type: 'running' },
        { type: 'error', status_code: 500, message: 'Separator ran but produced no vocals stem.' },
      ]);
    await expect(alignLyricsForced(req)).rejects.toThrow(/no vocals stem/);
  });

  test('throws when the stream ends without a terminal envelope', async () => {
    fetchHandler = () => ndjsonResponse([{ type: 'running' }]);
    await expect(alignLyricsForced(req)).rejects.toThrow();
  });
});
