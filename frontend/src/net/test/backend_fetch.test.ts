/**
 * Tests for backendFetch: transport failures surface a single "Server is
 * down" toast and a typed BackendUnreachableError; HTTP error responses and
 * aborts pass through untouched.
 */
import { afterEach, describe, expect, it } from 'bun:test';
import {
  backendFetch,
  BackendUnreachableError,
  isBackendUnreachable,
  SERVER_DOWN_TOAST_TEST_ID,
} from '../backend_fetch';
import { toastStore } from 'src/ui/toasts/toasts';

const realFetch = globalThis.fetch;

function clearToasts() {
  for (const t of [...toastStore.toasts]) toastStore.dismiss(t.id);
}

function serverDownToasts() {
  return toastStore.toasts.filter((t) => t.testId === SERVER_DOWN_TOAST_TEST_ID);
}

afterEach(() => {
  globalThis.fetch = realFetch;
  clearToasts();
});

describe('backendFetch', () => {
  it('throws BackendUnreachableError and toasts on transport failure', async () => {
    clearToasts();
    globalThis.fetch = (() => Promise.reject(new TypeError('Failed to fetch'))) as unknown as typeof fetch;
    const err = await backendFetch('/api/x').then(
      () => null,
      (e) => e
    );
    expect(isBackendUnreachable(err)).toBe(true);
    expect(serverDownToasts()).toHaveLength(1);
  });

  it('rethrows an AbortError untouched, with no toast', async () => {
    clearToasts();
    globalThis.fetch = (() =>
      Promise.reject(new DOMException('aborted', 'AbortError'))) as unknown as typeof fetch;
    const err = await backendFetch('/api/x').then(
      () => null,
      (e) => e
    );
    expect(err).toBeInstanceOf(DOMException);
    expect((err as DOMException).name).toBe('AbortError');
    expect(err).not.toBeInstanceOf(BackendUnreachableError);
    expect(serverDownToasts()).toHaveLength(0);
  });

  it('passes through HTTP error responses without throwing or toasting', async () => {
    clearToasts();
    globalThis.fetch = (() =>
      Promise.resolve(new Response('nope', { status: 500 }))) as unknown as typeof fetch;
    const res = await backendFetch('/api/x');
    expect(res.status).toBe(500);
    expect(serverDownToasts()).toHaveLength(0);
  });

  it('does not stack a second toast while one is already showing', async () => {
    clearToasts();
    globalThis.fetch = (() => Promise.reject(new TypeError('Failed to fetch'))) as unknown as typeof fetch;
    await backendFetch('/api/a').catch(() => {});
    await backendFetch('/api/b').catch(() => {});
    expect(serverDownToasts()).toHaveLength(1);
  });

  it('silent mode throws the typed error but shows no toast', async () => {
    clearToasts();
    globalThis.fetch = (() => Promise.reject(new TypeError('Failed to fetch'))) as unknown as typeof fetch;
    const err = await backendFetch('/api/x', undefined, { silent: true }).then(
      () => null,
      (e) => e
    );
    expect(isBackendUnreachable(err)).toBe(true);
    expect(serverDownToasts()).toHaveLength(0);
  });
});
