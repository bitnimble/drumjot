import { toastStore } from 'src/ui/toasts/toasts';

/**
 * Thrown when a request to the backend API never reached the server (DNS
 * failure, connection refused, CORS preflight reject, offline), i.e. the
 * `fetch` promise itself rejected, as opposed to the server answering with a
 * 4xx/5xx (which resolves with `res.ok === false`). Callers can detect this
 * with {@link isBackendUnreachable} to skip their own per-request error toast,
 * since {@link backendFetch} has already surfaced the generic "Server is down"
 * notice.
 */
export class BackendUnreachableError extends Error {
  constructor() {
    super('Server is down');
    this.name = 'BackendUnreachableError';
  }
}

export function isBackendUnreachable(err: unknown): err is BackendUnreachableError {
  return err instanceof BackendUnreachableError;
}

/** `data-testid` on the generic outage toast; also the dedupe key. */
export const SERVER_DOWN_TOAST_TEST_ID = 'server-down-toast';

function reportServerDown(): void {
  // Don't stack identical notices when several backend calls fail together
  // (a page firing list + a user action at once). One sticky toast is enough;
  // once the user dismisses it, the next failure re-notifies.
  if (toastStore.toasts.some((t) => t.testId === SERVER_DOWN_TOAST_TEST_ID)) return;
  toastStore.showError('Server is down', {
    title:
      'The transcription backend is unreachable. Loading and editing local files still works.',
    testId: SERVER_DOWN_TOAST_TEST_ID,
  });
}

/**
 * `fetch` wrapper for every call to the backend API (`/api/...`). Behaves
 * exactly like `fetch` on success and on HTTP error responses (a 4xx/5xx
 * resolves normally, callers keep inspecting `res.ok`/status as before).
 *
 * The one difference is a *transport* failure (the backend is down /
 * unreachable, so `fetch` rejects): it surfaces a single generic, throttled
 * "Server is down" toast and rethrows a {@link BackendUnreachableError} so the
 * caller can bail without double-reporting. A user-initiated cancel
 * (`AbortError`) is rethrown untouched; that isn't the server being down.
 *
 * Pass `{ silent: true }` for background / fire-and-forget probes (liveness
 * checks, the recent-runs list) that should fail quietly without a toast; they
 * still get the typed `BackendUnreachableError`.
 */
export async function backendFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
  opts: { silent?: boolean } = {}
): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (err) {
    // A user/programmatic cancel isn't "the server is down".
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    if (!opts.silent) reportServerDown();
    throw new BackendUnreachableError();
  }
}
