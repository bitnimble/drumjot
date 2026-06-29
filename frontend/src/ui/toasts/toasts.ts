import { makeAutoObservable, runInAction } from 'mobx';
import type { ReactNode } from 'react';

export type ToastKind = 'success' | 'error' | 'warning';

export type Toast = {
  id: string;
  kind: ToastKind;
  /** Renderable body. ReactNode (not just string) so callers can embed
   *  links (e.g. the transcribe-done [debug.zip] anchor) without losing
   *  the toast's click-to-dismiss outer affordance. */
  message: ReactNode;
  /** Optional `title=` tooltip, used by callers that truncate their
   *  visible message and want the full text on hover. */
  title?: string;
  /** Optional `data-testid=` for Playwright probes. */
  testId?: string;
};

export type ShowToastOpts = {
  title?: string;
  testId?: string;
};

/** Success toasts auto-dismiss after this delay; error toasts are sticky. */
const SUCCESS_AUTO_DISMISS_MS = 4000;

/**
 * One-shot notification surface for the app. Long-running operations
 * keep their toolbar pill (e.g. "Transcribing… filename · separating
 * drums"); one-off events (success / error after a long-running op
 * finishes, or any other fire-and-forget notice) route through here
 * instead so the toolbar isn't permanently decorated with stale
 * dismiss-me-please banners.
 *
 * Module-level singleton, MobX-observable. Rendered by `ToastContainer`
 * mounted once at the top of `View`.
 */
class ToastStore {
  toasts: Toast[] = [];
  private nextId = 1;
  private timers = new Map<string, ReturnType<typeof setTimeout>>();

  constructor() {
    makeAutoObservable(this, { showSuccess: false, showError: false, showWarning: false });
  }

  showSuccess(message: ReactNode, opts: ShowToastOpts = {}): string {
    return this.show('success', message, opts);
  }

  showError(message: ReactNode, opts: ShowToastOpts = {}): string {
    return this.show('error', message, opts);
  }

  /** Sticky like an error (the user should read what was changed), but
   *  amber rather than red, the action succeeded, it just had side effects. */
  showWarning(message: ReactNode, opts: ShowToastOpts = {}): string {
    return this.show('warning', message, opts);
  }

  show(kind: ToastKind, message: ReactNode, opts: ShowToastOpts = {}): string {
    const id = `toast-${this.nextId++}`;
    const toast: Toast = { id, kind, message, ...opts };
    runInAction(() => {
      // Newest first; the container caps the visible stack so older
      // toasts roll off naturally without us hard-dropping them.
      this.toasts = [toast, ...this.toasts];
    });
    if (kind === 'success') {
      const timer = setTimeout(() => this.dismiss(id), SUCCESS_AUTO_DISMISS_MS);
      this.timers.set(id, timer);
    }
    return id;
  }

  dismiss(id: string): void {
    const timer = this.timers.get(id);
    if (timer !== undefined) {
      clearTimeout(timer);
      this.timers.delete(id);
    }
    runInAction(() => {
      this.toasts = this.toasts.filter((t) => t.id !== id);
    });
  }
}

export const toastStore = new ToastStore();
