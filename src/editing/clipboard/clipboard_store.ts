import { makeAutoObservable, observable } from 'mobx';
import type { ClipboardPayload } from './clipboard_payload';

/**
 * The in-app clipboard, DATA ONLY (a single observable). Holds the most recent
 * copy/cut made in THIS document so a paste needn't depend on the async system
 * clipboard (and as a fallback when the system clipboard holds non-Drumjot
 * data). The system clipboard is written in lockstep on copy with the same
 * timestamp; on paste the newer of the two wins. Every mutation lives on
 * {@link ClipboardPresenter}.
 */
export class ClipboardStore {
  /** The last copied cluster, or `undefined` before any copy this session. */
  payload: ClipboardPayload | undefined = undefined;

  constructor() {
    // `observable.ref`: each copy replaces the whole payload (its `notes` array
    // is immutable), so ref-equality suffices and MobX won't deep-proxy the
    // readonly note records.
    makeAutoObservable(this, { payload: observable.ref });
  }
}
