import { makeAutoObservable } from 'mobx';

/**
 * Undo/redo availability for the loaded song, DATA ONLY (observables). The
 * booleans drive toolbar button enablement; the keyboard path doesn't read
 * them (a no-op undo with an empty stack is harmless). Every mutation lives on
 * {@link HistoryPresenter}, which mirrors the live Loro `UndoManager`'s
 * `canUndo()` / `canRedo()` into these after each document change.
 */
export class HistoryStore {
  /** True when there's at least one local edit to undo. */
  canUndo: boolean = false;
  /** True when an undo has been performed and not yet superseded by a new edit. */
  canRedo: boolean = false;

  constructor() {
    makeAutoObservable(this);
  }
}
