import { reaction, runInAction } from 'mobx';
import { UndoManager, type LoroDoc } from 'loro-crdt';
import { HistoryStore } from './history_store';
import { JotEditorStore } from '../jot_editor_store';

/**
 * Undo/redo for the loaded song, built on Loro's local-peer {@link
 * UndoManager}. The reactive document already commits one Loro change per user
 * gesture (every façade `set` / `setAll` / variadic `delete` commits once, bulk
 * edits via the module-private `transact`), so each gesture is naturally one
 * undo step, no extra grouping here. Undo/redo replay through the same
 * `doc.subscribe` → MobX-cache path as every other write, so the score, mixer,
 * and selection update for free; this presenter only drives the document and
 * mirrors availability into {@link HistoryStore}.
 *
 * The manager is presenter-local (like an in-flight `AbortController`, not
 * store state) and is rebuilt whenever a new song loads: a fresh load swaps
 * `JotEditorStore.loroDoc`, the reaction below tears down the old manager +
 * subscription and attaches to the new doc. Because the doc is fully populated
 * before attach, the load itself isn't undoable, only edits after it are.
 *
 * Single writer of {@link HistoryStore}.
 */
export class HistoryPresenter {
  /** Live manager for the current doc, or undefined before the first load. */
  private manager: UndoManager | undefined;
  /** Tears down the current doc's availability subscription. */
  private unsubscribe: (() => void) | undefined;
  /** Disposes the doc-swap reaction (teardown / leak-test safety). */
  private readonly disposeReaction: () => void;

  constructor(
    private readonly historyStore: HistoryStore,
    private readonly jotEditorStore: JotEditorStore
  ) {
    // No observable state of its own (availability lives in HistoryStore), so
    // no makeAutoObservable; `refresh` wraps its store writes in runInAction.
    // Rebuild the manager whenever the backing doc swaps (song load / clear).
    // `fireImmediately` attaches to whatever doc is already loaded.
    this.disposeReaction = reaction(
      () => this.jotEditorStore.loroDoc,
      (doc) => this.attach(doc),
      { fireImmediately: true }
    );
  }

  /** Undo the last local edit. No-op (returns false) with an empty undo stack. */
  undo(): void {
    if (!this.manager?.canUndo()) return;
    this.manager.undo();
    this.refresh();
  }

  /** Redo the last undone edit. No-op with an empty redo stack. */
  redo(): void {
    if (!this.manager?.canRedo()) return;
    this.manager.redo();
    this.refresh();
  }

  /** Drop the manager + its subscription and build a fresh one over `doc`.
   *  `mergeInterval: 0` keeps every gesture a distinct undo step (Loro's
   *  default merges edits within 1s, which would fold rapid deliberate edits
   *  into one step). */
  private attach(doc: LoroDoc | undefined): void {
    this.unsubscribe?.();
    this.unsubscribe = undefined;
    this.manager?.free();
    this.manager = undefined;
    if (!doc) {
      this.refresh();
      return;
    }
    this.manager = new UndoManager(doc, { mergeInterval: 0 });
    // Any commit (local edit, or an undo/redo replay) can change availability;
    // recompute off the same event stream the cache rides.
    this.unsubscribe = doc.subscribe(() => this.refresh());
    this.refresh();
  }

  /** Mirror the manager's availability into the store. */
  private refresh(): void {
    const canUndo = this.manager?.canUndo() ?? false;
    const canRedo = this.manager?.canRedo() ?? false;
    runInAction(() => {
      this.historyStore.canUndo = canUndo;
      this.historyStore.canRedo = canRedo;
    });
  }

  /** Tear down the reaction + live manager (editor disposal / leak tests). */
  dispose(): void {
    this.disposeReaction();
    this.unsubscribe?.();
    this.unsubscribe = undefined;
    this.manager?.free();
    this.manager = undefined;
  }
}
