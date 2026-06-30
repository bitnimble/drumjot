import { runInAction } from 'mobx';
import type { EditingPresenter } from '../editing_presenter';
import { ClipboardStore } from './clipboard_store';
import {
  DRUMJOT_CLIPBOARD_MIME,
  deserializeClipboard,
  newerPayload,
  serializeClipboard,
} from './clipboard_payload';

/**
 * Cut / copy / paste orchestration for selected notes. Copy projects the
 * selection to a {@link ClipboardPayload} (via {@link EditingPresenter}) and
 * writes it BOTH to the in-app {@link ClipboardStore} and the system clipboard,
 * under a custom MIME only ({@link DRUMJOT_CLIPBOARD_MIME}), so a copy never
 * clobbers the user's `text/plain` clipboard. Paste reads both the in-app and
 * system payloads and places whichever was copied more recently. Cut is copy +
 * delete in one undo step (the delete).
 *
 * The handlers take a `DataTransfer` (the DOM clipboard event's `clipboardData`)
 * so the same methods serve both the real OS shortcuts / context menu (wired by
 * `useClipboardShortcuts`) and unit tests. Single writer of {@link ClipboardStore}.
 */
export class ClipboardPresenter {
  constructor(
    private readonly clipboardStore: ClipboardStore,
    private readonly editingPresenter: EditingPresenter
  ) {}

  /**
   * Copy the selection to the in-app store and (if `data` given) the system
   * clipboard. Returns true if anything was copied (caller should then
   * `preventDefault` so the browser doesn't also write the text selection).
   */
  copy(data: DataTransfer | null): boolean {
    const payload = this.editingPresenter.copySelectionPayload();
    if (!payload) return false;
    runInAction(() => {
      this.clipboardStore.payload = payload;
    });
    // Custom MIME only, deliberately no `text/plain`, so the user's text
    // clipboard is untouched.
    data?.setData(DRUMJOT_CLIPBOARD_MIME, serializeClipboard(payload));
    return true;
  }

  /** Cut = copy + delete the selection. The delete is the single undo step
   *  (copy writes no document state). Returns true if anything was cut. */
  cut(data: DataTransfer | null): boolean {
    if (!this.copy(data)) return false;
    this.editingPresenter.deleteSelection();
    return true;
  }

  /**
   * Begin a paste placement from the newer of the in-app and system payloads.
   * Returns true if a payload was found (caller should `preventDefault`). The
   * actual write happens later, on the click that commits the placement.
   */
  paste(data: DataTransfer | null): boolean {
    const system = deserializeClipboard(data?.getData(DRUMJOT_CLIPBOARD_MIME));
    const chosen = newerPayload(this.clipboardStore.payload, system);
    if (!chosen) return false;
    this.editingPresenter.beginPaste(chosen);
    return true;
  }
}
