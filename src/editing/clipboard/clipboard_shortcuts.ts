import React from 'react';
import { isTextEntryTarget } from '../keyboard/keymap';
import type { EditingStore } from '../editing_store';
import type { EditingPresenter } from '../editing_presenter';
import type { ClipboardPresenter } from './clipboard_presenter';

/**
 * Wire note cut / copy / paste to the DOM `copy` / `cut` / `paste` events
 * (rather than the keymap) so the real OS clipboard, its keyboard shortcuts,
 * AND the context-menu items all flow through one path with synchronous
 * `clipboardData` access. Each handler defers to {@link ClipboardPresenter} and
 * `preventDefault`s only when it actually handled the gesture, so an unrelated
 * copy (nothing selected, or a focused text field) stays native.
 *
 * Paste doesn't write immediately: it begins a placement that follows the
 * cursor. This hook also installs the Esc-to-cancel for that placement (only
 * while one is active), the click-to-commit lives on the bars row.
 */
export function useClipboardShortcuts(
  clipboardPresenter: ClipboardPresenter,
  editingStore: EditingStore,
  editingPresenter: EditingPresenter
): void {
  React.useEffect(() => {
    const onCopy = (e: ClipboardEvent) => {
      if (isTextEntryTarget(e.target)) return;
      if (clipboardPresenter.copy(e.clipboardData)) e.preventDefault();
    };
    const onCut = (e: ClipboardEvent) => {
      if (isTextEntryTarget(e.target)) return;
      if (clipboardPresenter.cut(e.clipboardData)) e.preventDefault();
    };
    const onPaste = (e: ClipboardEvent) => {
      if (isTextEntryTarget(e.target)) return;
      if (clipboardPresenter.paste(e.clipboardData)) e.preventDefault();
    };
    document.addEventListener('copy', onCopy);
    document.addEventListener('cut', onCut);
    document.addEventListener('paste', onPaste);
    return () => {
      document.removeEventListener('copy', onCopy);
      document.removeEventListener('cut', onCut);
      document.removeEventListener('paste', onPaste);
    };
  }, [clipboardPresenter]);

  // Esc cancels an in-flight paste placement. Installed only while a paste is
  // active so it never competes with other Escape handlers (menus, modals).
  const pasting = editingStore.pasteActive;
  React.useEffect(() => {
    if (!pasting) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      e.preventDefault();
      editingPresenter.cancelPaste();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [pasting, editingPresenter]);
}
