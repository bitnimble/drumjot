import React from 'react';
import { COMMANDS_BY_ID, type CommandContext } from './commands';

/**
 * Keyboard layer, abstracted from the action layer. A {@link Keymap} maps a
 * normalized key-combo string to a command id; the dispatcher resolves combo →
 * id → command and runs it. Remapping later is purely a matter of swapping the
 * keymap, the commands don't change.
 */
export type Keymap = Readonly<Record<string, string>>;

export const DEFAULT_KEYMAP: Keymap = {
  Delete: 'deleteSelection',
  Backspace: 'deleteSelection',
  Space: 'togglePlayPause',
  'Ctrl+g': 'group',
  'Ctrl+Shift+g': 'ungroup',
};

/** INPUT `type`s where a keystroke is meaningful text entry and shortcuts must
 *  yield. A range/checkbox/etc. input is absent, so (e.g.) Space still toggles
 *  transport while a slider has focus. */
const TEXT_ENTRY_INPUT_TYPES = new Set([
  'text',
  'search',
  'email',
  'url',
  'tel',
  'password',
  'number',
]);

/** Normalize a KeyboardEvent to a combo string used as a keymap key. Space is
 *  spelled `Space` (its `key` is a literal space); a bare key uses `key`
 *  directly (`Delete`, `Backspace`, …). A Ctrl/⌘ chord is spelled
 *  `Ctrl[+Shift]+<lowercased key>` (`Ctrl+g`, `Ctrl+Shift+g`), so a letter
 *  shortcut matches regardless of the Shift-cased `key` (`G` vs `g`) or
 *  platform (Ctrl vs ⌘). Shift/Alt WITHOUT Ctrl don't prefix, so existing bare
 *  bindings (Delete/Backspace/Space) keep matching even when Shift is held. */
export function eventCombo(e: Pick<KeyboardEvent, 'key' | 'code' | 'ctrlKey' | 'metaKey' | 'shiftKey'>): string {
  const base = e.code === 'Space' || e.key === ' ' ? 'Space' : e.key;
  if (!e.ctrlKey && !e.metaKey) return base;
  const key = base.length === 1 ? base.toLowerCase() : base;
  return `Ctrl+${e.shiftKey ? 'Shift+' : ''}${key}`;
}

/** True when the event target is a control that should swallow the keystroke
 *  (the user is typing / driving a native picker). */
function isTextEntryTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  const tag = el?.tagName;
  const isTextInput =
    tag === 'INPUT' && TEXT_ENTRY_INPUT_TYPES.has((el as HTMLInputElement).type);
  return isTextInput || tag === 'TEXTAREA' || tag === 'SELECT' || el?.isContentEditable === true;
}

/**
 * Install the global keydown dispatcher for the editor. One listener resolves
 * each keystroke through the keymap to a command and runs it, skipping
 * text-entry targets. Supersedes the ad-hoc per-shortcut listeners.
 */
export function useEditorKeymap(ctx: CommandContext, keymap: Keymap = DEFAULT_KEYMAP): void {
  // Keep the latest ctx without re-subscribing the listener every render.
  const ctxRef = React.useRef(ctx);
  ctxRef.current = ctx;
  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (isTextEntryTarget(e.target)) return;
      const id = keymap[eventCombo(e)];
      if (!id) return;
      const command = COMMANDS_BY_ID.get(id);
      if (!command) return;
      e.preventDefault();
      command.run(ctxRef.current);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [keymap]);
}
