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
  // `Mod` = Ctrl on Windows/Linux, Cmd on macOS (see `eventCombo`). Undo /
  // redo follow the cross-platform convention (Cmd/Ctrl+Z, Cmd/Ctrl+Shift+Z,
  // plus Ctrl+Y for Windows muscle memory). Cut/copy/paste are intentionally
  // NOT here: they ride the DOM `copy`/`cut`/`paste` events (see
  // `clipboard_presenter`) so the system clipboard + context menu integrate.
  'Mod+z': 'undo',
  'Mod+Shift+z': 'redo',
  'Mod+y': 'redo',
  'Mod+g': 'group',
  'Mod+Shift+g': 'ungroup',
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

/** Normalize a KeyboardEvent to a combo string used as a keymap key.
 *
 *  Space is spelled `Space` (its `key` is a literal space) and stays
 *  modifier-free for backwards compatibility. Otherwise the combo is the
 *  modifier prefix (`Mod` for Ctrl/Cmd, then `Alt`, then `Shift`, in that
 *  fixed order) joined to the key by `+`: `Mod+z`, `Mod+Shift+z`, `Delete`.
 *
 *  `Mod` unifies Ctrl (Windows/Linux) and Cmd (macOS) so one binding serves
 *  both platforms. Single-character keys are lower-cased so the binding is
 *  written `Mod+z` regardless of Shift (whose presence is carried by the
 *  explicit `Shift` token, not the letter case `Z`). */
export function eventCombo(
  e: Pick<KeyboardEvent, 'key' | 'code' | 'ctrlKey' | 'metaKey' | 'altKey' | 'shiftKey'>
): string {
  if (e.code === 'Space' || e.key === ' ') return 'Space';
  const parts: string[] = [];
  if (e.ctrlKey || e.metaKey) parts.push('Mod');
  if (e.altKey) parts.push('Alt');
  if (e.shiftKey) parts.push('Shift');
  parts.push(e.key.length === 1 ? e.key.toLowerCase() : e.key);
  return parts.join('+');
}

/** macOS runtime (render `Mod` as ⌘, not Ctrl). Best-effort + SSR-safe;
 *  defaults to non-mac. `navigator.platform` is deprecated but still the most
 *  reliable signal in current browsers; prefer `userAgentData` when present. */
function isMacPlatform(): boolean {
  if (typeof navigator === 'undefined') return false;
  const platform =
    (navigator as { userAgentData?: { platform?: string } }).userAgentData?.platform ??
    navigator.platform ??
    '';
  return /mac/i.test(platform);
}

/** Render a normalized combo (see {@link eventCombo}) as a human-readable
 *  shortcut for a menu pill: `Mod+Shift+z` → `⌘⇧Z` on macOS, `Ctrl+Shift+Z`
 *  elsewhere. macOS concatenates its glyphs; other platforms join with `+`. */
export function formatCombo(combo: string): string {
  const mac = isMacPlatform();
  const symbols: Record<string, string> = mac
    ? { Mod: '⌘', Shift: '⇧', Alt: '⌥' }
    : { Mod: 'Ctrl', Shift: 'Shift', Alt: 'Alt' };
  const parts = combo.split('+').map((p) => symbols[p] ?? (p.length === 1 ? p.toUpperCase() : p));
  return mac ? parts.join('') : parts.join('+');
}

/** Display label of the FIRST shortcut bound to `commandId` in `keymap`, or
 *  undefined if none. Reads the live keymap registry, so a future rebind shows
 *  the new key with no hardcoded shortcut text anywhere. */
export function shortcutForCommand(
  commandId: string,
  keymap: Keymap = DEFAULT_KEYMAP
): string | undefined {
  for (const [combo, id] of Object.entries(keymap)) {
    if (id === commandId) return formatCombo(combo);
  }
  return undefined;
}

/** True when the event target is a control that should swallow the keystroke
 *  (the user is typing / driving a native picker). Shared with the clipboard
 *  handlers so copy/cut/paste over a focused text field stay native. */
export function isTextEntryTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  const tag = el?.tagName;
  const isTextInput =
    tag === 'INPUT' && TEXT_ENTRY_INPUT_TYPES.has((el as HTMLInputElement).type);
  return isTextInput || tag === 'TEXTAREA' || tag === 'SELECT' || el?.isContentEditable === true;
}

/** Controls that natively activate on Space: a real `<button>`, a `<summary>`,
 *  or an ARIA `role="button"`. A Space keybinding must yield to a focused one so
 *  the keystroke activates the control instead of ALSO toggling transport (and
 *  preventDefault stealing the press from the control entirely). */
function isActivatableControl(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (el == null) return false;
  return el.tagName === 'BUTTON' || el.tagName === 'SUMMARY' || el.getAttribute?.('role') === 'button';
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
      const combo = eventCombo(e);
      // A focused button/summary natively activates on Space; let it, rather than
      // also firing the Space transport toggle (whose preventDefault would then
      // steal the press from the button entirely).
      if (combo === 'Space' && isActivatableControl(e.target)) return;
      const id = keymap[combo];
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
