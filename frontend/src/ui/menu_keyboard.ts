import React from 'react';

/**
 * Shared WAI-ARIA `menu` roving-focus keyboard behaviour for the floating
 * panels rendered by `DropdownButton` and `ContextMenu`. Their rows (plain
 * `<button>`s, `ActionMenuItem` / `ToggleMenuItem`, submenu triggers) were only
 * reachable by Tab; this adds the arrow-key contract a `role="menu"` advertises:
 *   - ArrowDown / ArrowUp move focus between the panel's rows (wrapping),
 *   - Home / End jump to the first / last row.
 *
 * There is deliberately NO focus-on-open: the first ArrowDown from outside the
 * rows already lands on the first one, and grabbing focus on open would put a
 * focus ring on an item after a plain mouse click and, for a submenu trigger
 * (whose `onFocus` opens its flyout), auto-open a submenu the user didn't ask
 * for.
 *
 * Scope: only the panel's OWN rows participate. Rows inside a nested open
 * submenu (a descendant `role="menu"`) are excluded so arrowing in the parent
 * doesn't walk into a fly-out; that submenu runs its own copy of this behaviour.
 */
function menuRowsOf(panel: HTMLElement): HTMLElement[] {
  const rows = Array.from(
    panel.querySelectorAll<HTMLElement>('button:not([disabled]), [role^="menuitem"]:not([disabled])')
  );
  return rows.filter((row) => {
    if (row.getAttribute('aria-disabled') === 'true') return false;
    // Exclude rows owned by a nested submenu panel (a descendant menu that
    // isn't `panel` itself).
    return row.closest('[role="menu"]') === panel;
  });
}

/**
 * Wires an arrow / Home / End roving-focus handler onto a menu panel. Returns
 * the `onKeyDown` to spread onto the panel element.
 */
export function useMenuKeyboard(panelRef: React.RefObject<HTMLDivElement | null>): {
  onKeyDown: (e: React.KeyboardEvent<HTMLDivElement>) => void;
} {
  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>): void => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp' && e.key !== 'Home' && e.key !== 'End') {
      return;
    }
    const panel = panelRef.current;
    if (!panel) return;
    const rows = menuRowsOf(panel);
    if (rows.length === 0) return;
    e.preventDefault();
    const current = rows.indexOf(document.activeElement as HTMLElement);
    let next: number;
    if (e.key === 'Home') {
      next = 0;
    } else if (e.key === 'End') {
      next = rows.length - 1;
    } else if (e.key === 'ArrowDown') {
      next = current < 0 ? 0 : (current + 1) % rows.length;
    } else {
      next = current < 0 ? rows.length - 1 : (current - 1 + rows.length) % rows.length;
    }
    rows[next]?.focus();
  };

  return { onKeyDown };
}
