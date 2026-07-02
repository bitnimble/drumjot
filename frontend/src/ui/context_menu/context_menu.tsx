import React from 'react';
import { createPortal } from 'react-dom';
import { dropdownStyles } from 'src/ui/dropdown/dropdown';
import { useMenuKeyboard } from 'src/ui/menu_keyboard';

/**
 * A right-click context menu: the SAME floating panel + menu-item components as
 * the header dropdowns ({@link DropdownButton}), but opened at an arbitrary
 * cursor position instead of anchored to a trigger button. Reuses
 * `dropdownStyles.dropdownPanel`, so callers fill it with the regular
 * {@link ActionMenuItem} / `DropdownSection` rows.
 *
 * Portaled to `document.body` and positioned `fixed` at `(x, y)`, clamped into
 * the viewport. Closes on outside pointer-down or Escape; menu items dismiss it
 * by calling `onClose` from their `onClick`.
 */
export function ContextMenu({
  x,
  y,
  onClose,
  children,
}: {
  x: number;
  y: number;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const panelRef = React.useRef<HTMLDivElement>(null);
  const [pos, setPos] = React.useState<{ top: number; left: number }>({ top: y, left: x });
  // Arrow/Home/End roving focus over the menu rows.
  const { onKeyDown: onPanelKeyDown } = useMenuKeyboard(panelRef);

  React.useEffect(() => {
    const onPointerDown = (e: MouseEvent) => {
      if (panelRef.current?.contains(e.target as Node)) return;
      onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [onClose]);

  // Clamp into the viewport once the panel's real size is known (one extra
  // pass; only runs on open / when the cursor anchor changes, never a render
  // hot path). Mirrors `DropdownButton`'s clamp.
  React.useLayoutEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;
    const margin = 8;
    const maxLeft = window.innerWidth - panel.offsetWidth - margin;
    const maxTop = window.innerHeight - panel.offsetHeight - margin;
    setPos({
      top: Math.max(margin, Math.min(y, maxTop)),
      left: Math.max(margin, Math.min(x, maxLeft)),
    });
  }, [x, y]);

  return createPortal(
    <div
      ref={panelRef}
      className={dropdownStyles.dropdownPanel}
      role="menu"
      style={{ position: 'fixed', top: pos.top, left: pos.left }}
      onContextMenu={(e) => e.preventDefault()}
      onKeyDown={onPanelKeyDown}
    >
      {children}
    </div>,
    document.body
  );
}
