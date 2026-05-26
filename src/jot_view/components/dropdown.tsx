import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { createPortal } from 'react-dom';
import styles from './dropdown.module.css';

export { styles as dropdownStyles };

/**
 * Registry of close callbacks for every currently-open DropdownButton.
 * Opening a dropdown closes the others so at most one panel is visible.
 *
 * The document-level outside-click handler can't do this on its own: the
 * trigger's `onMouseDown` stops propagation (needed so triggers inside the
 * mixer don't start a marquee selection), which also blocks the native
 * mousedown from reaching the document listener that other open dropdowns
 * rely on for outside-click detection.
 */
const openDropdownCloseCallbacks = new Set<() => void>();

/**
 * A button that toggles a floating panel of related controls. Used by
 * the toolbar's grouped menus ("Load", "Transcribe") and the per-row
 * overflow menus on audio tracks. Closes on outside click or Escape.
 * `children` is a render prop receiving a `close` callback so menu items
 * that complete an action can dismiss the panel while sticky controls
 * (option checkboxes) can leave it open.
 *
 * Wrapped in `observer` so observable reads inside the children render
 * prop are tracked against THIS component's reactive context. Without
 * that, an enclosing observer only sees the closure being created; it
 * never dereferences the observable properties itself; so MobX has no
 * subscriber when those properties change. The store mutation lands but
 * any controlled inputs inside the panel stay stale until some
 * unrelated re-render rebuilds the closure.
 */
export const DropdownButton = observer(
  ({
    label,
    title,
    className,
    panelClassName,
    onOpen,
    children,
  }: {
    label: React.ReactNode;
    title?: string;
    className?: string;
    panelClassName?: string;
    /** Called once each time the panel transitions from closed to open.
     *  Used by callers that need to refresh data on open without forcing
     *  a parent re-render. */
    onOpen?: () => void;
    children: (close: () => void) => React.ReactNode;
  }) => {
    const [open, setOpen] = React.useState(false);
    const [anchor, setAnchor] = React.useState<{ top: number; left: number } | null>(null);
    const wrapperRef = React.useRef<HTMLDivElement>(null);
    const triggerRef = React.useRef<HTMLButtonElement>(null);
    const panelRef = React.useRef<HTMLDivElement>(null);
    const onOpenRef = React.useRef(onOpen);
    onOpenRef.current = onOpen;

    React.useEffect(() => {
      if (!open) return;
      const myClose = () => setOpen(false);
      // Snapshot before iterating: each close() schedules a state update
      // whose cleanup will mutate the set.
      const others = Array.from(openDropdownCloseCallbacks);
      openDropdownCloseCallbacks.clear();
      others.forEach((close) => close());
      openDropdownCloseCallbacks.add(myClose);

      onOpenRef.current?.();
      // Anchor the portaled panel to the trigger's viewport position and
      // keep it pinned as ancestors scroll / the window resizes.
      const reposition = () => {
        const rect = triggerRef.current?.getBoundingClientRect();
        if (!rect) return;
        setAnchor({ top: rect.bottom + 6, left: rect.left });
      };
      reposition();
      const onPointerDown = (e: MouseEvent) => {
        const target = e.target as Node;
        if (wrapperRef.current?.contains(target)) return;
        if (panelRef.current?.contains(target)) return;
        setOpen(false);
      };
      const onKey = (e: KeyboardEvent) => {
        if (e.key === 'Escape') setOpen(false);
      };
      document.addEventListener('mousedown', onPointerDown);
      document.addEventListener('keydown', onKey);
      // Capture phase so we catch scrolls from any ancestor scroller, not
      // just window.
      window.addEventListener('scroll', reposition, true);
      window.addEventListener('resize', reposition);
      return () => {
        openDropdownCloseCallbacks.delete(myClose);
        document.removeEventListener('mousedown', onPointerDown);
        document.removeEventListener('keydown', onKey);
        window.removeEventListener('scroll', reposition, true);
        window.removeEventListener('resize', reposition);
      };
    }, [open]);

    return (
      <div className={styles.dropdown} ref={wrapperRef}>
        <button
          ref={triggerRef}
          type="button"
          className={className}
          title={title}
          aria-haspopup="menu"
          aria-expanded={open}
          // Stop propagation so a dropdown trigger placed inside a
          // marquee-listening container (the mixer) doesn't kick off a
          // selection drag on every click. Toolbar triggers live outside
          // such containers, so this is a no-op there.
          onMouseDown={(e) => e.stopPropagation()}
          onClick={(e) => {
            e.stopPropagation();
            setOpen((o) => !o);
          }}
        >
          {label}
        </button>
        {open &&
          anchor &&
          createPortal(
            <div
              ref={panelRef}
              className={classNames(styles.dropdownPanel, panelClassName)}
              role="menu"
              style={{ position: 'fixed', top: anchor.top, left: anchor.left }}
            >
              {children(() => setOpen(false))}
            </div>,
            document.body
          )}
      </div>
    );
  }
);

/**
 * A nested menu item inside a {@link DropdownButton} panel. Renders as a
 * regular dropdown row with a trailing ▸; clicking toggles a fly-out
 * panel anchored to its right edge. Outside-click + Escape are handled
 * by the enclosing DropdownButton, so we only need local open/close
 * state.
 */
export const SubmenuItem = ({
  label,
  children,
}: {
  label: React.ReactNode;
  children: (close: () => void) => React.ReactNode;
}) => {
  const [open, setOpen] = React.useState(false);
  return (
    <div className={styles.submenu}>
      <button
        type="button"
        className={classNames(styles.dropdownItem, styles.submenuTrigger)}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span>{label}</span>
        <span aria-hidden="true" className={styles.submenuArrow}>
          ▸
        </span>
      </button>
      {open && (
        <div className={styles.submenuPanel} role="menu">
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
};

/**
 * Section within a {@link DropdownButton} panel: a subtle uppercase
 * heading followed by its child items and a thin divider to separate it
 * from the next section. The trailing divider on the last section is
 * suppressed via a `.dropdownPanel > .dropdownDivider:last-child` CSS
 * rule so callers don't need to special-case the final section.
 */
export const DropdownSection = ({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) => (
  <>
    <span className={styles.dropdownSectionHeading}>{label}</span>
    {children}
    <span className={styles.dropdownDivider} aria-hidden="true" />
  </>
);

/**
 * Dropdown menu row with a leading tick (or blank gutter) so the user
 * sees the toggle's current state at a glance. Acts like a regular
 * `.dropdownItem` (hover background, focus ring) but the panel stays
 * open across clicks; the row's purpose is to flip the toggle, not
 * dismiss the menu.
 */
export const ToggleMenuItem = ({
  label,
  active,
  onToggle,
  title,
  disabled,
}: {
  label: React.ReactNode;
  active: boolean;
  onToggle: () => void;
  title?: string;
  disabled?: boolean;
}) => (
  <button
    type="button"
    className={classNames(styles.dropdownItem, styles.toggleMenuItem)}
    role="menuitemcheckbox"
    aria-checked={active}
    onClick={onToggle}
    disabled={disabled}
    title={title}
  >
    <span className={styles.toggleMenuTick} aria-hidden="true">
      {active ? '✓' : ''}
    </span>
    <span>{label}</span>
  </button>
);
