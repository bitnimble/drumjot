import classNames from 'classnames';
import { Check, ChevronRight } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { createPortal } from 'react-dom';
import { useMenuKeyboard } from 'src/ui/menu_keyboard';
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
 * A registry of "close me" callbacks for the submenus opened within a
 * single panel. When any submenu inside the panel opens, it registers
 * with the *nearest* registry (provided via {@link SubmenuRegistryContext})
 * so siblings under the same parent get closed; but ancestor submenus
 * (which provide their own registry to their children) are NOT closed,
 * because they live one level up in the React tree.
 *
 * Previously this was a single module-level Set, which couldn't tell a
 * sibling from an ancestor: opening Examples inside Load closed Load
 * itself, which then unmounted the Examples panel before its click
 * could resolve.
 */
export type SubmenuRegistry = {
  /** Register a "close me" callback. Closes every other already-
   *  registered submenu (sibling-exclusivity). Returns an unregister
   *  fn the consumer should call when its submenu closes. */
  register: (close: () => void) => () => void;
};

function createSubmenuRegistry(): SubmenuRegistry {
  const closers = new Set<() => void>();
  return {
    register: (close) => {
      // Close any currently-registered sibling. Snapshot first because
      // each `close()` call ultimately removes itself from the set on
      // its own cleanup, which would mutate the iterator otherwise.
      const others = Array.from(closers);
      closers.clear();
      others.forEach((c) => c());
      closers.add(close);
      return () => {
        closers.delete(close);
      };
    },
  };
}

export const SubmenuRegistryContext =
  React.createContext<SubmenuRegistry | null>(null);

/**
 * Hook a self-managed open/close submenu into the parent panel's
 * registry so it gets closed when a sibling opens. Safe to call
 * unconditionally; if there's no parent registry (e.g. a submenu
 * mounted outside any DropdownButton, in a test harness), the hook
 * is a no-op.
 *
 * Shared by the generic {@link SubmenuItem} and bespoke flyouts like
 * `RecentTranscriptionsPicker` so every submenu (generic or custom)
 * participates in sibling-exclusivity the same way.
 */
export function useParentSubmenuRegistry(
  open: boolean,
  setOpen: (open: boolean) => void,
): void {
  const parent = React.useContext(SubmenuRegistryContext);
  React.useEffect(() => {
    if (!open || !parent) return;
    return parent.register(() => setOpen(false));
  }, [open, parent, setOpen]);
}

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
    // Top-level registry for any submenu opened directly inside this
    // panel. Stable across renders so children don't churn their effect
    // hooks; the registry's own state lives in its closure-bound Set.
    const submenuRegistry = React.useMemo(createSubmenuRegistry, []);
    // Arrow/Home/End roving focus over the panel's menu rows.
    const { onKeyDown: onPanelKeyDown } = useMenuKeyboard(panelRef);

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

    // Keep the panel within the viewport horizontally: a trigger near the right
    // edge (e.g. the Layers panel's ⋯) would otherwise open a left-anchored
    // panel that runs off-screen. Runs after layout so the panel's real width is
    // known; only shifts when it would overflow, so left-side dropdowns (the
    // toolbar menus) are untouched. Converges in one extra render (the clamp
    // equals itself on the next pass).
    React.useLayoutEffect(() => {
      if (!open || !anchor) return;
      const panel = panelRef.current;
      if (!panel) return;
      const margin = 8;
      const maxLeft = window.innerWidth - panel.offsetWidth - margin;
      const clamped = Math.max(margin, Math.min(anchor.left, maxLeft));
      if (clamped !== anchor.left) {
        setAnchor((a) => (a ? { ...a, left: clamped } : a));
      }
    }, [open, anchor]);

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
            <SubmenuRegistryContext.Provider value={submenuRegistry}>
              <div
                ref={panelRef}
                className={classNames(styles.dropdownPanel, panelClassName)}
                role="menu"
                style={{ position: 'fixed', top: anchor.top, left: anchor.left }}
                onKeyDown={onPanelKeyDown}
              >
                {children(() => setOpen(false))}
              </div>
            </SubmenuRegistryContext.Provider>,
            document.body
          )}
      </div>
    );
  }
);

/**
 * A nested menu item inside a {@link DropdownButton} panel. Renders as
 * a regular dropdown row with a trailing chevron; clicking toggles a
 * fly-out panel anchored to its right edge.
 *
 * Sibling-exclusivity: when this submenu opens, it registers with the
 * nearest {@link SubmenuRegistryContext}; opening it closes any other
 * submenu registered in the same panel, but NOT this submenu's
 * ancestors (they hold their own registry one level up). The submenu
 * also exposes its own registry to its children so the same rule
 * applies recursively.
 *
 * Outside-click + Escape are handled by the enclosing DropdownButton,
 * so we only need local open/close state.
 */
/** Hover dwell before a SubmenuItem opens automatically. Short enough
 *  that intentional hovers feel responsive, long enough that brushing
 *  past on the way to a sibling doesn't open the wrong panel. */
const SUBMENU_HOVER_OPEN_DELAY_MS = 180;

export const SubmenuItem = ({
  label,
  title,
  disabled,
  children,
}: {
  label: React.ReactNode;
  title?: string;
  disabled?: boolean;
  children: (close: () => void) => React.ReactNode;
}) => {
  const [open, setOpen] = React.useState(false);
  useParentSubmenuRegistry(open, setOpen);
  // Registry exposed to *this submenu's* children, so its own
  // descendants can be siblings-exclusive among themselves without
  // closing us.
  const childRegistry = React.useMemo(createSubmenuRegistry, []);
  // The fly-out runs its own copy of the roving-focus behaviour: the parent
  // panel's arrow-nav excludes descendant-submenu rows (see menu_keyboard.ts),
  // so each submenu handles arrow/Home/End over its own rows.
  const panelRef = React.useRef<HTMLDivElement>(null);
  const { onKeyDown: onPanelKeyDown } = useMenuKeyboard(panelRef);

  // Hover-to-open. Mouse-enter on the trigger arms a short timer; if
  // the pointer leaves (or the panel opens some other way) before it
  // fires, the timer is cancelled. Once opened, hover plays no further
  // role; closing is via click-outside / Escape / sibling open /
  // re-clicking the trigger, same as before.
  const hoverTimerRef = React.useRef<number | null>(null);
  const clearHoverTimer = () => {
    if (hoverTimerRef.current !== null) {
      window.clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
  };
  React.useEffect(() => clearHoverTimer, []);
  const handleMouseEnter = () => {
    if (disabled || open) return;
    clearHoverTimer();
    hoverTimerRef.current = window.setTimeout(() => {
      hoverTimerRef.current = null;
      setOpen(true);
    }, SUBMENU_HOVER_OPEN_DELAY_MS);
  };

  return (
    <div className={styles.submenu}>
      <button
        type="button"
        className={classNames(styles.dropdownItem, styles.submenuTrigger)}
        aria-haspopup="menu"
        aria-expanded={open}
        title={title}
        disabled={disabled}
        onClick={() => {
          clearHoverTimer();
          // Click only ever opens. Closing happens via outside-click,
          // Escape, a sibling submenu opening, or selecting an item;
          // re-clicking the trigger while open would be a disruptive
          // toggle when the user's mouse is already on its way to a
          // child item.
          setOpen(true);
        }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={clearHoverTimer}
        onFocus={handleMouseEnter}
        onBlur={clearHoverTimer}
      >
        <span>{label}</span>
        <ChevronRight
          size={14}
          aria-hidden="true"
          className={styles.submenuArrow}
        />
      </button>
      {open && !disabled && (
        <SubmenuRegistryContext.Provider value={childRegistry}>
          <div className={styles.submenuPanel} role="menu" ref={panelRef} onKeyDown={onPanelKeyDown}>
            {children(() => setOpen(false))}
          </div>
        </SubmenuRegistryContext.Provider>
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
 * A one-shot action row inside a {@link DropdownButton} panel: a label on the
 * left and (optionally) a keyboard-shortcut pill on the right. Unlike
 * {@link ToggleMenuItem} it carries no tick and reads as `menuitem`; the caller
 * typically dismisses the panel from `onClick` (via the render-prop `close`).
 * `disabled` greys it out but keeps it visible (an unavailable action still
 * shows where it lives + its shortcut).
 */
export const ActionMenuItem = ({
  label,
  onClick,
  disabled,
  shortcut,
  title,
  testId,
}: {
  label: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  /** Display text for the trailing shortcut pill (e.g. `Ctrl+Z`). Pull it from
   *  the keymap registry (see `shortcutForCommand`) so a rebind reflects here
   *  rather than hardcoding it. Omitted = no pill. */
  shortcut?: string;
  title?: string;
  testId?: string;
}) => (
  <button
    type="button"
    className={classNames(styles.dropdownItem, styles.actionMenuItem)}
    role="menuitem"
    onClick={onClick}
    disabled={disabled}
    title={title}
    data-testid={testId}
  >
    <span>{label}</span>
    {shortcut && (
      <span className={styles.menuItemShortcut} data-testid={testId ? `${testId}-shortcut` : undefined}>
        {shortcut}
      </span>
    )}
  </button>
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
  testId,
  role = 'menuitemcheckbox',
}: {
  label: React.ReactNode;
  active: boolean;
  onToggle: () => void;
  title?: string;
  disabled?: boolean;
  testId?: string;
  /** `menuitemradio` for a mutually-exclusive set (one tick at a time),
   *  otherwise the default independent checkbox. `aria-checked` is
   *  carried the same way for both. */
  role?: 'menuitemcheckbox' | 'menuitemradio';
}) => (
  <button
    type="button"
    className={classNames(styles.dropdownItem, styles.toggleMenuItem)}
    role={role}
    aria-checked={active}
    onClick={onToggle}
    disabled={disabled}
    title={title}
    data-testid={testId}
  >
    <span className={styles.toggleMenuTick} aria-hidden="true">
      {active && <Check size={12} aria-hidden="true" />}
    </span>
    <span>{label}</span>
  </button>
);
