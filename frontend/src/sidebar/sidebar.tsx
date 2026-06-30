import classNames from 'classnames';
import { Pin, PinOff } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { SidebarStoreContext, SidebarPresenterContext } from './sidebar_contexts';
import { SIDEBAR_PANELS } from './panels/sidebar_panels';
import type { SidebarPresenter } from './sidebar_presenter';
import styles from './sidebar.module.css';

/**
 * Collapsible right-hand sidebar, an absolute overlay anchored to the right
 * edge of the score region (so it sits below the toolbar and above the minimap
 * / transport, never spanning the window chrome). A persistent vertical rail of
 * icon buttons sits at the far right; selecting an item opens the panel to its
 * left. The rail's top holds the float/pin button, then a divider, then one
 * button per registered panel (see {@link SIDEBAR_PANELS}).
 *
 * The panel renders in the same place at the same width whether **floating** or
 * **pinned**, so toggling pin never moves or resizes it. The only difference is
 * what the score does underneath: floating overlays it (the score keeps full
 * width, virtualization still renders the bars beneath the panel); pinning
 * reserves the panel's width in the score region's right padding (jot_editor.tsx
 * sets `data-sidebar-pinned` on `.scoreRegion`, which the CSS keys the padding
 * off; an attribute rather than a `:has()` on the panel, since `:has()` over the
 * score subtree forced a style recalc on every note mutation), so the score
 * narrows and its scroll virtualization stops rendering the bars under the dock.
 * A floating panel is dismissed by an outside click or Escape; a pinned one
 * stays until re-toggled.
 */
export const Sidebar = observer(function Sidebar() {
  const store = React.useContext(SidebarStoreContext);
  const presenter = React.useContext(SidebarPresenterContext);
  // Outside-click / Escape dismissal, active only while a panel is floating.
  useFloatingDismiss(!!store?.expanded && !store?.pinned, presenter ?? null);
  if (!store || !presenter) return null;
  const { expanded, activePanel, pinned } = store;
  const active = SIDEBAR_PANELS.find((p) => p.id === activePanel);

  return (
    <aside className={styles.sidebar} data-testid="sidebar" data-expanded={expanded || undefined}>
      {expanded && active && (
        <div
          className={styles.panel}
          data-testid="sidebar-panel"
          data-sidebar-mode={pinned ? 'pinned' : 'floating'}
          data-sidebar-float
        >
          {active.render()}
        </div>
      )}
      <div className={styles.rail}>
        <button
          type="button"
          className={classNames(styles.railButton, expanded && pinned && styles.railButtonActive)}
          onClick={() => presenter.togglePin()}
          aria-label={expanded && pinned ? 'Unpin panel' : 'Pin panel'}
          aria-pressed={expanded && pinned}
          title={
            expanded && pinned
              ? 'Unpin panel (float over the score)'
              : 'Pin panel (dock beside the score)'
          }
          data-testid="sidebar-pin-toggle"
        >
          {expanded && pinned ? <PinOff size={18} /> : <Pin size={18} />}
        </button>
        <div className={styles.railDivider} role="separator" aria-orientation="horizontal" />
        {SIDEBAR_PANELS.map((panel) => {
          const isActive = expanded && activePanel === panel.id;
          const Icon = panel.Icon;
          return (
            <button
              key={panel.id}
              type="button"
              className={classNames(styles.railButton, isActive && styles.railButtonActive)}
              onClick={() => presenter.selectPanel(panel.id)}
              aria-label={panel.label}
              aria-pressed={isActive}
              title={panel.label}
              data-testid={`sidebar-item-${panel.id}`}
            >
              <Icon size={18} />
            </button>
          );
        })}
      </div>
    </aside>
  );
});

/** Movement (px) past which an empty-score press is a marquee drag, not a
 *  dismissing click. Mirrors the tap/drag threshold used by the score's own
 *  touch handling. */
const MARQUEE_DRAG_PX = 8;

/**
 * Dismiss a floating sidebar panel on an outside interaction. While `active`,
 * a document-level pointer/key listener collapses the panel (via
 * `presenter.collapse()`) when the user clicks away, with deliberate
 * exceptions so the panel survives the interactions that feed it:
 *
 * - clicks inside the panel or on the rail are ignored;
 * - clicking a note (which drives the Note properties panel) keeps it open;
 * - a marquee selection drag on empty score keeps it open, but a plain click
 *   on empty score (which clears the selection) dismisses it;
 * - a click anywhere else (toolbar, minimap, transport, page chrome) dismisses;
 * - Escape dismisses.
 *
 * Pinned panels never mount this (they stay until re-toggled).
 */
function useFloatingDismiss(active: boolean, presenter: SidebarPresenter | null): void {
  const presenterRef = React.useRef(presenter);
  presenterRef.current = presenter;
  React.useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    const { signal } = controller;
    const dismiss = () => presenterRef.current?.collapse();

    const onMouseDown = (e: MouseEvent) => {
      const target = e.target instanceof Element ? e.target : null;
      if (!target) return;
      // Interactions that keep the floating panel open.
      if (target.closest('[data-sidebar-float]')) return; // inside the panel body
      if (target.closest('[data-testid="sidebar"]')) return; // the rail (buttons own their state)
      if (target.closest('[data-note-id]')) return; // selecting a note
      // Transient popups the panel spawns (the Layers ⋯ menu, a colour picker)
      // portal to <body>, so they read as "outside" the panel even though they
      // belong to it. Exclude the standard popup ARIA roles so interacting with
      // them doesn't dismiss the panel underneath.
      if (target.closest('[role="menu"], [role="dialog"], [role="listbox"]')) return;
      if (target.closest('[data-jot-scroller]')) {
        // Empty score: a plain click clears the selection (dismiss), a drag is
        // a marquee selection (keep). Can't tell yet, so resolve on release.
        watchEmptyScorePress(e, signal, dismiss);
        return;
      }
      // Anywhere else (toolbar, minimap, transport, page chrome).
      dismiss();
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') dismiss();
    };
    document.addEventListener('mousedown', onMouseDown, { capture: true, signal });
    document.addEventListener('keydown', onKeyDown, { signal });
    return () => controller.abort();
  }, [active]);
}

/** Resolve an empty-score press into dismiss (plain click) vs keep (marquee
 *  drag) once it ends. Watches movement until release; a drag past
 *  {@link MARQUEE_DRAG_PX} cancels the dismissal. */
function watchEmptyScorePress(
  downEvent: MouseEvent,
  outerSignal: AbortSignal,
  dismiss: () => void
): void {
  const startX = downEvent.clientX;
  const startY = downEvent.clientY;
  const inner = new AbortController();
  const stop = () => inner.abort();
  outerSignal.addEventListener('abort', stop, { once: true });
  const onMove = (e: MouseEvent) => {
    if (Math.hypot(e.clientX - startX, e.clientY - startY) > MARQUEE_DRAG_PX) stop(); // marquee: keep open
  };
  const onUp = () => {
    stop();
    dismiss(); // released without dragging ⇒ plain click ⇒ dismiss
  };
  document.addEventListener('mousemove', onMove, { signal: inner.signal });
  document.addEventListener('mouseup', onUp, { signal: inner.signal });
}
