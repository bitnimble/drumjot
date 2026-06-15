import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { createPortal } from 'react-dom';
import { perfProbe } from 'src/utils/perf_probe';
import { ViewportStoreContext } from '../viewport/viewport_contexts';

/**
 * Selection popover that escapes the `.jotContainer { overflow: hidden }`
 * clip by rendering into `document.body` via `createPortal`. Without
 * the portal the popover is clipped at the score's bottom edge and
 * stops short of the minimap / playback bar below; with it the popover
 * can extend over any sibling chrome (subject only to the window's own
 * edge).
 *
 * Position is computed at render time from the anchor's
 * `getBoundingClientRect()`, applied as inline `position: fixed`
 * coords. Re-renders when the score scrolls or zooms, the wrapping
 * `observer(...)` HOC subscribes to `store.scrollX` / `store.scrollY`
 * / `store.zoom`, so MobX re-fires whenever the anchor's screen
 * position changes under the (transform-driven) scroll. The bar's
 * `bounding rect` returns the post-transform viewport coordinates, so
 * one read per render is enough; no per-frame imperative updates.
 *
 * Above-flip is reused from the previous in-DOM implementation, with
 * the only change being the bottom limit, now the window edge rather
 * than the score-scroller bottom. The popover can extend through the
 * minimap / playback area, so we flip only when it would overrun the
 * window itself.
 *
 * Reading `getBoundingClientRect` at render time is the popover-
 * anchoring exception called out in AGENTS.md §5.9: a single rect
 * read per popover-open re-render, not a per-frame layout loop.
 */
type PopoverPortalProps = {
  anchorRef: React.RefObject<HTMLElement>;
  show: boolean;
  className: string;
  /** Class added on top of `className` when the popover flipped above
   *  the anchor. Optional: positioning + transform are handled inline,
   *  so consumers only pass a flipped class when there's *visual*
   *  chrome that differs (e.g. a tail pointing the other way). */
  flippedClassName?: string;
  children: React.ReactNode;
  /** Extra `<div>` props applied to the portaled wrapper (refs, mouse
   *  handlers, etc.). The wrapper's `ref` is reserved for internal
   *  measurement; consumers that need a label ref should pass it
   *  through this prop. */
  extraProps?: React.HTMLAttributes<HTMLDivElement> & {
    ref?: React.Ref<HTMLDivElement>;
  };
};

/**
 * Hidden-state gate. There is one PopoverPortal per note (and per
 * filtered-onset ghost), so on a large score the tree holds thousands of
 * them, but at most one is ever `show`n at a time (the selected/hovered
 * note's label). This wrapper reads NO observables and runs NO hooks when
 * hidden, it just returns `null`, so a zoom / scroll tick (which mutates
 * `store.zoom` / `store.scrollX`) does not wake one observer per note. The
 * subscribing logic lives in {@link PopoverPortalShown}, which only mounts
 * for the popover that's actually open. Re-rendered by its parent (NoteView
 * etc.) when `show` flips, so it doesn't need to be an observer itself.
 *
 * This split is load-bearing for zoom performance: before it, every hidden
 * popover subscribed to `store.zoom` and re-rendered on every wheel tick,
 * turning a zoom into a multi-thousand-node synchronous reconciliation.
 */
export function PopoverPortal(props: PopoverPortalProps) {
  if (!props.show) return null;
  return <PopoverPortalShown {...props} />;
}

const PopoverPortalShown = observer(function PopoverPortalShown({
  anchorRef,
  className,
  flippedClassName,
  children,
  extraProps,
}: PopoverPortalProps) {
  perfProbe('PopoverPortal');
  const viewport = React.useContext(ViewportStoreContext);
  // Read these for MobX reactivity even though we don't use the values
  // directly, the bounding-rect read in the render below picks up the
  // new post-transform position whenever the score scrolls or zooms.
  // Only the open popover is mounted, so this is one subscription, not one
  // per note (see {@link PopoverPortal}).
  void viewport?.scrollX;
  void viewport?.scrollY;
  void viewport?.zoom;

  const labelRef = React.useRef<HTMLDivElement | null>(null);
  const [flip, setFlip] = React.useState(false);

  React.useLayoutEffect(() => {
    const anchor = anchorRef.current;
    const label = labelRef.current;
    if (!anchor || !label) return;
    const aRect = anchor.getBoundingClientRect();
    const lRect = label.getBoundingClientRect();
    const SAFE = 8;
    const GAP = 16;
    // Window bounds, the popover is portaled to `document.body` and
    // sits above every app-shell sibling, so the only edge it can't
    // cross is the viewport itself.
    const overflowsBelow = aRect.bottom + GAP + lRect.height > window.innerHeight - SAFE;
    const fitsAbove = aRect.top - GAP - lRect.height > SAFE;
    setFlip(overflowsBelow && fitsAbove);
  }, [anchorRef, viewport?.scrollX, viewport?.scrollY, viewport?.zoom]);

  const anchor = anchorRef.current;
  if (!anchor) return null;
  const aRect = anchor.getBoundingClientRect();
  const GAP = 16;
  const top = flip ? aRect.top - GAP : aRect.bottom + GAP;
  const left = aRect.left + aRect.width / 2;
  const { ref: forwardedRef, style: extraStyle, ...restProps } = extraProps ?? {};
  // Merge consumer ref + our own measurement ref so the layout effect
  // can size against the same node the consumer holds onto.
  const setRef = (node: HTMLDivElement | null) => {
    labelRef.current = node;
    if (typeof forwardedRef === 'function') forwardedRef(node);
    else if (forwardedRef && typeof forwardedRef === 'object') {
      (forwardedRef as React.MutableRefObject<HTMLDivElement | null>).current = node;
    }
  };
  return createPortal(
    <div
      {...restProps}
      ref={setRef}
      className={classNames(className, flip && flippedClassName)}
      data-popover="note-label"
      style={{
        position: 'fixed',
        top,
        left,
        transform: flip ? 'translate(-50%, -100%)' : 'translateX(-50%)',
        margin: 0,
        zIndex: 1100,
        ...extraStyle,
      }}
    >
      {children}
    </div>,
    document.body,
  );
});
