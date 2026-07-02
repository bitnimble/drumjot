import classNames from 'classnames';
import { X } from 'lucide-react';
import React from 'react';
import { createPortal } from 'react-dom';
import { useModalManager } from './modal_manager';
import styles from './modal.module.css';

export { styles as modalStyles };

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

function focusablesIn(panel: HTMLElement): HTMLElement[] {
  return Array.from(panel.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (el) => el.offsetParent != null || el === document.activeElement
  );
}

/**
 * Low-level modal shell: a portaled backdrop + centred panel that owns the
 * cross-cutting behaviour every modal needs and no call site should
 * re-implement, top-of-stack portal (so it escapes the score's clipping /
 * z-index contexts), `role="dialog"` + `aria-modal`, backdrop-click and
 * Escape to close (Escape via the {@link ModalProvider} stack, so only the
 * topmost modal closes), a stable size, and focus management: on open, focus
 * moves into the panel (respecting an `autoFocus`ed control, else the first
 * focusable, else the panel itself); Tab/Shift+Tab are trapped to wrap within
 * the panel; and focus is restored to the previously-focused element on close.
 *
 * Content is composed from {@link ModalHeader}, {@link ModalBody}, and
 * {@link ModalFooter} (or anything else); the common confirm/cancel shape
 * is pre-assembled as {@link ConfirmModal}.
 */
export const Modal: React.FC<{
  open: boolean;
  onClose: () => void;
  /** Accessible name. Provide one of `ariaLabel` / `ariaLabelledBy`. */
  ariaLabel?: string;
  ariaLabelledBy?: string;
  /** Panel width in px (capped to the viewport) or any CSS width value. */
  width?: number | string;
  /** Cap the panel height to the viewport (for tall, scrolling bodies). */
  maxHeight?: boolean;
  panelClassName?: string;
  testId?: string;
  children: React.ReactNode;
}> = ({
  open,
  onClose,
  ariaLabel,
  ariaLabelledBy,
  width,
  maxHeight,
  panelClassName,
  testId,
  children,
}) => {
  const manager = useModalManager();
  const panelRef = React.useRef<HTMLDivElement>(null);

  // Register with the manager while open so Escape / closeActive can reach
  // this modal as the top of the stack. Keep the latest onClose in a ref so
  // re-renders don't churn the registration (which would reorder the stack).
  const onCloseRef = React.useRef(onClose);
  onCloseRef.current = onClose;
  React.useEffect(() => {
    if (!open || !manager) return;
    return manager.register(() => onCloseRef.current());
  }, [open, manager]);

  // Focus management: on open, pull focus into the panel and remember what
  // had it; on close (cleanup), hand focus back. An `autoFocus`ed control
  // (ConfirmModal's Cancel/confirm) has already grabbed focus by the time this
  // runs, so only reach for the first focusable when focus isn't in the panel.
  React.useEffect(() => {
    if (!open) return;
    const panel = panelRef.current;
    if (!panel) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    if (!panel.contains(document.activeElement)) {
      (focusablesIn(panel)[0] ?? panel).focus();
    }
    return () => previouslyFocused?.focus?.();
  }, [open]);

  if (!open) return null;

  const onPanelKeyDown = (e: React.KeyboardEvent<HTMLDivElement>): void => {
    if (e.key !== 'Tab') return;
    const panel = panelRef.current;
    if (!panel) return;
    const focusables = focusablesIn(panel);
    if (focusables.length === 0) {
      // Nothing tabbable: keep focus pinned on the panel rather than letting
      // Tab escape to the page behind the modal.
      e.preventDefault();
      panel.focus();
      return;
    }
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && (active === first || active === panel)) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && active === last) {
      e.preventDefault();
      first.focus();
    }
  };

  const panelStyle: React.CSSProperties = {};
  if (width !== undefined) {
    panelStyle.width = typeof width === 'number' ? `min(${width}px, 100%)` : width;
  }
  if (maxHeight) {
    panelStyle.maxHeight = 'calc(100dvh - 48px)';
  }

  return createPortal(
    <div
      className={styles.backdrop}
      role="dialog"
      aria-modal="true"
      aria-label={ariaLabel}
      aria-labelledby={ariaLabelledBy}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      data-testid={testId}
    >
      <div
        ref={panelRef}
        className={classNames(styles.panel, panelClassName)}
        style={panelStyle}
        tabIndex={-1}
        onKeyDown={onPanelKeyDown}
      >
        {children}
      </div>
    </div>,
    document.body
  );
};

/**
 * Standard modal header: an optional leading status icon, the title, and a
 * trailing close button. Pass `title` for the common case or `children` for
 * a bespoke title node.
 */
export const ModalHeader: React.FC<{
  title?: React.ReactNode;
  titleId?: string;
  /** Leading status glyph (e.g. a warning triangle), tinted + non-shrinking. */
  icon?: React.ReactNode;
  onClose: () => void;
  closeLabel?: string;
  closeTestId?: string;
  children?: React.ReactNode;
}> = ({ title, titleId, icon, onClose, closeLabel = 'Close', closeTestId, children }) => (
  <header className={styles.header}>
    {icon && (
      <span className={styles.headerIcon} aria-hidden="true">
        {icon}
      </span>
    )}
    <h3 className={styles.title} id={titleId}>
      {children ?? title}
    </h3>
    <button
      type="button"
      className={styles.close}
      onClick={onClose}
      aria-label={closeLabel}
      data-testid={closeTestId}
    >
      <X size={18} aria-hidden="true" />
    </button>
  </header>
);

export const ModalBody: React.FC<{
  className?: string;
  testId?: string;
  children: React.ReactNode;
}> = ({ className, testId, children }) => (
  <div className={classNames(styles.body, className)} data-testid={testId}>
    {children}
  </div>
);

export const ModalFooter: React.FC<{
  /** `'end'` right-aligns the actions (the confirm/cancel pair). */
  align?: 'start' | 'end';
  className?: string;
  testId?: string;
  children: React.ReactNode;
}> = ({ align = 'start', className, testId, children }) => (
  <footer
    className={classNames(align === 'end' ? styles.footerEnd : styles.footer, className)}
    data-testid={testId}
  >
    {children}
  </footer>
);

/**
 * The confirm/cancel modal: a title, a body (`children`), and a right-aligned
 * Cancel + confirm footer. Covers the "are you sure?" prompts (discard edits,
 * replace score) where the only variation is the copy, the confirm variant,
 * and which action takes initial focus.
 */
export const ConfirmModal: React.FC<{
  open: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  title: React.ReactNode;
  ariaLabel?: string;
  confirmLabel: React.ReactNode;
  cancelLabel?: React.ReactNode;
  /** Confirm button paint: `'danger'` for destructive choices. */
  confirmVariant?: 'primary' | 'danger';
  /** Which button receives initial focus (default the safe `cancel`). */
  autoFocus?: 'confirm' | 'cancel';
  width?: number;
  testId?: string;
  /** testid for the header X (defaults to none). */
  closeTestId?: string;
  /** testid for the footer Cancel button. */
  cancelTestId?: string;
  confirmTestId?: string;
  children: React.ReactNode;
}> = ({
  open,
  onConfirm,
  onCancel,
  title,
  ariaLabel,
  confirmLabel,
  cancelLabel = 'Cancel',
  confirmVariant = 'primary',
  autoFocus = 'cancel',
  width = 420,
  testId,
  closeTestId,
  cancelTestId,
  confirmTestId,
  children,
}) => (
  <Modal
    open={open}
    onClose={onCancel}
    ariaLabel={ariaLabel}
    width={width}
    testId={testId}
  >
    <ModalHeader
      title={title}
      onClose={onCancel}
      closeLabel="Cancel"
      closeTestId={closeTestId}
    />
    <ModalBody>{children}</ModalBody>
    <ModalFooter align="end">
      <button
        type="button"
        className={styles.secondaryButton}
        onClick={onCancel}
        autoFocus={autoFocus === 'cancel'}
        data-testid={cancelTestId}
      >
        {cancelLabel}
      </button>
      <button
        type="button"
        className={confirmVariant === 'danger' ? styles.dangerButton : styles.primaryButton}
        onClick={onConfirm}
        autoFocus={autoFocus === 'confirm'}
        data-testid={confirmTestId}
      >
        {confirmLabel}
      </button>
    </ModalFooter>
  </Modal>
);
