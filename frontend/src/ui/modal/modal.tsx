import classNames from 'classnames';
import { X } from 'lucide-react';
import React from 'react';
import { createPortal } from 'react-dom';
import { useModalManager } from './modal_manager';
import styles from './modal.module.css';

export { styles as modalStyles };

/**
 * Low-level modal shell: a portaled backdrop + centred panel that owns the
 * cross-cutting behaviour every modal needs and no call site should
 * re-implement, top-of-stack portal (so it escapes the score's clipping /
 * z-index contexts), `role="dialog"` + `aria-modal`, backdrop-click and
 * Escape to close (Escape via the {@link ModalProvider} stack, so only the
 * topmost modal closes), and a stable size.
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

  // Register with the manager while open so Escape / closeActive can reach
  // this modal as the top of the stack. Keep the latest onClose in a ref so
  // re-renders don't churn the registration (which would reorder the stack).
  const onCloseRef = React.useRef(onClose);
  onCloseRef.current = onClose;
  React.useEffect(() => {
    if (!open || !manager) return;
    return manager.register(() => onCloseRef.current());
  }, [open, manager]);

  if (!open) return null;

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
      <div className={classNames(styles.panel, panelClassName)} style={panelStyle}>
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
