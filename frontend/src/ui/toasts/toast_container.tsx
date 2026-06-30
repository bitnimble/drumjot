import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import styles from './toasts.module.css';
import { toastStore } from './toasts';

/** Cap on simultaneously-visible toasts. Older toasts beyond this index
 *  stay in the store (still tracked, still auto-dismissing) but aren't
 *  painted; they surface as the visible ones drop off. */
const MAX_VISIBLE_TOASTS = 3;

/**
 * Bottom-right toast stack. Mount once at the top level of the app;
 * reads `toastStore` directly so dispatchers don't have to thread props
 * down through the toolbar / mixer / score subtrees.
 *
 * Click anywhere on a toast to dismiss. Success toasts also auto-dismiss
 * via a timer in the store (see `toasts.ts`); errors are sticky.
 */
export const ToastContainer = observer(() => {
  const visible = toastStore.toasts.slice(0, MAX_VISIBLE_TOASTS);
  if (visible.length === 0) return null;
  return (
    <div className={styles.toastContainer} aria-live="polite">
      {visible.map((toast) => (
        <div
          key={toast.id}
          className={classNames(
            styles.toast,
            toast.kind === 'success' && styles.toastSuccess,
            toast.kind === 'error' && styles.toastError,
            toast.kind === 'warning' && styles.toastWarning,
          )}
          role="status"
          title={toast.title}
          data-testid={toast.testId}
          onClick={(e) => {
            // The transcribe-done toast embeds a `[debug.zip]` <a>; let
            // its native click through so the download fires instead of
            // dismissing on the same click that started the download.
            const tag = (e.target as HTMLElement | null)?.tagName;
            if (tag === 'A') return;
            toastStore.dismiss(toast.id);
          }}
        >
          {toast.message}
        </div>
      ))}
    </div>
  );
});
