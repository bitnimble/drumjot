import React from 'react';

/**
 * Centralised modal coordination. Every open {@link Modal} registers a
 * "close me" callback here, newest on top, so the app has one place that
 * knows the modal stack. That buys:
 *
 *  - a single Escape handler that closes the *topmost* modal (and stops
 *    the keystroke before it reaches editor shortcuts / paste-cancel), and
 *  - {@link ModalManager.closeActive} for "dismiss whatever's open" from
 *    anywhere (a global command, a route change).
 *
 * This mirrors the registry pattern already used for dropdowns
 * (`dropdown.tsx`): open/close is transient UI state, so it lives in
 * React, not a persisted store. Modals that carry a payload (a drop plan,
 * a transcribe target) keep that payload in their own domain store and
 * still register here for stacking / Escape / global close.
 */
export type ModalManager = {
  /** Register a top-of-stack close callback; returns an unregister fn the
   *  Modal calls when it closes/unmounts. */
  register: (close: () => void) => () => void;
  /** Close the most-recently-opened modal, if any. Returns whether one
   *  was closed (so callers can decide whether the event was handled). */
  closeActive: () => boolean;
};

const ModalManagerContext = React.createContext<ModalManager | null>(null);

/**
 * Hosts the modal stack and installs the single document-level Escape
 * handler. Wrap the app (or the editor) once; all {@link Modal}s below it
 * coordinate through it.
 */
export const ModalProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Stack of close callbacks, newest last. A ref (not state) because the
  // Escape handler only ever reads the top; registering/unregistering must
  // not re-render every modal. The manager itself is therefore stable
  // (`useMemo([])`), so registration order can't churn.
  const stackRef = React.useRef<Array<() => void>>([]);

  const manager = React.useMemo<ModalManager>(
    () => ({
      register: (close) => {
        stackRef.current.push(close);
        return () => {
          const i = stackRef.current.lastIndexOf(close);
          if (i !== -1) stackRef.current.splice(i, 1);
        };
      },
      closeActive: () => {
        const top = stackRef.current[stackRef.current.length - 1];
        if (!top) return false;
        top();
        return true;
      },
    }),
    []
  );

  React.useEffect(() => {
    // Capture phase + stopPropagation so an open modal swallows Escape
    // before the editor keymap / in-flight paste-cancel sees it.
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (manager.closeActive()) {
        e.stopPropagation();
        e.preventDefault();
      }
    };
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [manager]);

  return <ModalManagerContext.Provider value={manager}>{children}</ModalManagerContext.Provider>;
};

/** Read the ambient modal manager. Null outside a {@link ModalProvider}
 *  (e.g. an isolated component test rendering a Modal on its own). */
export function useModalManager(): ModalManager | null {
  return React.useContext(ModalManagerContext);
}
