import React from 'react';

/**
 * Centralised modal coordination. Every open {@link Modal} registers a
 * "close me" callback here, newest on top, so the app has one place that
 * knows the modal stack. That buys three things no per-modal `useState`
 * could on its own:
 *
 *  - a single Escape handler that closes the *topmost* modal (and stops
 *    the keystroke before it reaches editor shortcuts / paste-cancel),
 *  - {@link ModalManager.closeActive} for "dismiss whatever's open" from
 *    anywhere (a global command, a route change), and
 *  - {@link useModalState}, so simple modals drive their open/close
 *    through the manager instead of each host threading its own boolean.
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
  /** Whether the named declarative modal is open. */
  isOpen: (id: string) => boolean;
  /** Open / close / toggle a declarative modal by id. */
  open: (id: string) => void;
  close: (id: string) => void;
  toggle: (id: string) => void;
};

const ModalManagerContext = React.createContext<ModalManager | null>(null);

/**
 * Hosts the modal stack + the declarative open-state map and installs the
 * single document-level Escape handler. Wrap the app (or the editor) once;
 * all {@link Modal}s below it coordinate through it.
 */
export const ModalProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Stack of close callbacks, newest last. A ref (not state) because the
  // Escape handler only ever reads the top; registering/unregistering must
  // not re-render every modal.
  const stackRef = React.useRef<Array<() => void>>([]);
  const [openIds, setOpenIds] = React.useState<ReadonlySet<string>>(() => new Set());

  const manager = React.useMemo<ModalManager>(() => {
    const closeActive = () => {
      const top = stackRef.current[stackRef.current.length - 1];
      if (!top) return false;
      top();
      return true;
    };
    return {
      register: (close) => {
        stackRef.current.push(close);
        return () => {
          const i = stackRef.current.lastIndexOf(close);
          if (i !== -1) stackRef.current.splice(i, 1);
        };
      },
      closeActive,
      isOpen: (id) => openIds.has(id),
      open: (id) =>
        setOpenIds((prev) => {
          if (prev.has(id)) return prev;
          const next = new Set(prev);
          next.add(id);
          return next;
        }),
      close: (id) =>
        setOpenIds((prev) => {
          if (!prev.has(id)) return prev;
          const next = new Set(prev);
          next.delete(id);
          return next;
        }),
      toggle: (id) =>
        setOpenIds((prev) => {
          const next = new Set(prev);
          if (next.has(id)) next.delete(id);
          else next.add(id);
          return next;
        }),
    };
  }, [openIds]);

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

/**
 * Drive a simple modal's open/close through the manager rather than a
 * host-local `useState`. Returns the live boolean plus open/close/toggle
 * actions bound to `id`. For modals whose visibility already lives on a
 * domain store (lyrics, transcribe), keep that store and pass its boolean
 * to {@link Modal} directly instead of using this hook.
 */
export function useModalState(id: string): {
  open: boolean;
  openModal: () => void;
  closeModal: () => void;
  toggleModal: () => void;
} {
  const manager = useModalManager();
  if (!manager) {
    throw new Error('useModalState must be used within a <ModalProvider>');
  }
  return {
    open: manager.isOpen(id),
    openModal: () => manager.open(id),
    closeModal: () => manager.close(id),
    toggleModal: () => manager.toggle(id),
  };
}
