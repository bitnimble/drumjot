import { observer } from 'mobx-react-lite';
import React from 'react';
import { Modal, ModalBody, ModalFooter, ModalHeader, modalStyles } from 'src/ui/modal/modal';
import { formatBytes } from './capability_manifest';
import { CapabilityTree, useCapabilityInstall } from './capability_install';
import styles from './capability_panel.module.css';

/** Persisted (per-install) flag so the setup modal shows only on first launch. */
const SETUP_SEEN_KEY = 'drumjot.setupSeen';

function setupSeen(): boolean {
  try {
    return localStorage.getItem(SETUP_SEEN_KEY) != null;
  } catch {
    return false;
  }
}

/**
 * First-run capability setup, shown once in the desktop shell: the shared
 * capability picker in a one-off modal with a cumulative-size footer + Skip.
 * Persists a "seen" flag so it doesn't reappear on later launches; the same
 * picker lives in Settings → Capabilities. Renders nothing in the web build.
 */
export const DesktopFirstRun = observer(function DesktopFirstRun() {
  const [open, setOpen] = React.useState(() => !setupSeen());
  const install = useCapabilityInstall();
  const dismiss = (): void => {
    try {
      localStorage.setItem(SETUP_SEEN_KEY, '1');
    } catch {
      // private mode / storage disabled, fall back to in-session dismissal.
    }
    setOpen(false);
  };
  if (!install.available || !open) return null;

  return (
    <Modal
      open
      onClose={dismiss}
      ariaLabel="Set up Drumjot"
      width={560}
      maxHeight
      testId="desktop-first-run"
    >
      <ModalHeader title="Set up Drumjot" onClose={dismiss} closeLabel="Skip setup" />
      <ModalBody>
        <p className={styles.intro}>
          Drumjot is ready for writing and editing right now. Optional features download what they
          need; pick any to install now, or skip and install later from Settings.
        </p>
        <CapabilityTree controller={install} />
      </ModalBody>
      <ModalFooter>
        <span className={styles.total}>
          {install.enoughSpace === false
            ? 'Not enough disk space for this selection'
            : install.totalBytes > 0
              ? `Total: ${formatBytes(install.totalBytes)}`
              : 'Nothing selected'}
        </span>
        <button type="button" className={modalStyles.secondaryButton} onClick={dismiss}>
          Skip for now
        </button>
        <button
          type="button"
          className={modalStyles.primaryButton}
          disabled={install.totalBytes === 0 || install.installing || install.enoughSpace === false}
          onClick={install.install}
        >
          {install.installing ? 'Installing…' : 'Install'}
        </button>
      </ModalFooter>
    </Modal>
  );
});
