import { observer } from 'mobx-react-lite';
import React from 'react';
import { Modal, ModalBody, ModalFooter, ModalHeader, modalStyles } from 'src/ui/modal/modal';
import { formatBytes } from './capability_manifest';
import { CapabilityTree, useCapabilityInstall } from './capability_install';
import styles from './capability_panel.module.css';

/**
 * First-run capability setup, shown once in the desktop shell: the shared
 * capability picker in a one-off modal with a cumulative-size footer + Skip.
 * The same picker lives in Settings → Capabilities. Renders nothing in the web
 * build.
 */
export const DesktopFirstRun = observer(function DesktopFirstRun() {
  const [open, setOpen] = React.useState(true);
  const install = useCapabilityInstall();
  if (!install.available || !open) return null;

  return (
    <Modal
      open
      onClose={() => setOpen(false)}
      ariaLabel="Set up Drumjot"
      width={560}
      maxHeight
      testId="desktop-first-run"
    >
      <ModalHeader title="Set up Drumjot" onClose={() => setOpen(false)} closeLabel="Skip setup" />
      <ModalBody>
        <p className={styles.intro}>
          Drumjot is ready for writing and editing right now. Optional features download what they
          need; pick any to install now, or skip and install later from Settings.
        </p>
        <CapabilityTree controller={install} />
      </ModalBody>
      <ModalFooter>
        <span className={styles.total}>
          {install.totalBytes > 0 ? `Total: ${formatBytes(install.totalBytes)}` : 'Nothing selected'}
        </span>
        <button type="button" className={modalStyles.secondaryButton} onClick={() => setOpen(false)}>
          Skip for now
        </button>
        <button
          type="button"
          className={modalStyles.primaryButton}
          disabled={install.totalBytes === 0 || install.installing}
          onClick={install.install}
        >
          {install.installing ? 'Installing…' : 'Install'}
        </button>
      </ModalFooter>
    </Modal>
  );
});
