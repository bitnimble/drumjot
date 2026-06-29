import { observer } from 'mobx-react-lite';
import React from 'react';
import modal from 'src/ui/modal/modal.module.css';
import { CapabilityList } from './capability_list';
import { desktopCapabilities } from './desktop_services';
import styles from './capability_panel.module.css';

/**
 * First-run capability setup, shown once in the desktop shell: the shared
 * {@link CapabilityList} in a one-off modal with a Skip. The same list lives in
 * Settings → Capabilities. Renders nothing in the web build.
 */
export const DesktopFirstRun = observer(function DesktopFirstRun() {
  const [open, setOpen] = React.useState(true);
  if (desktopCapabilities() == null || !open) return null;

  return (
    <div className={modal.backdrop}>
      <div className={styles.panel}>
        <div className={modal.header}>
          <h2 className={modal.title}>Set up Drumjot</h2>
        </div>
        <div className={modal.body}>
          <p className={styles.intro}>
            Drumjot is ready for writing and editing right now. Optional features download what
            they need the first time you use them.
          </p>
          <CapabilityList />
        </div>
        <div className={modal.footer}>
          <button className={styles.skip} onClick={() => setOpen(false)}>
            Skip for now
          </button>
        </div>
      </div>
    </div>
  );
});
