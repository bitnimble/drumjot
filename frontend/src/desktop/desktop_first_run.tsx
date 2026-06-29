import { observer } from 'mobx-react-lite';
import React from 'react';
import modal from 'src/ui/modal/modal.module.css';
import { CAPABILITIES, formatBytes } from './capability_manifest';
import { desktopCapabilities } from './desktop_services';
import styles from './capability_panel.module.css';

/**
 * First-run capability setup, shown once in the desktop shell. Lists the
 * optional capabilities with what each does + its incremental download, and
 * lets the user install some or skip straight into a fully working editor.
 * Shares one capability store with the point-of-use gate (desktopCapabilities),
 * so an install here or there is reflected everywhere. Renders nothing in the
 * web build.
 */
export const DesktopFirstRun = observer(function DesktopFirstRun() {
  const deps = desktopCapabilities();
  const [open, setOpen] = React.useState(true);
  if (deps == null || !open) return null;
  const { store, presenter } = deps;

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
          {CAPABILITIES.map((cap) => {
            const status = store.statusOf(cap.id);
            return (
              <div key={cap.id} className={styles.row}>
                <div className={styles.info}>
                  <span className={styles.name}>{cap.name}</span>
                  <span className={styles.desc}>{cap.description}</span>
                </div>
                <div className={styles.action}>
                  {status === 'ready' ? (
                    <span className={styles.ready}>Installed</span>
                  ) : status === 'installing' ? (
                    <span className={styles.status}>Installing…</span>
                  ) : cap.kind === 'credentials' ? (
                    <span className={styles.status}>Needs API key</span>
                  ) : (
                    <button className={styles.install} onClick={() => void presenter.install(cap.id)}>
                      Install · {formatBytes(presenter.incrementalBytes([cap.id]))}
                    </button>
                  )}
                </div>
              </div>
            );
          })}
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
