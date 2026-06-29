import { observer } from 'mobx-react-lite';
import React from 'react';
import modal from 'src/ui/modal/modal.module.css';
import { CAPABILITIES } from './capability_manifest';
import { CapabilityPresenter } from './capability_presenter';
import { CapabilityStore } from './capability_store';
import { TauriBridge } from './desktop_bridge';
import { isTauri } from './is_tauri';
import styles from './capability_panel.module.css';

function formatBytes(bytes: number): string {
  if (bytes <= 0) return 'free';
  const gb = bytes / 1_000_000_000;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${Math.round(bytes / 1_000_000)} MB`;
}

/**
 * First-run capability setup, shown once in the desktop shell. Lists the
 * optional capabilities with what each does + its incremental download, and
 * lets the user install some or skip straight into a fully working editor.
 * Renders nothing in the web build (no Tauri), so it never touches the browser
 * app or the e2e suite.
 */
export const DesktopFirstRun = observer(function DesktopFirstRun() {
  const enabled = isTauri();
  const depsRef = React.useRef<{
    store: CapabilityStore;
    presenter: CapabilityPresenter;
  } | null>(null);
  if (enabled && depsRef.current == null) {
    const store = new CapabilityStore();
    depsRef.current = {
      store,
      presenter: new CapabilityPresenter({ store, bridge: new TauriBridge() }),
    };
  }
  const [open, setOpen] = React.useState(true);

  React.useEffect(() => {
    if (enabled) {
      void depsRef.current?.presenter.refresh();
    }
  }, [enabled]);

  if (!enabled || !open || depsRef.current == null) return null;
  const { store, presenter } = depsRef.current;

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
                    <button
                      className={styles.install}
                      onClick={() => void presenter.install(cap.id)}
                    >
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
