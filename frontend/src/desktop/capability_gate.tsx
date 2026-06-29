import { observer } from 'mobx-react-lite';
import modal from 'src/ui/modal/modal.module.css';
import { capabilityById, formatBytes } from './capability_manifest';
import { desktopCapabilities } from './desktop_services';
import styles from './capability_panel.module.css';

/**
 * Point-of-use install prompt. Shows when something (e.g. a transcribe) needs a
 * capability that isn't installed yet (`store.pendingGate`): confirm to download
 * + install with live progress, or cancel. Desktop-only; renders nothing in the
 * web build.
 */
export const CapabilityGate = observer(function CapabilityGate() {
  const deps = desktopCapabilities();
  if (deps == null) return null;
  const { store, presenter } = deps;
  const id = store.pendingGate;
  if (id == null) return null;

  const cap = capabilityById(id);
  const installing = store.statusOf(id) === 'installing';
  const error = store.errors.get(id);
  const size = formatBytes(presenter.incrementalBytes([id]));

  return (
    <div className={modal.backdrop}>
      <div className={styles.panel}>
        <div className={modal.header}>
          <h2 className={modal.title}>Install {cap.name}?</h2>
        </div>
        <div className={modal.body}>
          <p className={styles.desc}>{cap.description}</p>
          {installing ? (
            <p className={styles.status}>Installing… {store.installLog.get(id) ?? ''}</p>
          ) : error ? (
            <p className={styles.status}>Install failed: {error}</p>
          ) : (
            <p className={styles.intro}>Downloads about {size} once, then runs offline.</p>
          )}
        </div>
        <div className={modal.footer}>
          <button className={styles.skip} disabled={installing} onClick={() => presenter.cancelGate()}>
            Cancel
          </button>
          <button
            className={styles.install}
            disabled={installing}
            onClick={() => void presenter.confirmGate()}
          >
            {error ? 'Retry' : `Download · ${size}`}
          </button>
        </div>
      </div>
    </div>
  );
});
