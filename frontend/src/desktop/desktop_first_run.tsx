import { observer } from 'mobx-react-lite';
import React from 'react';
import modal from 'src/ui/modal/modal.module.css';
import { CAPABILITIES, type CapabilityId, capabilityById, formatBytes } from './capability_manifest';
import { desktopCapabilities } from './desktop_services';
import styles from './capability_panel.module.css';

/**
 * First-run capability setup, shown once in the desktop shell. Top-level
 * capabilities are rows; a capability that requires another (e.g. Japanese
 * lyrics → lyrics) renders as a checkbox under its parent, showing only its
 * incremental download so the size doesn't read as the full closure on top of
 * the parent. Shares one capability store with the point-of-use gate
 * (desktopCapabilities). Renders nothing in the web build.
 */
export const DesktopFirstRun = observer(function DesktopFirstRun() {
  const deps = desktopCapabilities();
  const [open, setOpen] = React.useState(true);
  const [selected, setSelected] = React.useState<ReadonlySet<CapabilityId>>(new Set());
  if (deps == null || !open) return null;
  const { store, presenter } = deps;

  const topLevel = CAPABILITIES.filter((c) => c.requires.length === 0);
  const depsChildren = (id: CapabilityId): CapabilityId[] =>
    CAPABILITIES.filter((c) => c.kind === 'deps' && c.requires.includes(id)).map((c) => c.id);
  const toggle = (id: CapabilityId): void =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

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
          {topLevel.map((cap) => {
            const children = depsChildren(cap.id);
            // Install the deepest wanted child (its closure pulls the parent in
            // one uv sync); else just the parent.
            const wantedChild = children.find((id) => selected.has(id) || store.isReady(id));
            const target = wantedChild ?? cap.id;
            const installing =
              store.statusOf(cap.id) === 'installing' ||
              children.some((id) => store.statusOf(id) === 'installing');
            const size = presenter.incrementalBytes([target]);
            return (
              <div key={cap.id} className={styles.row}>
                <div className={styles.info}>
                  <span className={styles.name}>{cap.name}</span>
                  <span className={styles.desc}>{cap.description}</span>
                  {children.map((id) => {
                    const childReady = store.isReady(id);
                    const childInstalling = store.statusOf(id) === 'installing';
                    const delta =
                      presenter.incrementalBytes([id]) - presenter.incrementalBytes([cap.id]);
                    return (
                      <label key={id} className={styles.subOption}>
                        <input
                          type="checkbox"
                          checked={childReady || selected.has(id)}
                          disabled={childReady || childInstalling}
                          onChange={() => toggle(id)}
                        />
                        {capabilityById(id).name}
                        <span className={styles.delta}>
                          {childReady ? 'installed' : `+${formatBytes(delta)}`}
                        </span>
                      </label>
                    );
                  })}
                </div>
                <div className={styles.action}>
                  {installing ? (
                    <span className={styles.status}>Installing…</span>
                  ) : cap.kind === 'credentials' ? (
                    <span className={styles.status}>Needs API key</span>
                  ) : size === 0 ? (
                    <span className={styles.ready}>Installed</span>
                  ) : (
                    <button className={styles.install} onClick={() => void presenter.install(target)}>
                      Install · {formatBytes(size)}
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
