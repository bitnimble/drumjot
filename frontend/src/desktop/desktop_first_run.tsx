import { observer } from 'mobx-react-lite';
import React from 'react';
import modal from 'src/ui/modal/modal.module.css';
import { CAPABILITIES, type CapabilityId, capabilityById, formatBytes } from './capability_manifest';
import { desktopCapabilities } from './desktop_services';
import styles from './capability_panel.module.css';

const requiresChildren = (id: CapabilityId): CapabilityId[] =>
  CAPABILITIES.filter((c) => c.kind === 'deps' && c.requires.includes(id)).map((c) => c.id);

function collect(seed: CapabilityId, next: (id: CapabilityId) => CapabilityId[]): Set<CapabilityId> {
  const out = new Set<CapabilityId>();
  const visit = (id: CapabilityId): void => {
    for (const n of next(id)) {
      if (!out.has(n)) {
        out.add(n);
        visit(n);
      }
    }
  };
  visit(seed);
  return out;
}

/**
 * First-run capability setup, shown once in the desktop shell. Capabilities
 * render as a requires-tree: a root (Stem separation) with its dependents
 * (transcription, lyrics, Japanese) as nested checkboxes, each showing only its
 * incremental download (+delta) rather than the full closure stacked on top.
 * The root's Install button installs the root + every checked dependent in one
 * uv sync. Shares one capability store with the point-of-use gate. Renders
 * nothing in the web build.
 */
export const DesktopFirstRun = observer(function DesktopFirstRun() {
  const deps = desktopCapabilities();
  const [open, setOpen] = React.useState(true);
  const [selected, setSelected] = React.useState<ReadonlySet<CapabilityId>>(new Set());
  if (deps == null || !open) return null;
  const { store, presenter } = deps;

  const toggle = (id: CapabilityId): void =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
        // A dependent can't stay selected without the thing it requires.
        for (const dep of collect(id, requiresChildren)) next.delete(dep);
      } else {
        next.add(id);
        // Selecting a dependent implies its prerequisites.
        for (const req of collect(id, (x) => capabilityById(x).requires)) next.add(req);
      }
      return next;
    });

  // Root + every dependent the user picked (or that's already installed); the
  // size sums the lot, deduped, via incrementalBytes.
  const installSet = (rootId: CapabilityId): CapabilityId[] => {
    const set = [rootId];
    for (const dep of collect(rootId, requiresChildren)) {
      if (selected.has(dep) || store.isReady(dep)) set.push(dep);
    }
    return set;
  };

  const renderChild = (id: CapabilityId, depth: number): React.ReactNode => {
    const ready = store.isReady(id);
    const installing = store.statusOf(id) === 'installing';
    const parent = capabilityById(id).requires[0];
    const delta =
      parent != null
        ? presenter.incrementalBytes([id]) - presenter.incrementalBytes([parent])
        : presenter.incrementalBytes([id]);
    return (
      <React.Fragment key={id}>
        <label className={styles.subOption} style={{ marginLeft: depth * 16 }}>
          <input
            type="checkbox"
            checked={ready || selected.has(id)}
            disabled={ready || installing}
            onChange={() => toggle(id)}
          />
          {capabilityById(id).name}
          <span className={styles.delta}>{ready ? 'installed' : `+${formatBytes(delta)}`}</span>
        </label>
        {requiresChildren(id).map((c) => renderChild(c, depth + 1))}
      </React.Fragment>
    );
  };

  const roots = CAPABILITIES.filter((c) => c.requires.length === 0);

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
          {roots.map((cap) => {
            const set = installSet(cap.id);
            const size = presenter.incrementalBytes(set);
            const installing = set.some((id) => store.statusOf(id) === 'installing');
            return (
              <div key={cap.id} className={styles.row}>
                <div className={styles.info}>
                  <span className={styles.name}>{cap.name}</span>
                  <span className={styles.desc}>{cap.description}</span>
                  {requiresChildren(cap.id).map((c) => renderChild(c, 0))}
                </div>
                <div className={styles.action}>
                  {installing ? (
                    <span className={styles.status}>Installing…</span>
                  ) : cap.kind === 'credentials' ? (
                    <span className={styles.status}>Needs API key</span>
                  ) : size === 0 ? (
                    <span className={styles.ready}>Installed</span>
                  ) : (
                    <button className={styles.install} onClick={() => void presenter.installAll(set)}>
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
