import { observer } from 'mobx-react-lite';
import React from 'react';
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
 * The capability install list: a requires-tree of install rows (a root like
 * Stem separation with its dependents as nested +delta checkboxes; one Install
 * syncs the whole selection). Shared by the first-run setup and the Settings →
 * Capabilities tab; owns its own selection state. Renders nothing in the web
 * build (no Tauri capabilities).
 */
export const CapabilityList = observer(function CapabilityList() {
  const deps = desktopCapabilities();
  const [selected, setSelected] = React.useState<ReadonlySet<CapabilityId>>(new Set());
  if (deps == null) return null;
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
    <>
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
    </>
  );
});
