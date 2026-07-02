import { observer } from 'mobx-react-lite';
import React from 'react';
import {
  CAPABILITIES,
  type CapabilityId,
  capabilityById,
  capabilityClosure,
} from './capability_manifest';
import { type CapabilityStatus } from './capability_store';
import { desktopCapabilities } from './desktop_services';
import { Spinner } from 'src/ui/spinner/spinner';
import styles from './capability_panel.module.css';

/** Selection + install state for the capability picker, shared by the first-run
 *  panel and Settings → Capabilities. Direct user picks live in `selected`;
 *  everything their requires-closure pulls in is derived (checked + locked). */
export type CapabilityInstallController = {
  available: boolean;
  isChecked: (id: CapabilityId) => boolean;
  isDisabled: (id: CapabilityId) => boolean;
  statusOf: (id: CapabilityId) => CapabilityStatus;
  toggle: (id: CapabilityId) => void;
  /** Cumulative incremental download for the whole selection (deduped). */
  totalBytes: number;
  installing: boolean;
  /** undefined = free space unknown (don't block); false = the selection won't
   *  fit on the data-root volume. */
  enoughSpace: boolean | undefined;
  install: () => void;
};

export function useCapabilityInstall(): CapabilityInstallController {
  const deps = desktopCapabilities();
  const [selected, setSelected] = React.useState<ReadonlySet<CapabilityId>>(new Set());
  const store = deps?.store;
  const presenter = deps?.presenter;

  const selectedClosure = capabilityClosure([...selected]);
  // A capability pulled in by *another* selection (e.g. separation under a
  // selected transcription) is locked: checked, but the user can't uncheck it.
  const impliedByOther = (id: CapabilityId): boolean =>
    [...selected].some((s) => s !== id && capabilityClosure([s]).has(id));

  return {
    available: deps != null,
    isChecked: (id) => (store?.isReady(id) ?? false) || selectedClosure.has(id),
    isDisabled: (id) => (store?.isReady(id) ?? false) || impliedByOther(id),
    statusOf: (id) => store?.statusOf(id) ?? 'not-installed',
    toggle: (id) =>
      setSelected((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      }),
    totalBytes: presenter != null ? presenter.incrementalBytes([...selected]) : 0,
    installing: [...selectedClosure].some((id) => store?.statusOf(id) === 'installing'),
    enoughSpace: presenter != null ? presenter.hasEnoughSpaceFor([...selected]) : undefined,
    install: () => {
      if (presenter != null && presenter.hasEnoughSpaceFor([...selected]) !== false) {
        void presenter.installAll([...selected]);
      }
    },
  };
}

/**
 * Capability picker as a checkbox tree: top-level features (Stem separation,
 * Transcription, Lyrics, AI assist) with pure sub-features nested (Japanese
 * under Lyrics). Shared prerequisites deduplicate, selecting Transcription
 * checks + locks Stem separation, since it's a subset. No per-row sizes; the
 * cumulative total lives in the dialog footer (see {@link useCapabilityInstall}).
 */
export const CapabilityTree = observer(function CapabilityTree({
  controller,
}: {
  controller: CapabilityInstallController;
}) {
  const childrenOf = (id: CapabilityId): CapabilityId[] =>
    CAPABILITIES.filter((c) => c.uiParent === id).map((c) => c.id);

  const renderNode = (id: CapabilityId, depth: number): React.ReactNode => {
    const cap = capabilityById(id);
    const status = controller.statusOf(id);
    const credentials = cap.kind === 'credentials';
    // `installing` renders a spinner (below), not a text badge.
    const badge =
      status === 'ready'
        ? 'Installed'
        : credentials
          ? 'Needs API key'
          : status === 'error'
            ? 'Failed'
            : '';
    return (
      <React.Fragment key={id}>
        <div className={styles.node} style={{ paddingInlineStart: depth * 22 }}>
          {credentials ? (
            <span className={styles.nodeInfo}>
              <span className={styles.name}>{cap.name}</span>
              <span className={styles.desc}>{cap.description}</span>
            </span>
          ) : (
            <label className={styles.nodeLabel}>
              <input
                type="checkbox"
                checked={controller.isChecked(id)}
                disabled={controller.isDisabled(id)}
                onChange={() => controller.toggle(id)}
              />
              <span className={styles.nodeInfo}>
                <span className={styles.name}>{cap.name}</span>
                <span className={styles.desc}>{cap.description}</span>
              </span>
            </label>
          )}
          {status === 'installing' ? (
            <span
              className={styles.nodeStatus}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}
            >
              <Spinner size={12} label="Installing" />
              Installing…
            </span>
          ) : (
            badge !== '' && <span className={styles.nodeStatus}>{badge}</span>
          )}
        </div>
        {childrenOf(id).map((c) => renderNode(c, depth + 1))}
      </React.Fragment>
    );
  };

  const roots = CAPABILITIES.filter((c) => c.uiParent == null);
  return <div className={styles.tree}>{roots.map((c) => renderNode(c.id, 0))}</div>;
});
