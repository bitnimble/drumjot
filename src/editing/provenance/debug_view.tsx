import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import styles from './debug_view.module.css';
import { ProvenanceStoreContext } from './provenance_contexts';

/**
 * Debug-bundle viewer body for the right sidebar's Debug panel. Reads the
 * loaded transcriber debug bundle off {@link ProvenanceStore} (populated by
 * the `Load > Load zip` flow, which auto-detects the bundle) and renders, top
 * to bottom in the portrait sidebar column: a run summary, the per-stage
 * timings, then the captured log stream.
 *
 * Read-only: the score + audio tracks the bundle carries are operated through
 * the existing toolbar / gutter controls. When no bundle is loaded the panel
 * shows a muted empty state (the rail item is always present, mirroring the
 * other sidebar panels).
 */
export const DebugView = observer(function DebugView() {
  const provenance = React.useContext(ProvenanceStoreContext);
  const bundle = provenance?.lastDebugBundle;
  if (!bundle) {
    return <p className={styles.empty}>No debug bundle loaded.</p>;
  }
  const stages = bundle.stage_timings ?? [];
  const logs = bundle.logs ?? [];
  const totalElapsed = bundle.elapsed_seconds;
  return (
    <div className={styles.debugView}>
      <div className={styles.summary}>
        {bundle.filename && <div className={styles.summaryFilename}>{bundle.filename}</div>}
        <div className={styles.summaryStats}>
          {stages.length} stage{stages.length === 1 ? '' : 's'} · {logs.length} log line
          {logs.length === 1 ? '' : 's'}
          {typeof totalElapsed === 'number' ? ` · ${totalElapsed.toFixed(2)}s total` : ''}
        </div>
        {bundle.started_at && <div className={styles.summaryStats}>{bundle.started_at}</div>}
      </div>

      <section className={styles.section}>
        <h3 className={styles.sectionTitle}>Stage timings</h3>
        {stages.length === 0 ? (
          <p className={styles.empty}>No stage timings recorded.</p>
        ) : (
          <ul className={styles.stageList}>
            {stages.map((s, i) => (
              <li key={i} className={styles.stageRow}>
                <span className={styles.stageName}>{s.stage}</span>
                <span className={styles.stageElapsed}>{s.elapsed_seconds.toFixed(2)}s</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className={styles.section}>
        <h3 className={styles.sectionTitle}>Logs</h3>
        {logs.length === 0 ? (
          <p className={styles.empty}>No logs captured.</p>
        ) : (
          <ul className={styles.logList}>
            {logs.map((entry, i) => (
              <li key={i} className={styles.logRow}>
                <div className={styles.logMeta}>
                  <span className={styles.logTimestamp}>+{entry.elapsed_seconds.toFixed(2)}s</span>
                  <span
                    className={classNames(
                      styles.logLevel,
                      entry.level === 'WARNING' && styles.logLevelWARNING,
                      entry.level === 'ERROR' && styles.logLevelERROR
                    )}
                  >
                    {entry.level}
                  </span>
                  <span className={styles.logLogger}>{entry.logger}</span>
                </div>
                <div className={styles.logMessage}>{entry.message}</div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
});
