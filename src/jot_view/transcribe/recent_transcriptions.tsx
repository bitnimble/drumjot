import { ChevronDown, ChevronRight, RefreshCw } from 'lucide-react';
import React from 'react';
import { TranscriptionSummary } from 'src/jot_view/transcribe/transcriber';
import { useParentSubmenuRegistry } from 'src/ui/dropdown/dropdown';
import styles from './recent_transcriptions.module.css';

const LIMIT = 5;

type Variant = 'menu' | 'cta';

/** Format one row of the picker. Compresses three pieces of context into
 *  the row's tooltip: the original upload filename, when the run was
 *  originally requested, and when its artifacts were most-recently
 *  regenerated (with the resume stage tagged on if the most-recent run
 *  was a resume). Also used by the Resume-tab Select in the toolbar. */
export function formatTranscriptionSummary(s: TranscriptionSummary): string {
  const filename = s.original_filename ?? s.folder;
  const requested = formatTimestamp(s.requested_at);
  const lastRun = s.last_run_at ? formatTimestamp(s.last_run_at) : null;
  let detail = `requested ${requested}`;
  if (lastRun && lastRun !== requested) {
    detail += `, last run ${lastRun}`;
    if (s.last_resume_stage) {
      detail += ` (from ${s.last_resume_stage})`;
    }
  }
  return `${filename}; ${detail}`;
}

function formatTimestamp(iso: string): string {
  // The backend stamps timestamps as UTC ISO (Z-suffixed); see
  // `mint_request_folder_name` + `_parse_folder_timestamp` in
  // transcriber/app/pipeline/resume.py. Parse with `Date` and emit the
  // user's local wall clock so the picker doesn't surprise operators in
  // non-UTC timezones.
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/**
 * Recent-transcriptions popover used both inside the toolbar's Load menu
 * (variant="menu") and on the empty-state welcome card (variant="cta").
 *
 * Self-managed open state, outside-click + Escape close, lazy fetch on
 * first open. Recent runs are sorted server-side by
 * `last_run_at ?? requested_at` desc, so a transcription that was just
 * resumed bubbles to the top even if its original request was older.
 */
export const RecentTranscriptionsPicker = ({
  variant,
  triggerLabel,
  triggerTitle,
  items,
  loaded,
  loading,
  onRefresh,
  onPick,
  onAfterPick,
}: {
  variant: Variant;
  triggerLabel: string;
  triggerTitle?: string;
  items: readonly TranscriptionSummary[];
  loaded: boolean;
  loading: boolean;
  onRefresh: () => void;
  onPick: (folder: string) => void;
  /** Invoked after a successful pick, e.g. to close an enclosing menu. */
  onAfterPick?: () => void;
}) => {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);

  // Hook into the nearest parent panel's submenu registry so opening
  // this picker closes any sibling submenu (and vice versa), matching
  // SubmenuItem's behaviour. No-op when rendered outside a dropdown
  // panel (the cta variant on the empty-state card).
  useParentSubmenuRegistry(open, setOpen);

  // Stable ref so the open-effect doesn't refire when the parent passes
  // a new closure on every render.
  const onRefreshRef = React.useRef(onRefresh);
  onRefreshRef.current = onRefresh;

  React.useEffect(() => {
    if (!open) return;
    if (!loaded && !loading) {
      onRefreshRef.current();
    }
    const onPointerDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open, loaded, loading]);

  const visible = items.slice(0, LIMIT);
  const containerClass = variant === 'cta' ? styles.containerCta : styles.containerMenu;
  const triggerClass = variant === 'cta' ? styles.triggerCta : styles.triggerMenu;
  const panelClass = variant === 'cta' ? styles.panelCta : styles.panelMenu;

  return (
    <div className={containerClass} ref={ref}>
      <button
        type="button"
        className={triggerClass}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        title={triggerTitle}
      >
        <span>{triggerLabel}</span>
        {variant === 'cta' ? (
          <ChevronDown size={14} aria-hidden="true" className={styles.icon} />
        ) : (
          <ChevronRight size={14} aria-hidden="true" className={styles.iconMuted} />
        )}
      </button>
      {open && (
        <div className={panelClass} role="menu">
          <div className={styles.header}>
            <span className={styles.title}>Recent transcriptions</span>
            <button
              type="button"
              className={styles.refresh}
              onClick={() => onRefresh()}
              disabled={loading}
              title="Re-fetch the list of recent transcriptions from the server."
              aria-label="Refresh recent transcriptions"
            >
              {loading ? (
                <span className={styles.spinner} aria-hidden="true" />
              ) : (
                <RefreshCw size={12} aria-hidden="true" className={styles.icon} />
              )}
            </button>
          </div>
          {loading && visible.length === 0 ? (
            <div className={styles.loading}>
              <span className={styles.spinner} aria-hidden="true" />
              <span>Loading…</span>
            </div>
          ) : visible.length === 0 ? (
            <div className={styles.empty}>No recent transcriptions.</div>
          ) : (
            visible.map((s) => (
              <button
                key={s.folder}
                type="button"
                className={styles.item}
                onClick={() => {
                  onPick(s.folder);
                  setOpen(false);
                  onAfterPick?.();
                }}
                title={formatTranscriptionSummary(s)}
              >
                {s.original_filename ?? s.folder}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
};
