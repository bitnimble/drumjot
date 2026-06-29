import classNames from 'classnames';
import React from 'react';
import { GutterResizeHandle } from 'src/ui/gutter_resize_handle/gutter_resize_handle';
import styles from './track_gutter.module.css';

/** A track's in-flight indicator: drives the shared gutter spinner with a
 *  tooltip + optional test id. Absent ⇒ no spinner. Used by every track type
 *  (audio split / audio transcribe / lyrics align) so the busy affordance is
 *  identical across rows and lives in exactly one place. */
export type TrackBusy = { tooltip: string; testId?: string };

/** The shared spinner shown beside a track's secondary label while work is in
 *  flight. The sole spinner implementation for track gutters. */
export const TrackBusySpinner = ({ busy }: { busy: TrackBusy }) => (
  <span
    className={styles.spinner}
    title={busy.tooltip}
    aria-label={busy.tooltip}
    role="status"
    data-testid={busy.testId}
  />
);

/**
 * The shared chrome for every track row's gutter head: the sticky, fixed-width
 * gutter column holding a drag handle, a resize handle, and a content column
 * with a header (label + optional spinner + optional overflow menu) and an
 * optional body (the per-track controls).
 *
 * Each track type renders only its differentiated *slots* (label spans, the
 * controls body, its overflow menu); the structure, geometry, sticky/scroll
 * behaviour, and busy spinner live here so audio / instrument / lyrics rows
 * can't drift apart.
 */
export const TrackGutter = ({
  variant = 'cream',
  height,
  dragHandle,
  onResizeGutterStart,
  dim,
  labelDirection = 'column',
  labelClassName,
  labelTitle,
  contentAlign = 'between',
  headerAlign = 'start',
  primary,
  secondary,
  busy,
  overflow,
  body,
}: {
  /** Background tint: `cream` (audio / lyrics) or `creamStrong` (instrument). */
  variant?: 'cream' | 'creamStrong';
  /** Fixed row height in px; omit to size from content (instrument rows). */
  height?: number;
  /** The drag handle element (per-view, e.g. `MixerDragHandle`). */
  dragHandle: React.ReactNode;
  onResizeGutterStart: (e: React.PointerEvent<HTMLDivElement>) => void;
  /** Dim the label (e.g. a muted/inaudible track). */
  dim?: boolean;
  /** `column` stacks primary over secondary (audio / lyrics); `row` lays the
   *  primary slot inline (instrument: lane + name). */
  labelDirection?: 'column' | 'row';
  /** Extra class merged onto the label box (per-view typography/weight). */
  labelClassName?: string;
  /** `title=` hover tooltip for the label box (e.g. the full instrument name). */
  labelTitle?: string;
  contentAlign?: 'between' | 'center';
  headerAlign?: 'start' | 'center';
  /** Primary label line (slot). */
  primary: React.ReactNode;
  /** Optional secondary label line (slot); the spinner sits beside it. */
  secondary?: React.ReactNode;
  busy?: TrackBusy;
  /** Optional overflow-menu trigger (slot), pinned to the header's right. */
  overflow?: React.ReactNode;
  /** Optional controls row beneath the header (slot). */
  body?: React.ReactNode;
}) => (
  <div
    className={classNames(styles.gutter, variant === 'creamStrong' ? styles.creamStrong : styles.cream)}
    style={height !== undefined ? { height } : undefined}
  >
    {dragHandle}
    <GutterResizeHandle onResizeStart={onResizeGutterStart} />
    <div
      className={classNames(
        styles.content,
        contentAlign === 'center' ? styles.contentCenter : styles.contentBetween
      )}
    >
      <div className={classNames(styles.header, headerAlign === 'center' ? styles.headerCenter : styles.headerStart)}>
        <div
          className={classNames(
            styles.label,
            labelDirection === 'row' ? styles.labelRow : styles.labelColumn,
            dim && styles.labelDim,
            labelClassName
          )}
          title={labelTitle}
        >
          {primary}
          {(secondary !== undefined || busy) && (
            <div className={styles.secondaryRow}>
              {secondary}
              {busy && <TrackBusySpinner busy={busy} />}
            </div>
          )}
        </div>
        {overflow && <span className={styles.overflowSlot}>{overflow}</span>}
      </div>
      {body}
    </div>
  </div>
);
