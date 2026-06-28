import { observer } from 'mobx-react-lite';
import React from 'react';
import { ActionMenuItem } from 'src/ui/dropdown/dropdown';
import { ContextMenu } from 'src/ui/context_menu/context_menu';
import type { BpmMarker, TempoEditPresenter } from 'src/editing/playback/tempo_edit_presenter';
import styles from './score.module.css';

/**
 * One editable BPM pill in the timeline header. Reads its value off the live
 * {@link BpmMarker} (so an external edit / undo reflows it); clicking the text
 * opens an inline numeric editor (Enter / blur saves, Escape cancels, clearing
 * it deletes the event); right-clicking opens the same dropdown-style menu the
 * header uses, with a "Delete BPM change" item for events. A pill freshly
 * minted by "Change BPM here" mounts with `autoFocus` so the user can type the
 * new value immediately.
 */
export const BpmPill = observer(function BpmPill({
  marker,
  presenter,
  className,
  autoFocus,
  onAutoFocusConsumed,
}: {
  marker: BpmMarker;
  presenter: TempoEditPresenter;
  className?: string;
  autoFocus?: boolean;
  onAutoFocusConsumed?: () => void;
}) {
  const [editing, setEditing] = React.useState(false);
  const [text, setText] = React.useState(String(marker.bpm));
  const [menu, setMenu] = React.useState<{ x: number; y: number } | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const autoFocused = React.useRef(false);

  const startEdit = () => {
    setText(String(marker.bpm));
    setEditing(true);
  };

  // A pill created via "Change BPM here" enters edit mode once, immediately.
  React.useEffect(() => {
    if (autoFocus && !autoFocused.current) {
      autoFocused.current = true;
      startEdit();
      onAutoFocusConsumed?.();
    }
  }, [autoFocus, onAutoFocusConsumed]); // eslint-disable-line

  React.useEffect(() => {
    if (!editing) return;
    const el = inputRef.current;
    if (el) {
      el.focus();
      el.select();
    }
  }, [editing]);

  const commit = () => {
    setEditing(false);
    presenter.commitMarker(marker.source, text);
  };
  const cancel = () => {
    setEditing(false);
    setText(String(marker.bpm));
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        className={`${className ?? ''} ${styles.timelineHeaderBpmInput}`}
        value={text}
        inputMode="numeric"
        size={Math.max(3, text.length)}
        data-noseek
        data-testid="bpm-pill-input"
        onPointerDown={(e) => e.stopPropagation()}
        onChange={(e) => setText(e.target.value.replace(/[^0-9]/g, ''))}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commit();
          } else if (e.key === 'Escape') {
            e.preventDefault();
            cancel();
          }
        }}
      />
    );
  }

  return (
    <>
      <span
        className={`${className ?? ''} ${styles.timelineHeaderBpmEditable}`}
        data-testid="bpm-pill"
        title="Click to edit, right-click to delete"
        onClick={(e) => {
          // stopPropagation (not `data-noseek`) keeps the click off the
          // bars-row seek handler, so the pill isn't mistaken for a note by
          // selectors that key on `data-noseek`.
          e.stopPropagation();
          startEdit();
        }}
        onContextMenu={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setMenu({ x: e.clientX, y: e.clientY });
        }}
      >
        {marker.bpm} bpm
      </span>
      {menu && (
        <ContextMenu x={menu.x} y={menu.y} onClose={() => setMenu(null)}>
          <ActionMenuItem
            label="Edit BPM"
            testId="bpm-menu-edit"
            onClick={() => {
              setMenu(null);
              startEdit();
            }}
          />
          {presenter.canDelete(marker.source) && (
            <ActionMenuItem
              label="Delete BPM change"
              testId="bpm-menu-delete"
              onClick={() => {
                setMenu(null);
                if (marker.source.kind === 'event') presenter.deleteEvent(marker.source.id);
              }}
            />
          )}
        </ContextMenu>
      )}
    </>
  );
});
