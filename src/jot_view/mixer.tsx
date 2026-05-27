import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { RenderedJot, StructuralBar, ViewConfig } from 'src/jot';
import { AudioTrack, AudioTrackId, AudioTrackRole, jotPlayer } from 'src/playback';
import { waveformWorker, BarSlice } from 'src/playback/waveform_worker_client';
import {
  BarBeat,
  WaveformChunk,
  buildChunkLayout,
} from './waveform_chunks';
import { GutterResizeHandle } from './components/gutter_resize_handle';
import { ClearButton, MuteButton, SoloButton } from './components/icon_button';
import { DropdownButton, dropdownStyles } from './components/dropdown';
import {
  JotViewStoreContext,
  NoteProvenanceContext,
  RenderedJotContext,
  UniformWaveformsContext,
} from './contexts';
import { LyricsRow } from './lyrics_row';
import styles from './mixer.module.css';
import { Playhead } from './playback';
import { BarView, FilteredOnsetView, seekFromClick } from './score';
import { TrackKey, VOLUME_STEP } from './store';

export type VoiceControls = {
  mutedPitches: ReadonlySet<string>;
  soloedPitches: ReadonlySet<string>;
  /** True if the row would currently make sound; false = muted via M or solo exclusion. */
  isPitchAudible: (pitch: string) => boolean;
  /** Current row fader value, 0..1 (1 = full). */
  volumeFor: (pitch: string) => number;
  onSetVolume: (pitch: string, v: number) => void;
  onToggleMute: (pitch: string) => void;
  onToggleSolo: (pitch: string) => void;
  /** Drum section master M/S. The master acts at the bus, not by editing
   * the per-row M/S sets; `masterAudible` reflects the resolved state
   * (master mute + cross-domain solo) so the master row can dim itself. */
  masterMuted: boolean;
  masterSoloed: boolean;
  masterAudible: boolean;
  onToggleMasterMute: () => void;
  onToggleMasterSolo: () => void;
};

export type AudioTrackControls = {
  mutedAudioTracks: ReadonlySet<AudioTrackId>;
  soloedAudioTracks: ReadonlySet<AudioTrackId>;
  isAudioTrackAudible: (id: AudioTrackId) => boolean;
  volumeFor: (id: AudioTrackId) => number;
  onSetVolume: (id: AudioTrackId, v: number) => void;
  onToggleMute: (id: AudioTrackId) => void;
  onToggleSolo: (id: AudioTrackId) => void;
  /** Drop a loaded audio track (button in the gutter clears the slot). */
  onClear: (id: AudioTrackId) => void;
  /** Overflow menu: run stage 1 (`stems_all`) on this track,
   *  isolating drums + drumless backing from a full-mix recording. */
  onSplitFromMix: (id: AudioTrackId) => void;
  /** Overflow menu: run stage 2 (`stems_per`) on this track,
   *  splitting a drum-only recording into per-instrument pieces. */
  onSplitDrumPieces: (id: AudioTrackId) => void;
  /** Audio section master M/S; same semantics as on {@link VoiceControls}. */
  masterMuted: boolean;
  masterSoloed: boolean;
  masterAudible: boolean;
  onToggleMasterMute: () => void;
  onToggleMasterSolo: () => void;
};

/**
 * Drag-source identifier carried on the DataTransfer of a mixer-row
 * drag. A custom MIME type lets us reject foreign drops (files,
 * external pages) so the gutter never tries to swallow them.
 */
const MIXER_DRAG_MIME = 'application/x-drumjot-mixer-row';

/** Fixed row height shared by the gutter (label + filename + button
 *  cluster) and the bars-row waveform on the right. Sized to fit the
 *  worst-case gutter content: a 2-line clamped name (~32px) + the
 *  filename row (~14px) + the M/S/X button cluster (~22px) + the
 *  gutter's 8px vertical padding. Bumping this also bumps the
 *  waveform height, which is a desirable side effect, taller peaks
 *  read more clearly. */
const AUDIO_TRACK_HEIGHT = 76;

/** Audio-track display name: filename with its extension stripped. */
function audioTrackLabel(filename: string): string {
  return filename.replace(/\.[^./\\]+$/, '') || filename;
}

/** Common drag/drop props passed to every mixer row. */
type MixerRowDragProps = {
  idx: number;
  dragFromIdx: number | undefined;
  dropTargetIdx: number | undefined;
  onDragStartIdx: (i: number) => void;
  onDropTargetIdx: (i: number | undefined) => void;
  onMoveTrack: (from: number, to: number) => void;
  onResetDrag: () => void;
  /** Length of the mixer list (used by the end-of-list drop zone). */
  mixerLength: number;
  /**
   * True when this row starts a new group (its `groupId` differs from
   * the previous row's, or it's not in a group at all and follows a
   * row that was). The row renders a small top margin so adjacent
   * groups read as distinct clusters; same-group rows render flush.
   * The first row in the mixer never receives this — nothing above it
   * to gap against.
   */
  groupStart: boolean;
  /**
   * True when this row ends a group — it has a `groupId` AND the next
   * row has a different (or no) `groupId`. Together with `groupStart`
   * it lets the row know it's on the outer edge of a real group (vs a
   * solo row that just happens to follow a different cluster), so the
   * outer border can render thicker than a regular inter-row separator.
   */
  groupEnd: boolean;
  /** True iff this row is part of a group (`key.groupId !== undefined`). */
  inGroup: boolean;
  /** Pointer-down handler for the gutter-edge resize affordance. */
  onResizeGutterStart: (e: React.PointerEvent<HTMLDivElement>) => void;
};

/**
 * The unified mixer that replaced the old separate "audio tracks" and
 * "voice staves" sections. Renders the two section masters at the top,
 * then one row per entry in `trackOrder` — an audio track or a single
 * drum-instrument pitch, freely interleavable. Drag-and-drop on each
 * row's gutter handle rewrites the order via `JotViewStore.moveTrack`;
 * the topmost drum-pitch row hosts the pattern/tuplet bracket overlay
 * so they read as a single piece of score chrome regardless of where
 * the user has moved the rows.
 */
export const MixerView = observer(
  ({
    jot,
    config,
    trackOrder,
    highlightedPattern,
    onPatternClick,
    onSeek,
    onMoveTrack,
    voiceControls,
    audioTrackControls,
    onResizeGutterStart,
  }: {
    jot: RenderedJot;
    config: ViewConfig;
    trackOrder: readonly TrackKey[];
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    onSeek: (x: number) => void;
    onMoveTrack: (from: number, to: number) => void;
    voiceControls: VoiceControls;
    audioTrackControls: AudioTrackControls;
    /** Pointer-down handler for the gutter resize affordance painted on
     * the right edge of every row's gutter. */
    onResizeGutterStart: (e: React.PointerEvent<HTMLDivElement>) => void;
  }) => {
    // The drop indicator is rendered above the row at `dropTargetIdx`
    // when it lies in [0, length]; `length` is the "after the last row"
    // slot. `dragFromIdx` short-circuits a hover-over-self update so the
    // indicator doesn't flash on the row the user picked up.
    const [dragFromIdx, setDragFromIdx] = React.useState<number | undefined>(undefined);
    const [dropTargetIdx, setDropTargetIdx] = React.useState<number | undefined>(undefined);
    const resetDrag = () => {
      setDragFromIdx(undefined);
      setDropTargetIdx(undefined);
    };

    // The topmost drum-pitch row in the user's mixer order hosts the
    // tuplet brackets and the lead-in label — chrome that belongs to
    // the score as a whole, not to any one instrument.
    const firstPitchIdx = trackOrder.findIndex((k) => k.kind === 'pitch');
    // Drum pitches in mixer order. Pattern brackets are drawn on every
    // row whose pitch participates in the pattern; this list lets each
    // row know whether it's the topmost / bottommost participant for a
    // given span so the bracket reads as one continuous outline across
    // rows (with non-participating rows in between visually skipped).
    const pitchOrder = React.useMemo(
      () => trackOrder.flatMap((k) => (k.kind === 'pitch' ? [k.pitch] : [])),
      [trackOrder]
    );

    return (
      <div className={styles.mixer}>
        <GutterMasterRow
          label="Audio master"
          title="Master volume for all loaded audio (backing) tracks together. Multiplies on top of each track's own fader; takes effect instantly, including mid-playback."
          value={jotPlayer.audioTrackMasterVolume}
          onChange={(v) => jotPlayer.setAudioTrackMasterVolume(v)}
          muted={audioTrackControls.masterMuted}
          soloed={audioTrackControls.masterSoloed}
          audible={audioTrackControls.masterAudible}
          onToggleMute={audioTrackControls.onToggleMasterMute}
          onToggleSolo={audioTrackControls.onToggleMasterSolo}
          testId="audio-track-master"
          onResizeGutterStart={onResizeGutterStart}
        />
        <GutterMasterRow
          label="Drums master"
          title="Master volume for all drum/instrument rows together. Multiplies on top of each row's own fader; takes effect instantly, including mid-playback."
          value={jotPlayer.drumMasterVolume}
          onChange={(v) => jotPlayer.setDrumMasterVolume(v)}
          muted={voiceControls.masterMuted}
          soloed={voiceControls.masterSoloed}
          audible={voiceControls.masterAudible}
          onToggleMute={voiceControls.onToggleMasterMute}
          onToggleSolo={voiceControls.onToggleMasterSolo}
          testId="drum-master"
          onResizeGutterStart={onResizeGutterStart}
        />
        {trackOrder.map((key, idx) => {
          // Reuse a stable React key per row so dragging doesn't tear
          // down + remount expensive children (the AudioTrackWaveformCanvas
          // would otherwise re-decode peaks on every reorder).
          const reactKey =
            key.kind === 'audio'
              ? `audio:${key.id}`
              : key.kind === 'pitch'
                ? `pitch:${key.pitch}`
                : 'lyrics';
          // A row begins a new "group" — and so renders with a small
          // top gap — whenever its `groupId` differs from the previous
          // row's. Solo (groupId undefined) rows are each their own
          // group. The first row never gets a gap (nothing above it).
          const prevGroupId = idx > 0 ? trackOrder[idx - 1].groupId : undefined;
          const nextGroupId = idx < trackOrder.length - 1 ? trackOrder[idx + 1].groupId : undefined;
          const groupStart = idx > 0 && key.groupId !== prevGroupId;
          // groupEnd / inGroup only fire on rows that are actually part
          // of a `groupId` cluster (paired audio↔pitch today). Solo rows
          // crossing a group boundary still get `groupStart` for the
          // inter-cluster gap, but aren't treated as a one-row group.
          const inGroup = key.groupId !== undefined;
          const groupEnd = inGroup && key.groupId !== nextGroupId;
          const rowProps = {
            idx,
            dragFromIdx,
            dropTargetIdx,
            onDragStartIdx: setDragFromIdx,
            onDropTargetIdx: setDropTargetIdx,
            onMoveTrack,
            onResetDrag: resetDrag,
            mixerLength: trackOrder.length,
            groupStart,
            groupEnd,
            inGroup,
            onResizeGutterStart,
          };
          if (key.kind === 'audio') {
            const track = jotPlayer.audioTracks.get(key.id);
            // The reaction in JotViewStore drops dead audio ids on the
            // same MobX tick, so this gap is one-frame at most. Render
            // nothing rather than crash if the maps race.
            if (!track) return null;
            return (
              <AudioTrackRow
                key={reactKey}
                id={key.id}
                track={track}
                jot={jot}
                controls={audioTrackControls}
                onSeek={onSeek}
                {...rowProps}
              />
            );
          }
          if (key.kind === 'lyrics') {
            return <LyricsRow key={reactKey} jot={jot} onSeek={onSeek} {...rowProps} />;
          }
          return (
            <PitchRow
              key={reactKey}
              pitch={key.pitch}
              jot={jot}
              config={config}
              showBrackets={idx === firstPitchIdx}
              pitchOrder={pitchOrder}
              highlightedPattern={highlightedPattern}
              onPatternClick={onPatternClick}
              onSeek={onSeek}
              voiceControls={voiceControls}
              {...rowProps}
            />
          );
        })}
        {/* "Drop at the very end" zone — without this the user can't
            move a row past the last existing row because the indicator
            target would clamp to its own bottom edge. Kept thin so it
            barely affects layout when no drag is in flight. */}
        <MixerEndDropZone
          idx={trackOrder.length}
          dragFromIdx={dragFromIdx}
          dropTargetIdx={dropTargetIdx}
          onDropTargetIdx={setDropTargetIdx}
          onMoveTrack={onMoveTrack}
          onResetDrag={resetDrag}
        />
      </div>
    );
  }
);

/**
 * Shared drag-target behaviour for the row gutter: a drag-over either
 * marks "drop above this row" (top half) or "drop below this row"
 * (bottom half), `onDrop` commits the move. Returns the props/style
 * fragments the row should spread onto its wrapper + a boolean for
 * whether the drop indicator should render above this row.
 */
function useMixerRowDropTarget({
  idx,
  dragFromIdx,
  dropTargetIdx,
  onDropTargetIdx,
  onMoveTrack,
  onResetDrag,
}: {
  idx: number;
  dragFromIdx: number | undefined;
  dropTargetIdx: number | undefined;
  onDropTargetIdx: (i: number | undefined) => void;
  onMoveTrack: (from: number, to: number) => void;
  onResetDrag: () => void;
}) {
  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    if (dragFromIdx === undefined) return;
    if (!e.dataTransfer.types.includes(MIXER_DRAG_MIME)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    const rect = e.currentTarget.getBoundingClientRect();
    const isTopHalf = e.clientY < rect.top + rect.height / 2;
    const target = isTopHalf ? idx : idx + 1;
    if (target !== dropTargetIdx) onDropTargetIdx(target);
  };
  const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    // Don't clear when the pointer just crossed into a child element;
    // only when it actually leaves the row bounds (relatedTarget
    // outside the gutter element).
    const next = e.relatedTarget as Node | null;
    if (next && e.currentTarget.contains(next)) return;
    if (dropTargetIdx === idx || dropTargetIdx === idx + 1) onDropTargetIdx(undefined);
  };
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    if (dragFromIdx === undefined) return;
    const data = e.dataTransfer.getData(MIXER_DRAG_MIME);
    if (!data) return;
    e.preventDefault();
    const from = parseInt(data, 10);
    if (Number.isFinite(from) && dropTargetIdx !== undefined) {
      onMoveTrack(from, dropTargetIdx);
    }
    onResetDrag();
  };
  const isDropIndicatorAbove = dropTargetIdx === idx && dragFromIdx !== undefined;
  const isDropIndicatorBelow = dropTargetIdx === idx + 1 && dragFromIdx !== undefined;
  return { onDragOver, onDragLeave, onDrop, isDropIndicatorAbove, isDropIndicatorBelow };
}

/**
 * A small "drop after the last row" zone. The per-row drop logic
 * already covers "before me" and "after me", but it bottoms out at the
 * last row's "after" position; this acts as the explicit final slot so
 * the indicator renders cleanly between the last row and the bottom of
 * the mixer.
 */
const MixerEndDropZone = ({
  idx,
  dragFromIdx,
  dropTargetIdx,
  onDropTargetIdx,
  onMoveTrack,
  onResetDrag,
}: {
  idx: number;
  dragFromIdx: number | undefined;
  dropTargetIdx: number | undefined;
  onDropTargetIdx: (i: number | undefined) => void;
  onMoveTrack: (from: number, to: number) => void;
  onResetDrag: () => void;
}) => {
  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    if (dragFromIdx === undefined) return;
    if (!e.dataTransfer.types.includes(MIXER_DRAG_MIME)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (dropTargetIdx !== idx) onDropTargetIdx(idx);
  };
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    if (dragFromIdx === undefined) return;
    const data = e.dataTransfer.getData(MIXER_DRAG_MIME);
    if (!data) return;
    e.preventDefault();
    const from = parseInt(data, 10);
    if (Number.isFinite(from)) onMoveTrack(from, idx);
    onResetDrag();
  };
  const showIndicator = dropTargetIdx === idx && dragFromIdx !== undefined;
  return (
    <div
      className={classNames(styles.mixerEndDrop, showIndicator && styles.mixerDropIndicator)}
      onDragOver={onDragOver}
      onDrop={onDrop}
    />
  );
};

/**
 * Drag handle (≡) parked on the leftmost edge of every mixer row's
 * gutter. Only this element is `draggable`, so the user can still click
 * mute/solo, drag the volume slider, etc. without accidentally lifting
 * the whole row.
 */
const MixerDragHandle = ({
  idx,
  onDragStartIdx,
  onResetDrag,
  ariaLabel,
}: {
  idx: number;
  onDragStartIdx: (i: number) => void;
  onResetDrag: () => void;
  ariaLabel: string;
}) => {
  return (
    <div
      className={styles.mixerDragHandle}
      draggable={true}
      // The page-level mousedown listener (createJotView's marquee
      // selection) calls `preventDefault()`, which also cancels the
      // subsequent native dragstart — so without this stop the row
      // never lifts and the user just gets a marquee instead. The
      // handle's own mousedown still fires; only the bubbled handler
      // is suppressed.
      onMouseDown={(e) => e.stopPropagation()}
      onDragStart={(e) => {
        e.dataTransfer.setData(MIXER_DRAG_MIME, String(idx));
        // Some browsers refuse the drag with no plain-text payload.
        e.dataTransfer.setData('text/plain', String(idx));
        e.dataTransfer.effectAllowed = 'move';
        onDragStartIdx(idx);
      }}
      onDragEnd={() => {
        // dragend fires whether or not the drop took — clear the
        // ephemeral state either way so a cancelled drag (Escape, drop
        // outside) doesn't leave the indicator stuck.
        onResetDrag();
      }}
      title={`${ariaLabel} (drag to reorder)`}
      aria-label={`Reorder ${ariaLabel}`}
      role="button"
    >
      ⋮⋮
    </div>
  );
};

/** Per-menu-item availability for the {@link AudioTrackOverflowMenu}.
 *  `enabled` drives the disabled prop; `reason` is the tooltip shown
 *  on the disabled item so the user sees why the action is blocked. */
type AudioTrackMenuItemState = {
  enabled: boolean;
  reason: string;
};

/** Compute whether the "Split into drums + backing" item is actionable
 *  for an audio track of the given role. Stage 1 (`stems_all`) only
 *  makes sense on a recording that may contain non-drum content; running
 *  it on an already-isolated drum stem, a drumless backing, or a single
 *  drum piece all produce garbage or noop. `unknown` defaults to enabled
 *  (ad-hoc loads where we couldn't classify; let the user try). */
export function splitFromMixState(role: AudioTrackRole | undefined): AudioTrackMenuItemState {
  switch (role ?? 'unknown') {
    case 'full-mix':
      return { enabled: true, reason: 'Isolate drums and a drumless backing from this recording.' };
    case 'unknown':
      return {
        enabled: true,
        reason: 'Try isolating drums and a drumless backing from this recording.',
      };
    case 'drums':
      return { enabled: false, reason: 'Already drums-only.' };
    case 'no-drums':
      return { enabled: false, reason: 'No drums to split.' };
    case 'drum-piece':
      return { enabled: false, reason: 'Already a single drum piece.' };
  }
}

/** Compute whether the "Split into kick / snare / hi-hat / cymbals" item
 *  is actionable for the given role. Stage 2 (`stems_per`) requires an
 *  already-isolated drum stem; the model was trained on isolated drums
 *  only and produces garbage when fed a full mix. */
export function splitDrumPiecesState(role: AudioTrackRole | undefined): AudioTrackMenuItemState {
  switch (role ?? 'unknown') {
    case 'drums':
      return { enabled: true, reason: 'Split this drum recording into per-instrument pieces.' };
    case 'unknown':
      return {
        enabled: true,
        reason: 'Try splitting this recording into per-instrument drum pieces.',
      };
    case 'full-mix':
      return { enabled: false, reason: 'Isolate drums first.' };
    case 'no-drums':
      return { enabled: false, reason: 'No drums to split.' };
    case 'drum-piece':
      return { enabled: false, reason: 'Already a single drum piece.' };
  }
}

/** Per-row overflow menu on audio tracks. Hosts the two separation
 *  operations (stage 1, stage 2) with enable state derived from the
 *  track's {@link AudioTrackRole}. When both items would be disabled
 *  the trigger button isn't rendered at all; those rows have nothing
 *  actionable and a dead button would just be visual clutter. */
const AudioTrackOverflowMenu = ({
  id,
  role,
  trackLabel,
  onSplitFromMix,
  onSplitDrumPieces,
}: {
  id: AudioTrackId;
  role: AudioTrackRole | undefined;
  trackLabel: string;
  onSplitFromMix: (id: AudioTrackId) => void;
  onSplitDrumPieces: (id: AudioTrackId) => void;
}) => {
  const mixState = splitFromMixState(role);
  const piecesState = splitDrumPiecesState(role);
  if (!mixState.enabled && !piecesState.enabled) return null;
  return (
    <DropdownButton
      label="⋯"
      className={styles.overflowTrigger}
      panelClassName={styles.overflowPanel}
      title={`More actions for ${trackLabel}`}
    >
      {(close) => (
        <>
          <AudioTrackMenuItem
            label="Split into drums + backing"
            state={mixState}
            onClick={() => {
              onSplitFromMix(id);
              close();
            }}
            testId={`audio-track-split-mix-${id}`}
          />
          <AudioTrackMenuItem
            label="Split into kick / snare / hi-hat / cymbals"
            state={piecesState}
            onClick={() => {
              onSplitDrumPieces(id);
              close();
            }}
            testId={`audio-track-split-pieces-${id}`}
          />
        </>
      )}
    </DropdownButton>
  );
};

const AudioTrackMenuItem = ({
  label,
  state,
  onClick,
  testId,
}: {
  label: string;
  state: AudioTrackMenuItemState;
  onClick: () => void;
  testId?: string;
}) => (
  <button
    type="button"
    className={dropdownStyles.dropdownItem}
    role="menuitem"
    disabled={!state.enabled}
    title={state.reason}
    onClick={onClick}
    data-testid={testId}
  >
    {label}
  </button>
);

const AudioTrackRow = observer(
  ({
    id,
    track,
    jot,
    controls,
    onSeek,
    idx,
    dragFromIdx,
    dropTargetIdx,
    onDragStartIdx,
    onDropTargetIdx,
    onMoveTrack,
    onResetDrag,
    groupStart,
    groupEnd,
    inGroup,
    onResizeGutterStart,
  }: {
    id: AudioTrackId;
    track: AudioTrack;
    jot: RenderedJot;
    controls: AudioTrackControls;
    onSeek: (x: number) => void;
  } & MixerRowDragProps) => {
    // Voice-level total beats for the bars-row width (in beats — the
    // row's pixel width is `voiceBeats × --px-per-beat` via CSS calc).
    // Reading the structural cache (not `jot.resolved`) keeps this row
    // stable across zoom changes; pixel width updates via CSS variable
    // on the score root. The waveform canvas reads the zoom-dependent
    // pixel width itself so only IT re-renders on zoom.
    const structureVoice = jot.structure.voices[0];
    // Total quarter-note span of the voice, lead-in lives in
    // `voice.bars` as negative-indexed bars, so summing `bar.beats`
    // covers both the pre-drum and the drum content with one pass.
    let voiceBeats = 0;
    if (structureVoice) {
      for (const b of structureVoice.bars) voiceBeats += b.beats;
    }
    const audible = controls.isAudioTrackAudible(id);
    const muted = controls.mutedAudioTracks.has(id);
    const soloed = controls.soloedAudioTracks.has(id);
    const label = audioTrackLabel(track.filename);
    const lc = `"${track.filename}"`;
    const drop = useMixerRowDropTarget({
      idx,
      dragFromIdx,
      dropTargetIdx,
      onDropTargetIdx,
      onMoveTrack,
      onResetDrag,
    });
    const isDragging = dragFromIdx === idx;
    return (
      <div
        className={classNames(
          styles.musicTrack,
          groupStart && styles.mixerRowGroupStart,
          groupEnd && styles.mixerRowGroupEnd,
          inGroup && styles.mixerRowInGroup,
          isDragging && styles.mixerRowDragging,
          drop.isDropIndicatorAbove && styles.mixerDropIndicatorAbove,
          drop.isDropIndicatorBelow && styles.mixerDropIndicatorBelow
        )}
        data-testid={`audio-track-row-${id}`}
        onDragOver={drop.onDragOver}
        onDragLeave={drop.onDragLeave}
        onDrop={drop.onDrop}
      >
        <div className={styles.musicTrackGutter} style={{ height: AUDIO_TRACK_HEIGHT }}>
          <MixerDragHandle
            idx={idx}
            onDragStartIdx={onDragStartIdx}
            onResetDrag={onResetDrag}
            ariaLabel={`${label} audio track`}
          />
          <GutterResizeHandle onResizeStart={onResizeGutterStart} />
          <div className={styles.musicTrackContent}>
            <div className={styles.musicTrackHeader}>
              <div
                className={classNames(
                  styles.musicTrackLabel,
                  !audible && styles.musicTrackLabelDim
                )}
              >
                <span className={styles.musicTrackName} title={label}>
                  {label}
                </span>
                <span className={styles.musicTrackFile} title={track.filename}>
                  {track.filename}
                </span>
              </div>
              <AudioTrackOverflowMenu
                id={id}
                role={track.role}
                trackLabel={label}
                onSplitFromMix={controls.onSplitFromMix}
                onSplitDrumPieces={controls.onSplitDrumPieces}
              />
            </div>
            <div className={styles.musicTrackButtons}>
              <RowVolumeSlider
                value={controls.volumeFor(id)}
                onChange={(v) => controls.onSetVolume(id, v)}
                label={`${label} audio track`}
              />
              {/* Clear sits before Mute/Solo so the M/S pair stays flush
                  with the gutter's right edge, aligned with the M/S
                  column on the instrument rows below (both gutters share
                  a width). */}
              <ClearButton
                onClear={() => controls.onClear(id)}
                label={`Remove the ${lc} audio track`}
                testId={`audio-track-clear-${id}`}
              />
              <MuteButton
                active={muted}
                onToggle={() => controls.onToggleMute(id)}
                offTitle={`Mute ${lc} audio track`}
                onTitle={`Unmute ${lc} audio track`}
                testId={`audio-track-mute-${id}`}
              />
              <SoloButton
                active={soloed}
                onToggle={() => controls.onToggleSolo(id)}
                offTitle={`Solo ${lc} audio track`}
                onTitle={`Unsolo ${lc} audio track`}
                testId={`audio-track-solo-${id}`}
              />
            </div>
          </div>
        </div>
        <div
          className={styles.musicTrackBarsRow}
          style={
            {
              ['--voice-beats' as string]: voiceBeats,
              height: AUDIO_TRACK_HEIGHT,
            } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          <AudioTrackWaveformCanvas
            jot={jot}
            track={track}
            height={AUDIO_TRACK_HEIGHT}
            dim={!audible}
            testId={`audio-track-waveform-${id}`}
          />
          <Playhead onSeek={onSeek} />
        </div>
      </div>
    );
  }
);

/**
 * One drum-instrument row in the unified mixer — exactly one DSL pitch
 * (kick, snare, hi-hat, …). Mirrors `AudioTrackRow`: same gutter
 * geometry, M/S/volume controls, drag handle, bars-row + barlines +
 * beat dividers; the lane content is this pitch's notes (drawn through
 * `BarView` with `pitches=[pitch]`). The topmost drum row in the mixer
 * (`showBrackets={true}`) also paints the pattern + tuplet brackets so
 * the score chrome stays visible regardless of where the user has
 * dragged the rows.
 *
 * Multi-voice jots: pitches can belong to any voice (e.g. kick lives in
 * the "Feet" voice). The bar geometry is taken from voice[0] (every voice
 * shares the same bar grid), and per-bar tracks are looked up across all
 * voices for this pitch — so the row works whether the pitch lives in
 * voice 0 or 1.
 */
const PitchRow = observer(
  ({
    pitch,
    jot,
    config,
    showBrackets,
    pitchOrder,
    highlightedPattern,
    onPatternClick,
    onSeek,
    voiceControls,
    idx,
    dragFromIdx,
    dropTargetIdx,
    onDragStartIdx,
    onDropTargetIdx,
    onMoveTrack,
    onResetDrag,
    groupStart,
    groupEnd,
    inGroup,
    onResizeGutterStart,
  }: {
    pitch: string;
    jot: RenderedJot;
    config: ViewConfig;
    showBrackets: boolean;
    pitchOrder: readonly string[];
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    onSeek: (x: number) => void;
    voiceControls: VoiceControls;
  } & MixerRowDragProps) => {
    const structure = jot.structure;
    const voice0 = structure.voices[0];
    if (!voice0) return null;
    const trackHeight = config.trackHeight as number;
    // Look up the first instrument and color found for this pitch
    // across all voices, so the gutter label is correct even when the
    // pitch lives in voice[1] (e.g. kick under the "Feet" voice).
    let instrumentName: string | undefined;
    for (const v of structure.voices) {
      if (instrumentName) break;
      for (const bar of v.bars) {
        const t = bar.tracks[pitch];
        if (t?.instrument.name) {
          instrumentName = t.instrument.name;
          break;
        }
      }
    }
    // Replace voice[0]'s per-bar tracks with this pitch's track wherever
    // it appears across the jot's voices. Bar geometry (time, beats,
    // patternSpans, tupletSpans) is untouched — only `tracks` changes
    // — so BarView reads the same beat-coord layout as before. Reading
    // the structural cache (not `jot.resolved`) keeps these bar refs
    // stable across zoom changes; the surrounding container's
    // `--px-per-beat` CSS variable does the actual rescaling.
    const bars: StructuralBar[] = voice0.bars.map((b, i) => {
      let track = b.tracks[pitch];
      if (!track) {
        for (let v = 1; v < structure.voices.length; v++) {
          const t = structure.voices[v].bars[i]?.tracks[pitch];
          if (t) {
            track = t;
            break;
          }
        }
      }
      return { ...b, tracks: track ? { [pitch]: track } : {} };
    });
    // Voice-level totals for the bars-row width (in beats, the row's
    // pixel width is `voiceBeats × --px-per-beat` via CSS calc).
    // Lead-in lives in `voice.bars` as negative-indexed bars so a single
    // sum over `bar.beats` covers both pre-drum and drum content.
    let voiceBeats = 0;
    for (const b of voice0.bars) voiceBeats += b.beats;
    // Cumulative quarter-note span of the leading negative-indexed bars
    //; used to size the centered "lead-in" label overlay across them.
    let leadInBarsBeats = 0;
    for (const b of voice0.bars) {
      if (b.index >= 0) break;
      leadInBarsBeats += b.beats;
    }

    // Filtered-onset ghost overlays (debug bundle + checkbox gated).
    // Resolve once per row so the per-entry render below is just a map.
    const provenance = React.useContext(NoteProvenanceContext);
    const showFiltered = provenance?.showFiltered ?? false;
    const rejectedForPitch = showFiltered ? (provenance!.rejectedByPitch.get(pitch) ?? []) : [];
    // Cumulative beat offsets so each rejected entry can be positioned
    // absolutely in the bars row without walking back through bar
    // widths on every render. Same scale (quarter-note beats) as the
    // CSS-var positioning the kept notes use.
    const barBeatStart: number[] = [];
    {
      let acc = 0;
      for (let i = 0; i < bars.length; i++) {
        barBeatStart.push(acc);
        acc += bars[i].beats;
      }
    }
    // Pitch's lane colour for the ghost dashed outline. Best-effort
    // lookup: a pitch with no kept notes has no `tracks[pitch]` entry
    // in any bar — falls back to neutral grey then.
    let pitchColor = 'var(--color-text-faint-strong)';
    for (const b of bars) {
      const t = b.tracks[pitch];
      if (t?.color) {
        pitchColor = t.color;
        break;
      }
    }

    const audible = voiceControls.isPitchAudible(pitch);
    const muted = voiceControls.mutedPitches.has(pitch);
    const soloed = voiceControls.soloedPitches.has(pitch);
    const drop = useMixerRowDropTarget({
      idx,
      dragFromIdx,
      dropTargetIdx,
      onDropTargetIdx,
      onMoveTrack,
      onResetDrag,
    });
    const isDragging = dragFromIdx === idx;
    const labelText = instrumentName ?? `Pitch ${pitch}`;
    return (
      <div
        className={classNames(
          styles.pitchRow,
          groupStart && styles.mixerRowGroupStart,
          groupEnd && styles.mixerRowGroupEnd,
          inGroup && styles.mixerRowInGroup,
          isDragging && styles.mixerRowDragging,
          drop.isDropIndicatorAbove && styles.mixerDropIndicatorAbove,
          drop.isDropIndicatorBelow && styles.mixerDropIndicatorBelow
        )}
        data-testid={`pitch-row-${pitch}`}
        onDragOver={drop.onDragOver}
        onDragLeave={drop.onDragLeave}
        onDrop={drop.onDrop}
      >
        <div className={styles.pitchRowGutter}>
          <MixerDragHandle
            idx={idx}
            onDragStartIdx={onDragStartIdx}
            onResetDrag={onResetDrag}
            ariaLabel={labelText}
          />
          <GutterResizeHandle onResizeStart={onResizeGutterStart} />
          {/* Two-row stack: the label gets the full gutter width on top
              so long instrument names breathe instead of getting clipped
              with `…`; the slider + M/S sit on a second line below. */}
          <div className={styles.pitchRowContent}>
            <div
              className={classNames(styles.pitchRowLabel, !audible && styles.musicTrackLabelDim)}
              title={instrumentName ? `${instrumentName} (pitch ${pitch})` : `Pitch ${pitch}`}
            >
              <span className={styles.gutterPitch}>{pitch}</span>
              {instrumentName && <span className={styles.pitchRowName}>{instrumentName}</span>}
            </div>
            <div className={styles.pitchRowControls}>
              <RowVolumeSlider
                value={voiceControls.volumeFor(pitch)}
                onChange={(v) => voiceControls.onSetVolume(pitch, v)}
                label={labelText}
              />
              {/* Reserves the Clear-button slot so the slider width
                  matches the audio-track row (which has X/M/S, vs M/S
                  here). aria-hidden + non-interactive — purely layout. */}
              <span className={styles.controlSpacer} aria-hidden="true" />
              <MuteButton
                active={muted}
                onToggle={() => voiceControls.onToggleMute(pitch)}
                offTitle={`Mute ${pitch}`}
                onTitle={`Unmute ${pitch}`}
              />
              <SoloButton
                active={soloed}
                onToggle={() => voiceControls.onToggleSolo(pitch)}
                offTitle={`Solo ${pitch}`}
                onTitle={`Unsolo ${pitch}`}
              />
            </div>
          </div>
        </div>
        <div
          className={styles.barsRow}
          data-bars-row
          style={
            {
              ['--voice-beats' as string]: voiceBeats,
            } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          {/* Lead-in label overlay floating across the negative-indexed
              bars. The bars themselves carry the hatched background
              (`.barLeadIn` / `.barLeadInLast`); this overlay just adds
              the centered "lead-in" caption. Topmost-row only
              (`showBrackets`) so the label doesn't repeat on every
              pitch row. */}
          {leadInBarsBeats > 0 && showBrackets && (
            <div
              className={styles.leadInOverlay}
              style={
                {
                  ['--lead-in-bars-beats' as string]: leadInBarsBeats,
                } as React.CSSProperties
              }
            >
              <span className={styles.leadInLabel}>lead-in</span>
            </div>
          )}
          {(() => {
            // Cumulative quarter-note position of each bar's left edge
            // within the voice. Drives the bar's absolute left via
            // `--bar-start-beat`; see `.bar` in score.module.css.
            const startBeats = new Array<number>(bars.length);
            let cursor = 0;
            for (let i = 0; i < bars.length; i++) {
              startBeats[i] = cursor;
              cursor += bars[i].beats;
            }
            return bars.map((bar, i) => (
              <BarView
                key={i}
                bar={bar}
                barStartBeat={startBeats[i]}
                pitches={[pitch]}
                config={config}
                isAnacrusis={bar.index === 0}
                highlightedPattern={highlightedPattern}
                onPatternClick={onPatternClick}
                isPitchAudible={voiceControls.isPitchAudible}
                showBrackets={showBrackets}
                rowPitch={pitch}
                pitchOrder={pitchOrder}
              />
            ));
          })()}
          {rejectedForPitch.map((entry, i) => {
            // The MIDI lays `leadBars` empty bar-0-sized blocks before
            // struct bar 0, so the struct bar index maps to the
            // rendered jot's bars array as `leadBars + entry.bar`.
            // Out-of-range entries are already filtered out upstream.
            const barIdx = provenance!.leadBars + entry.bar;
            if (barIdx < 0 || barIdx >= bars.length) return null;
            // beat_in_bar is 1-indexed in the provenance (per the
            // transcriber's OnsetCandidate convention); the CSS calc
            // expects a 0-indexed beat offset within the bar.
            const beatInBar = Math.max(0, entry.beat_in_bar - 1);
            const beatOffset = barBeatStart[barIdx] + beatInBar;
            return (
              <FilteredOnsetView
                key={`f-${entry.bar}-${i}-${entry.detected_time_sec}`}
                entry={entry}
                beatOffset={beatOffset}
                color={pitchColor}
                trackHeight={trackHeight as number}
              />
            );
          })}
          <Playhead onSeek={onSeek} />
        </div>
      </div>
    );
  }
);

/**
 * Tiled waveform row for one audio track, aligned to the score's bar
 * timeline. The row is split into `BEATS_PER_CHUNK`-beat windows;
 * each window renders as its own absolutely-positioned canvas tile
 * (see {@link AudioTrackWaveformChunk}). Tiling buys:
 *
 *  1. **Unbounded effective resolution.** Each tile picks its own
 *     backing-store size, so the cross-browser 16 384 px per-axis
 *     canvas cap no longer rolls a long or zoomed track off into
 *     lower-resolution rendering. `BEATS_PER_CHUNK = 4` (see
 *     `waveform_chunks.ts`) keeps the worst-case backing well under
 *     the cap (max zoom × max densityFactor × DPR 3 ≈ 8 600 px).
 *  2. **Stable chunk identity across zoom.** Chunks key on a beat-
 *     aligned bucket index, so a zoom change only resizes existing
 *     chunks via JS-recomputed `left` / `width`; React never unmounts
 *     / remounts them. No bucket-transition churn, no stretched-
 *     bitmap holdover gap.
 *  3. **Parent-driven visibility (no `IntersectionObserver`).** This
 *     component is an `observer()` that reads `scrollX` +
 *     `_viewportWidth` from `JotViewStore` (no DOM layout reads, see
 *     AGENTS.md §5.9) and only mounts the chunks whose CSS box
 *     currently intersects the viewport (plus a prefetch margin).
 *     Off-screen chunks unmount cleanly; the worker keeps the PCM,
 *     so re-entering the viewport draws fresh from stored peaks in
 *     ~5 ms.
 *
 * Chunk layout is memoised on `jot`, so scroll / zoom re-renders of
 * this observer only walk the filtered visibility check, not the
 * structure.
 */

/**
 * Score-px margin around the visible viewport that still counts as
 * "in viewport" for chunk-mount purposes. Generous enough that a
 * moderate horizontal scroll never reveals a blank tile before its
 * draw completes (one chunk at typical zoom ≈ 4 beats × 112 px/beat
 * ≈ 450 score-px, so a 1200 px margin covers ~2-3 chunks of lookahead
 * on either side of the visible range).
 */
const CHUNK_VIEWPORT_MARGIN_PX = 1200;

const AudioTrackWaveformCanvas = observer(
  ({
    jot,
    track,
    height,
    dim,
    testId,
  }: {
    jot: RenderedJot;
    track: AudioTrack;
    height: number;
    dim: boolean;
    testId?: string;
  }) => {
    const store = React.useContext(JotViewStoreContext);
    const uniformWaveforms = React.useContext(UniformWaveformsContext);
    const padBeats = React.useContext(RenderedJotContext)?.config.barNotePaddingBeats ?? 0.125;
    // Per-track stem-colour resolution: same logic the legacy single
    // canvas used; hoisted to the parent so every chunk reuses the
    // result instead of re-walking the structure on each mount.
    let pitchColor: string | undefined;
    if (track.pitch) {
      outer: for (const v of jot.structure.voices) {
        for (const b of v.bars) {
          const t = b.tracks[track.pitch];
          if (t?.color) {
            pitchColor = t.color;
            break outer;
          }
        }
      }
    }
    // Beat-stable chunk layout (zoom-invariant). Memoed on `jot` so
    // scroll / zoom re-renders of this observer don't rebuild it.
    const layout = React.useMemo(() => buildChunkLayout(jot), [jot]);
    const livePxPerBeat = useLiveJotPxPerBeat();

    if (!store || layout.chunks.length === 0) return null;

    // Visibility: derive the score-px x-range currently on screen
    // from `JotViewStore` observables. The score uses a virtualised
    // scroll model (`.scrollViewport` translated by `(-scrollX, 0)`),
    // so `[scrollX, scrollX + viewportWidth]` is exactly the score-px
    // window the user sees. Each chunk's score-px left mirrors the
    // formula the chunk component below uses for its inline `left`
    // (`chunk.startBeat * livePxPerBeat + padBeats * livePxPerBeat`);
    // any chunk whose box intersects the viewport plus prefetch
    // margin is mounted, anything else is unmounted.
    const scrollX = store.scrollX;
    const viewportWidth = store._viewportWidth;
    if (viewportWidth <= 0 || livePxPerBeat <= 0) return null;
    const visibleLeft = scrollX - CHUNK_VIEWPORT_MARGIN_PX;
    const visibleRight = scrollX + viewportWidth + CHUNK_VIEWPORT_MARGIN_PX;
    const padPx = padBeats * livePxPerBeat;

    const visibleChunks: WaveformChunk[] = [];
    for (const c of layout.chunks) {
      const left = c.startBeat * livePxPerBeat + padPx;
      const right = left + c.totalBeats * livePxPerBeat;
      if (right > visibleLeft && left < visibleRight) visibleChunks.push(c);
    }
    if (visibleChunks.length === 0) return null;

    // Per-track amplitude scale for uniform mode (resolved once at
    // track registration, identical for every chunk of this track,
    // so neighbouring chunks render at the same vertical scale and no
    // amplitude seam shows at the chunk boundary).
    const ampScale = uniformWaveforms ? waveformWorker.getAmpScale(track.id) : 1;

    return (
      <>
        {visibleChunks.map((chunk, i) => (
          <AudioTrackWaveformChunk
            key={chunk.key}
            track={track}
            chunk={chunk}
            bars={layout.bars}
            height={height}
            dim={dim}
            pitchColor={pitchColor}
            ampScale={ampScale}
            testId={i === 0 ? testId : undefined}
          />
        ))}
      </>
    );
  }
);

/**
 * One tile in the tiled waveform row. Owns the `<canvas>` and its
 * rasterised bitmap (sized to `chunk.totalBeats × livePxPerBeat`,
 * snapped to integer CSS px).
 *
 * Visibility is decided by the parent
 * (`AudioTrackWaveformCanvas`), which only mounts chunks intersecting
 * the viewport; so mount = visible, and there's no
 * `IntersectionObserver` round-trip to wait through. The first render
 * after mount draws immediately so a newly-visible chunk paints on
 * the same frame as the parent's visibility decision; subsequent
 * renders triggered by zoom / `drumsT0Sec` / etc. rAF-coalesce so a
 * sustained wheel-zoom gesture triggers at most one worker call per
 * displayed frame.
 */
const AudioTrackWaveformChunk = observer(
  ({
    track,
    chunk,
    bars,
    height,
    dim,
    pitchColor,
    ampScale,
    testId,
  }: {
    track: AudioTrack;
    chunk: WaveformChunk;
    bars: BarBeat[];
    height: number;
    dim: boolean;
    pitchColor: string | undefined;
    ampScale: number;
    testId?: string;
  }) => {
    const canvasRef = React.useRef<HTMLCanvasElement>(null);
    // Live drum↔audio offset; chunks re-render (and so re-rasterise
    // on the next rAF) when the user nudges the Offset control.
    const drumsT0Sec = jotPlayer.drumsT0Sec;
    const livePxPerBeat = useLiveJotPxPerBeat();
    const padBeats = React.useContext(RenderedJotContext)?.config.barNotePaddingBeats ?? 0.125;
    // Globally-unique worker-side slot identifier for this tile.
    // `chunk.key` alone collides across audio tracks (it's
    // `startBeat / BEATS_PER_CHUNK`, defined per-voice); prefixing
    // with `track.id` (a string per audio track) disambiguates.
    const chunkKey = `${track.id}:${chunk.key}`;

    // Snap the chunk's CSS left / width to integer CSS pixels in JS so
    // the canvas's backing-store width (= cssWidth × dpr) and the peak
    // buffer length (= 2 × cssWidth) are both whole integers, and so
    // adjacent chunks share an *exactly* aligned boundary (chunk N+1's
    // left = chunk N's right by construction, no asymmetric rounding
    // gap or overlap). Without this, each chunk's CSS width came from
    // `round(right_edge) - round(left_edge)` in CSS, and the two edges
    // would round in different directions for adjacent chunks (one to
    // -0.5, one to +0.5), leaving each chunk's canvas bitmap stretched
    // by a slightly different ratio, which renders as a visible
    // brightness / density step at the chunk boundary. Same snapped
    // width feeds the canvas backing-store, the inline CSS width, and
    // the peak buffer length so all three agree to the pixel.
    const chunkLayout = React.useMemo(() => {
      const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
      const padPx = Math.round(padBeats * livePxPerBeat * dpr) / dpr;
      const leftRaw = chunk.startBeat * livePxPerBeat + padPx;
      const rightRaw = leftRaw + chunk.totalBeats * livePxPerBeat;
      const left = Math.round(leftRaw);
      const right = Math.round(rightRaw);
      return { left, width: Math.max(0, right - left) };
    }, [chunk.startBeat, chunk.totalBeats, livePxPerBeat, padBeats]);

    // Transfer control of the `<canvas>` to the worker once on mount;
    // release on unmount. After this point the main thread can no
    // longer draw into the canvas (any attempt throws); the worker
    // owns the bitmap, sized via `canvas.width` / `canvas.height` set
    // on its `OffscreenCanvas` handle inside `renderChunk`. CSS box
    // dimensions are still controlled here via inline `style.left` /
    // `style.width` (CSS properties of the `<canvas>` element are
    // separate from the backing bitmap).
    React.useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      if (typeof canvas.transferControlToOffscreen !== 'function') {
        console.warn(
          '[mixer] OffscreenCanvas not supported; waveform chunk will not render',
        );
        return;
      }
      const offscreen = canvas.transferControlToOffscreen();
      waveformWorker.attachChunk(chunkKey, offscreen, track.id);
      return () => {
        waveformWorker.releaseChunk(chunkKey);
      };
    }, [chunkKey, track.id]);

    // First render after mount paints immediately so a newly-visible
    // chunk shows up on the same frame as the parent's visibility
    // decision; subsequent paints (zoom tick, drumsT0Sec change, etc.)
    // rAF-coalesce so a sustained wheel-zoom gesture triggers at most
    // one worker call per displayed frame. The paint itself is
    // fire-and-forget: the worker computes peaks and paints into the
    // chunk's `OffscreenCanvas` directly, no bytes cross back to the
    // main thread.
    const isFirstDrawRef = React.useRef(true);

    React.useEffect(() => {
      if (chunk.totalBeats <= 0 || livePxPerBeat <= 0) return;
      const widthPx = chunkLayout.width;
      if (widthPx <= 0) return;
      const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
      // `BEATS_PER_CHUNK` is sized so the worst-case backing
      // dimensions stay well under the 16 384 px cross-browser
      // canvas cap; this clamp is defensive only and shouldn't fire
      // in normal use.
      const MAX_CANVAS_DIM = 16384;
      // `widthPx` is integer (from `chunkLayout.width`) so
      // `widthPx * dpr` is also integer, no `Math.floor` needed, and
      // the backing-store width is exactly `cssWidth × dpr`, so
      // `image-rendering: pixelated` displays the bitmap 1:1 with no
      // stretch.
      const backingW = Math.min(Math.max(1, widthPx * dpr), MAX_CANVAS_DIM);
      const backingH = Math.min(Math.max(1, Math.floor(height * dpr)), MAX_CANVAS_DIM);
      // Effective per-beat scale for this bitmap. Differs from
      // `livePxPerBeat` by at most ~0.5 CSS px / chunk.totalBeats
      // (because `widthPx` is rounded to integer above); using the
      // bitmap's actual per-beat ratio for the bar slice mapping
      // keeps each column's audio time aligned to the chunk's CSS
      // box, so a transient at beat B in the source audio lands
      // exactly under beat B in the snapped chunk geometry.
      const renderedScale = widthPx / chunk.totalBeats;
      // Bar slices in chunk-local pixel coordinates: bars to the left
      // of the chunk get a negative `x`, bars to the right get `x >=
      // widthPx`; the worker's clamp drops both groups without an
      // explicit filter on our side.
      const barSlices: BarSlice[] = bars.map((b) => ({
        x: (b.startBeat - chunk.startBeat) * renderedScale,
        width: b.beats * renderedScale,
        startSec: b.startSec,
        durationSec: b.durationSec,
      }));
      const fire = () => {
        waveformWorker.renderChunk(
          chunkKey,
          barSlices,
          widthPx,
          height,
          backingW,
          backingH,
          drumsT0Sec,
          pitchColor ?? '#5BA8E8',
          ampScale,
        );
      };
      if (isFirstDrawRef.current) {
        isFirstDrawRef.current = false;
        fire();
        return;
      }
      const id = requestAnimationFrame(fire);
      return () => cancelAnimationFrame(id);
    }, [
      chunkKey,
      chunk,
      bars,
      height,
      drumsT0Sec,
      livePxPerBeat,
      pitchColor,
      ampScale,
      chunkLayout.width,
    ]);

    // Canvas `left` / `width` come from `chunkLayout` (JS-snapped to
    // integer CSS px). Inline styles override CSS, and we need the
    // canvas's CSS box to match the bitmap's pixel count
    // (`widthPx * dpr` backing) exactly so adjacent chunks share a
    // pixel-perfect boundary and `image-rendering: pixelated`
    // displays the bitmap 1:1. During the one-rAF gap between a zoom
    // event and the next rasterisation, the chunk's CSS width grows
    // with `livePxPerBeat` while the bitmap's intrinsic size is
    // still at the previous scale; the canvas element scales the
    // bitmap nearest-neighbour (via image-rendering: pixelated)
    // until the rAF redraw catches up.
    return (
      <canvas
        ref={canvasRef}
        className={classNames(styles.musicTrackWaveformChunk, dim && styles.musicTrackWaveformDim)}
        style={
          {
            height,
            left: `${chunkLayout.left}px`,
            width: `${chunkLayout.width}px`,
          } as React.CSSProperties
        }
        data-testid={testId}
      />
    );
  }
);

/**
 * Read the live `pxPerBeat` off the active jot via the MobX scope
 * already used by surrounding `observer`s. Used by chunk draws to
 * sample the current zoom at rasterisation time without forcing the
 * chunk's render path to take a `jot` prop (the bars/structure don't
 * change with zoom; passing the whole jot just to read this one
 * field would dirty the chunks on every wheel tick).
 */
function useLiveJotPxPerBeat(): number {
  // `RenderedJotContext` is provided at the JotView root; null only
  // outside the View (tests). In that case fall back to 1, which is
  // safe (chunks just don't render).
  const jot = React.useContext(RenderedJotContext);
  return jot?.pxPerBeat ?? 1;
}

/**
 * A per-section master fader that sits in the sticky lane gutter,
 * directly above the section it controls (the loaded audio tracks, or
 * the drum/instrument staff). Gutter-aligned (same sticky column width
 * — `--gutter-width` — as the per-row M/S/volume controls below it) so
 * it reads as the "header" for that column. Reads/writes the global
 * observable `jotPlayer`; all pointer events are kept from bubbling so
 * dragging the fader doesn't start the page marquee or trip
 * seek-on-click.
 */
const GutterMasterRow = observer(
  ({
    label,
    title,
    value,
    onChange,
    muted,
    soloed,
    audible,
    onToggleMute,
    onToggleSolo,
    testId,
    onResizeGutterStart,
  }: {
    label: string;
    title: string;
    value: number;
    onChange: (v: number) => void;
    muted: boolean;
    soloed: boolean;
    /** True when the section's bus would currently make sound (master
     * mute / cross-domain solo can drop it). Dims the row to match the
     * per-row label-dim treatment when the section is silent. */
    audible: boolean;
    onToggleMute: () => void;
    onToggleSolo: () => void;
    testId?: string;
    onResizeGutterStart: (e: React.PointerEvent<HTMLDivElement>) => void;
  }) => {
    const stop = (e: React.MouseEvent) => e.stopPropagation();
    const pct = Math.round(value * 100);
    return (
      <div className={styles.gutterMasterRow}>
        <div className={styles.gutterMasterGutter} title={title} data-testid={testId}>
          <span
            className={classNames(styles.gutterMasterLabel, !audible && styles.musicTrackLabelDim)}
          >
            {label}
          </span>
          <div className={styles.gutterMasterControls}>
            <input
              type="range"
              className={styles.gutterMasterSlider}
              min={0}
              max={1}
              step={VOLUME_STEP}
              value={value}
              onChange={(e) => onChange(parseFloat(e.target.value))}
              onClick={stop}
              onMouseDown={stop}
              onMouseUp={stop}
              aria-label={`${label} volume`}
              title={`${label} volume: ${pct}%`}
              style={{ ['--value' as string]: value } as React.CSSProperties}
            />
            <span className={styles.gutterMasterValue}>{pct}%</span>
            <MuteButton
              active={muted}
              onToggle={onToggleMute}
              offTitle={`Mute ${label}`}
              onTitle={`Unmute ${label}`}
              testId={testId ? `${testId}-mute` : undefined}
            />
            <SoloButton
              active={soloed}
              onToggle={onToggleSolo}
              offTitle={`Solo ${label}`}
              onTitle={`Unsolo ${label}`}
              testId={testId ? `${testId}-solo` : undefined}
            />
          </div>
          <GutterResizeHandle onResizeStart={onResizeGutterStart} />
        </div>
      </div>
    );
  }
);

/**
 * Compact horizontal volume fader shared by the pitch gutter and the
 * audio-track gutter. Range is 0..1 (pure attenuation). All mouse events are
 * kept from bubbling so dragging the fader doesn't start the page-level
 * marquee selection or trip the seek-on-click handler.
 */
const RowVolumeSlider = ({
  value,
  onChange,
  label,
}: {
  value: number;
  onChange: (v: number) => void;
  label: string;
}) => {
  const stop = (e: React.MouseEvent) => e.stopPropagation();
  return (
    <input
      type="range"
      className={styles.rowVolume}
      min={0}
      max={1}
      step={VOLUME_STEP}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      onClick={stop}
      onMouseDown={stop}
      onMouseUp={stop}
      title={`${label} volume: ${Math.round(value * 100)}%`}
      aria-label={`${label} volume`}
      style={{ ['--value' as string]: value } as React.CSSProperties}
    />
  );
};
