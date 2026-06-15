import { observer } from 'mobx-react-lite';
import React from 'react';
import { ViewConfig } from 'src/editing/viewport/view_config';
import { jotPlayer } from 'src/editing/playback/player';
import { MixerStoreContext } from './mixer_contexts';
import { LyricsRow } from '../lyrics/lyrics_row';
import styles from './mixer.module.css';
import { TrackKey } from 'src/editing/tracks/tracks';
import { GutterMasterRow } from './gutter_controls';
import { MixerEndDropZone } from './mixer_drag';
import { InstrumentRow } from './instrument_row';

import type { VoiceControls, AudioTrackControls } from './mixer_controls';
// Re-exported so existing `from '.../mixer/mixer'` importers (jot_editor) keep
// working; the definitions live in the leaf `mixer_controls.ts`.
import { AudioTrackRow } from './audio_track_row';
export type { VoiceControls, AudioTrackControls };




/**
 * The unified mixer that replaced the old separate "audio tracks" and
 * "voice staves" sections. Renders the two section masters at the top,
 * then one row per entry in `trackOrder` — an audio track or a single
 * drum-instrument pitch, freely interleavable. Drag-and-drop on each
 * row's gutter handle rewrites the order via `JotEditorStore.moveTrack`;
 * the topmost instrument row hosts the pattern/tuplet bracket overlay
 * so they read as a single piece of score chrome regardless of where
 * the user has moved the rows.
 */
export const MixerView = observer(
  ({
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

    // The topmost instrument row in the user's mixer order hosts the
    // tuplet brackets and the lead-in label; chrome that belongs to
    // the score as a whole, not to any one instrument. Both this index
    // and the mixer-ordered pitch list are MobX computeds on the store,
    // so MixerView (already an observer) gets correct invalidation when
    // the user reorders rows.
    const mixer = React.useContext(MixerStoreContext);
    const firstInstrumentIdx = mixer?.firstInstrumentIdx ?? -1;
    // Drum pitches in mixer order. Pattern brackets are drawn on every
    // row whose pitch participates in the pattern; this list lets each
    // row know whether it's the topmost / bottommost participant for a
    // given span so the bracket reads as one continuous outline across
    // rows (with non-participating rows in between visually skipped).
    const pitchOrder: readonly string[] = mixer?.pitchOrder ?? [];

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
              : key.kind === 'instrument'
                ? `instrument:${key.pitch}`
                : `lyrics:${key.id}`;
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
            // The reaction in JotEditorStore drops dead audio ids on the
            // same MobX tick, so this gap is one-frame at most. Render
            // nothing rather than crash if the maps race.
            if (!track) return null;
            return (
              <AudioTrackRow
                key={reactKey}
                id={key.id}
                track={track}
                controls={audioTrackControls}
                onSeek={onSeek}
                {...rowProps}
              />
            );
          }
          if (key.kind === 'lyrics') {
            return <LyricsRow key={reactKey} id={key.id} onSeek={onSeek} {...rowProps} />;
          }
          return (
            <InstrumentRow
              key={reactKey}
              pitch={key.pitch}
              config={config}
              showBrackets={idx === firstInstrumentIdx}
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
