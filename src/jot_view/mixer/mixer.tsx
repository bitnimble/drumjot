import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { RenderedJot, StructuralBar, ViewConfig } from 'src/jot';
import { AudioTrack, AudioTrackId, jotPlayer } from 'src/playback';
import { waveformWorker, BarSlice } from 'src/playback/waveform_worker_client';
import { InstrumentTrack } from 'src/tracks';
import { BarBeat, WaveformChunk, buildChunkLayout } from './waveform_chunks';
import { GutterResizeHandle } from '../components/gutter_resize_handle';
import { MuteButton, SoloButton } from '../components/icon_button';
import { MixerStoreContext, NoteProvenanceContext, RenderedJotContext, UniformWaveformsContext, ViewportStoreContext } from '../contexts';
import { LyricsRow } from '../lyrics/lyrics_row';
import styles from './mixer.module.css';
import { Playhead } from '../playback/playhead';
import { seekFromClick } from '../score/seek';
import { FilteredOnsetView } from '../score/filtered_onset_view';
import { BarView } from '../score/bar_view';
import { TrackKey } from '../store';
import { ViewportStore } from '../viewport/viewport_store';
import { barsRowWidthSeed, intersectsBeatRange } from '../utils/windowing';
import { AudioTrackOverflowMenu, InstrumentRowOverflowMenu } from './overflow_menus';
import { GutterMasterRow, RowVolumeSlider } from './gutter_controls';
import { MixerRowDragProps, useMixerRowDropTarget, MixerEndDropZone, MixerDragHandle } from './mixer_drag';

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
  /** Drop a loaded audio track (exposed in the row's overflow menu). */
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


/**
 * The unified mixer that replaced the old separate "audio tracks" and
 * "voice staves" sections. Renders the two section masters at the top,
 * then one row per entry in `trackOrder` — an audio track or a single
 * drum-instrument pitch, freely interleavable. Drag-and-drop on each
 * row's gutter handle rewrites the order via `JotViewStore.moveTrack`;
 * the topmost instrument row hosts the pattern/tuplet bracket overlay
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
            return <LyricsRow key={reactKey} id={key.id} jot={jot} onSeek={onSeek} {...rowProps} />;
          }
          return (
            <InstrumentRow
              key={reactKey}
              pitch={key.pitch}
              jot={jot}
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
    // Voice-level total beats for the bars-row width (in beats, the
    // row's pixel width is `voiceBeats × --px-per-beat` via CSS calc).
    // `jot.voiceBeats` reads off the structural cache (not
    // `jot.resolved`) so the value is stable across zoom changes; pixel
    // width updates via CSS variable on the score root. The waveform
    // canvas reads the zoom-dependent pixel width itself so only IT
    // re-renders on zoom.
    const voiceBeats = jot.voiceBeats;
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
    const mixer = React.useContext(MixerStoreContext);
    const splitStatus = mixer?.audioTrackSplitStatuses.get(id);
    const splittingTitle =
      splitStatus?.kind === 'mix'
        ? 'Splitting into drums + backing…'
        : splitStatus?.kind === 'pieces'
          ? 'Splitting into per-instrument pieces…'
          : undefined;
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
                <span className={styles.musicTrackFileRow}>
                  <span className={styles.musicTrackFile} title={track.filename}>
                    {track.filename}
                  </span>
                  {splitStatus && (
                    <span
                      className={styles.musicTrackSplitSpinner}
                      title={splittingTitle}
                      aria-label={splittingTitle}
                      role="status"
                      data-testid={`audio-track-split-spinner-${id}`}
                    />
                  )}
                </span>
              </div>
              <AudioTrackOverflowMenu
                track={track}
                trackLabel={label}
                onSplitFromMix={controls.onSplitFromMix}
                onSplitDrumPieces={controls.onSplitDrumPieces}
                onClear={controls.onClear}
              />
            </div>
            <div className={styles.musicTrackButtons}>
              <RowVolumeSlider
                value={controls.volumeFor(id)}
                onChange={(v) => controls.onSetVolume(id, v)}
                label={`${label} audio track`}
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
          data-bars-row
          style={
            {
              ['--voice-beats' as string]: voiceBeats,
              ['--bars-row-width' as string]: barsRowWidthSeed(jot, voiceBeats),
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
 * One drum-instrument row in the unified mixer; exactly one DSL pitch
 * (kick, snare, hi-hat, …). Mirrors `AudioTrackRow`: same gutter
 * geometry, M/S/volume controls, drag handle, bars-row + barlines +
 * beat dividers; the lane content is this pitch's notes (drawn through
 * `BarView` with `pitches=[pitch]`). The topmost instrument row in the
 * mixer (`showBrackets={true}`) also paints the pattern + tuplet
 * brackets so the score chrome stays visible regardless of where the
 * user has dragged the rows.
 *
 * Multi-voice jots: pitches can belong to any voice (e.g. kick lives in
 * the "Feet" voice). The bar geometry is taken from voice[0] (every voice
 * shares the same bar grid), and per-bar tracks are looked up across all
 * voices for this pitch, so the row works whether the pitch lives in
 * voice 0 or 1.
 */
/**
 * The windowed bar list for one instrument row. Split out of
 * {@link InstrumentRow} so the only thing that re-renders on a scroll /
 * zoom tick is this bar map, the row gutter (label, fader, M/S, overflow
 * menu) reads no scroll observable and stays put. Mirrors the
 * waveform-chunk visibility pattern ({@link AudioTrackWaveformCanvas}):
 * read the visible beat window from the store and render only the bars
 * whose span intersects it (plus the buffer baked into
 * `visibleBeatRange`).
 *
 * Bars key on the clone-stable `bar.index` (not the array position) so
 * the window sliding by one bar reuses every surviving bar's DOM instead
 * of re-keying the whole list. The per-bar props handed to {@link
 * BarView} are referentially stable across scroll (the caller memoises
 * `pitches` / `colorForPitch`), so `BarView`'s `observer` memo holds and
 * an unchanged visible bar pays nothing on a scroll tick that doesn't
 * move the window, only newly-revealed bars mount.
 */
const WindowedBarList = observer(function WindowedBarList({
  viewport,
  pitchBars,
  startBeats,
  pitch,
  config,
  showBrackets,
  pitchOrder,
  highlightedPattern,
  onPatternClick,
  isPitchAudible,
  pitches,
  colorForPitch,
}: {
  viewport: ViewportStore | null;
  pitchBars: readonly StructuralBar[];
  startBeats: readonly number[];
  pitch: string;
  config: ViewConfig;
  showBrackets: boolean;
  pitchOrder: readonly string[];
  highlightedPattern: string | undefined;
  onPatternClick: (name: string) => void;
  isPitchAudible: (pitch: string) => boolean;
  pitches: string[];
  colorForPitch: (pitch: string) => string | undefined;
}) {
  const range = viewport?.visibleBeatRange ?? null;
  return (
    <>
      {pitchBars.map((bar, i) => {
        const startBeat = startBeats[i];
        if (!intersectsBeatRange(range, startBeat, bar.beats)) return null;
        return (
          <BarView
            key={bar.index}
            bar={bar}
            barStartBeat={startBeat}
            pitches={pitches}
            config={config}
            isAnacrusis={bar.index === 0}
            highlightedPattern={highlightedPattern}
            onPatternClick={onPatternClick}
            isPitchAudible={isPitchAudible}
            showBrackets={showBrackets}
            rowPitch={pitch}
            pitchOrder={pitchOrder}
            colorForPitch={colorForPitch}
          />
        );
      })}
    </>
  );
});

const InstrumentRow = observer(
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
    const voice0 = jot.primaryStructuralVoice;
    if (!voice0) return null;
    const trackHeight = config.trackHeight as number;
    // Per-pitch derived data (bars, voice-wide totals, cumulative
    // bar-start offsets, label color/instrument name); all memoised on
    // the jot via `barsForPitch(pitch)`, so each row reads its slice
    // from the MobX cache instead of recomputing on every render.
    // `barBeatStart` and `startBeats` are the same array; the keyed
    // names just disambiguate the two historical use sites.
    const {
      bars: pitchBars,
      voiceBeats,
      leadInBarsBeats,
      barBeatStart,
      startBeats,
      instrumentName,
    } = jot.barsForPitch(pitch);
    // Resolve the row's note colour through the store-owned
    // `InstrumentTrack`. The structural `barsForPitch().pitchColor` is
    // now palette-only (overrides moved off the jot in the colour-
    // picker refactor), so layering happens here: the InstrumentTrack
    // returns the override if set, otherwise the jot's palette default,
    // otherwise the neutral fallback grey. Reading it inside this
    // observer is the dependency that drives a row re-render when the
    // user picks a new colour.
    const mixer = React.useContext(MixerStoreContext);
    const viewport = React.useContext(ViewportStoreContext);
    const instrumentTrack = mixer?.getInstrumentTrack(pitch);
    const pitchColor = instrumentTrack?.color ?? 'var(--color-text-faint-strong)';

    // Filtered-onset ghost overlays (debug bundle + checkbox gated).
    // Resolve once per row so the per-entry render below is just a map.
    const provenance = React.useContext(NoteProvenanceContext);
    const showFiltered = provenance?.showFiltered ?? false;
    const rejectedForPitch = showFiltered ? (provenance!.rejectedByPitch.get(pitch) ?? []) : [];

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
    // Stable per-bar props so the windowed bar list's scroll re-renders
    // don't bust `BarView`'s observer memo for bars that didn't move.
    const pitchesMemo = React.useMemo(() => [pitch], [pitch]);
    const colorForPitch = React.useCallback(
      (p: string) => mixer?.getInstrumentTrack(p).color,
      [mixer]
    );
    return (
      <div
        className={classNames(
          styles.instrumentRow,
          groupStart && styles.mixerRowGroupStart,
          groupEnd && styles.mixerRowGroupEnd,
          inGroup && styles.mixerRowInGroup,
          isDragging && styles.mixerRowDragging,
          drop.isDropIndicatorAbove && styles.mixerDropIndicatorAbove,
          drop.isDropIndicatorBelow && styles.mixerDropIndicatorBelow
        )}
        data-testid={`instrument-row-${pitch}`}
        onDragOver={drop.onDragOver}
        onDragLeave={drop.onDragLeave}
        onDrop={drop.onDrop}
      >
        <div className={styles.instrumentRowGutter}>
          <MixerDragHandle
            idx={idx}
            onDragStartIdx={onDragStartIdx}
            onResetDrag={onResetDrag}
            ariaLabel={labelText}
          />
          <GutterResizeHandle onResizeStart={onResizeGutterStart} />
          {/* Two-row stack mirroring the audio-track row: header (label +
              overflow trigger) on top, slider + M/S on a second line
              below. */}
          <div className={styles.instrumentRowContent}>
            <div className={styles.instrumentRowHeader}>
              <div
                className={classNames(styles.instrumentRowLabel, !audible && styles.musicTrackLabelDim)}
                title={instrumentName ? `${instrumentName} (pitch ${pitch})` : `Pitch ${pitch}`}
              >
                <span className={styles.gutterPitch}>{pitch}</span>
                {instrumentName && <span className={styles.instrumentRowName}>{instrumentName}</span>}
              </div>
              {instrumentTrack && (
                <InstrumentRowOverflowMenu
                  instrumentTrack={instrumentTrack}
                  trackLabel={labelText}
                />
              )}
            </div>
            <div className={styles.instrumentRowControls}>
              <RowVolumeSlider
                value={voiceControls.volumeFor(pitch)}
                onChange={(v) => voiceControls.onSetVolume(pitch, v)}
                label={labelText}
              />
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
              ['--bars-row-width' as string]: barsRowWidthSeed(jot, voiceBeats),
            } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          {/* Lead-in label overlay floating across the negative-indexed
              bars. The bars themselves carry the hatched background
              (`.barLeadIn` / `.barLeadInLast`); this overlay just adds
              the centered "lead-in" caption. Topmost-row only
              (`showBrackets`) so the label doesn't repeat on every
              instrument row. */}
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
          {/* Cumulative quarter-note position of each bar's left edge
              within the voice (drives the bar's absolute left via
              `--bar-start-beat`; see `.bar` in score.module.css) is
              precomputed by `jot.barsForPitch(pitch)` as `startBeats`,
              so this map is just a render. */}
          <WindowedBarList
            viewport={viewport ?? null}
            pitchBars={pitchBars}
            startBeats={startBeats}
            pitch={pitch}
            config={config}
            showBrackets={showBrackets}
            pitchOrder={pitchOrder}
            highlightedPattern={highlightedPattern}
            onPatternClick={onPatternClick}
            isPitchAudible={voiceControls.isPitchAudible}
            pitches={pitchesMemo}
            colorForPitch={colorForPitch}
          />
          {rejectedForPitch.map((entry, i) => {
            // The MIDI lays `leadBars` empty bar-0-sized blocks before
            // struct bar 0, so the struct bar index maps to the
            // rendered jot's bars array as `leadBars + entry.bar`.
            // Out-of-range entries are already filtered out upstream.
            const barIdx = provenance!.leadBars + entry.bar;
            if (barIdx < 0 || barIdx >= pitchBars.length) return null;
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
    const viewport = React.useContext(ViewportStoreContext);
    const uniformWaveforms = React.useContext(UniformWaveformsContext);
    const padBeats = React.useContext(RenderedJotContext)?.config.barNotePaddingBeats ?? 0.125;
    // Waveform tint reads straight off the AudioTrack instance; the
    // class's `color` getter resolves the user override -> grouped
    // instrument inheritance -> neutral chain itself (see
    // `resolveAudioInheritedColor`), and is MobX-observable so picker
    // commits repaint chunks reactively. Always returns a `#rrggbb`
    // string the chunk worker can consume directly.
    const pitchColor = track.color;
    // Beat-stable chunk layout (zoom-invariant). Memoed on `jot` so
    // scroll / zoom re-renders of this observer don't rebuild it.
    const layout = React.useMemo(() => buildChunkLayout(jot), [jot]);
    const livePxPerBeat = useLiveJotPxPerBeat();

    if (!viewport || layout.chunks.length === 0) return null;

    // Visibility: derive the score-px x-range currently on screen
    // from `JotViewStore` observables. The score uses a virtualised
    // scroll model (`.scrollViewport` translated by `(-scrollX, 0)`),
    // so `[scrollX, scrollX + viewportWidth]` is exactly the score-px
    // window the user sees. Each chunk's score-px left mirrors the
    // formula the chunk component below uses for its inline `left`
    // (`chunk.startBeat * livePxPerBeat + padBeats * livePxPerBeat`);
    // any chunk whose box intersects the viewport plus prefetch
    // margin is mounted, anything else is unmounted.
    const scrollX = viewport.scrollX;
    const viewportWidth = viewport._viewportWidth;
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
