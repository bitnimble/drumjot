import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { RenderedJot, StructuralBar, ViewConfig } from 'src/jot';
import {
  AudioTrack,
  AudioTrackId,
  computeWaveformPeaksForJot,
  jotPlayer,
} from 'src/playback';
import { ClearButton, MuteButton, SoloButton } from './components/icon_button';
import { NoteProvenanceContext } from './contexts';
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
};

/**
 * Drag-source identifier carried on the DataTransfer of a mixer-row
 * drag. A custom MIME type lets us reject foreign drops (files,
 * external pages) so the gutter never tries to swallow them.
 */
const MIXER_DRAG_MIME = 'application/x-drumjot-mixer-row';

const AUDIO_TRACK_HEIGHT = 56;

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
    // pattern/tuplet bracket overlay. Brackets describe score structure
    // (not a specific instrument), so anchoring them to whichever drum
    // row currently sits at the top of the drum block keeps the overlay
    // visible no matter how the user rearranges the mixer.
    const firstPitchIdx = trackOrder.findIndex((k) => k.kind === 'pitch');

    return (
      <div className={styles.mixer}>
        <GutterMasterRow
          label="Audio"
          title="Master volume for all loaded audio (backing) tracks together. Multiplies on top of each track's own fader; takes effect instantly, including mid-playback."
          value={jotPlayer.audioTrackMasterVolume}
          onChange={(v) => jotPlayer.setAudioTrackMasterVolume(v)}
          testId="audio-track-master"
        />
        <GutterMasterRow
          label="Drums"
          title="Master volume for all drum/instrument rows together. Multiplies on top of each row's own fader; takes effect instantly, including mid-playback."
          value={jotPlayer.drumMasterVolume}
          onChange={(v) => jotPlayer.setDrumMasterVolume(v)}
          testId="drum-master"
        />
        {trackOrder.map((key, idx) => {
          // Reuse a stable React key per row so dragging doesn't tear
          // down + remount expensive children (the AudioTrackWaveformCanvas
          // would otherwise re-decode peaks on every reorder).
          const reactKey = key.kind === 'audio' ? `audio:${key.id}` : `pitch:${key.pitch}`;
          // A row begins a new "group" — and so renders with a small
          // top gap — whenever its `groupId` differs from the previous
          // row's. Solo (groupId undefined) rows are each their own
          // group. The first row never gets a gap (nothing above it).
          const prevGroupId = idx > 0 ? trackOrder[idx - 1].groupId : undefined;
          const groupStart = idx > 0 && key.groupId !== prevGroupId;
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
          return (
            <PitchRow
              key={reactKey}
              pitch={key.pitch}
              jot={jot}
              config={config}
              showBrackets={idx === firstPitchIdx}
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
  }: {
    id: AudioTrackId;
    track: AudioTrack;
    jot: RenderedJot;
    controls: AudioTrackControls;
    onSeek: (x: number) => void;
  } & MixerRowDragProps) => {
    const voice = jot.resolved.voices[0];
    const width = (voice?.width ?? 0) as number;
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
          isDragging && styles.mixerRowDragging,
          drop.isDropIndicatorAbove && styles.mixerDropIndicatorAbove,
          drop.isDropIndicatorBelow && styles.mixerDropIndicatorBelow,
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
          <div className={styles.musicTrackContent}>
            <div className={classNames(styles.musicTrackLabel, !audible && styles.musicTrackLabelDim)}>
              <span className={styles.musicTrackName}>{label}</span>
              <span className={styles.musicTrackFile} title={track.filename}>
                {track.filename}
              </span>
            </div>
            <div className={styles.musicTrackButtons}>
              <RowVolumeSlider
                value={controls.volumeFor(id)}
                onChange={(v) => controls.onSetVolume(id, v)}
                label={`${label} audio track`}
              />
              {/* Clear sits first so Mute/Solo stay flush with the gutter's
                  right edge — lining up with the M/S column on the
                  instrument rows below (both gutters share a width). */}
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
          style={{ width, height: AUDIO_TRACK_HEIGHT }}
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          <AudioTrackWaveformCanvas
            jot={jot}
            track={track}
            width={width}
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
  }: {
    pitch: string;
    jot: RenderedJot;
    config: ViewConfig;
    showBrackets: boolean;
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
    // Voice-level totals for the bars-row width (in beats — the row's
    // pixel width is `voiceBeats × --px-per-beat` via CSS calc). Lead-in
    // contributes `leadInSec × bpm/60` quarter notes at the row's tempo.
    const leadInBeats = voice0.leadInSec * (voice0.leadInBpm / 60);
    let voiceBeats = leadInBeats;
    for (const b of voice0.bars) voiceBeats += b.beats;

    // Filtered-onset ghost overlays (debug bundle + checkbox gated).
    // Resolve once per row so the per-entry render below is just a map.
    const provenance = React.useContext(NoteProvenanceContext);
    const showFiltered = provenance?.showFiltered ?? false;
    const rejectedForPitch = showFiltered
      ? provenance!.rejectedByPitch.get(pitch) ?? []
      : [];
    // Cumulative beat offsets so each rejected entry can be positioned
    // absolutely in the bars row without walking back through bar
    // widths on every render. Same scale (quarter-note beats) as the
    // CSS-var positioning the kept notes use.
    const barBeatStart: number[] = [];
    {
      let acc = leadInBeats;
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
          isDragging && styles.mixerRowDragging,
          drop.isDropIndicatorAbove && styles.mixerDropIndicatorAbove,
          drop.isDropIndicatorBelow && styles.mixerDropIndicatorBelow,
        )}
        data-testid={`pitch-row-${pitch}`}
        onDragOver={drop.onDragOver}
        onDragLeave={drop.onDragLeave}
        onDrop={drop.onDrop}
      >
        <div className={styles.pitchRowGutter} style={{ height: trackHeight }}>
          <MixerDragHandle
            idx={idx}
            onDragStartIdx={onDragStartIdx}
            onResetDrag={onResetDrag}
            ariaLabel={labelText}
          />
          <div
            className={classNames(styles.pitchRowLabel, !audible && styles.musicTrackLabelDim)}
            title={instrumentName ? `${instrumentName} (pitch ${pitch})` : `Pitch ${pitch}`}
          >
            <span className={styles.gutterPitch}>{pitch}</span>
            {instrumentName && <span className={styles.pitchRowName}>{instrumentName}</span>}
          </div>
          <div className={styles.musicTrackButtons}>
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
        <div
          className={styles.barsRow}
          data-bars-row
          style={
            {
              ['--voice-beats' as string]: voiceBeats,
              height: trackHeight,
            } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          {leadInBeats > 0 && (
            <div
              className={styles.leadIn}
              style={
                {
                  ['--lead-in-beats' as string]: leadInBeats,
                  height: trackHeight,
                } as React.CSSProperties
              }
              title={`Lead-in: ${voice0.leadInSec.toFixed(
                2,
              )}s of pre-roll before the first beat — keeps the drum notation aligned with a loaded audio-track waveform.`}
            >
              {showBrackets && (
                <span className={styles.leadInLabel}>lead-in</span>
              )}
            </div>
          )}
          {bars.map((bar, i) => (
            <BarView
              key={i}
              bar={bar}
              pitches={[pitch]}
              config={config}
              isAnacrusis={bar.index === 0}
              highlightedPattern={highlightedPattern}
              onPatternClick={onPatternClick}
              isPitchAudible={voiceControls.isPitchAudible}
              showBrackets={showBrackets}
            />
          ))}
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
 * Canvas-rendered waveform for one audio track, aligned to the score's
 * bar timeline. Peaks are recomputed in a `useEffect` whenever the
 * (zoom-dependent) total bar width changes or the underlying track
 * swaps — same cadence the score uses to re-flow under
 * `viewConfig.barWidth`.
 */
const AudioTrackWaveformCanvas = observer(
  ({
    jot,
    track,
    width,
    height,
    dim,
    testId,
  }: {
    jot: RenderedJot;
    track: AudioTrack;
    width: number;
    height: number;
    dim: boolean;
    testId?: string;
  }) => {
    const canvasRef = React.useRef<HTMLCanvasElement>(null);
    // The live drum↔audio offset (Offset control). Reading it here under
    // `observer` re-renders the waveform when the user nudges the offset
    // so it stays aligned with where the audio actually plays.
    const startOffsetSec = jotPlayer.startOffsetSec;

    React.useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas || width <= 0) return;
      const dpr = window.devicePixelRatio || 1;
      // Browsers cap a canvas's backing-store dimensions (and total
      // area). A long score at high zoom × dpr easily blows past that
      // and throws "Canvas exceeds max size". Clamp the backing store;
      // the element stays CSS-sized to `width`, so past the cap it just
      // renders at reduced horizontal resolution instead of crashing.
      // 16384 is the safe cross-browser per-axis limit (Safari/iOS is
      // the tightest; Chrome/Firefox allow more).
      const MAX_CANVAS_DIM = 16384;
      const backingW = Math.min(Math.max(1, Math.floor(width * dpr)), MAX_CANVAS_DIM);
      const backingH = Math.min(Math.max(1, Math.floor(height * dpr)), MAX_CANVAS_DIM);
      canvas.width = backingW;
      canvas.height = backingH;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      // Map CSS-pixel drawing coords (0..width, 0..height) onto the
      // possibly-clamped backing store. Reduces to ctx.scale(dpr, dpr)
      // when nothing was clamped.
      ctx.setTransform(backingW / width, 0, 0, backingH / height, 0, 0);

      const { peaks } = computeWaveformPeaksForJot(
        jot,
        track.buffer,
        startOffsetSec,
      );

      ctx.clearRect(0, 0, width, height);
      const mid = height / 2;
      ctx.fillStyle = dim ? '#d3c8b6' : '#5BA8E8';
      // Each pixel column is a vertical line from min*scale to max*scale.
      // A single fillRect per column is faster than building a Path2D
      // for thousands of segments and lets us keep the colour-by-column
      // option open if we ever want to tint clipped peaks differently.
      const scale = mid * 0.95;
      for (let p = 0; p < width; p++) {
        const mn = peaks[p * 2];
        const mx = peaks[p * 2 + 1];
        if (mn === 0 && mx === 0) continue;
        const y0 = mid - mx * scale;
        const y1 = mid - mn * scale;
        ctx.fillRect(p, y0, 1, Math.max(1, y1 - y0));
      }
    }, [jot, track, width, height, dim, startOffsetSec]);

    if (width <= 0) return null;
    return (
      <canvas
        ref={canvasRef}
        className={styles.musicTrackWaveform}
        style={{ width, height }}
        data-testid={testId}
      />
    );
  }
);

/**
 * A per-section master fader that sits in the sticky lane gutter,
 * directly above the section it controls (the loaded audio tracks, or
 * the drum/instrument staff). Gutter-aligned (same 132px sticky column
 * as the per-row M/S/volume controls below it) so it reads as the
 * "header" for that column. Reads/writes the global observable
 * `jotPlayer`; all pointer events are kept from bubbling so dragging
 * the fader doesn't start the page marquee or trip seek-on-click.
 */
const GutterMasterRow = observer(
  ({
    label,
    title,
    value,
    onChange,
    testId,
  }: {
    label: string;
    title: string;
    value: number;
    onChange: (v: number) => void;
    testId?: string;
  }) => {
    const stop = (e: React.MouseEvent) => e.stopPropagation();
    const pct = Math.round(value * 100);
    return (
      <div className={styles.gutterMasterRow}>
        <div className={styles.gutterMasterGutter} title={title} data-testid={testId}>
          <span className={styles.gutterMasterLabel}>{label}</span>
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
            aria-label={`${label} master volume`}
            title={`${label} master volume: ${pct}%`}
          />
          <span className={styles.gutterMasterValue}>{pct}%</span>
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
    />
  );
};
