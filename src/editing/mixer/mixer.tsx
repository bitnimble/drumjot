import { observer } from 'mobx-react-lite';
import React from 'react';
import { ViewConfig } from 'src/editing/viewport/view_config';
import { jotPlayer } from 'src/editing/playback/player';
import { LayersStoreContext } from '../layers/layers_contexts';
import type { LayersTrackView } from '../layers/layers_store';
import { MergeLayersContext } from './mixer_contexts';
import { LyricsTrackView } from '../lyrics/lyrics_track_view';
import styles from './mixer.module.css';
import { GutterMasterRow } from './gutter_controls';
import { InstrumentTrackView } from './instrument_track_view';

import type { LayerControls, AudioTrackControls } from './mixer_controls';
// Re-exported so existing `from '.../mixer/mixer'` importers (jot_editor) keep
// working; the definitions live in the leaf `mixer_controls.ts`.
import { AudioTrackView } from './audio_track_view';
export type { LayerControls, AudioTrackControls };




/**
 * The unified mixer that replaced the old separate "audio tracks" and
 * "layer staves" sections. Renders the two section masters at the top,
 * then the rows of `jot.ordering` (via `LayersStore.layout`): per layer,
 * a colour band holding its groups + per-track rows (instrument / audio /
 * lyrics), freely interleavable. Row order + grouping is owned by the
 * doc (written by `LayersPresenter`; the Layers panel and the gutter both
 * call it). The topmost instrument row hosts the pattern/tuplet bracket
 * overlay so they read as a single piece of score chrome regardless of
 * how the user has arranged the rows.
 */
export const MixerView = observer(
  ({
    config,
    highlightedPattern,
    onPatternClick,
    onSeek,
    layerControls,
    audioTrackControls,
    onResizeGutterStart,
  }: {
    config: ViewConfig;
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    onSeek: (x: number) => void;
    layerControls: LayerControls;
    audioTrackControls: AudioTrackControls;
    /** Pointer-down handler for the gutter resize affordance painted on
     * the right edge of every row's gutter. */
    onResizeGutterStart: (e: React.PointerEvent<HTMLDivElement>) => void;
  }) => {
    // The topmost instrument row in the user's mixer order hosts the
    // tuplet brackets and the lead-in label; chrome that belongs to
    // the score as a whole, not to any one instrument. Both this index
    // and the mixer-ordered lane list are MobX computeds on the store,
    // so MixerView (already an observer) gets correct invalidation when
    // the user reorders rows.
    // All rows (instrument / audio / lyrics) render from the ordering
    // read-model (`LayersStore.layout`): per layer, a colour band containing
    // its groups + per-track rows. Audio / lyrics tracks are folded into
    // `ordering` by `LayersPresenter`, so they appear here like any other row.
    const layers = React.useContext(LayersStoreContext);
    // "Visually merge layers" (View menu): collapse same-lane tracks across
    // layers into one flat per-lane row (no bands); else render layer-first.
    const merge = React.useContext(MergeLayersContext);
    const layout = layers?.layout ?? [];
    const mergedSlots = merge ? (layers?.mergedLayout ?? []) : [];
    // Flat instrument-lane order (for pattern-bracket top/bottom spanning) and
    // the topmost instrument track (hosts the tuplet brackets + lead-in label),
    // from whichever structure is being rendered.
    const renderSlots = merge
      ? mergedSlots
      : layout.flatMap((l) => l.slots);
    const instrumentLaneOrder: readonly string[] = renderSlots.flatMap((s) =>
      s.tracks.flatMap((t) => (t.kind === 'instrument' ? [t.lane] : []))
    );
    const firstInstrumentTrackId = (() => {
      for (const s of renderSlots) {
        for (const t of s.tracks) if (t.kind === 'instrument') return t.id;
      }
      return undefined;
    })();
    // No-op drag wiring: the gutter drag handle is inert for now (row order +
    // grouping is edited through the Layers panel, which writes `ordering`).
    const noDrag = {
      idx: -1,
      dragFromIdx: undefined,
      dropTargetIdx: undefined,
      onDragStartIdx: () => {},
      onDropTargetIdx: () => {},
      onMoveTrack: () => {},
      onResetDrag: () => {},
      mixerLength: 0,
      groupStart: false,
      groupEnd: false,
      onResizeGutterStart,
    };

    // One row of any kind. Instrument: `layerId` set = per-track (this layer's
    // notes); merged rows pass `layerId` undefined + the layers the collapsed
    // lane aggregates. Audio / lyrics rows render their session track. The
    // gutter drag is a no-op for now (panel DnD writes `ordering`).
    const renderTrackRow = (
      t: LayersTrackView,
      layerId: string | undefined,
      mergeLaneLayerIds: readonly string[] | undefined,
      inGroup: boolean
    ) => {
      if (t.kind === 'instrument') {
        return (
          <InstrumentTrackView
            key={`${layerId ?? 'merged'} ${t.lane}`}
            lane={t.lane}
            layerId={layerId}
            mergeLaneLayerIds={mergeLaneLayerIds}
            config={config}
            showBrackets={firstInstrumentTrackId === t.id}
            laneOrder={instrumentLaneOrder}
            highlightedPattern={highlightedPattern}
            onPatternClick={onPatternClick}
            onSeek={onSeek}
            layerControls={layerControls}
            {...noDrag}
            inGroup={inGroup}
          />
        );
      }
      if (t.kind === 'audio') {
        const track = jotPlayer.audioTracks.get(t.audioId);
        if (!track) return null;
        return (
          <AudioTrackView
            key={`audio:${t.audioId}`}
            id={t.audioId}
            track={track}
            controls={audioTrackControls}
            onSeek={onSeek}
            {...noDrag}
            inGroup={inGroup}
          />
        );
      }
      return (
        <LyricsTrackView key={`lyrics:${t.lyricsId}`} id={t.lyricsId} onSeek={onSeek} {...noDrag} inGroup={inGroup} />
      );
    };

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
          muted={layerControls.masterMuted}
          soloed={layerControls.masterSoloed}
          audible={layerControls.masterAudible}
          onToggleMute={layerControls.onToggleMasterMute}
          onToggleSolo={layerControls.onToggleMasterSolo}
          testId="drum-master"
          onResizeGutterStart={onResizeGutterStart}
        />
        {/* All rows render from `ordering` (LayersStore). Merge view: flat
            per-lane rows (no bands), each collapsing its lane across layers.
            Else layer-first: a tinted band per layer holding its groups
            (heading + indent) and per-track rows (instrument / audio / lyrics). */}
        {merge
          ? mergedSlots.map((slot, si) => (
              <React.Fragment key={`merged-${si}`}>
                {slot.kind === 'group' && (
                  <div className={styles.layerGroupLabel}>{slot.name}</div>
                )}
                {slot.tracks.map((t) =>
                  renderTrackRow(
                    t,
                    undefined,
                    layers?.layerIdsForLane(t.kind === 'instrument' ? t.lane : ''),
                    slot.kind === 'group'
                  )
                )}
              </React.Fragment>
            ))
          : layout.map((layer, li) => (
              <div
                key={layer.id}
                className={styles.layerBand}
                data-testid={`score-layer-${layer.id}`}
                style={
                  layer.color
                    ? { background: `color-mix(in srgb, ${layer.color} 12%, transparent)` }
                    : undefined
                }
              >
                {layout.length > 1 && (
                  <div className={styles.layerBandLabel}>{layer.name ?? `Layer ${li + 1}`}</div>
                )}
                {layer.slots.map((slot, si) => (
                  <React.Fragment key={si}>
                    {slot.kind === 'group' && (
                      <div className={styles.layerGroupLabel}>{slot.name}</div>
                    )}
                    {slot.tracks.map((t) =>
                      renderTrackRow(t, layer.id, undefined, slot.kind === 'group')
                    )}
                  </React.Fragment>
                ))}
              </div>
            ))}
      </div>
    );
  }
);
