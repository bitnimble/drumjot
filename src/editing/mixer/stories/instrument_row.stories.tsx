import type { Meta, StoryObj } from '@storybook/react-vite';
import { runInAction } from 'mobx';
import React from 'react';
import { buildJotModel } from '../../jot_editor_store';
import { rockJot } from 'src/fakes/fakes';
import { fn } from 'storybook/test';
import { MixerStoreContext } from '../mixer_contexts';
import { StructuralContext } from '../../jot_editor_contexts';
import { ViewportStoreContext } from '../../viewport/viewport_contexts';
import { JotEditorStore } from '../../jot_editor_store';
import { MixerStore } from '../mixer_store';
import { ViewportStore } from '../../viewport/viewport_store';
import type { TrackKey } from 'src/editing/tracks/tracks';
import type { LayerControls } from '../mixer_controls';
import { InstrumentRow } from '../instrument_row';
import { Gallery, Variant } from 'src/ui/stories/_variants';

/**
 * A single drum-instrument (note) row from the unified mixer: the gutter
 * (drag handle, label, volume + mute/solo) on the left and the notated
 * bars on the right. Driven by REAL Document/Mixer/Viewport stores so the
 * label colour + overflow menu resolve through the store the way they do
 * in the app; the per-lane M/S/volume contract (`layerControls`) is
 * stubbed so its callbacks report to the Actions panel.
 *
 * The viewport is left unsized, so `visibleBeatRange` is null and the row
 * renders the whole song (the windowing's "not laid out yet → draw
 * everything" path) rather than an empty strip.
 */
const meta: Meta = {
  title: 'Mixer/InstrumentRow',
  parameters: { layout: 'fullscreen' },
};
export default meta;

type Story = StoryObj;

const NOOP_DRAG = {
  idx: 0,
  dragFromIdx: undefined,
  dropTargetIdx: undefined,
  onDragStartIdx: fn(),
  onDropTargetIdx: fn(),
  onMoveTrack: fn(),
  onResetDrag: fn(),
  mixerLength: 1,
  groupStart: false,
  groupEnd: false,
  inGroup: false,
  onResizeGutterStart: fn(),
};

/** One InstrumentRow backed by a fresh real store trio, pointed at the
 *  first real lane of the loaded jot. `muted` flips the row's audibility
 *  so the gutter renders its dimmed state. */
function Row({ muted = false }: { muted?: boolean }) {
  const { jotEditorStore, structural, mixer, viewport, lane, layerControls } = React.useMemo(() => {
    const jotEditorStore = new JotEditorStore();
    const model = buildJotModel(rockJot, jotEditorStore.viewConfig);
    runInAction(() => {
      jotEditorStore.source = rockJot;
      jotEditorStore.structural = model.structural;
      jotEditorStore.palette = model.palette;
      jotEditorStore.tempo = model.tempo;
    });
    const { structural } = model;
    const mixer = new MixerStore(jotEditorStore);
    const viewport = new ViewportStore(jotEditorStore);
    const lanes = mixer.jotLanes;
    runInAction(() => {
      mixer.trackOrder = lanes.map((p): TrackKey => ({ kind: 'instrument', lane: p }));
    });
    const lane = lanes[0];
    const layerControls: LayerControls = {
      mutedLanes: muted ? new Set([lane]) : new Set(),
      soloedLanes: new Set(),
      isLaneAudible: () => !muted,
      volumeFor: () => 1,
      onSetVolume: fn(),
      onToggleMute: fn(),
      onToggleSolo: fn(),
      masterMuted: false,
      masterSoloed: false,
      masterAudible: true,
      onToggleMasterMute: fn(),
      onToggleMasterSolo: fn(),
    };
    return { jotEditorStore, structural, mixer, viewport, lane, layerControls };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <StructuralContext.Provider value={structural}>
      <MixerStoreContext.Provider value={mixer}>
        <ViewportStoreContext.Provider value={viewport}>
          <InstrumentRow
            lane={lane}
            config={jotEditorStore.viewConfig}
            showBrackets
            laneOrder={mixer.jotLanes}
            highlightedPattern={undefined}
            onPatternClick={fn()}
            onSeek={fn()}
            layerControls={layerControls}
            {...NOOP_DRAG}
          />
        </ViewportStoreContext.Provider>
      </MixerStoreContext.Provider>
    </StructuralContext.Provider>
  );
}

/** Default + muted states in one place. */
export const All: Story = {
  render: () => (
    <Gallery>
      <Variant label="Default">
        <Row />
      </Variant>
      <Variant label="Muted (dimmed gutter)">
        <Row muted />
      </Variant>
    </Gallery>
  ),
};
