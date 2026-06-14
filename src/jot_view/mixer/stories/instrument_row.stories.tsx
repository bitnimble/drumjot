import type { Meta, StoryObj } from '@storybook/react-vite';
import { runInAction } from 'mobx';
import React from 'react';
import { RenderedJot } from 'src/jot';
import { rockJot } from 'src/fakes';
import { fn } from 'storybook/test';
import { MixerStoreContext } from '../mixer_contexts';
import { ViewportStoreContext } from '../../viewport/viewport_contexts';
import { DocumentStore } from '../../document/document_store';
import { MixerStore } from '../mixer_store';
import { ViewportStore } from '../../viewport/viewport_store';
import type { TrackKey } from '../../store';
import type { VoiceControls } from '../mixer_controls';
import { InstrumentRow } from '../instrument_row';

/**
 * A single drum-instrument (note) row from the unified mixer: the gutter
 * (drag handle, label, volume + mute/solo) on the left and the notated
 * bars on the right. Driven by REAL Document/Mixer/Viewport stores so the
 * label colour + overflow menu resolve through the store the way they do
 * in the app; the per-pitch M/S/volume contract (`voiceControls`) is
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

/** Build the store trio + a stubbed VoiceControls and pick the first real
 *  pitch from the loaded jot, so the row always points at a valid lane. */
function useInstrumentRowHarness(over: Partial<VoiceControls> = {}) {
  return React.useMemo(() => {
    const documentStore = new DocumentStore();
    runInAction(() => {
      documentStore.currentJot = new RenderedJot(rockJot, documentStore.viewConfig);
    });
    const mixer = new MixerStore(documentStore);
    const viewport = new ViewportStore(documentStore);
    const pitches = mixer.jotPitches;
    runInAction(() => {
      mixer.trackOrder = pitches.map((pitch): TrackKey => ({ kind: 'instrument', pitch }));
    });
    const voiceControls: VoiceControls = {
      mutedPitches: new Set(),
      soloedPitches: new Set(),
      isPitchAudible: () => true,
      volumeFor: () => 1,
      onSetVolume: fn(),
      onToggleMute: fn(),
      onToggleSolo: fn(),
      masterMuted: false,
      masterSoloed: false,
      masterAudible: true,
      onToggleMasterMute: fn(),
      onToggleMasterSolo: fn(),
      ...over,
    };
    return { documentStore, mixer, viewport, pitches, voiceControls };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

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

function Row({
  voiceControlsOver,
  showBrackets = true,
}: {
  voiceControlsOver?: Partial<VoiceControls>;
  showBrackets?: boolean;
}) {
  const { documentStore, mixer, viewport, pitches, voiceControls } =
    useInstrumentRowHarness(voiceControlsOver);
  return (
    <MixerStoreContext.Provider value={mixer}>
      <ViewportStoreContext.Provider value={viewport}>
        <InstrumentRow
          pitch={pitches[0]}
          jot={documentStore.currentJot!}
          config={documentStore.viewConfig}
          showBrackets={showBrackets}
          pitchOrder={pitches}
          highlightedPattern={undefined}
          onPatternClick={fn()}
          onSeek={fn()}
          voiceControls={voiceControls}
          {...NOOP_DRAG}
        />
      </ViewportStoreContext.Provider>
    </MixerStoreContext.Provider>
  );
}

export const Default: Story = {
  render: () => <Row />,
};

export const Muted: Story = {
  render: () => {
    // Pre-mute the row's pitch so the gutter renders its dimmed state.
    const Wrapper = () => {
      const { documentStore, mixer, viewport, pitches, voiceControls } = useInstrumentRowHarness({
        isPitchAudible: () => false,
      });
      const muted = new Set([pitches[0]]);
      const controls: VoiceControls = { ...voiceControls, mutedPitches: muted };
      return (
        <MixerStoreContext.Provider value={mixer}>
          <ViewportStoreContext.Provider value={viewport}>
            <InstrumentRow
              pitch={pitches[0]}
              jot={documentStore.currentJot!}
              config={documentStore.viewConfig}
              showBrackets
              pitchOrder={pitches}
              highlightedPattern={undefined}
              onPatternClick={fn()}
              onSeek={fn()}
              voiceControls={controls}
              {...NOOP_DRAG}
            />
          </ViewportStoreContext.Provider>
        </MixerStoreContext.Provider>
      );
    };
    return <Wrapper />;
  },
};
