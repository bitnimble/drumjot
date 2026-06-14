import type { Meta, StoryObj } from '@storybook/react-vite';
import { runInAction } from 'mobx';
import React from 'react';
import { RenderedJot } from 'src/jot/resolved_jot';
import { rockJot } from 'src/fakes/fakes';
import { fn } from 'storybook/test';
import { MixerStoreContext } from '../mixer_contexts';
import { ViewportStoreContext } from '../../viewport/viewport_contexts';
import { DocumentStore } from '../../document/document_store';
import { MixerStore } from '../mixer_store';
import { ViewportStore } from '../../viewport/viewport_store';
import type { TrackKey } from '../../store';
import type { VoiceControls } from '../mixer_controls';
import { InstrumentRow } from '../instrument_row';
import { Gallery, Variant } from '../../components/stories/_variants';

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
 *  first real pitch of the loaded jot. `muted` flips the row's audibility
 *  so the gutter renders its dimmed state. */
function Row({ muted = false }: { muted?: boolean }) {
  const { documentStore, mixer, viewport, pitch, voiceControls } = React.useMemo(() => {
    const documentStore = new DocumentStore();
    runInAction(() => {
      documentStore.currentJot = new RenderedJot(rockJot, documentStore.viewConfig);
    });
    const mixer = new MixerStore(documentStore);
    const viewport = new ViewportStore(documentStore);
    const pitches = mixer.jotPitches;
    runInAction(() => {
      mixer.trackOrder = pitches.map((p): TrackKey => ({ kind: 'instrument', pitch: p }));
    });
    const pitch = pitches[0];
    const voiceControls: VoiceControls = {
      mutedPitches: muted ? new Set([pitch]) : new Set(),
      soloedPitches: new Set(),
      isPitchAudible: () => !muted,
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
    return { documentStore, mixer, viewport, pitch, voiceControls };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <MixerStoreContext.Provider value={mixer}>
      <ViewportStoreContext.Provider value={viewport}>
        <InstrumentRow
          pitch={pitch}
          jot={documentStore.currentJot!}
          config={documentStore.viewConfig}
          showBrackets
          pitchOrder={mixer.jotPitches}
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
