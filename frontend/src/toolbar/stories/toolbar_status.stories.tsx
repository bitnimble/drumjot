import type { Meta, StoryObj } from '@storybook/react-vite';
import { LyricsAlignStore, type LyricsAlignStatus } from 'src/editing/lyrics/lyrics_align_store';
import { LyricsAlignStoreContext } from 'src/editing/lyrics/lyrics_contexts';
import { TranscribeStore, type TranscribeTrackStatus } from 'src/editing/transcribe/transcribe_store';
import { TranscribeStoreContext } from 'src/editing/transcribe/transcribe_contexts';
import { LyricsAlignBusyPill, TranscribeBusyPill } from '../toolbar_status';
import { Gallery, Variant } from 'src/ui/stories/_variants';

/**
 * The toolbar's right-aligned busy pills, shown while a backend job is in
 * flight. Both read their store from context, so each variant wraps the pill in
 * a provider with a stub store. (DrumLoadingIndicator reads the `jotPlayer`
 * singleton directly and isn't represented here.)
 */
const meta: Meta = {
  title: 'Toolbar/Status pills',
};
export default meta;

type Story = StoryObj;

/** A store stub whose single in-flight track has the given status. */
const withTrack = (status: TranscribeTrackStatus) => {
  const store = new TranscribeStore();
  store.trackStatuses.set('a', status);
  return store;
};

const TranscribePill = ({ status }: { status: TranscribeTrackStatus }) => (
  <TranscribeStoreContext.Provider value={withTrack(status)}>
    <TranscribeBusyPill />
  </TranscribeStoreContext.Provider>
);

/** A store stub whose single in-flight row has the given align status. */
const withAlign = (status: LyricsAlignStatus) => {
  const store = new LyricsAlignStore();
  store.lyricsAlignStatuses.set('a', status);
  return store;
};

const AlignPill = ({ status }: { status: LyricsAlignStatus }) => (
  <LyricsAlignStoreContext.Provider value={withAlign(status)}>
    <LyricsAlignBusyPill />
  </LyricsAlignStoreContext.Provider>
);

/** Every busy-pill state in one place. */
export const All: Story = {
  render: () => (
    <Gallery>
      <Variant label="Lyrics align, queued">
        <AlignPill status={{ phase: 'queued', detail: 'waiting for GPU' }} />
      </Variant>
      <Variant label="Lyrics align, running">
        <AlignPill status={{ phase: 'aligning', detail: 'aligning' }} />
      </Variant>
      <Variant label="Transcribe, starting (filename only)">
        <TranscribePill status={{ filename: 'my-song.flac' }} />
      </Variant>
      <Variant label="Transcribe, on a stage">
        <TranscribePill status={{ filename: 'my-song.flac', stage: 'onsets' }} />
      </Variant>
      <Variant label="Transcribe, with substage detail">
        <TranscribePill
          status={{ filename: 'my-song.flac', stage: 'stems_per', substage: 'separating 3/5 (latest: snare)' }}
        />
      </Variant>
    </Gallery>
  ),
};
