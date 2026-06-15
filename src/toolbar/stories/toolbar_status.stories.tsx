import type { Meta, StoryObj } from '@storybook/react-vite';
import type { TranscribeStatus } from 'src/editing/transcribe/transcribe_store';
import { LyricsAlignBusyPill, TranscribeBusyPill } from '../toolbar_status';
import { Gallery, Variant } from 'src/ui/stories/_variants';

/**
 * The toolbar's right-aligned busy pills, shown while a backend job is in
 * flight. Both are pure prop-driven `observer`s, so they render in
 * isolation without any store wiring. (The third indicator,
 * DrumLoadingIndicator, reads the `jotPlayer` singleton's load state
 * directly and isn't prop-driven, so it's not represented here.)
 */
const meta: Meta = {
  title: 'Toolbar/Status pills',
};
export default meta;

type Story = StoryObj;

const uploading = (
  over: Partial<Extract<TranscribeStatus, { phase: 'uploading' }>>
): TranscribeStatus => ({
  phase: 'uploading',
  filename: 'my-song.flac',
  ...over,
});

/** Every busy-pill state in one place. */
export const All: Story = {
  render: () => (
    <Gallery>
      <Variant label="Lyrics align — queued">
        <LyricsAlignBusyPill phase="queued" />
      </Variant>
      <Variant label="Lyrics align — running">
        <LyricsAlignBusyPill phase="aligning" />
      </Variant>
      <Variant label="Transcribe — starting (filename only)">
        <TranscribeBusyPill status={uploading({})} />
      </Variant>
      <Variant label="Transcribe — on a stage">
        <TranscribeBusyPill status={uploading({ stage: 'onsets' })} />
      </Variant>
      <Variant label="Transcribe — with substage detail">
        <TranscribeBusyPill
          status={uploading({ stage: 'stems_per', substage: 'separating 3/5 (latest: snare)' })}
        />
      </Variant>
    </Gallery>
  ),
};
