import type { Meta, StoryObj } from '@storybook/react-vite';
import type { TranscribeStatus } from '../../store';
import { LyricsAlignBusyPill, TranscribeBusyPill } from '../toolbar_status';

/**
 * The toolbar's right-aligned busy pills, shown while a backend job is in
 * flight. Both are pure prop-driven `observer`s, so they render in
 * isolation without any store wiring, each story just supplies the state
 * the pill reacts to. (The third indicator, DrumLoadingIndicator, reads
 * the `jotPlayer` singleton's load state directly and isn't prop-driven,
 * so it's not represented here.)
 */
const meta: Meta = {
  title: 'Toolbar/Status pills',
};
export default meta;

type Story = StoryObj;

/** Lyrics-alignment pill, waiting behind another GPU job. */
export const LyricsAlignQueued: Story = {
  render: () => <LyricsAlignBusyPill phase="queued" />,
};

/** Lyrics-alignment pill, actively running. */
export const LyricsAlignRunning: Story = {
  render: () => <LyricsAlignBusyPill phase="aligning" />,
};

const uploading = (over: Partial<Extract<TranscribeStatus, { phase: 'uploading' }>>): TranscribeStatus => ({
  phase: 'uploading',
  filename: 'my-song.flac',
  ...over,
});

/** Transcribe pill before the first stage event arrives (filename only). */
export const TranscribeStarting: Story = {
  render: () => <TranscribeBusyPill status={uploading({})} />,
};

/** Transcribe pill mid-pipeline, on a named stage. */
export const TranscribeOnStage: Story = {
  render: () => <TranscribeBusyPill status={uploading({ stage: 'onsets' })} />,
};

/** Transcribe pill with in-stage substage detail. */
export const TranscribeWithSubstage: Story = {
  render: () => (
    <TranscribeBusyPill
      status={uploading({ stage: 'stems_per', substage: 'separating 3/5 (latest: snare)' })}
    />
  ),
};
