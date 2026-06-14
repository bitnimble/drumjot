import type { Meta, StoryObj } from '@storybook/react-vite';
import { runInAction } from 'mobx';
import React from 'react';
import { RenderedJot } from 'src/jot/resolved_jot';
import { rockJot } from 'src/fakes/fakes';
import { DocumentStore } from '../../document/document_store';
import { PlaybackStore } from '../playback_store';
import { PlaybackPresenter } from '../playback_presenter';
import { PlaybackBar } from '../playback';

/**
 * The bottom transport bar, driven by REAL stores + presenter (the point
 * of the store/presenter split: a component can be mounted in isolation
 * with its own peer instances). Play/stop/offset go through a live
 * PlaybackPresenter; the player singleton stays idle so nothing actually
 * sounds, but the wiring is genuine.
 */
// Typed loosely: each story builds its own store/presenter trio and
// drives the component through `render`, so we don't bind the required
// props as static `args`.
const meta: Meta = {
  title: 'Playback/PlaybackBar',
  component: PlaybackBar as Meta['component'],
  parameters: { layout: 'fullscreen' },
};
export default meta;

type Story = StoryObj;

/** Build a fresh document/playback store + presenter trio, optionally
 *  seeding a loaded jot so the transport controls light up. */
function usePlaybackHarness(withJot: boolean) {
  return React.useMemo(() => {
    const documentStore = new DocumentStore();
    const playback = new PlaybackStore(documentStore);
    const presenter = new PlaybackPresenter(playback, documentStore);
    if (withJot) {
      runInAction(() => {
        documentStore.currentJot = new RenderedJot(rockJot, documentStore.viewConfig);
      });
    }
    return { documentStore, playback, presenter };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

export const WithJotLoaded: Story = {
  render: () => {
    const { documentStore, playback, presenter } = usePlaybackHarness(true);
    return <PlaybackBar documentStore={documentStore} playback={playback} presenter={presenter} />;
  },
};

export const NoJot: Story = {
  render: () => {
    const { documentStore, playback, presenter } = usePlaybackHarness(false);
    return <PlaybackBar documentStore={documentStore} playback={playback} presenter={presenter} />;
  },
};
