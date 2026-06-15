import type { Meta, StoryObj } from '@storybook/react-vite';
import { runInAction } from 'mobx';
import React from 'react';
import { buildJotModel } from '../../jot_view_store';
import { rockJot } from 'src/fakes/fakes';
import { JotViewStore } from '../../jot_view_store';
import { PlaybackStore } from '../playback_store';
import { PlaybackPresenter } from '../playback_presenter';
import { PlaybackBar } from '../playback';
import { Gallery, Variant } from 'src/ui/stories/_variants';

/**
 * The bottom transport bar, driven by REAL stores + presenter (the point
 * of the store/presenter split: a component can be mounted in isolation
 * with its own peer instances). Play/stop/offset go through a live
 * PlaybackPresenter; the player singleton stays idle so nothing actually
 * sounds, but the wiring is genuine.
 */
const meta: Meta = {
  title: 'Playback/PlaybackBar',
  parameters: { layout: 'fullscreen' },
};
export default meta;

type Story = StoryObj;

/** One PlaybackBar backed by a fresh document/playback store + presenter
 *  trio; `withJot` seeds a loaded jot so the transport controls light up. */
function Bar({ withJot }: { withJot: boolean }) {
  const { jotViewStore, playback, presenter } = React.useMemo(() => {
    const jotViewStore = new JotViewStore();
    const playback = new PlaybackStore(jotViewStore);
    const presenter = new PlaybackPresenter(playback, jotViewStore);
    if (withJot) {
      const model = buildJotModel(rockJot, jotViewStore.viewConfig);
      runInAction(() => {
        jotViewStore.source = rockJot;
        jotViewStore.structural = model.structural;
        jotViewStore.palette = model.palette;
        jotViewStore.tempo = model.tempo;
      });
    }
    return { jotViewStore, playback, presenter };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return <PlaybackBar jotViewStore={jotViewStore} playback={playback} presenter={presenter} />;
}

/** Both transport states in one place. */
export const All: Story = {
  render: () => (
    <Gallery>
      <Variant label="Jot loaded (transport active)">
        <Bar withJot />
      </Variant>
      <Variant label="No jot (empty state)">
        <Bar withJot={false} />
      </Variant>
    </Gallery>
  ),
};
