import { readFileSync } from 'node:fs';
import { expect, test } from 'bun:test';
import { runInAction } from 'mobx';
import { loadParadbZip } from 'src/schema/rlrr/paradb';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import { PlaybackStore } from 'src/editing/playback/playback_store';
import { PARADB_MAP_EXPECTED_EPOCHS, PARADB_MAP_PATH } from 'src/editing/test/paradb.helper';

/**
 * Epoch derivation on a real full-length song. Loads the `E2E_PARADB_MAP`
 * pack (a complete chart with a real audio lead-in AND a longer synthetic
 * rendered lead-in), runs it through the actual import → structure → tempo
 * pipeline, and checks `PlaybackStore.epochs` against the hand-verified
 * anchors. This pins down the contract the seek/playback engines depend on:
 *
 *   fullLeadIn  <=  songLeadIn  <=  drums (0)
 *
 * `songLeadIn` is the recording's audio lead-in (seeded from the jot's
 * `globalMetadata.songLeadIn`, exactly as `PlaybackPresenter` does on load);
 * `fullLeadIn` is the rendered left edge, the first bar's `startSec` in the
 * tempo timeline, which includes the view-only virtual lead-in bar. A unit
 * test (not e2e) so it asserts the exact numbers deterministically with no
 * browser / audio context. Skipped when `E2E_PARADB_MAP` is unset (the pack
 * is large + machine-local, never committed).
 */
test.skipIf(!PARADB_MAP_PATH)(
  'PlaybackStore.epochs on the full ParaDB-map song matches the verified anchors',
  async () => {
    const bytes = readFileSync(PARADB_MAP_PATH as string);
    const file = new File([bytes as BlobPart], 'map.zip', { type: 'application/zip' });
    const { jot } = await loadParadbZip(file);

    // Real load pipeline: build the reactive doc + structure/tempo peers.
    const store = new JotEditorStore();
    store.loadSource(jot);

    // Mirror PlaybackPresenter's seed-on-load: the live `songLeadIn` epoch
    // comes from `globalMetadata.songLeadIn`, clamped to <= 0.
    const raw = jot.globalMetadata.songLeadIn;
    const seed = typeof raw === 'number' ? Math.min(0, raw) : 0;
    const playback = new PlaybackStore(store);
    runInAction(() => {
      playback.songLeadInSec = seed;
    });

    expect(playback.epochs).toEqual({ ...PARADB_MAP_EXPECTED_EPOCHS });
  },
);
