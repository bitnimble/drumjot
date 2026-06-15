import React from 'react';
import { BarTiming } from 'src/jot_view/playback/timeline';
import type { StructuralPresenter } from 'src/jot_view/structure/structural_presenter';
import type { TempoPresenter } from 'src/jot_view/playback/tempo_presenter';
import type { PaletteStore } from 'src/jot_view/palette/palette_store';

/**
 * The loaded song's peer domains, provided once at the JotView level so deep
 * consumers read the one they need directly instead of threading a `jot` prop
 * down through MixerView → InstrumentRow → BarView → NoteView. Each is `null`
 * outside the View (e.g. tests / a filtered-ghost render); consumers fall back
 * to a sensible default in that case.
 *
 * - {@link StructuralContext}: structure / layout (`pxPerBeat`, `voices`,
 *   `voiceBeats`, `barsForPitch`) + the user-applied Beat-offset
 *   (`effectiveDrumOffsetBeats`) the provenance drift visualisation reads.
 * - {@link TempoContext}: per-bar tempos + the audio timeline.
 * - {@link PaletteContext}: per-pitch colours + the legend.
 */
export const StructuralContext = React.createContext<StructuralPresenter | null>(null);
export const TempoContext = React.createContext<TempoPresenter | null>(null);
export const PaletteContext = React.createContext<PaletteStore | null>(null);

/**
 * Per-bar audio-time timings (start + duration, in seconds) for the
 * current jot, keyed by {@link StructBar.index}. Computed once at
 * the JotView level so deep consumers (today: NoteProvenanceDetails'
 * "Final position" row) can read a bar's absolute audio time without
 * depending on the playback timeline; the player's timeline is
 * `EMPTY_TIMELINE` until the first Play, but the math only needs the
 * jot's structure + tempos, so building it eagerly here makes the
 * lookup work even on an idle score.
 *
 * Keyed by `bar.index` rather than by `StructBar` reference because
 * the rendering chain shallow-clones bars (InstrumentRow rewrites
 * `tracks` for its lane), the original reference doesn't survive the
 * walk down to NoteView. `bar.index` is preserved across those clones
 * and across `drumOffsetBeats` reflows, so it's the stable key.
 *
 * `null` outside the View or when the jot has no voices/bars.
 */
export const BarTimingsContext = React.createContext<
  ReadonlyMap<number, BarTiming> | null
>(null);
