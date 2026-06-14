import React from 'react';
import { RenderedJot } from 'src/jot';
import { BarTiming } from 'src/jot_view/playback';
import { SelectionStore } from 'src/selection';
import { GridLineSettings } from './store';

/**
 * Cross-cutting React contexts with no single feature home, derived
 * view-data and view-wide toggles read across several features. Each
 * feature's own contexts (mixer / viewport / lyrics / provenance /
 * playback) live next to that feature's store/presenter in
 * `<feature>/<feature>_contexts.ts`.
 */

/**
 * Routes the active {@link SelectionStore} to deep score chrome (today:
 * `NoteView`) without threading props through `JotView → MixerView →
 * InstrumentRow → BarView`. `null` outside the view so a `NoteView` rendered
 * in isolation just no-ops the click-to-select interaction.
 */
export const SelectionContext = React.createContext<SelectionStore | null>(null);

/**
 * Per-bar audio-time timings (start + duration, in seconds) for the
 * current jot, keyed by {@link StructuralBar.index}. Computed once at
 * the JotView level so deep consumers (today: NoteProvenanceDetails'
 * "Final position" row) can read a bar's absolute audio time without
 * depending on the playback timeline — the player's timeline is
 * `EMPTY_TIMELINE` until the first Play, but the math only needs the
 * jot's structure + tempos, so building it eagerly here makes the
 * lookup work even on an idle score.
 *
 * Keyed by `bar.index` rather than by `StructuralBar` reference
 * because the rendering chain shallow-clones bars (InstrumentRow rewrites
 * `tracks` for its lane) — the original reference doesn't survive the
 * walk down to NoteView. `bar.index` is preserved across those clones
 * and across `drumOffsetBeats` reflows, so it's the stable key.
 *
 * `null` outside the View or when the jot has no voices/bars.
 */
export const BarTimingsContext = React.createContext<
  ReadonlyMap<number, BarTiming> | null
>(null);

/**
 * The active {@link RenderedJot} for the current view. Provided once at
 * the JotView level so deep consumers (today: NoteProvenanceDetails'
 * timing-drift visualization, which reads `effectiveDrumOffsetBeats` to
 * account for the user-applied Beat-offset slider as a separate stage in
 * the detected → final chain) don't have to thread the jot down through
 * MixerView → InstrumentRow → BarView → NoteView.
 *
 * `null` outside the View; consumers should fall back to a sensible
 * "no offset / nothing to show" default in that case.
 */
export const RenderedJotContext = React.createContext<RenderedJot | null>(null);

/**
 * Grid-line toggles surfaced through the View dropdown. Threaded as
 * context so the deep `BarView` can read each setting without every
 * intermediate ({MixerView} → {InstrumentRow} → {BarView}) carrying a prop.
 * Defaults match the store's initial state so a BarView rendered outside
 * the View (e.g. unit tests, future embedded usage) still has the
 * classic look.
 */
export const GridLineSettingsContext = React.createContext<GridLineSettings>({
  mainBeat: true,
  subBeat16: false,
  subBeatQuarterTriplet: false,
  subBeatTriplet: false,
  subBeat48: false,
});

/**
 * Toolbar toggle: render audio-track waveforms with per-track
 * normalisation so the median non-silent peak fills most of the row,
 * regardless of source amplitude. Defaults to `false` so a canvas
 * rendered outside the View still shows the accurate signal level.
 * Read by `AudioTrackWaveformCanvas`.
 */
export const UniformWaveformsContext = React.createContext<boolean>(false);
