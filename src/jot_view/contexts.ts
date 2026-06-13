import React from 'react';
import { NoteProvenanceEntry } from 'src/debug_zip';
import { RenderedJot } from 'src/jot';
import { BarTiming } from 'src/playback';
import { SelectionStore } from 'src/selection';
import { GridLineSettings, JotViewStore } from './store';
import { ProvenanceStore } from './stores/provenance_store';
import { LyricsAlignStore } from './stores/lyrics_align_store';
import { ViewportStore } from './stores/viewport_store';
import { MixerStore } from './stores/mixer_store';
import { JotViewerPresenter } from './jot_viewer_presenter';

/**
 * Routes the active {@link SelectionStore} to deep score chrome (today:
 * `NoteView`) without threading props through `JotView → MixerView →
 * InstrumentRow → BarView`. `null` outside the view so a `NoteView` rendered
 * in isolation just no-ops the click-to-select interaction.
 */
export const SelectionContext = React.createContext<SelectionStore | null>(null);

/**
 * Routes the active {@link JotViewStore} to deep consumers that need to
 * read or drive view-level state (today: `FilteredOnsetView`'s pinned-
 * popover identity). `null` outside the view; consumers should no-op or
 * fall back to local state.
 */
export const JotViewStoreContext = React.createContext<JotViewStore | null>(null);

/**
 * Routes the {@link ProvenanceStore} to deep consumers that read debug-
 * bundle / provenance state (today: `FilteredOnsetView`'s pinned-popover
 * identity). `null` outside the view.
 */
export const ProvenanceStoreContext = React.createContext<ProvenanceStore | null>(null);

/**
 * Routes the {@link LyricsAlignStore} to deep consumers that read lyrics
 * align state (today: `LyricsRow`'s per-row align spinner). `null`
 * outside the view.
 */
export const LyricsAlignStoreContext = React.createContext<LyricsAlignStore | null>(null);

/**
 * Routes the {@link ViewportStore} to deep consumers that read scroll /
 * zoom / visible-range state (today: score `WindowedTicks` / `PopoverPortal`
 * and `WindowedLyricLines`). `null` outside the view.
 */
export const ViewportStoreContext = React.createContext<ViewportStore | null>(null);

/**
 * Routes the {@link MixerStore} to deep consumers that read mixer state
 * (today: `MixerView`'s row order, the per-row audio-split status, the
 * per-instrument colour view-models). `null` outside the view.
 */
export const MixerStoreContext = React.createContext<MixerStore | null>(null);

/**
 * Routes the {@link JotViewerPresenter} to deep consumers that need to
 * invoke actions (today: `FilteredOnsetView` pinning a popover). `null`
 * outside the view; consumers no-op when absent.
 */
export const JotViewerPresenterContext = React.createContext<JotViewerPresenter | null>(null);

/**
 * Routes the loaded debug bundle's per-note provenance to two deep
 * consumers: `NoteView` (looks up its own entry via `byTick` to render
 * the `Debug details` collapsible in the selection label) and `InstrumentRow`
 * (reads `rejectedByPitch` + `leadBars` + `showFiltered` to render
 * filtered onsets as ghost overlays). `null` outside the View, or when
 * no bundle is loaded — both consumers no-op in that case.
 */
export type NoteProvenanceContextValue = {
  /** Keyed by `${pitch}:${tick}` — exact-match lookup from NoteView. */
  byTick: Map<string, NoteProvenanceEntry>;
  /**
   * Per-pitch rejected onsets used by InstrumentRow to render the dashed
   * ghost overlays. Out-of-range entries are pre-filtered out (they
   * have no anchored bar to render against).
   */
  rejectedByPitch: Map<string, NoteProvenanceEntry[]>;
  /**
   * The `lead_bars` field from the provenance file. The MIDI lays
   * `lead_bars` empty bar-0-sized blocks before bar 0 to absorb the
   * audio lead-in, so a struct bar `b` maps to the rendered jot's
   * `bars[lead_bars + b]`.
   */
  leadBars: number;
  /** Toolbar checkbox state — true when the user opted into rendering
   * the rejected-onset overlays. NoteView's Debug details remain
   * available regardless (they are per-kept-note, not gated). */
  showFiltered: boolean;
  /**
   * Global beat-grid alignment offset (seconds) the beat tracker
   * applied. Same value for every entry, surfaced here so the
   * per-note Debug details panel can show it as "Grid align" without
   * threading the whole provenance file through. `null` when the
   * sidecar didn't record one (older bundles).
   */
  beatAlignmentOffsetSec: number | null;
  /** Coarse envelope-phase alignment shift (`align_beats_to_envelope`),
   * separated from the combined `beatAlignmentOffsetSec` so the popup
   * can show the two alignment passes as distinct stages. `null` when
   * the bundle predates provenance format v3 or the pass didn't apply
   * a shift. */
  beatAlignCoarseOffsetSec: number | null;
  /** Fine median onset-snap alignment shift (`align_beats_to_onsets`).
   * `null` like {@link beatAlignCoarseOffsetSec}. */
  beatAlignFineOffsetSec: number | null;
  /**
   * Bundle-manifest mapping from pitch letter (and the synthetic
   * `no_drums` key) to the audio filename inside the bundle, e.g.
   * `k` → `stem_k.mp3`. Sourced from `DebugBundleManifest.mapping`.
   * Used by the per-onset timing visualization to pick the right
   * loaded audio track for a given pitch (the isolated stem shows
   * the drum hit far more clearly than the full mix or the
   * `no_drums` backing). Empty when the bundle didn't ship a mapping
   * — the visualization falls back to filename heuristics in that
   * case.
   */
  audioFilenameByPitch: ReadonlyMap<string, string>;
};

export const NoteProvenanceContext =
  React.createContext<NoteProvenanceContextValue | null>(null);

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

/**
 * Whether the score auto-scrolls to keep the playhead centred during
 * playback, and the toggle that flips it. Read by two distant
 * consumers: `PlayheadAutoScroller` (skips the per-frame `scrollLeft`
 * write when `follow` is false) and the `FollowToggle` button stacked
 * above the playhead label. Threading through `JotView →
 * TimelineHeader → Playhead → PlayheadLabel` for one boolean + one
 * handler is more noise than it's worth, hence the context. Defaults
 * to `{ follow: true, toggle: noop }` so a Playhead rendered outside
 * the View still behaves like today's always-follow build.
 */
export type FollowPlayheadContextValue = {
  follow: boolean;
  toggle: () => void;
};

export const FollowPlayheadContext = React.createContext<FollowPlayheadContextValue>({
  follow: true,
  toggle: () => {},
});
