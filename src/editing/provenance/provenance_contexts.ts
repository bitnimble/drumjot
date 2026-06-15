import React from 'react';
import { NoteProvenanceEntry } from 'src/editing/provenance/debug_zip';
import { ProvenanceStore } from './provenance_store';
import { ProvenancePresenter } from './provenance_presenter';

/**
 * Routes the {@link ProvenanceStore} to deep consumers that read debug-
 * bundle / provenance state (today: `FilteredOnsetView`'s pinned-popover
 * identity). `null` outside the view.
 */
export const ProvenanceStoreContext = React.createContext<ProvenanceStore | null>(null);

/**
 * Routes the {@link ProvenancePresenter} to deep consumers that mutate
 * debug-bundle / provenance state (today: `FilteredOnsetView` pinning a
 * popover via `setPinnedFilteredOnsetKey`). `null` outside the view;
 * consumers no-op when absent.
 */
export const ProvenancePresenterContext =
  React.createContext<ProvenancePresenter | null>(null);

/**
 * Routes the loaded debug bundle's per-note provenance to two deep
 * consumers: `NoteView` (looks up its own entry via `byTick` to render
 * the `Debug details` collapsible in the selection label) and `InstrumentTrackView`
 * (reads `rejectedByLane` + `leadBars` + `showFiltered` to render
 * filtered onsets as ghost overlays). `null` outside the View, or when
 * no bundle is loaded — both consumers no-op in that case.
 */
export type NoteProvenanceContextValue = {
  /** Keyed by `${lane}:${tick}` — exact-match lookup from NoteView. */
  byTick: Map<string, NoteProvenanceEntry>;
  /**
   * Per-lane rejected onsets used by InstrumentTrackView to render the dashed
   * ghost overlays. Out-of-range entries are pre-filtered out (they
   * have no anchored bar to render against).
   */
  rejectedByLane: Map<string, NoteProvenanceEntry[]>;
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
   * Bundle-manifest mapping from lane letter (and the synthetic
   * `no_drums` key) to the audio filename inside the bundle, e.g.
   * `k` → `stem_k.mp3`. Sourced from `DebugBundleManifest.mapping`.
   * Used by the per-onset timing visualization to pick the right
   * loaded audio track for a given lane (the isolated stem shows
   * the drum hit far more clearly than the full mix or the
   * `no_drums` backing). Empty when the bundle didn't ship a mapping
   * — the visualization falls back to filename heuristics in that
   * case.
   */
  audioFilenameByLane: ReadonlyMap<string, string>;
};

export const NoteProvenanceContext =
  React.createContext<NoteProvenanceContextValue | null>(null);
