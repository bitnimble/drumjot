import React from 'react';
import { NoteProvenanceEntry } from 'src/debug_zip';
import { BarTiming } from 'src/playback';
import { SelectionStore } from 'src/selection';

/**
 * Routes the active {@link SelectionStore} to deep score chrome (today:
 * `NoteView`) without threading props through `JotView → MixerView →
 * PitchRow → BarView`. `null` outside the view so a `NoteView` rendered
 * in isolation just no-ops the click-to-select interaction.
 */
export const SelectionContext = React.createContext<SelectionStore | null>(null);

/**
 * Routes the loaded debug bundle's per-note provenance to two deep
 * consumers: `NoteView` (looks up its own entry via `byTick` to render
 * the `Debug details` collapsible in the selection label) and `PitchRow`
 * (reads `rejectedByPitch` + `leadBars` + `showFiltered` to render
 * filtered onsets as ghost overlays). `null` outside the View, or when
 * no bundle is loaded — both consumers no-op in that case.
 */
export type NoteProvenanceContextValue = {
  /** Keyed by `${pitch}:${tick}` — exact-match lookup from NoteView. */
  byTick: Map<string, NoteProvenanceEntry>;
  /**
   * Per-pitch rejected onsets used by PitchRow to render the dashed
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
   * applied. Same value for every entry — surfaced here so the
   * per-note Debug details panel can show it as "Grid align" without
   * threading the whole provenance file through. `null` when the
   * sidecar didn't record one (older bundles).
   */
  beatAlignmentOffsetSec: number | null;
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
 * because the rendering chain shallow-clones bars (PitchRow rewrites
 * `tracks` for its lane) — the original reference doesn't survive the
 * walk down to NoteView. `bar.index` is preserved across those clones
 * and across `drumOffsetBeats` reflows, so it's the stable key.
 *
 * `null` outside the View or when the jot has no voices/bars.
 */
export const BarTimingsContext = React.createContext<
  ReadonlyMap<number, BarTiming> | null
>(null);
