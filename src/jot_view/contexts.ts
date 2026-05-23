import React from 'react';
import { NoteProvenanceEntry } from 'src/debug_zip';
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
};

export const NoteProvenanceContext =
  React.createContext<NoteProvenanceContextValue | null>(null);
