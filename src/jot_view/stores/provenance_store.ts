import { makeAutoObservable } from 'mobx';
import {
  DebugBundleManifest,
  NoteProvenanceEntry,
  NoteProvenanceFile,
} from 'src/debug_zip';
import type { NoteProvenanceContextValue } from '../contexts';

/**
 * Map a transcriber-side provenance pitch tag onto the jot's pitch
 * letter. The transcriber's hi-hat split (`transcriber/app/pipeline/
 * hihat_split.py`) routes open-hi-hat onsets through synthetic pitch
 * `H` so the filter LLM can see closed (`h`) and open (`H`) hits as
 * separate lanes; from_midi.ts then folds those back into the standard
 * `h:o` notation. Provenance lookups (debug-details popover, "show
 * filtered" ghost overlays) have to canonicalise the same way so the
 * jot's `note.pitch = 'h'` finds entries the provenance stored under
 * `'H'`. Adding new synthetic-pitch routes (e.g. a future ride-bell
 * split) means adding a case here.
 */
function canonicalProvenancePitch(transcriberPitch: string): string {
  if (transcriberPitch === 'H') return 'h';
  return transcriberPitch;
}

/**
 * Transcriber debug-bundle state: the manifest (logs + stage timings)
 * behind the DebugPanel, the per-note onset provenance behind the
 * selection label + filtered-onset ghosts, and the DebugPanel's own
 * open/height chrome.
 *
 * Pure data: observables + derived computeds. The bundle-loading
 * orchestration that populates these (applyDebugBundle, clearNoteProvenance)
 * lives on the presenter.
 */
export class ProvenanceStore {
  /**
   * Last loaded transcriber debug bundle (`.zip`), if any. Carries the
   * captured logs + per-stage timings produced server-side during a
   * transcribe run, so the UI's DebugPanel can show what happened end-
   * to-end without requiring a `docker compose logs` round trip.
   * Replaced when a new bundle is loaded; otherwise survives jot/audio
   * changes.
   */
  lastDebugBundle: DebugBundleManifest | undefined = undefined;
  /**
   * Per-note debug provenance from the loaded debug bundle, if the
   * bundle came from a filter-mode transcribe run. Keyed by DSL pitch
   * letter → list of every detected onset (kept and rejected). The
   * NoteView selection label looks up its provenance by matching
   * `note.metadata.midi.tick` against entries' `tick`; the
   * FilteredOnsetView renders the `kept=false` entries as ghost
   * overlays gated by {@link showFilteredOnsets}. `undefined` until a
   * filter-mode bundle is loaded; cleared when a new (non-bundle) song
   * replaces the current one.
   */
  noteProvenance: NoteProvenanceFile | undefined = undefined;
  /**
   * Toolbar checkbox: show rejected onsets as dashed ghost overlays.
   * Only meaningful when {@link noteProvenance} is loaded; the checkbox
   * is hidden when there's nothing to show. Default off so a freshly
   * loaded bundle reads as just "the score" until the operator opts in.
   */
  showFilteredOnsets: boolean = false;
  /** Identifies which filtered-onset popover is pinned open. The key is
   * `${pitch}:${detected_time_sec}` (rejected onsets have `tick === null`,
   * so we can't use it); `undefined` means none pinned. Hover-only popovers
   * don't go through here. */
  pinnedFilteredOnsetKey: string | undefined = undefined;
  /** Whether the DebugPanel is expanded, small UI state, kept here so
   * the toolbar toggle and the panel itself stay in sync. */
  debugPanelOpen: boolean = false;
  /** Height of the DebugPanel (px) when expanded; adjusted by dragging
   * the resize handle along its top edge. */
  debugPanelHeight: number = 280;

  constructor() {
    makeAutoObservable(this);
  }

  /**
   * Bundle the per-note debug provenance into the shape
   * {@link NoteProvenanceContextValue} consumers expect, or `null` when no
   * filter-mode bundle is loaded. Memoised through the MobX computed
   * graph so the `audioFilenameByPitch` Map (rebuilt from the manifest's
   * plain-object `mapping`) is only re-constructed when the underlying
   * provenance / bundle / toggle changes.
   */
  get provenanceContextValue(): NoteProvenanceContextValue | null {
    const provenance = this.noteProvenance;
    if (!provenance) return null;
    return {
      byTick: this.noteProvenanceByTick,
      rejectedByPitch: this.filteredOnsetsByPitch,
      leadBars: provenance.lead_bars ?? 0,
      showFiltered: this.showFilteredOnsets,
      beatAlignmentOffsetSec: provenance.beat_alignment_offset_sec ?? null,
      beatAlignCoarseOffsetSec: provenance.beat_align_coarse_offset_sec ?? null,
      beatAlignFineOffsetSec: provenance.beat_align_fine_offset_sec ?? null,
      // Bundle manifest mapping is `Record<string, string>`; rebuild it
      // as a Map for ergonomic .get() lookups inside the per-onset
      // timing visualization. Empty when the current bundle didn't ship
      // a manifest (hand-authored jots, legacy bundles).
      audioFilenameByPitch: new Map(Object.entries(this.lastDebugBundle?.mapping ?? {})),
    };
  }

  /**
   * Pre-indexed view onto `noteProvenance` for the per-note selection
   * label lookup. Keyed by `${pitch}:${tick}` so `NoteView` can attach
   * provenance to its note in O(1) instead of scanning the per-pitch
   * list on every render. Recomputed when `noteProvenance` changes.
   *
   * Pitch keys are canonicalised through {@link canonicalProvenancePitch}
   * so the rendered jot's pitch letter (what `NoteView` builds the
   * lookup key from) matches the provenance regardless of any synthetic
   * routing pitches the transcriber pipeline used (today: `H` for open
   * hi-hat, which `from_midi.ts` collapses back into `h:o`).
   */
  get noteProvenanceByTick(): Map<string, NoteProvenanceEntry> {
    const out = new Map<string, NoteProvenanceEntry>();
    const provenance = this.noteProvenance;
    if (!provenance) return out;
    for (const [pitch, entries] of Object.entries(provenance.per_pitch)) {
      const jotPitch = canonicalProvenancePitch(pitch);
      for (const entry of entries) {
        if (entry.tick === null || !entry.kept) continue;
        out.set(`${jotPitch}:${entry.tick}`, entry);
      }
    }
    return out;
  }

  /**
   * Per-pitch list of rejected onsets the {@link FilteredOnsetView}
   * renders. Built once from `noteProvenance` and cached via MobX so
   * the per-instrument row doesn't re-filter on every render. Out-of-range
   * entries (those that fell outside the beat-tracked region) are
   * dropped, they have no displayable bar to anchor against.
   */
  get filteredOnsetsByPitch(): Map<string, NoteProvenanceEntry[]> {
    const out = new Map<string, NoteProvenanceEntry[]>();
    const provenance = this.noteProvenance;
    if (!provenance) return out;
    for (const [pitch, entries] of Object.entries(provenance.per_pitch)) {
      const rejected = entries.filter((e) => !e.kept && !e.out_of_range);
      if (rejected.length === 0) continue;
      // Canonicalise so the consuming instrument row (which keys by the
      // jot's `note.pitch`) finds entries even when the transcriber
      // routed them through a synthetic pitch like `H` for open hat;
      // merge into the existing bucket rather than overwriting so the
      // closed (`h`) and open (`H` → `h`) rejected lists land together.
      const jotPitch = canonicalProvenancePitch(pitch);
      const existing = out.get(jotPitch);
      if (existing) {
        existing.push(...rejected);
      } else {
        out.set(jotPitch, rejected);
      }
    }
    return out;
  }
}
