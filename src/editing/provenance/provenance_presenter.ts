import { makeAutoObservable } from 'mobx';
import { DebugBundleManifest, NoteProvenanceFile } from 'src/editing/provenance/debug_zip';
import { ProvenanceStore } from './provenance_store';
import type { Resettable } from '../session_reset';

/**
 * Mutations over {@link ProvenanceStore}, the debug-bundle / per-note
 * provenance state behind the filtered-onset overlays and the sidebar Debug
 * panel.
 */
export class ProvenancePresenter implements Resettable {
  readonly provenance: ProvenanceStore;

  constructor(provenance: ProvenanceStore) {
    this.provenance = provenance;
    makeAutoObservable(this, { provenance: false });
  }

  /** Replace the toolbar's `Show filtered` checkbox state. */
  setShowFilteredOnsets(show: boolean) {
    this.provenance.showFilteredOnsets = show;
  }

  setPinnedFilteredOnsetKey(key: string | undefined) {
    this.provenance.pinnedFilteredOnsetKey = key;
  }

  /**
   * Mount a freshly-loaded debug bundle's provenance: the manifest (logs
   * + stage timings behind the sidebar Debug panel) and the per-note onset
   * provenance behind the selection label / filtered-onset ghosts. Always
   * resets the visibility toggle so a new bundle reads as just "the score"
   * until the operator opts into the ghost overlays. `noteProvenance` is
   * cleared when the bundle didn't ship one (legacy / hand-built zips) so
   * the previous bundle's provenance can't leak onto the new score.
   */
  loadDebugBundle(
    manifest: DebugBundleManifest,
    noteProvenance: NoteProvenanceFile | undefined
  ) {
    this.provenance.lastDebugBundle = manifest;
    this.provenance.noteProvenance = noteProvenance;
    this.provenance.showFilteredOnsets = false;
  }

  /** Drop the debug bundle's per-note provenance + reset the visibility
   * toggle. Called from every loader that replaces the current song
   * outside the bundle path so stale debug info doesn't leak onto the
   * new score. */
  clearNoteProvenance() {
    this.provenance.noteProvenance = undefined;
    this.provenance.showFilteredOnsets = false;
  }

  /**
   * Session reset: drop the loaded bundle's per-song debug state (manifest,
   * per-note provenance, the filtered-onset overlay toggle + pin). The
   * sidebar's open/active-panel state is UI chrome owned by SidebarStore, not
   * per-song, so it survives the load. The debug-bundle loader runs this and
   * then mounts its own manifest/provenance afterwards.
   */
  reset(): void {
    this.provenance.lastDebugBundle = undefined;
    this.provenance.noteProvenance = undefined;
    this.provenance.showFilteredOnsets = false;
    this.provenance.pinnedFilteredOnsetKey = undefined;
  }
}
