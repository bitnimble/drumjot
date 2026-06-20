import { makeAutoObservable } from 'mobx';
import { DebugBundleManifest, NoteProvenanceFile } from 'src/editing/provenance/debug_zip';
import { ProvenanceStore } from './provenance_store';
import { ViewportStore } from '../viewport/viewport_store';
import type { Resettable } from '../session_reset';

/**
 * Mutations over {@link ProvenanceStore}, the debug-bundle / per-note
 * provenance state behind the filtered-onset overlays and the DebugPanel.
 * Reads {@link ViewportStore} only to clamp the panel height against the
 * live viewport.
 */
export class ProvenancePresenter implements Resettable {
  readonly provenance: ProvenanceStore;
  readonly viewport: ViewportStore;

  constructor(provenance: ProvenanceStore, viewport: ViewportStore) {
    this.provenance = provenance;
    this.viewport = viewport;
    makeAutoObservable(this, { provenance: false, viewport: false });
  }

  /** Replace the toolbar's `Show filtered` checkbox state. */
  setShowFilteredOnsets(show: boolean) {
    this.provenance.showFilteredOnsets = show;
  }

  setPinnedFilteredOnsetKey(key: string | undefined) {
    this.provenance.pinnedFilteredOnsetKey = key;
  }

  /** Toggle the DebugPanel's open state without forgetting the bundle. */
  toggleDebugPanel() {
    this.provenance.debugPanelOpen = !this.provenance.debugPanelOpen;
  }

  /** Resize the DebugPanel. Clamped so it can't shrink past the header or
   * grow past the viewport (with headroom for the toolbar). */
  setDebugPanelHeight(px: number): void {
    const max = Math.max(120, this.viewport._viewportHeight - 160);
    this.provenance.debugPanelHeight = Math.min(max, Math.max(80, px));
  }

  /**
   * Mount a freshly-loaded debug bundle's provenance: the manifest (logs
   * + stage timings behind the DebugPanel) and the per-note onset
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
   * DebugPanel's open/height chrome is UI state, not per-song, so it
   * survives the load. The debug-bundle loader runs this and then mounts
   * its own manifest/provenance afterwards.
   */
  reset(): void {
    this.provenance.lastDebugBundle = undefined;
    this.provenance.noteProvenance = undefined;
    this.provenance.showFilteredOnsets = false;
    this.provenance.pinnedFilteredOnsetKey = undefined;
  }
}
