/**
 * Session-reset contract for the editor's persistent peer stores.
 *
 * Loading a song wholesale (a `.jot` / MIDI / ParaDB / debug-bundle load, or
 * a Save-file open) must return every persistent store to a clean state, as
 * if the page had just been refreshed, so the previous song's transient and
 * per-song state can't leak onto the new one (stale selection, a phantom
 * solo, lingering audio tracks, the previous recording's debug provenance,
 * …). Rather than have each loader hand-clear an ad-hoc subset of stores
 * (which drifted: the plain `.jot`/MIDI loaders historically forgot to drop
 * the previous song's audio tracks + lane mixer that the ParaDB/bundle
 * loaders did clear), every reset owner implements {@link Resettable} and is
 * registered once in the {@link SessionReset} registry at the composition
 * root. The load orchestrator fires {@link SessionReset.reset} exactly once
 * per load.
 *
 * **Where `reset()` lives.** For a pure-state store (return its observables
 * to their constructor defaults) the method sits on the store itself, the
 * sanctioned exception to "stores hold no mutation logic", alongside the
 * existing {@link JotEditorStore.loadSource} / `LyricsStore.clear`
 * precedent. For a domain with imperative teardown beyond store state
 * (playback must stop the engine; the mixer must tear down the audio graph)
 * the method sits on that domain's presenter, keeping the single-writer rule
 * intact. {@link SessionReset} holds whichever object owns reset for each
 * domain.
 *
 * **What does NOT reset.** Genuinely global / session-level preferences that
 * a page refresh is the only thing that should clear: the sidebar's
 * open/active-panel/pin state, the zoom level + viewport gutter, and the
 * transcribe form + recent-runs picker. Those stores are simply absent from
 * the registry. Per-song display settings that the user
 * asked to travel with the song (the grid-line overlay, the colour palette)
 * DO reset, and are then re-applied from a loaded save file's editor metadata
 * (see `src/editing/persistence/`).
 *
 * **Stores rebuilt, not reset.** The loaded song's structure/palette/tempo
 * peers (`StructureStore`, `PaletteStore`, `LayoutStore`, `StructuralPresenter`,
 * `TempoPresenter`) and the backing Loro document are torn down and
 * reconstructed by {@link JotEditorStore.loadSource} / `loadState` on each
 * load, so they need no reset entry.
 */
export interface Resettable {
  /** Return this domain to its fresh-session state. Called exactly once per
   *  wholesale song load, before the new song is installed. */
  reset(): void;
}

/**
 * The composition root's registry of every {@link Resettable} peer. Built
 * once in `createJotEditor` after all stores/presenters exist, then handed to
 * the load orchestrator (`JotEditorPresenter`), which fires {@link reset}
 * exactly once at the start of every wholesale load.
 */
export class SessionReset {
  private readonly targets: readonly Resettable[];

  constructor(targets: readonly Resettable[]) {
    this.targets = targets;
  }

  /** Reset every registered peer, in registration order. */
  reset(): void {
    for (const target of this.targets) target.reset();
  }
}
