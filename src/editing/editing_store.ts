import { makeAutoObservable } from 'mobx';

/** The current editing interaction mode.
 *  - `select`: the existing behaviour (click-to-seek, marquee, note select).
 *  - `insert`: hovering a lane shows a placeholder note; clicking commits it. */
export type EditMode = 'select' | 'insert';

/**
 * A preview note shown under the cursor while in insert mode, before the
 * user commits it with a click. Named "placeholder" (not "ghost") to avoid
 * confusion with a drumming *ghost note* (the `:g` modifier).
 */
export type PlaceholderNote = {
  /** Lane (drum instrument) the placeholder sits on. */
  lane: string;
  /** Owning bar id (where a committed note is inserted). */
  barId: string;
  /** Beat within the bar (quarter notes from the downbeat). */
  beat: number;
  /** Cumulative beat offset from the start of the layer's bars row; drives
   *  the rendered x via the same CSS calc the real notes use. */
  absBeat: number;
  /** The owning bar's length in beats; the upper clamp for snapping. */
  barBeats: number;
};

/**
 * Editing UI state: the current {@link EditMode} and the transient
 * insert-mode {@link placeholder}. Pure observable data; every mutation
 * lives on `EditingPresenter`.
 */
export class EditingStore {
  /** Current editing mode. Defaults to `select` (existing behaviour). */
  mode: EditMode = 'select';

  /** Insert-mode preview note under the cursor, or `undefined` when the
   *  cursor isn't over a lane (or not in insert mode). */
  placeholder: PlaceholderNote | undefined = undefined;

  /** When enabled, inserting and moving notes snaps to the grid at the
   *  resolution of the currently-enabled grid-line families. On by default;
   *  toggled from the Edit toolbar menu. */
  snappingEnabled: boolean = true;

  constructor() {
    makeAutoObservable(this);
  }
}
