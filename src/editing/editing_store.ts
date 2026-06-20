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
  /** The `||` layer of the row the cursor is over, so a committed note lands
   *  in the clicked row's layer (per-track view). `undefined` on a merged row
   *  (the merge view collapses layers), where insert falls back to the
   *  firstmost layer carrying the lane. */
  layerId?: string;
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
 * One placed note's live position during a drag-move OR a paste placement,
 * rendered as a preview glyph by whichever lane row owns {@link lane}. The set
 * of these (one per selected note for a move, one per copied note for a paste)
 * is the top-down source of truth for the preview, so no DOM is read to
 * position it.
 */
export type DragPreviewNote = {
  /** For a move, the id of the note being dragged (also hides its real glyph
   *  while dragging). For a paste, a synthetic `paste:<i>` key (no real glyph
   *  to hide; the notes don't exist until commit). */
  id: string;
  /** Lane the preview currently sits on (the row that renders it). */
  lane: string;
  /** Cumulative beat offset from the layer's start; drives the rendered x via
   *  the same CSS calc the insert placeholder + real notes use. */
  absBeat: number;
};

/**
 * Editing UI state: the current {@link EditMode}, the transient insert-mode
 * {@link placeholder}, and the live drag-move preview. Pure observable data;
 * every mutation lives on `EditingPresenter`.
 */
export class EditingStore {
  /** Current editing mode. Defaults to `select` (existing behaviour). */
  mode: EditMode = 'select';

  /** Insert-mode preview note under the cursor, or `undefined` when the
   *  cursor isn't over a lane (or not in insert mode). */
  placeholder: PlaceholderNote | undefined = undefined;

  /** True while a note drag-move is in progress. Drives hiding the dragged
   *  glyphs + the selection frame. Flips only at drag start / end (never per
   *  pointer move), so observers reading it don't churn during a drag. */
  dragActive: boolean = false;

  /** True while a paste placement is in flight: the copied cluster follows the
   *  cursor as preview glyphs ({@link dragPreview}), a click commits it, Esc
   *  cancels. Distinct from {@link dragActive}, which also hides the dragged
   *  real glyphs by id; a paste has no pre-existing glyphs to hide, so the
   *  bars-row pointer handlers route to the paste path on this flag instead. */
  pasteActive: boolean = false;

  /** Live preview of the dragged notes at their current (snapped) target lane
   *  + position. Rewritten on every pointer move during a drag; empty idle.
   *  Each lane row renders the entries whose {@link DragPreviewNote.lane}
   *  matches it. */
  dragPreview: DragPreviewNote[] = [];

  /** When enabled, inserting and moving notes snaps to the grid at the
   *  resolution of the currently-enabled grid-line families. On by default;
   *  toggled from the Edit toolbar menu. */
  snappingEnabled: boolean = true;

  constructor() {
    makeAutoObservable(this);
  }
}
