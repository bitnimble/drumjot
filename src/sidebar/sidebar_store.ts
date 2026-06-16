import { makeAutoObservable } from 'mobx';

/** The panels the right sidebar can show. Only `layers` exists today; the
 *  panel body itself is a stub until a later PR. */
export type SidebarPanelId = 'layers';

/**
 * Right-sidebar UI state: whether the panel is expanded and which panel is
 * active. Pure observable data, every mutation lives on
 * {@link SidebarPresenter}. Survives hot reload (a deliberate, store-owned UI
 * preference), so it belongs in a store rather than React-local state.
 */
export class SidebarStore {
  /** Whether the panel area is open (the rail of icons is always visible). */
  expanded: boolean = false;

  /** Which panel the rail last selected; shown when {@link expanded}. */
  activePanel: SidebarPanelId = 'layers';

  constructor() {
    makeAutoObservable(this);
  }
}
