import { makeAutoObservable } from 'mobx';

/** The panels the right sidebar can show. The store owns this (it's the type of
 *  its own `activePanel` state); the view registry in `panels/sidebar_panels`
 *  maps each id to its rail icon + panel body and is `satisfies`-checked against
 *  this union, so a new panel is a new member here plus a registry entry. */
export type SidebarPanelId = 'layers' | 'note_properties';

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
