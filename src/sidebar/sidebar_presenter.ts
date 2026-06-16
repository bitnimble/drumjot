import { makeAutoObservable } from 'mobx';
import { SidebarStore, type SidebarPanelId } from './sidebar_store';

/**
 * The only writer of {@link SidebarStore}. Owns the collapse/open toggle and
 * panel selection.
 */
export class SidebarPresenter {
  constructor(private readonly sidebar: SidebarStore) {
    makeAutoObservable<this, 'sidebar'>(this, { sidebar: false });
  }

  /** The collapse/open button: open ⇄ collapse the panel area. */
  toggleExpanded(): void {
    this.sidebar.expanded = !this.sidebar.expanded;
  }

  /**
   * Click a rail item: open its panel. Clicking the already-active panel while
   * it's open collapses it (VS Code-style toggle), so a single item still
   * opens and closes the sidebar.
   */
  selectPanel(id: SidebarPanelId): void {
    if (this.sidebar.expanded && this.sidebar.activePanel === id) {
      this.sidebar.expanded = false;
      return;
    }
    this.sidebar.activePanel = id;
    this.sidebar.expanded = true;
  }
}
