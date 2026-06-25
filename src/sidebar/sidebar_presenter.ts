import { makeAutoObservable } from 'mobx';
import { SidebarStore, type SidebarPanelId } from './sidebar_store';

/**
 * The only writer of {@link SidebarStore}. Owns panel selection and the
 * float/pin toggle.
 */
export class SidebarPresenter {
  constructor(private readonly sidebar: SidebarStore) {
    makeAutoObservable<this, 'sidebar'>(this, { sidebar: false });
  }

  /**
   * The rail's topmost button. With the panel open it flips placement between
   * floating (overlay) and pinned (docked, reflowing the score); with it
   * closed it opens the active panel pinned. It never closes the panel, that's
   * the panel rail items' job ({@link selectPanel}).
   */
  togglePin(): void {
    if (!this.sidebar.expanded) {
      this.sidebar.expanded = true;
      this.sidebar.pinned = true;
      return;
    }
    this.sidebar.pinned = !this.sidebar.pinned;
  }

  /**
   * Click a rail item: open its panel (in the current placement mode, floating
   * by default). Clicking the already-active panel while it's open collapses
   * it (VS Code-style toggle), so a single item still opens and closes the
   * sidebar.
   */
  selectPanel(id: SidebarPanelId): void {
    if (this.sidebar.expanded && this.sidebar.activePanel === id) {
      this.sidebar.expanded = false;
      return;
    }
    this.sidebar.activePanel = id;
    this.sidebar.expanded = true;
  }

  /**
   * Dismiss the open panel. Used by the floating panel's outside-click /
   * Escape handling; leaves {@link SidebarStore.pinned} untouched so the next
   * open restores the same placement.
   */
  collapse(): void {
    this.sidebar.expanded = false;
  }
}
