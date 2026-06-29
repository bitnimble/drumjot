import React from 'react';
import type { SidebarStore } from './sidebar_store';
import type { SidebarPresenter } from './sidebar_presenter';

/** Right-sidebar state + mutations, provided once at the app-shell level.
 *  `null` outside the View (tests / standalone renders). */
export const SidebarStoreContext = React.createContext<SidebarStore | null>(null);
export const SidebarPresenterContext = React.createContext<SidebarPresenter | null>(null);
