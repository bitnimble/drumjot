import React from 'react';
import type { HistoryStore } from './history_store';
import type { HistoryPresenter } from './history_presenter';

/**
 * Undo/redo availability + its mutations, provided at the JotEditor level so
 * the toolbar's Edit menu can read enablement and invoke undo/redo directly.
 * `null` outside the View (tests / standalone renders).
 */
export const HistoryStoreContext = React.createContext<HistoryStore | null>(null);
export const HistoryPresenterContext = React.createContext<HistoryPresenter | null>(null);
