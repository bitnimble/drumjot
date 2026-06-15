import React from 'react';
import type { EditingStore } from './editing_store';
import type { EditingPresenter } from './editing_presenter';

/**
 * Editing-mode state + mutations, provided once at the JotEditor level so the
 * floating mode toolbar and the bars-row insert handlers read them directly.
 * `null` outside the View (tests / standalone renders).
 */
export const EditingStoreContext = React.createContext<EditingStore | null>(null);
export const EditingPresenterContext = React.createContext<EditingPresenter | null>(null);
