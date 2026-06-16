import React from 'react';
import type { LayersStore } from './layers_store';
import type { LayersPresenter } from './layers_presenter';

/** The Layers read-model, provided once at the editor composition root.
 *  `null` outside the View (tests / standalone renders). */
export const LayersStoreContext = React.createContext<LayersStore | null>(null);

/** The Layers writer (rename / colour / reorder / move), provided alongside
 *  the store. `null` outside the View. */
export const LayersPresenterContext = React.createContext<LayersPresenter | null>(null);
