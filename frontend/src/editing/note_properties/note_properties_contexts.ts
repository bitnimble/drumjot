import React from 'react';
import type { NotePropertiesStore } from './note_properties_store';
import type { NotePropertiesPresenter } from './note_properties_presenter';

/** The Note properties read-model, provided once at the editor composition
 *  root. `null` outside the View (tests / standalone renders). */
export const NotePropertiesStoreContext = React.createContext<NotePropertiesStore | null>(null);

/** The Note properties writer (lane / bar-beat / volume / modifiers / …),
 *  provided alongside the store. `null` outside the View. */
export const NotePropertiesPresenterContext =
  React.createContext<NotePropertiesPresenter | null>(null);
