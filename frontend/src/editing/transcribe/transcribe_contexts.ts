import React from 'react';
import { TranscribeStore } from './transcribe_store';
import { TranscribePresenter } from './transcribe_presenter';

/**
 * Routes the {@link TranscribeStore} to deep consumers that read transcribe
 * state (the audio-track gutter's per-track transcribe spinner). `null`
 * outside the editor view.
 */
export const TranscribeStoreContext = React.createContext<TranscribeStore | null>(null);

/**
 * Routes the {@link TranscribePresenter} to deep consumers that kick off or
 * configure transcription (the audio-track overflow menu's "Transcribe" item,
 * the transcribe dialog, the recent-runs pickers). `null` outside the view.
 */
export const TranscribePresenterContext = React.createContext<TranscribePresenter | null>(null);
