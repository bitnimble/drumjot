import { makeAutoObservable, observable } from 'mobx';
import {
  BeatInput,
  LlmModel,
  OnsetBackend,
  TranscribeStage,
  TranscriptionSummary,
} from 'src/editing/transcribe/transcriber';
import { AudioTrackId } from 'src/editing/playback/audio_tracks';

/** Live per-track transcribe progress. Absence of an entry for a track means
 *  idle; presence means a transcription is in flight. Fed from the NDJSON
 *  progress stream so the track gutter spinner can show the current stage. */
export type TranscribeTrackStatus = {
  filename: string;
  /** Current pipeline stage (`stems_all`, `beats`, `transcribe`, …); absent
   *  until the first stage event arrives. */
  stage?: TranscribeStage;
  /** Optional in-stage detail (e.g. "filtering 3/5 instruments"). */
  substage?: string;
};

/** Long-running indicator for the wholesale-replace (recent / resume) flow,
 *  which has no owning track. Only the in-flight `uploading` phase is modelled;
 *  success and failure surface as toasts. */
export type TranscribeStatus =
  | { phase: 'idle' }
  | {
      phase: 'uploading';
      filename: string;
      stage?: TranscribeStage;
      substage?: string;
    };

export type TranscribeOptions = {
  debug: boolean;
  beatInput: BeatInput;
  /** Onset detector backend: `learned` (default, the MERT model) or `adtof`. */
  onsetBackend: OnsetBackend;
  /** Model for the three Opus-by-default classification stages. */
  llmModel: LlmModel;
  /** Run the optional `quantise` pipeline stage. False = no snap; every
   *  onset keeps its raw detected time, the MIDI emitter writes it as
   *  a near-grid tick + sub-slot offset, and the UI / playback honour
   *  the offset so nothing re-snaps on load. */
  quantise: boolean;
  /** Run the LLM residual pass inside the quantise stage. Disables that
   *  pass when false; geometric + envelope + grid still run. No-op
   *  when `quantise` is false. */
  quantiseUseLlm: boolean;
};

/**
 * Which transcribe dialog is open and what it targets:
 *  - `append`: transcribe one loaded audio track and insert the result into
 *    the current jot (an extra `||` layer). `audioTrackId` is the source.
 *  - `replace`: re-run a previous server-side run (`folder`) from a chosen
 *    `resumeStage` and replace the whole jot with its output.
 */
export type TranscribeDialogState =
  | { mode: 'append'; audioTrackId: AudioTrackId }
  | { mode: 'replace'; folder: string; resumeStage: TranscribeStage | undefined };

/**
 * Transcribe / resume UI state: per-track in-flight progress, the form
 * options, the recent-runs list, and the open dialog. Pure data (observables
 * only); the upload / resume / refresh orchestration and the in-flight
 * `AbortController`s live on the presenter.
 */
export class TranscribeStore {
  /** Per-track append-flow progress, keyed by audio track id. */
  trackStatuses: Map<AudioTrackId, TranscribeTrackStatus> = new Map();
  /** Wholesale-replace (recent / resume) flow progress; no owning track. */
  replaceStatus: TranscribeStatus = { phase: 'idle' };
  /** UI-controlled options for the next transcribe call. `debug=true`
   *  so the run is resumable. */
  transcribeOptions: TranscribeOptions = {
    debug: true,
    beatInput: 'full_mix',
    onsetBackend: 'learned',
    llmModel: 'claude-haiku-4-5-20251001',
    quantise: true,
    quantiseUseLlm: false,
  };
  /** Server-side picker of recent /transcribe runs that can be resumed. */
  recentTranscriptions: TranscriptionSummary[] = [];
  /** True once the recent-runs list has been fetched at least once. */
  recentTranscriptionsLoaded: boolean = false;
  /** True while an in-flight recent-runs refresh is resolving. */
  recentTranscriptionsLoading: boolean = false;
  /** The open transcribe dialog, or `undefined` when none is open. */
  dialog: TranscribeDialogState | undefined = undefined;

  constructor() {
    makeAutoObservable(this, { trackStatuses: observable.shallow });
  }
}
