import { makeAutoObservable } from 'mobx';
import {
  BeatInput,
  DrumSeparator,
  LlmModel,
  OnsetBackend,
  TranscribeStage,
  TranscriptionSummary,
} from 'src/editing/transcribe/transcriber';

/** Long-running transcribe indicator. Only the in-flight `uploading`
 *  phase is modelled here; success and failure surface as toasts. */
export type TranscribeStatus =
  | { phase: 'idle' }
  | {
      phase: 'uploading';
      filename: string;
      /** Current pipeline stage (`stems_all`, `beats`, `transcribe`, …)
       *  reported by the server's NDJSON progress stream. `undefined`
       *  until the first stage event arrives; the initial "uploading"
       *  read covers everything before the first stage starts. */
      stage?: TranscribeStage;
      /** Optional in-stage detail, e.g. "filtering 3/5 instruments
       *  (latest: snare)". Cleared whenever the stage advances. */
      substage?: string;
    };

export type TranscribeOptions = {
  debug: boolean;
  beatInput: BeatInput;
  /** Stage-2 separator: `mdx23c` (default) or the opt-in `larsnet`. */
  drumSeparator: DrumSeparator;
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
 * Transcribe / resume UI state: the in-flight status pill, the form
 * options, and the recent-runs picker. Pure data (observables only); the
 * upload / resume / refresh orchestration and the in-flight
 * `AbortController` live on the presenter.
 */
export class TranscribeStore {
  transcribeStatus: TranscribeStatus = { phase: 'idle' };
  /** UI-controlled options for the next transcribe call. `debug=true`
   *  so the run is resumable. */
  transcribeOptions: TranscribeOptions = {
    debug: true,
    beatInput: 'full_mix',
    drumSeparator: 'mdx23c',
    onsetBackend: 'learned',
    llmModel: 'claude-haiku-4-5-20251001',
    quantise: true,
    quantiseUseLlm: false,
  };
  /** Server-side picker of recent /transcribe runs that can be resumed.
   *  Populated by the presenter's refresh; an empty array before the
   *  first fetch (the picker shows "Loading…" in that state). */
  recentTranscriptions: TranscriptionSummary[] = [];
  /** True once the recent-runs list has been fetched at least once
   *  (success or empty). The Load → Recent submenu uses this to decide
   *  whether to issue the initial fetch on first open or use the cache. */
  recentTranscriptionsLoaded: boolean = false;
  /** True while an in-flight recent-runs refresh is resolving. Drives the
   *  spinner inside the Load → Recent submenu. */
  recentTranscriptionsLoading: boolean = false;
  /** Folder name of the currently-selected recent transcription, or
   *  `undefined` when nothing is selected. Drives the stage picker (we
   *  read `resumable_stages` off the matching summary). */
  selectedResumeFolder: string | undefined = undefined;
  /** Stage the user has picked to resume from. `undefined` until they
   *  pick one; reset whenever {@link selectedResumeFolder} changes so
   *  stale picks from one folder can't leak into another folder's
   *  request. */
  selectedResumeStage: TranscribeStage | undefined = undefined;
  /** Which flow the Transcribe dropdown is showing: a fresh upload
   *  (`new`) or resume-from-debug-folder (`resume`). Defaults to `new`
   *  since that's the only flow available before any runs exist. */
  transcribeMode: 'new' | 'resume' = 'new';

  constructor() {
    makeAutoObservable(this);
  }
}
