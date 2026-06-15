import { makeAutoObservable, runInAction } from 'mobx';
import { ParseError } from 'src/schema/dsl/parser/errors';
import {
  BeatInput,
  DrumSeparator,
  LlmModel,
  stemUrl,
  transcriber,
  TranscribeProgress,
  TranscribeStage,
} from 'src/jot_view/transcribe/transcriber';
import { transcribeSuccessToastMessage } from '../../ui/toasts/toasts_messages';
import { toastStore } from '../../ui/toasts/toasts';
import { TranscribeStore } from './transcribe_store';
import { JotViewPresenter } from '../jot_view_presenter';

/**
 * Dependencies the transcribe presenter orchestrates over.
 */
export type TranscribePresenterDeps = {
  transcribe: TranscribeStore;
  /** Sibling presenter: the transcribe flow auto-loads its result bundle
   *  (score + audio + provenance) through the shared document loader, and
   *  the recent-runs picker reuses the loading overlay. */
  jotViewPresenter: JotViewPresenter;
};

/**
 * Transcribe orchestration for the jot viewer: the `/transcribe` and
 * `/resume` flows, the streamed-progress pill, the recent-runs picker,
 * and the transcribe form options. The post-run artifact load (score,
 * audio tracks, note provenance) is delegated to {@link JotViewPresenter}.
 *
 * Formerly the catch-all `JotViewStore` orchestration; the per-domain
 * presenters (document / mixer / playback / provenance / lyrics /
 * viewport / settings) were split out around it.
 */
export class TranscribePresenter {
  /**
   * Controller for the in-flight `/transcribe` request, if any. The
   * "Stop" toolbar button calls `.abort()` here; the request's
   * AbortSignal is passed into `transcriber.transcribe` which forwards
   * it to `fetch` so the request is genuinely cancelled at the
   * network layer rather than just discarding the response.
   */
  transcribeController: AbortController | undefined;

  readonly transcribe: TranscribeStore;
  readonly jotViewPresenter: JotViewPresenter;

  constructor(deps: TranscribePresenterDeps) {
    this.transcribe = deps.transcribe;
    this.jotViewPresenter = deps.jotViewPresenter;
    makeAutoObservable(this, {
      transcribeController: false,
      transcribe: false,
      jotViewPresenter: false,
    });
  }

  /**
   * Upload an audio file to the transcriber service, parse the returned
   * Drumjot DSL, and load the resulting Jot. Updates `transcribeStatus`
   * so the toolbar can show progress / errors.
   *
   * A single in-flight transcription is tracked via `transcribeController`.
   * Calling `cancelTranscribe()` aborts the underlying `fetch` request and
   * surfaces a cancelled state on the toolbar; starting a new
   * transcription while one is in flight will abort the previous one
   * first (defensive - the UI disables the button during upload, but the
   * console-level `loadDsl` API doesn't).
   */
  async transcribeAudio(file: File): Promise<void> {
    if (this.transcribeController) {
      this.transcribeController.abort();
    }
    const controller = new AbortController();
    this.transcribeController = controller;
    runInAction(() => {
      this.transcribe.transcribeStatus = { phase: 'uploading', filename: file.name };
    });
    try {
      const response = await transcriber.transcribe(file, {
        debug: this.transcribe.transcribeOptions.debug,
        beatInput: this.transcribe.transcribeOptions.beatInput,
        drumSeparator: this.transcribe.transcribeOptions.drumSeparator,
        llmModel: this.transcribe.transcribeOptions.llmModel,
        quantise: this.transcribe.transcribeOptions.quantise,
        quantiseUseLlm: this.transcribe.transcribeOptions.quantiseUseLlm,
        signal: controller.signal,
        onProgress: (event) => this.applyProgress(file.name, event),
      });
      await this.applyTranscribeResponse(response, file.name, controller.signal);
    } catch (err) {
      this.handleTranscribeError(err, controller, 'Transcribe');
    } finally {
      if (this.transcribeController === controller) {
        this.transcribeController = undefined;
      }
      // The folder list has a new entry (the just-finished run); refresh
      // best-effort so the picker is up to date without the operator
      // having to reopen the dropdown.
      void this.refreshRecentTranscriptions();
    }
  }

  /**
   * Re-run the transcribe pipeline from a chosen stage against a
   * previously-cached debug folder. Same status / auto-load semantics as
   * {@link transcribeAudio}: progress pill while in flight, the response
   * either parses straight (DSL mode) or auto-loads the rebuilt debug
   * bundle (filter mode), and the resume controller shares
   * `transcribeController` so the Stop button cancels both flows.
   */
  async resumeTranscribe(folder: string, stage: TranscribeStage): Promise<void> {
    if (this.transcribeController) {
      this.transcribeController.abort();
    }
    const controller = new AbortController();
    this.transcribeController = controller;
    const label = `${folder} from ${stage}`;
    runInAction(() => {
      this.transcribe.transcribeStatus = { phase: 'uploading', filename: label };
    });
    try {
      const response = await transcriber.resume({
        resumeFolder: folder,
        resumeStage: stage,
        beatInput: this.transcribe.transcribeOptions.beatInput,
        drumSeparator: this.transcribe.transcribeOptions.drumSeparator,
        llmModel: this.transcribe.transcribeOptions.llmModel,
        quantise: this.transcribe.transcribeOptions.quantise,
        quantiseUseLlm: this.transcribe.transcribeOptions.quantiseUseLlm,
        signal: controller.signal,
        onProgress: (event) => this.applyProgress(label, event),
      });
      // The resumed run reuses the original folder, so the original
      // upload filename is the most informative pill label — fall back
      // to the resume folder name when the server doesn't know it.
      const fallbackName =
        this.transcribe.recentTranscriptions.find((t) => t.folder === folder)?.original_filename ?? folder;
      await this.applyTranscribeResponse(response, fallbackName, controller.signal);
    } catch (err) {
      this.handleTranscribeError(err, controller, 'Resume');
    } finally {
      if (this.transcribeController === controller) {
        this.transcribeController = undefined;
      }
      void this.refreshRecentTranscriptions();
    }
  }

  /**
   * Shared post-transcribe handling. The backend produces a MIDI
   * prediction; we auto-load the bundled debug.zip so the score (via
   * `from_midi.ts`), audio tracks, and note provenance hydrate in one
   * go without the user having to download and re-load the zip by hand.
   */
  private async applyTranscribeResponse(
    response: Awaited<ReturnType<typeof transcriber.transcribe>>,
    fallbackName: string,
    signal: AbortSignal
  ): Promise<void> {
    const bundleUrl = stemUrl(response.debug_zip_url ?? null);
    if (!bundleUrl) {
      runInAction(() => {
        this.transcribe.transcribeStatus = { phase: 'idle' };
      });
      toastStore.showError('Transcriber returned no debug bundle.');
      return;
    }
    const ok = await this.jotViewPresenter.autoLoadDebugBundle(bundleUrl, fallbackName, signal);
    if (!ok) {
      // The auto-loader already surfaced the specific failure as an
      // error toast; clear the busy pill back to idle and bail.
      runInAction(() => {
        this.transcribe.transcribeStatus = { phase: 'idle' };
      });
      return;
    }
    runInAction(() => {
      this.transcribe.transcribeStatus = { phase: 'idle' };
    });
    toastStore.showSuccess(
      transcribeSuccessToastMessage({
        filename: fallbackName,
        tempo: response.metadata.initial_tempo,
        hasTempoChanges: response.metadata.has_tempo_changes,
        hasTimeSigChanges: response.metadata.has_time_sig_changes,
        barCount: response.metadata.bars.length,
        debugDir: response.debug_dir ?? null,
        debugZipUrl: bundleUrl,
      }),
      {
        title: response.debug_dir
          ? `Debug artifacts saved to ${response.debug_dir} (under ./debug/ on the host with the default docker-compose mount).`
          : undefined,
      }
    );
  }

  /** Shared transcribe / resume failure handler. Routes aborts to idle
   *  (user cancelled), everything else to the error pill. */
  /**
   * Fold one streamed `TranscribeProgress` event into the live
   * `transcribeStatus` pill so the user sees the pipeline advancing
   * through each stage. `stage` events with `phase='start'` set the
   * current stage and clear any substage label from the previous one;
   * `substage` events overwrite the in-stage detail without changing
   * the stage itself. `phase='end'` is ignored for UI purposes — the
   * pill rolls straight from one stage's `start` to the next stage's
   * `start`, which reads more clearly than briefly showing "(done)".
   */
  private applyProgress(filename: string, event: TranscribeProgress): void {
    runInAction(() => {
      const status = this.transcribe.transcribeStatus;
      // If the request was aborted or already terminal (success/error)
      // before this late event fires, ignore — late progress shouldn't
      // resurrect the spinner over an idle/success/error pill.
      if (status.phase !== 'uploading') return;
      if (event.kind === 'stage' && event.phase === 'start') {
        this.transcribe.transcribeStatus = {
          phase: 'uploading',
          filename,
          stage: event.stage,
        };
      } else if (event.kind === 'substage') {
        this.transcribe.transcribeStatus = {
          phase: 'uploading',
          filename,
          stage: event.stage,
          substage: event.detail,
        };
      }
    });
  }

  private handleTranscribeError(err: unknown, controller: AbortController, verb: string): void {
    // AbortError surfaces as DOMException with name='AbortError' (and
    // wraps as TypeError in some runtimes when the fetch was already
    // aborted at start). Treat the user-initiated cancellation
    // distinctly from real errors so we don't show a scary red pill.
    const isAbort =
      controller.signal.aborted || (err instanceof DOMException && err.name === 'AbortError');
    if (isAbort) {
      runInAction(() => {
        this.transcribe.transcribeStatus = { phase: 'idle' };
      });
      return;
    }
    const message =
      err instanceof ParseError
        ? `${verb} returned invalid DSL: ${err.message}`
        : err instanceof Error
          ? err.message
          : String(err);
    runInAction(() => {
      this.transcribe.transcribeStatus = { phase: 'idle' };
    });
    toastStore.showError(`${verb} failed: ${message}`);
  }

  /**
   * Refresh the recent-transcriptions picker from the server. Failures
   * are logged but never surfaced — the picker just stays as-is, which
   * is the right behaviour when the backend is briefly unavailable.
   * Safe to call from a fire-and-forget context.
   */
  async refreshRecentTranscriptions(): Promise<void> {
    runInAction(() => {
      this.transcribe.recentTranscriptionsLoading = true;
    });
    try {
      const list = await transcriber.listTranscriptions();
      runInAction(() => {
        this.transcribe.recentTranscriptions = list;
        this.transcribe.recentTranscriptionsLoaded = true;
        // Drop the selection if its target folder vanished server-side
        // (e.g. operator pruned the debug dir between dropdown opens).
        if (
          this.transcribe.selectedResumeFolder !== undefined &&
          !list.some((s) => s.folder === this.transcribe.selectedResumeFolder)
        ) {
          this.transcribe.selectedResumeFolder = undefined;
          this.transcribe.selectedResumeStage = undefined;
        }
      });
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn('Could not refresh recent transcriptions:', err);
    } finally {
      runInAction(() => {
        this.transcribe.recentTranscriptionsLoading = false;
      });
    }
  }

  /**
   * Load a previously produced transcription's debug bundle straight from
   * the server's `/outputs/<folder>/debug.zip` without re-running any
   * pipeline stage. The bundle carries the kept-onset MIDI score, the
   * per-stem audio, and the run's logs / stage timings, so this is the
   * cheap way to reopen a finished run.
   *
   * Errors land on the shared status pill, mirroring the explicit
   * "Load debug bundle" file picker. Wrapped in `withLoading` so the
   * modal overlay reads as one continuous load even though the inner
   * `applyDebugBundle` may itself trigger nested audio-track loads.
   */
  async loadRecentTranscription(folder: string): Promise<void> {
    const url = stemUrl(`/outputs/${encodeURIComponent(folder)}/debug.zip`);
    if (!url) return;
    const summary = this.transcribe.recentTranscriptions.find((s) => s.folder === folder);
    const fallbackName = summary?.original_filename ?? folder;
    return this.jotViewPresenter.loadDebugBundleFromUrl(url, fallbackName);
  }

  /**
   * Abort the in-flight transcription, if any. No-op when nothing is
   * running. The next `transcribeAudio` call resumes normally.
   */
  cancelTranscribe() {
    if (!this.transcribeController) return;
    this.transcribeController.abort();
    this.transcribeController = undefined;
  }

  // --- transcribe (form options + resume picker) ---

  setDebug(enabled: boolean) {
    this.transcribe.transcribeOptions.debug = enabled;
  }

  setBeatInput(input: BeatInput) {
    this.transcribe.transcribeOptions.beatInput = input;
  }

  setDrumSeparator(separator: DrumSeparator) {
    this.transcribe.transcribeOptions.drumSeparator = separator;
  }

  setLlmModel(model: LlmModel) {
    this.transcribe.transcribeOptions.llmModel = model;
  }

  setQuantise(enabled: boolean) {
    this.transcribe.transcribeOptions.quantise = enabled;
  }

  setQuantiseUseLlm(enabled: boolean) {
    this.transcribe.transcribeOptions.quantiseUseLlm = enabled;
  }

  setSelectedResumeFolder(folder: string | undefined) {
    this.transcribe.selectedResumeFolder = folder;
    // Clearing the folder (or picking a different one) invalidates any
    // stage selection, different folders have different `resumable_stages`,
    // so a stale pick could land on a stage missing its prerequisites.
    this.transcribe.selectedResumeStage = undefined;
  }

  setSelectedResumeStage(stage: TranscribeStage | undefined) {
    this.transcribe.selectedResumeStage = stage;
  }

  setTranscribeMode(mode: 'new' | 'resume') {
    this.transcribe.transcribeMode = mode;
  }
}
