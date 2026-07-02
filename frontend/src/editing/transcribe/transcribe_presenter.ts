import { makeAutoObservable, runInAction } from 'mobx';
import { ParseError } from 'src/schema/dsl/parser/errors';
import {
  BeatInput,
  LlmModel,
  OnsetBackend,
  stemUrl,
  transcriber,
  TranscribeProgress,
  TranscribeStage,
} from 'src/editing/transcribe/transcriber';
import { fromMidi } from 'src/midi/from_midi';
import { Jot } from 'src/schema/dsl/dsl';
import { AudioTrackId } from 'src/editing/playback/audio_tracks';
import { jotPlayer } from 'src/editing/playback/player';
import { transcribeSuccessToastMessage } from '../../ui/toasts/toasts_messages';
import { toastStore } from '../../ui/toasts/toasts';
import { isBackendUnreachable } from 'src/net/backend_fetch';
import { backendClient } from 'src/net/backend_client';
import { type RunProgress } from 'src/net/backend';
import { desktopCapabilities } from 'src/desktop/desktop_services';
import { TranscribeStore } from './transcribe_store';
import { JotEditorPresenter } from '../jot_editor_presenter';
import { appendTranscription } from './append_transcription';

/**
 * Dependencies the transcribe presenter orchestrates over.
 */
export type TranscribePresenterDeps = {
  transcribe: TranscribeStore;
  /** Sibling presenter: the replace (recent / resume) flow auto-loads its
   *  result bundle through the shared document loader, and the append flow
   *  reads the loaded document off its store to mutate in place. */
  jotEditorPresenter: JotEditorPresenter;
};

/** Audio-track display name: filename with its extension stripped. Used both
 *  as the inserted layer's label and the toast/progress text. */
function audioTrackLabel(filename: string): string {
  return filename.replace(/\.[^./\\]+$/, '') || filename;
}

/**
 * Transcribe orchestration for the jot editor. Two flows:
 *
 *  - **append** (per audio track): re-upload a loaded track's source bytes,
 *    convert the predicted MIDI to a jot, and insert it into the *current*
 *    document as a new `||` layer ({@link appendTranscription}) without
 *    dropping the session's audio tracks / lyrics / mixer.
 *  - **replace** (recent / resume): re-run a previous server run from a chosen
 *    stage and load its debug bundle as a wholesale document replacement
 *    (delegated to {@link JotEditorPresenter}).
 *
 * Also owns the transcribe dialog state, the form options, and the recent-runs
 * refresh.
 */
export class TranscribePresenter {
  /** One in-flight append request per audio track (mirrors lyrics align):
   *  concurrent per-track transcriptions don't cancel each other, and a second
   *  transcribe on the SAME track aborts the first. Non-observable. */
  private readonly trackControllers: Map<AudioTrackId, AbortController> = new Map();
  /** The single in-flight replace (recent / resume) request, if any. */
  private replaceController: AbortController | undefined;

  readonly transcribe: TranscribeStore;
  readonly jotEditorPresenter: JotEditorPresenter;

  constructor(deps: TranscribePresenterDeps) {
    this.transcribe = deps.transcribe;
    this.jotEditorPresenter = deps.jotEditorPresenter;
    makeAutoObservable(this, {
      transcribe: false,
      jotEditorPresenter: false,
    });
  }

  // --- dialog ---

  /** Open the append dialog for an audio track (transcribe → insert here). On
   *  desktop, gate the transcription capability first: if it isn't installed,
   *  prompt to install rather than show a config dialog the user can't act on,
   *  and only open the dialog once it's ready. */
  async openAppendDialog(audioTrackId: AudioTrackId): Promise<void> {
    // Desktop-only: `desktopCapabilities()` is null on web + mobile (HTTP
    // backend, nothing to install), so the gate is skipped there.
    const caps = desktopCapabilities();
    if (caps != null && !(await caps.presenter.requestCapability('transcription'))) {
      return; // not installed and the user dismissed the install prompt
    }
    this.transcribe.dialog = { mode: 'append', audioTrackId };
    void this.refreshRecentTranscriptions();
  }

  /** Open the replace dialog for a previous run (resume → replace the jot). */
  openReplaceDialog(folder: string): void {
    this.transcribe.dialog = { mode: 'replace', folder, resumeStage: undefined };
    void this.refreshRecentTranscriptions();
  }

  closeDialog(): void {
    this.transcribe.dialog = undefined;
  }

  setDialogResumeStage(stage: TranscribeStage | undefined): void {
    const dialog = this.transcribe.dialog;
    if (dialog?.mode !== 'replace') return;
    this.transcribe.dialog = { ...dialog, resumeStage: stage };
  }

  /** Run the dialog's configured action and close it. */
  confirmDialog(): void {
    const dialog = this.transcribe.dialog;
    if (!dialog) return;
    this.closeDialog();
    if (dialog.mode === 'append') {
      void this.transcribeAudioTrack(dialog.audioTrackId);
    } else if (dialog.resumeStage !== undefined) {
      void this.resumeReplace(dialog.folder, dialog.resumeStage);
    }
  }

  // --- append flow (per audio track → insert into current jot) ---

  /**
   * Transcribe a loaded audio track and insert the result into the current
   * jot as a new layer. The track's source bytes are re-uploaded; progress
   * streams into the per-track status (gutter spinner + busy pill). On success
   * the predicted MIDI is converted and merged in place via
   * {@link appendTranscription}; a warning toast fires when pre-existing
   * content was changed.
   */
  // Private: the capability gate runs earlier at openAppendDialog (gate-as-early
  // -as-possible), so this must only be reached via confirmDialog, never called
  // directly (which would skip the gate).
  private async transcribeAudioTrack(id: AudioTrackId): Promise<void> {
    const track = jotPlayer.audioTracks.get(id);
    if (!track) return;
    const store = this.jotEditorPresenter.jotEditorStore;
    if (!store.loroDoc || !store.jot) {
      toastStore.showError('Load or create a jot before transcribing into it.');
      return;
    }
    const prev = this.trackControllers.get(id);
    if (prev) prev.abort();
    const controller = new AbortController();
    this.trackControllers.set(id, controller);
    runInAction(() => {
      this.transcribe.trackStatuses.set(id, { filename: track.filename });
    });
    const options = this.transcribe.transcribeOptions;
    try {
      // Transport (HTTP vs local sidecar) is the adapter's concern; the desktop
      // capability gate already ran at openAppendDialog. Resolve the client ONCE
      // so the run and the artifact read below use the same transport even if the
      // user toggles Local/Hosted mid-job (a sidecar path ref can't be read over
      // HTTP, and vice-versa).
      const backend = backendClient();
      const result = await backend.run(
        {
          op: 'transcribe',
          params: {
            debug: options.debug,
            beatInput: options.beatInput,
            onsetBackend: options.onsetBackend,
            llmModel: options.llmModel,
            quantise: options.quantise,
            quantiseUseLlm: options.quantiseUseLlm,
          },
        },
        { kind: 'blob', blob: track.sourceBlob, filename: track.filename },
        { signal: controller.signal, onProgress: (p) => this.applyTrackProgress(id, track.filename, p) },
      );
      const midiRef = result.artifacts.find((a) => a.role === 'midi')?.ref;
      if (!midiRef) {
        toastStore.showError('Transcriber returned no predicted MIDI.');
        return;
      }
      const bytes = await backend.resolveBytes(midiRef);
      // The job can complete in the window where a cancel is racing the result
      // frame (cancelJob may not reach the sidecar in time); don't merge a
      // transcription the user aborted.
      if (controller.signal.aborted) return;
      this.mergeAppendedJot(track.filename, fromMidi(bytes));
    } catch (err) {
      this.handleTranscribeError(err, controller, 'Transcribe');
    } finally {
      // Only tear down this track's controller + status if a newer transcribe
      // for the SAME track hasn't already superseded us (which installs its own
      // controller + status synchronously). An unconditional status delete here
      // would wipe the in-flight job's progress entry, so its spinner/stage pill
      // vanishes for the rest of the (still-running) job.
      if (this.trackControllers.get(id) === controller) {
        this.trackControllers.delete(id);
        runInAction(() => this.transcribe.trackStatuses.delete(id));
      }
      void this.refreshRecentTranscriptions();
    }
  }

  /** Merge a transcribed jot into the live document as a new layer, with the
   *  heads-up toast about any tempo replacement / pre-existing notes. Shared by
   *  the HTTP and desktop-sidecar append paths. */
  private mergeAppendedJot(filename: string, jot: Jot): void {
    const store = this.jotEditorPresenter.jotEditorStore;
    const doc = store.loroDoc;
    const model = store.jot;
    // The jot may have been unloaded/replaced while the request was in flight.
    if (!doc || !model) return;

    const layerName = audioTrackLabel(filename);
    const result = appendTranscription(doc, model, jot, { layerName });

    const parts: string[] = [];
    if (result.replacedTempoCount > 0) {
      parts.push(
        `replaced ${result.replacedTempoCount} existing tempo event${result.replacedTempoCount === 1 ? '' : 's'}`
      );
    }
    if (result.hadNotes) {
      parts.push('existing notes were kept but may need realignment to the new tempo/bar grid');
    }
    if (parts.length > 0) {
      toastStore.showWarning(`Inserted transcription of ${filename}. Heads up: ${parts.join('; ')}.`);
    } else {
      toastStore.showSuccess(`Transcribed ${filename} into the current jot.`);
    }
  }

  private applyTrackProgress(id: AudioTrackId, filename: string, progress: RunProgress): void {
    runInAction(() => {
      // Ignore late events once the track's status was cleared (abort / done).
      if (!this.transcribe.trackStatuses.get(id)) return;
      this.transcribe.trackStatuses.set(id, {
        filename,
        stage: progress.stage as TranscribeStage,
        substage: progress.message,
      });
    });
  }

  /** Abort the in-flight transcription for one audio track, if any. */
  cancelTrackTranscribe(id: AudioTrackId): void {
    const controller = this.trackControllers.get(id);
    if (!controller) return;
    controller.abort();
    this.trackControllers.delete(id);
    runInAction(() => this.transcribe.trackStatuses.delete(id));
  }

  // --- replace flow (recent / resume → wholesale document replace) ---

  /**
   * Re-run the pipeline from a chosen stage against a previous run's debug
   * folder and load the rebuilt bundle as a wholesale replacement of the
   * current document (score + audio tracks + provenance), mirroring the
   * historical Transcribe-dropdown Resume behaviour.
   */
  async resumeReplace(folder: string, stage: TranscribeStage): Promise<void> {
    if (this.replaceController) this.replaceController.abort();
    const controller = new AbortController();
    this.replaceController = controller;
    const label = `${folder} from ${stage}`;
    runInAction(() => {
      this.transcribe.replaceStatus = { phase: 'uploading', filename: label };
    });
    try {
      const response = await transcriber.resume({
        resumeFolder: folder,
        resumeStage: stage,
        beatInput: this.transcribe.transcribeOptions.beatInput,
        onsetBackend: this.transcribe.transcribeOptions.onsetBackend,
        llmModel: this.transcribe.transcribeOptions.llmModel,
        quantise: this.transcribe.transcribeOptions.quantise,
        quantiseUseLlm: this.transcribe.transcribeOptions.quantiseUseLlm,
        signal: controller.signal,
        onProgress: (event) => this.applyReplaceProgress(label, event),
      });
      const fallbackName =
        this.transcribe.recentTranscriptions.find((t) => t.folder === folder)?.original_filename ??
        folder;
      await this.applyReplaceResponse(response, fallbackName, controller.signal);
    } catch (err) {
      this.handleTranscribeError(err, controller, 'Resume');
    } finally {
      if (this.replaceController === controller) this.replaceController = undefined;
      void this.refreshRecentTranscriptions();
    }
  }

  /**
   * Shared post-replace handling: the backend produced a debug bundle; auto-
   * load it so the score, audio tracks, and provenance hydrate in one go.
   */
  private async applyReplaceResponse(
    response: Awaited<ReturnType<typeof transcriber.resume>>,
    fallbackName: string,
    signal: AbortSignal
  ): Promise<void> {
    const bundleUrl = stemUrl(response.debug_zip_url ?? null);
    if (!bundleUrl) {
      runInAction(() => {
        this.transcribe.replaceStatus = { phase: 'idle' };
      });
      toastStore.showError('Transcriber returned no debug bundle.');
      return;
    }
    const ok = await this.jotEditorPresenter.autoLoadDebugBundle(bundleUrl, fallbackName, signal);
    runInAction(() => {
      this.transcribe.replaceStatus = { phase: 'idle' };
    });
    if (!ok) return;
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

  private applyReplaceProgress(filename: string, event: TranscribeProgress): void {
    runInAction(() => {
      const status = this.transcribe.replaceStatus;
      if (status.phase !== 'uploading') return;
      if (event.kind === 'stage' && event.phase === 'start') {
        this.transcribe.replaceStatus = { phase: 'uploading', filename, stage: event.stage };
      } else if (event.kind === 'substage') {
        this.transcribe.replaceStatus = {
          phase: 'uploading',
          filename,
          stage: event.stage,
          substage: event.detail,
        };
      }
    });
  }

  /** Shared transcribe / resume failure handler. Routes aborts + transport
   *  failures to a quiet reset; real errors to an error toast. Only the replace
   *  flow owns `replaceStatus`, so an append error doesn't silence a concurrent
   *  replace's progress (append per-track statuses are cleared in their own
   *  per-track `finally`). */
  private handleTranscribeError(err: unknown, controller: AbortController, verb: string): void {
    const isAbort =
      controller.signal.aborted || (err instanceof DOMException && err.name === 'AbortError');
    if (this.replaceController === controller) {
      runInAction(() => {
        this.transcribe.replaceStatus = { phase: 'idle' };
      });
    }
    if (isAbort || isBackendUnreachable(err)) return;
    const message =
      err instanceof ParseError
        ? `${verb} returned invalid DSL: ${err.message}`
        : err instanceof Error
          ? err.message
          : String(err);
    toastStore.showError(`${verb} failed: ${message}`);
  }

  // --- recent runs ---

  /**
   * Refresh the recent-transcriptions list from the server. Failures are
   * logged but never surfaced. Safe to call fire-and-forget.
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
        // Drop a stale dialog selection if its folder vanished server-side.
        const dialog = this.transcribe.dialog;
        if (dialog?.mode === 'replace' && !list.some((s) => s.folder === dialog.folder)) {
          this.transcribe.dialog = undefined;
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

  // --- form options ---

  setDebug(enabled: boolean) {
    this.transcribe.transcribeOptions.debug = enabled;
  }

  setBeatInput(input: BeatInput) {
    this.transcribe.transcribeOptions.beatInput = input;
  }

  setOnsetBackend(backend: OnsetBackend) {
    this.transcribe.transcribeOptions.onsetBackend = backend;
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
}
