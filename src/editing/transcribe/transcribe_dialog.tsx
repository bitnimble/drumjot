import { X } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { jotPlayer } from 'src/editing/playback/player';
import { Checkbox } from 'src/ui/checkbox/checkbox';
import {
  BeatInput,
  LLM_MODEL_LABELS,
  LLM_MODEL_ORDER,
  LlmModel,
  ONSET_BACKEND_LABELS,
  ONSET_BACKEND_ORDER,
  OnsetBackend,
  STAGE_ORDER,
  TranscribeStage,
} from 'src/editing/transcribe/transcriber';
import { TranscribePresenterContext, TranscribeStoreContext } from './transcribe_contexts';
import styles from './transcribe_dialog.module.css';

/** Native `<select>` that releases focus once a value is committed, so the
 *  global spacebar play/pause shortcut isn't swallowed while focus lingers on
 *  a just-used dropdown (same reason as the toolbar's `Select`). */
const Select = ({ onChange, ...rest }: React.ComponentPropsWithoutRef<'select'>) => (
  <select
    {...rest}
    className={styles.select}
    onChange={(e) => {
      onChange?.(e);
      e.currentTarget.blur();
    }}
  />
);

/**
 * The transcribe options dialog, opened from an audio track's overflow menu
 * (`append` mode) or a recent-runs picker (`replace` mode). Shared option set;
 * the `replace` flow adds a resume-stage picker. Confirming kicks off the
 * matching presenter flow.
 */
export const TranscribeDialog = observer(() => {
  const store = React.useContext(TranscribeStoreContext);
  const presenter = React.useContext(TranscribePresenterContext);
  const dialog = store?.dialog;
  if (!store || !presenter || !dialog) return null;

  const options = store.transcribeOptions;
  const isReplace = dialog.mode === 'replace';
  const targetName =
    dialog.mode === 'append'
      ? (jotPlayer.audioTracks.get(dialog.audioTrackId)?.filename ?? 'audio track')
      : (store.recentTranscriptions.find((s) => s.folder === dialog.folder)?.original_filename ??
        dialog.folder);

  const resumableSet = isReplace
    ? new Set(
        store.recentTranscriptions.find((s) => s.folder === dialog.folder)?.resumable_stages ?? []
      )
    : new Set<TranscribeStage>();
  const resumeStage = dialog.mode === 'replace' ? dialog.resumeStage : undefined;
  const canConfirm = !isReplace || (resumeStage !== undefined && resumableSet.has(resumeStage));

  return (
    <div
      className={styles.backdrop}
      role="dialog"
      aria-modal="true"
      aria-label={isReplace ? 'Re-run transcription' : 'Transcribe audio track'}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) presenter.closeDialog();
      }}
      data-testid="transcribe-dialog"
    >
      <div className={styles.panel}>
        <header className={styles.header}>
          <h3 className={styles.title}>
            {isReplace ? 'Re-run transcription' : 'Transcribe audio track'}
          </h3>
          <button
            type="button"
            className={styles.close}
            onClick={() => presenter.closeDialog()}
            aria-label="Close transcribe dialog"
            data-testid="transcribe-dialog-close"
          >
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <div className={styles.body}>
          <p className={styles.subtitle}>
            {isReplace ? (
              <>
                Re-runs <strong>{targetName}</strong> from a chosen stage and replaces the current
                jot with its output.
              </>
            ) : (
              <>
                Transcribes <strong>{targetName}</strong> and inserts the result into the current
                jot as a new layer.
              </>
            )}
          </p>

          <label className={styles.field} title="Which audio feeds the beat tracker. `full mix` is madmom's training distribution; `drum stem` can help on tracks with heavy non-drum syncopation.">
            <span>Beat input</span>
            <Select
              value={options.beatInput}
              onChange={(e) => presenter.setBeatInput(e.target.value as BeatInput)}
            >
              <option value="full_mix">full mix</option>
              <option value="drum_stem">drum stem</option>
            </Select>
          </label>

          <label className={styles.field} title="Onset detector. Drumjot Model (the trained MERT model) emits all drum classes per stem; ADTOF is the prior Frame-RNN detector.">
            <span>Onset detector</span>
            <Select
              value={options.onsetBackend}
              onChange={(e) => presenter.setOnsetBackend(e.target.value as OnsetBackend)}
            >
              {ONSET_BACKEND_ORDER.map((b) => (
                <option key={b} value={b}>
                  {ONSET_BACKEND_LABELS[b]}
                </option>
              ))}
            </Select>
          </label>

          <label className={styles.field} title="Anthropic model used by the classification stages (filter; hihat split; cymbal split).">
            <span>Model</span>
            <Select
              value={options.llmModel}
              onChange={(e) => presenter.setLlmModel(e.target.value as LlmModel)}
            >
              {LLM_MODEL_ORDER.map((m) => (
                <option key={m} value={m}>
                  {LLM_MODEL_LABELS[m]}
                </option>
              ))}
            </Select>
          </label>

          <label className={styles.field} title="Run the optional quantise stage. Off keeps every onset's raw detected time as a near-grid tick + sub-slot offset.">
            <span>Quantise</span>
            <Checkbox
              checked={options.quantise}
              onChange={(e) => presenter.setQuantise(e.target.checked)}
            />
          </label>

          <label
            className={`${styles.field} ${styles.subField}`}
            title="Run the LLM residual pass inside the quantise stage. No-op when Quantise is off."
          >
            <span>Include LLM adjustment</span>
            <Checkbox
              checked={options.quantise && options.quantiseUseLlm}
              disabled={!options.quantise}
              onChange={(e) => presenter.setQuantiseUseLlm(e.target.checked)}
            />
          </label>

          {isReplace && (
            <label className={styles.field} title="Pick the pipeline stage to resume from. Stages whose prerequisites are missing for this run are disabled.">
              <span>From stage</span>
              <Select
                value={resumeStage ?? ''}
                onChange={(e) =>
                  presenter.setDialogResumeStage((e.target.value || undefined) as TranscribeStage | undefined)
                }
              >
                <option value="">Select stage…</option>
                {STAGE_ORDER.map((stage) => (
                  <option key={stage} value={stage} disabled={!resumableSet.has(stage)}>
                    {stage}
                  </option>
                ))}
              </Select>
            </label>
          )}
        </div>
        <footer className={styles.footer}>
          <span className={styles.footerSpacer} />
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() => presenter.closeDialog()}
            data-testid="transcribe-dialog-cancel"
          >
            Cancel
          </button>
          <button
            type="button"
            className={styles.primaryButton}
            onClick={() => presenter.confirmDialog()}
            disabled={!canConfirm}
            data-testid="transcribe-dialog-confirm"
          >
            {isReplace ? 'Re-run & replace' : 'Transcribe'}
          </button>
        </footer>
      </div>
    </div>
  );
});
