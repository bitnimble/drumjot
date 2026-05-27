import { AlertTriangle } from 'lucide-react';
import React from 'react';
import styles from './audio_worklet_warning_modal.module.css';

/**
 * The two failure modes worth distinguishing to the user. `insecure-context`
 * is fixable client-side (switch to localhost / 127.0.0.1 / HTTPS) and
 * gets specific guidance; `unsupported` is a hard browser limitation and
 * gets a plain "won't work" notice. Computed once at module load by
 * {@link detectAudioWorkletState}; the AudioWorklet API doesn't change
 * after page load, so a static value is correct.
 */
export type AudioWorkletState = 'available' | 'insecure-context' | 'unsupported';

/**
 * Probe whether the browser will expose AudioWorklet on a future
 * AudioContext. `BaseAudioContext.audioWorklet` is only present in
 * secure contexts (localhost / 127.0.0.1 / HTTPS), so a missing
 * `AudioWorkletNode` constructor in an insecure context is
 * attributable to the context, not the browser. In a secure context
 * a missing constructor means the browser genuinely doesn't support it.
 *
 * Pure / side-effect free: no AudioContext is constructed, no audio
 * device is opened. Safe to call at module load.
 */
export function detectAudioWorkletState(): AudioWorkletState {
  if (typeof window === 'undefined') return 'available'; // SSR / tests
  if (!window.isSecureContext) return 'insecure-context';
  if (typeof AudioWorkletNode === 'undefined') return 'unsupported';
  return 'available';
}

/**
 * Modal that warns the user when audio-track playback won't work due
 * to a missing AudioWorklet. Dismissible; the warning is informational,
 * not blocking; drum (MIDI) playback still works. Re-shown on every
 * page load if the condition persists (no persistent dismissal across
 * reloads) because the limitation is a real ongoing issue worth keeping
 * the user aware of.
 */
export const AudioWorkletWarningModal: React.FC<{
  state: AudioWorkletState;
  open: boolean;
  onClose: () => void;
}> = ({ state, open, onClose }) => {
  if (!open || state === 'available') return null;

  return (
    <div
      className={styles.modalBackdrop}
      role="dialog"
      aria-modal="true"
      aria-label="Audio playback unavailable"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      data-testid="audio-worklet-warning-modal"
    >
      <div className={styles.modalPanel}>
        <header className={styles.modalHeader}>
          <span className={styles.warningIcon} aria-hidden="true">
            <AlertTriangle size={18} />
          </span>
          <h3 className={styles.modalTitle}>Audio playback unavailable</h3>
          <button
            type="button"
            className={styles.modalClose}
            onClick={onClose}
            aria-label="Close warning"
            data-testid="audio-worklet-warning-close"
          >
            ×
          </button>
        </header>
        <div className={styles.modalBody}>
          {state === 'insecure-context' ? (
            <>
              <p>
                This page isn't running in a <strong>secure context</strong>,
                so the browser won't expose AudioWorklet. Loaded audio
                tracks won't play, and pitch-preserved speed change is
                disabled.
              </p>
              <p>
                Open the app via <code>localhost</code>,{' '}
                <code>127.0.0.1</code>, or an HTTPS URL instead of a LAN
                IP / plain HTTP, and AudioWorklet will be available.
              </p>
              <p className={styles.note}>
                Drum (MIDI) playback is unaffected.
              </p>
            </>
          ) : (
            <>
              <p>
                This browser doesn't support <strong>AudioWorklet</strong>,
                so loaded audio tracks won't play and pitch-preserved
                speed change is unavailable.
              </p>
              <p>
                Try a recent version of Chrome, Firefox, Edge, or Safari.
              </p>
              <p className={styles.note}>
                Drum (MIDI) playback is unaffected.
              </p>
            </>
          )}
        </div>
        <footer className={styles.modalFooter}>
          <button
            type="button"
            className={styles.primaryButton}
            onClick={onClose}
            autoFocus
            data-testid="audio-worklet-warning-dismiss"
          >
            Got it
          </button>
        </footer>
      </div>
    </div>
  );
};
