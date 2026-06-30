import React from 'react';

/**
 * Render the success message for a finished transcribe (or resumed
 * transcribe) run. Mirrors the layout the old `TranscribeStatusPill`
 * success branch used: a "Loaded <file>" sentence with bpm / bar /
 * change details, plus a trailing `[debug.zip]` download link when a
 * bundle URL is available. Lives next to `ToastContainer` since the
 * embedded anchor's `stopPropagation` partners with the container's
 * click-to-dismiss behaviour.
 */
export function transcribeSuccessToastMessage(opts: {
  filename: string;
  tempo: number;
  hasTempoChanges: boolean;
  hasTimeSigChanges: boolean;
  barCount: number;
  debugDir?: string | null;
  debugZipUrl?: string | null;
}): React.ReactNode {
  let detail = `@ ${opts.tempo.toFixed(0)} bpm, ${opts.barCount} bars`;
  if (opts.hasTempoChanges) detail += ', tempo changes';
  if (opts.hasTimeSigChanges) detail += ', time-sig changes';
  if (opts.debugDir) detail += `, debug @ ${opts.debugDir}`;
  return (
    <>
      Loaded {opts.filename} {detail}
      {opts.debugZipUrl && (
        <>
          {' '}
          <a
            href={opts.debugZipUrl}
            download
            data-noseek="true"
            onClick={(e) => e.stopPropagation()}
            title="Download the debug bundle (.zip) for this run."
          >
            [debug.zip]
          </a>
        </>
      )}
    </>
  );
}
