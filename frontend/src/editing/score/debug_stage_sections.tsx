import React from 'react';
import { NoteProvenanceEntry } from 'src/editing/provenance/debug_zip';
import { NotePosition } from 'src/editing/score/note_position';
import {
  formatSignedBeats,
  formatSignedMs,
  formatSignedSlots,
} from './provenance_format';
import type { ProvenanceTiming, RenderedNoteContext } from './provenance_timing_math';
import styles from './score.module.css';

/** Per-lane acoustic measurements the cymbal_split + hihat_split
 *  passes capture. `null`/`undefined` everywhere else (other lanes,
 *  bundles predating v4). At least one populated field means the
 *  "Acoustic properties" subsection is worth surfacing. */
export function entryHasAcousticFields(entry: NoteProvenanceEntry): boolean {
  return (
    entry.decay_s != null ||
    entry.flatness != null ||
    entry.centroid_hz != null ||
    entry.gap_s != null ||
    entry.attack_s != null ||
    entry.late_rms != null ||
    entry.pre_rms != null ||
    entry.tail_end_s != null
  );
}

/** Render the per-onset acoustic measurements the cymbal / hi-hat
 *  classifiers saw. Only rows whose value is populated are emitted; the
 *  caller is responsible for skipping the whole section when none are.
 *  Reuses the `debugStageSections` grid + `debugDetailsList`
 *  contents-flow pair so labels line up with the Timing subsection's
 *  rows above. */
export function renderAcousticSection(entry: NoteProvenanceEntry): React.ReactNode {
  const rows: React.ReactNode[] = [];
  const row = (key: string, label: string, value: string) => {
    rows.push(
      <React.Fragment key={key}>
        <dt>{label}</dt>
        <dd>{value}</dd>
      </React.Fragment>,
    );
  };
  if (entry.attack_s != null) {
    // ms reads better than fractional seconds for typical attacks
    // (5-50ms range on hi-hats); ride / crash attacks fall in the
    // same range so the unit choice is uniform.
    row('attack', 'Attack rise', `${(entry.attack_s * 1000).toFixed(1)} ms`);
  }
  if (entry.decay_s != null) {
    row('decay', 'Decay (-20 dB)', `${entry.decay_s.toFixed(3)} s`);
  }
  if (entry.tail_end_s != null) {
    row('tail', 'Ring tail end', `${entry.tail_end_s.toFixed(3)} s`);
  }
  if (entry.late_rms != null) {
    row('late', 'Late RMS ratio', entry.late_rms.toFixed(3));
  }
  if (entry.pre_rms != null) {
    row('pre', 'Pre RMS ratio', entry.pre_rms.toFixed(3));
  }
  if (entry.flatness != null) {
    row('flat', 'Spectral flatness', entry.flatness.toFixed(4));
  }
  if (entry.centroid_hz != null) {
    row('cen', 'Spectral centroid', `${(entry.centroid_hz / 1000).toFixed(2)} kHz`);
  }
  if (entry.gap_s != null) {
    row('gap', 'Gap to neighbour', `${entry.gap_s.toFixed(3)} s`);
  }
  return (
    <div className={styles.debugStageSections}>
      <dl className={styles.debugDetailsList}>{rows}</dl>
    </div>
  );
}

type DebugStageSectionsProps = {
  entry: NoteProvenanceEntry;
  rendered: RenderedNoteContext | undefined;
  timing: ProvenanceTiming;
  gridDivision: number;
  slotsPerQuarterNote: number;
};

/**
 * Render the Debug details body as a vertical stack of per-stage
 * sections (Onset detection → Beat-grid alignment → Quantise → MIDI
 * render → User adjustment → Final → Filter). Lifted out of the
 * component body so the JSX stays readable; every value is
 * pre-computed by {@link computeProvenanceTiming} and passed in via the
 * `timing` struct.
 *
 * Stages with no relevant data for this onset are omitted entirely;
 * the section only renders when at least one row inside has something
 * worth showing. That way the popup stays compact for a clean
 * round-trip and only grows when shifts actually accumulated.
 */
export function renderDebugStageSections(p: DebugStageSectionsProps): React.ReactNode {
  const { entry, rendered, timing, gridDivision, slotsPerQuarterNote } = p;
  const {
    originalBar, originalBarIndex, originalQuantizedBeat, originalQuantizedSec,
    currentQuantizedBeat, displayedBarIndex, finalSec, snapBeats, snapMs,
    envelopeRefineSec, rawModelSec, coarseAlignSec, fineAlignSec, combinedAlignSec,
    hasAlignSplit, quantisePasses, hasPerPassQuantise, fallbackQuantSec,
    fallbackQuantSlots, anchorDriftSec, anchorDriftBeats, drumOffsetSec,
    drumOffsetBeats, barGridFrameShiftSec, unknownDriftSec, unknownDriftBeats,
    unknownDriftMs, secToOrigBeats, origBeatsToSlots,
  } = timing;
  const velocity = rendered?.note.velocity;

  const renderSignedMs = (ms: number): string => formatSignedMs(ms);
  const renderSignedBeats = (b: number): string => formatSignedBeats(b);
  // Format an integer slot count as a per-slot label like "+3/48".
  const formatSlots = (slots: number): string => formatSignedSlots(slots, gridDivision);

  // Helper: render a `<dt>/<dd>` pair displaying a per-stage shift in
  // {beats, slots, ms} triple form. The `slots` annotation is omitted
  // when neither an explicit count nor a derivable one resolves.
  const shiftRow = (
    label: string,
    deltaSec: number,
    deltaBeats: number | undefined,
    explicitSlots?: number,
  ) => {
    const derivedSlots = explicitSlots ?? origBeatsToSlots(deltaBeats);
    return (
      <>
        <dt>{label}</dt>
        <dd>
          {deltaBeats !== undefined && `${renderSignedBeats(deltaBeats)} beats `}
          {derivedSlots !== undefined && `· ${formatSlots(derivedSlots)} `}
          ({renderSignedMs(deltaSec * 1000)})
        </dd>
      </>
    );
  };

  return (
    <div className={styles.debugStageSections}>
      <section className={styles.stageGroup}>
        <h4 className={styles.stageHeading}>Onset detection</h4>
        <dl className={styles.debugDetailsList}>
          <dt>Detected beat</dt>
          <dd>
            {/* `entry.bar` is the transcriber's 0-indexed structural
              bar. We display the rendered jot's `bar.index` for the
              same bar (resolved via the array-position lookup above)
              so the value matches where the note actually lives in
              the score; `from_midi.ts` can fold leading all-rest
              drum bars into the lead-in, so the naïve `entry.bar + 1`
              mapping no longer always equals the displayed bar. Falls
              back to `entry.bar + 1` when the lookup couldn't resolve
              (no rendered jot context). */}
            {new NotePosition({
              barIndex: originalBarIndex ?? entry.bar + 1,
              beatInBar: entry.beat_in_bar,
              slotsPerQuarter: slotsPerQuarterNote,
              timeSig: originalBar
                ? { count: originalBar.tsCount, unit: originalBar.tsUnit }
                : undefined,
              audioSec: entry.detected_time_sec,
            }).toString()}
          </dd>
          {entry.amplitude !== null && entry.amplitude !== undefined && (
            <>
              <dt>Amplitude</dt>
              <dd>{entry.amplitude.toFixed(3)}</dd>
            </>
          )}
          <dt>Onset confidence</dt>
          <dd>{entry.strength.toFixed(3)}</dd>
          {entry.midi_note !== null && entry.midi_note !== undefined && (
            <>
              <dt>MIDI note</dt>
              <dd>{entry.midi_note}</dd>
            </>
          )}
          {/* Raw MIDI velocity is preserved on `note.metadata.midi.velocity`
              by from_midi.ts; the same file's [A7] step maps it into the
              `:a` (≥100) / `:g` (<40) modifiers visible in the description
              above. Surface the raw value so the operator can see exactly
              why a note picked up (or didn't) the accent/ghost decoration. */}
          {typeof velocity === 'number' && (
            <>
              <dt>Velocity</dt>
              <dd>{velocity}</dd>
            </>
          )}
          {envelopeRefineSec !== undefined && Math.abs(envelopeRefineSec) > 1e-9 && (
            <>
              <dt>Raw model peak</dt>
              <dd>{rawModelSec !== null && rawModelSec !== undefined ? `${rawModelSec.toFixed(3)}s` : ', '}</dd>
              {shiftRow(
                'Envelope refine',
                envelopeRefineSec,
                secToOrigBeats(envelopeRefineSec),
              )}
            </>
          )}
        </dl>
      </section>

      {(hasAlignSplit || Math.abs(combinedAlignSec) > 1e-9) && (
        <section className={styles.stageGroup}>
          <h4 className={styles.stageHeading}>Beat-grid alignment</h4>
          <dl className={styles.debugDetailsList}>
            {hasAlignSplit ? (
              <>
                {coarseAlignSec !== null &&
                  shiftRow(
                    'Coarse · envelope phase',
                    coarseAlignSec,
                    secToOrigBeats(coarseAlignSec),
                  )}
                {fineAlignSec !== null &&
                  shiftRow(
                    'Fine · onset-snap',
                    fineAlignSec,
                    secToOrigBeats(fineAlignSec),
                  )}
              </>
            ) : (
              shiftRow(
                'Beat alignment (combined)',
                combinedAlignSec,
                secToOrigBeats(combinedAlignSec),
              )
            )}
          </dl>
        </section>
      )}

      {/* Bar-grid frame shift: explicit attribution of the previously-
        * "Unknown drift" residual to its source, the sum of envelope
        * refine + combined alignment. Both shift the frame the chain
        * is measured against (envelope refine: raw model peak →
        * canonical onset time; alignment: pre- → post-alignment bar
        * grid) without moving the onset's final audio-time position,
        * so the chain math keeps them out of `accountedSec` while
        * this row makes their combined contribution visible. See the
        * `barGridFrameShiftSec` block in NoteProvenanceDetails for
        * the full rationale. */}
      {barGridFrameShiftSec !== undefined &&
        Math.abs(barGridFrameShiftSec) > 1e-9 && (
          <section className={styles.stageGroup}>
            <h4 className={styles.stageHeading}>Bar-grid frame shift</h4>
            <dl className={styles.debugDetailsList}>
              {shiftRow(
                'Frame shift (envelope refine + align)',
                barGridFrameShiftSec,
                secToOrigBeats(barGridFrameShiftSec),
              )}
            </dl>
          </section>
        )}

      {(hasPerPassQuantise ||
        (fallbackQuantSec !== undefined && Math.abs(fallbackQuantSec) > 1e-9) ||
        entry.off_grid === true ||
        (entry.quantised_residual_slots !== null &&
          entry.quantised_residual_slots !== undefined)) && (
        <section className={styles.stageGroup}>
          <h4 className={styles.stageHeading}>Quantise</h4>
          <dl className={styles.debugDetailsList}>
            {hasPerPassQuantise &&
              quantisePasses.map((pass) =>
                pass.slots === null || pass.slots === undefined ? null : (
                  <React.Fragment key={pass.key}>
                    <dt>{pass.label.replace(/^Quantise · /, '')}</dt>
                    <dd>{formatSlots(pass.slots)}</dd>
                  </React.Fragment>
                ),
              )}
            {!hasPerPassQuantise &&
              fallbackQuantSec !== undefined &&
              Math.abs(fallbackQuantSec) > 1e-9 &&
              shiftRow(
                'Quantise (combined)',
                fallbackQuantSec,
                secToOrigBeats(fallbackQuantSec),
                fallbackQuantSlots,
              )}
            {entry.quantised_residual_slots !== null &&
              entry.quantised_residual_slots !== undefined && (
                <>
                  <dt>Sub-slot residual</dt>
                  <dd>{renderSignedBeats(entry.quantised_residual_slots)} slot</dd>
                </>
              )}
            {entry.off_grid === true && (
              <>
                <dt>Off-grid</dt>
                <dd>yes (no free slot within match band)</dd>
              </>
            )}
            {originalQuantizedBeat !== undefined && originalBarIndex !== undefined && (
              <>
                <dt>Quantized to</dt>
                <dd>
                  {new NotePosition({
                    barIndex: originalBarIndex,
                    beatInBar: originalQuantizedBeat,
                    slotsPerQuarter: slotsPerQuarterNote,
                    timeSig: originalBar
                ? { count: originalBar.tsCount, unit: originalBar.tsUnit }
                : undefined,
                    audioSec: originalQuantizedSec,
                  }).toString()}
                </dd>
              </>
            )}
          </dl>
        </section>
      )}

      {(snapBeats !== undefined || anchorDriftSec !== undefined) && (
        <section className={styles.stageGroup}>
          <h4 className={styles.stageHeading}>MIDI render</h4>
          <dl className={styles.debugDetailsList}>
            {anchorDriftSec !== undefined &&
              shiftRow('Bar anchor drift', anchorDriftSec, anchorDriftBeats)}
            {snapBeats !== undefined && snapMs !== undefined && (
              <>
                <dt>1/48 snap</dt>
                <dd>
                  {`${renderSignedBeats(snapBeats)} beats `}
                  {origBeatsToSlots(snapBeats) !== undefined &&
                    `· ${formatSlots(origBeatsToSlots(snapBeats)!)} `}
                  ({renderSignedMs(snapMs)})
                </dd>
              </>
            )}
          </dl>
        </section>
      )}

      {(drumOffsetBeats !== undefined || unknownDriftSec !== undefined) && (
        <section className={styles.stageGroup}>
          <h4 className={styles.stageHeading}>User adjustment</h4>
          <dl className={styles.debugDetailsList}>
            {drumOffsetBeats !== undefined && (
              <>
                <dt>Drum offset</dt>
                <dd>
                  {renderSignedBeats(drumOffsetBeats)} beats{' '}
                  · {formatSlots(drumOffsetBeats * slotsPerQuarterNote)}
                  {drumOffsetSec !== undefined && ` (${renderSignedMs(drumOffsetSec * 1000)})`}
                </dd>
              </>
            )}
            {unknownDriftSec !== undefined && unknownDriftMs !== undefined && (
              <>
                <dt>Unknown drift</dt>
                <dd>
                  {unknownDriftBeats !== undefined &&
                    `${renderSignedBeats(unknownDriftBeats)} beats `}
                  {origBeatsToSlots(unknownDriftBeats) !== undefined &&
                    `· ${formatSlots(origBeatsToSlots(unknownDriftBeats)!)} `}
                  ({renderSignedMs(unknownDriftMs)})
                </dd>
              </>
            )}
          </dl>
        </section>
      )}

      {finalSec !== undefined &&
        currentQuantizedBeat !== undefined &&
        displayedBarIndex !== undefined && (
          <section className={styles.stageGroup}>
            <h4 className={styles.stageHeading}>Final</h4>
            <dl className={styles.debugDetailsList}>
              <dt>Position</dt>
              <dd>
                {new NotePosition({
                  barIndex: displayedBarIndex,
                  beatInBar: currentQuantizedBeat,
                  slotsPerQuarter: slotsPerQuarterNote,
                  timeSig: rendered
                    ? { count: rendered.bar.tsCount, unit: rendered.bar.tsUnit }
                    : undefined,
                  audioSec: finalSec,
                  offsetMs: rendered?.note.offsetMs,
                }).toString()}
              </dd>
            </dl>
          </section>
        )}

      <section className={styles.stageGroup}>
        <h4 className={styles.stageHeading}>Filter</h4>
        <dl className={styles.debugDetailsList}>
          <dt>Decision</dt>
          <dd>
            {entry.kept
              ? 'kept'
              : (() => {
                  const source =
                    entry.rejected_by ?? (entry.out_of_range ? 'out of range' : 'rejected');
                  const reasonLabel = entry.reason_code
                    ? (FILTER_REASON_LABELS[entry.reason_code] ?? entry.reason_code)
                    : null;
                  return reasonLabel ? (
                    <>
                      {source} · <span className={styles.reasonCode}>{reasonLabel}</span>
                    </>
                  ) : (
                    source
                  );
                })()}
          </dd>
          {!entry.kept && !entry.out_of_range && entry.reason_text && (
            <>
              <dt>Reason</dt>
              <dd>
                <span className={styles.reasonText}>{entry.reason_text}</span>
              </dd>
            </>
          )}
        </dl>
      </section>
    </div>
  );
}

/**
 * Display labels for the filter LLM's short reason codes. Keep in
 * lockstep with `REASON_CODES` in
 * `transcriber/app/pipeline/filter_llm.py`; unknown codes fall through
 * to the raw value so a new backend code still renders something
 * meaningful before the frontend catches up.
 */
const FILTER_REASON_LABELS: Record<string, string> = {
  bleed: 'Bleed from louder instrument',
  double_trigger: 'Detector double-trigger',
  noise: 'Isolated noise',
  custom: 'Custom',
};
