import { ChevronDown, ChevronRight } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { NoteProvenanceEntry } from 'src/editing/provenance/debug_zip';
import { DEFAULT_GRID_DIVISION } from 'src/grid/grid';
import { jotPlayer } from 'src/editing/playback/player';
import { BarTimingsContext, StructuralContext } from '../jot_editor_contexts';
import {
  entryHasAcousticFields,
  renderAcousticSection,
  renderDebugStageSections,
} from './debug_stage_sections';
import { OnsetTimingVisualization, StageShift } from './onset_timing_visualization';
import { computeProvenanceTiming, RenderedNoteContext } from './provenance_timing_math';
import styles from './score.module.css';

/**
 * Collapsible "Debug details" block that surfaces a single onset's full
 * provenance (detected time, strength, beat-tracker placement, MIDI
 * quantization, filter decision, MIDI tick, …) inside the selection
 * label. Shared between {@link NoteView} (where it appears under the
 * human-readable description with the post-quantization comparison
 * filled in) and {@link FilteredOnsetView} (where it IS the label,
 * filtered onsets have no rendered counterpart so the rendered/snap
 * rows are hidden).
 *
 * Toggle state is local to this component, so it resets every time the
 * label remounts. Closing the popover (de-selecting / un-hovering the
 * note) collapses the details again next time, acceptable for v1
 * since the block is short.
 */
export const NoteProvenanceDetails = observer(
  ({
    entry,
    rendered,
    startOpen = false,
  }: {
    entry: NoteProvenanceEntry;
    /** Present when hosted by a kept note; absent for filtered ghosts. */
    rendered?: RenderedNoteContext;
    /** Open by default (used by FilteredOnsetView so the user immediately
     * sees why the onset was rejected; for kept notes the toggle is
     * collapsed by default so it doesn't crowd the basic description). */
    startOpen?: boolean;
  }) => {
    const [open, setOpen] = React.useState(startOpen);
    // Subsection toggles. Both default open so opening the Debug
    // details panel surfaces everything at once; collapse to focus
    // on the other half.
    const [timingOpen, setTimingOpen] = React.useState(true);
    const [acousticOpen, setAcousticOpen] = React.useState(true);
    // Stop the container's mousedown handler so clicks on the toggle don't
    // begin a marquee selection (which would clear the surrounding note's
    // selection and immediately unmount this component).
    const stop = (e: React.MouseEvent) => e.stopPropagation();

    // Eager per-bar timings table provided once at JotEditor. Lets the
    // "Final position" row resolve a bar's absolute jot-time start without
    // waiting for the player's timeline to be built, pre-Play the
    // player's timeline is `EMPTY_TIMELINE`, but the math doesn't actually
    // need any playback state.
    const barTimings = React.useContext(BarTimingsContext);
    // The current song's structure, used to read `effectiveDrumOffsetBeats`,
    // the user-applied Beat-offset slider value, as a labelled stage in
    // the detected → final timing-drift chain.
    const structural = React.useContext(StructuralContext);
    // Grid density the jot was produced at; drives every slot readout in
    // this panel. Falls back to the default when no structure is in
    // context (filtered-ghost rendering).
    const gridDivision = structural ? structural.gridDivision : DEFAULT_GRID_DIVISION;
    const slotsPerQuarterNote = gridDivision / 4;

    const timing = computeProvenanceTiming({
      entry,
      rendered,
      barTimings,
      structural,
      songLeadInSec: jotPlayer.songLeadInSec,
      gridDivision,
      slotsPerQuarterNote,
    });
    const {
      displayedBarIndex,
      finalSec,
      snapBeats,
      snapSec,
      envelopeRefineSec,
      rawModelSec,
      coarseAlignSec,
      fineAlignSec,
      hasAlignSplit,
      combinedAlignSec,
      quantisePasses,
      hasPerPassQuantise,
      fallbackQuantSec,
      fallbackQuantSlots,
      anchorDriftSec,
      anchorDriftBeats,
      drumOffsetSec,
      drumOffsetBeats,
      secToOrigBeats,
      slotsToOrigSec,
      origBeatsToSlots,
    } = timing;

    // ─── Build the ordered chain in pipeline order ───
    //
    // Each entry is one named shift that actually advances the onset
    // along the detected → final chain in audio time; the trailing
    // unknown-drift residual is appended inside
    // `OnsetTimingVisualization` so it always lands at the chain's end.
    // Pipeline order: quantise passes → bar anchor drift → MIDI snap →
    // drum offset.
    //
    // **Not in the chain:** envelope refine and beat-grid alignment.
    // Both are frame shifts (pre-canonical for env refine, bar-grid for
    // alignment), so they don't accumulate into `finalSec` and adding
    // them as forward chain stages would create the same -64..-68 ms
    // residual mismatch this code's `barGridFrameShiftSec` attribution
    // is meant to surface explicitly. The two are still visible as
    // their own rows in the textual debug details (under "Onset
    // detection" → envelope refine and "Beat-grid alignment" →
    // coarse/fine), and their summed contribution is the "Bar-grid
    // frame shift" row.
    const stages: StageShift[] = [];
    // Frame-shift stages first: they render at the top of the diff rows
    // and are anchored to fixed audio-time positions (envelope refine
    // literally spans `rawModelSec → detectedSec`; alignment shifts the
    // grid without a literal "from" position, anchored at the detected
    // line by convention). Excluded from `accountedSec` above, so the
    // chain math and the trailing unknown-drift residual are unaffected.
    if (
      envelopeRefineSec !== undefined &&
      Math.abs(envelopeRefineSec) > 1e-9 &&
      rawModelSec !== null &&
      rawModelSec !== undefined
    ) {
      stages.push({
        key: 'env-refine',
        label: 'Envelope refine',
        className: styles.timingVizDiffBarEnvRefine,
        deltaSec: envelopeRefineSec,
        deltaBeats: secToOrigBeats(envelopeRefineSec),
        anchorSec: rawModelSec,
      });
    }
    if (hasAlignSplit) {
      if (coarseAlignSec !== null && Math.abs(coarseAlignSec) > 1e-9) {
        stages.push({
          key: 'align-coarse',
          label: 'Coarse · envelope phase',
          className: styles.timingVizDiffBarAlignCoarse,
          deltaSec: coarseAlignSec,
          deltaBeats: secToOrigBeats(coarseAlignSec),
          anchorSec: entry.detected_time_sec,
        });
      }
      if (fineAlignSec !== null && Math.abs(fineAlignSec) > 1e-9) {
        stages.push({
          key: 'align-fine',
          label: 'Fine · onset-snap',
          className: styles.timingVizDiffBarAlignFine,
          deltaSec: fineAlignSec,
          deltaBeats: secToOrigBeats(fineAlignSec),
          anchorSec: entry.detected_time_sec,
        });
      }
    } else if (Math.abs(combinedAlignSec) > 1e-9) {
      stages.push({
        key: 'align-combined',
        label: 'Beat alignment (combined)',
        className: styles.timingVizDiffBarAlignCoarse,
        deltaSec: combinedAlignSec,
        deltaBeats: secToOrigBeats(combinedAlignSec),
        anchorSec: entry.detected_time_sec,
      });
    }
    if (hasPerPassQuantise) {
      for (const p of quantisePasses) {
        if (p.slots === null || p.slots === undefined || p.slots === 0) continue;
        const deltaSec = slotsToOrigSec(p.slots);
        if (deltaSec === undefined) continue;
        stages.push({
          key: p.key,
          label: p.label,
          className: QUANT_PASS_CLASSNAMES[p.key] ?? '',
          deltaSec,
          deltaBeats: secToOrigBeats(deltaSec),
          deltaSlots: p.slots,
        });
      }
    } else if (fallbackQuantSec !== undefined && Math.abs(fallbackQuantSec) > 1e-9) {
      stages.push({
        key: 'quantise-combined',
        label: 'Quantise (combined)',
        className: styles.timingVizDiffBarQuantGeo,
        deltaSec: fallbackQuantSec,
        deltaBeats: secToOrigBeats(fallbackQuantSec),
        deltaSlots: fallbackQuantSlots,
      });
    }
    if (anchorDriftSec !== undefined && Math.abs(anchorDriftSec) > 1e-9) {
      stages.push({
        key: 'anchor',
        label: 'Bar anchor drift',
        className: styles.timingVizDiffBarAnchor,
        deltaSec: anchorDriftSec,
        deltaBeats: anchorDriftBeats,
      });
    }
    if (snapSec !== undefined && Math.abs(snapSec) > 1e-9) {
      stages.push({
        key: 'midi-snap',
        label: 'MIDI 1/48 snap',
        className: styles.timingVizDiffBarMidiSnap,
        deltaSec: snapSec,
        deltaBeats: snapBeats,
        deltaSlots: origBeatsToSlots(snapBeats),
      });
    }
    if (drumOffsetSec !== undefined && Math.abs(drumOffsetSec) > 1e-9) {
      stages.push({
        key: 'drum-offset',
        label: 'Drum offset',
        className: styles.timingVizDiffBarDrumOffset,
        deltaSec: drumOffsetSec,
        deltaBeats: drumOffsetBeats,
        // `drumOffsetBeats` is in quarter notes; 1 qn = gridDivision/4
        // slots independent of the bar's time signature.
        deltaSlots:
          drumOffsetBeats !== undefined ? drumOffsetBeats * slotsPerQuarterNote : undefined,
      });
    }

    const hasAcoustic = entryHasAcousticFields(entry);
    const renderSubsectionToggle = (
      label: string,
      isOpen: boolean,
      onToggle: () => void,
    ) => (
      <button
        type="button"
        className={styles.debugDetailsToggle}
        onClick={(e) => {
          stop(e);
          onToggle();
        }}
        onMouseDown={stop}
        aria-expanded={isOpen}
      >
        {isOpen ? (
          <ChevronDown size={12} aria-hidden="true" />
        ) : (
          <ChevronRight size={12} aria-hidden="true" />
        )}
        {label}
      </button>
    );
    return (
      <div className={styles.debugDetails} onMouseDown={stop}>
        <button
          type="button"
          className={styles.debugDetailsToggle}
          onClick={(e) => {
            stop(e);
            setOpen((o) => !o);
          }}
          onMouseDown={stop}
          aria-expanded={open}
        >
          {open ? (
            <ChevronDown size={12} aria-hidden="true" />
          ) : (
            <ChevronRight size={12} aria-hidden="true" />
          )}
          Debug details
        </button>
        {open && (
          <>
            {renderSubsectionToggle(
              'Timing',
              timingOpen,
              () => setTimingOpen((o) => !o),
            )}
            {timingOpen && rendered && (
              <OnsetTimingVisualization
                entry={entry}
                rendered={rendered}
                gridDivision={gridDivision}
                stages={stages}
                finalSec={finalSec}
                displayedBarIndex={displayedBarIndex}
              />
            )}
            {timingOpen &&
              renderDebugStageSections({
                entry,
                rendered,
                timing,
                gridDivision,
                slotsPerQuarterNote,
              })}
            {hasAcoustic && renderSubsectionToggle(
              'Acoustic properties',
              acousticOpen,
              () => setAcousticOpen((o) => !o),
            )}
            {hasAcoustic && acousticOpen && renderAcousticSection(entry)}
          </>
        )}
      </div>
    );
  }
);

/** Per-quantise-pass bar colour, keyed by the pass's stable key (as
 * minted in {@link computeProvenanceTiming}). Kept here rather than on
 * the pass struct so the timing-math module stays free of CSS-module
 * imports. */
const QUANT_PASS_CLASSNAMES: Record<string, string> = {
  'q-geo': styles.timingVizDiffBarQuantGeo,
  'q-env': styles.timingVizDiffBarQuantEnv,
  'q-grid': styles.timingVizDiffBarQuantGrid,
  'q-llm': styles.timingVizDiffBarQuantLlm,
};
