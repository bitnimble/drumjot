import { NoteProvenanceEntry } from 'src/editing/provenance/debug_zip';
import type { BarTiming } from 'src/editing/playback/timeline';
import type { StructBar, StructNote } from 'src/editing/structure/structure_store';
import type { StructuralPresenter } from 'src/editing/structure/structural_presenter';
import { NoteProvenanceContextValue } from 'src/editing/provenance/provenance_contexts';
import { TICKS_PER_BEAT } from 'src/midi/to_midi';
import { midiGridTicks, renderedBeatInBar } from './provenance_format';

/**
 * Rendering context for a kept note. Present only when the provenance
 * panel is hosted by a kept note (i.e. by NoteView); lets the panel
 * compare the detector's view of the onset against where the score
 * actually drew it after all post-detection processing.
 */
export type RenderedNoteContext = {
  note: StructNote;
  bar: StructBar;
  provenance: NoteProvenanceContextValue;
};

/**
 * One quantise pass contribution (provenance format v3). `slots` is
 * `null`/`undefined` when the pass didn't run for this onset, `0` when it
 * ran but didn't shift. `className` is intentionally absent, it's a CSS
 * concern the caller layers on when building the visualization.
 */
export type QuantPass = {
  key: string;
  label: string;
  slots: number | null | undefined;
};

/**
 * The full detected → final timing decomposition for one onset. Every
 * field is derived purely from the provenance entry, the file-level
 * alignment fields, the eager per-bar timings, and the live drum-offset
 * slider, with no React or CSS. {@link NoteProvenanceDetails} wires this
 * struct into both the timing visualization (assembling coloured stage
 * bars) and the textual per-stage sections.
 */
export type ProvenanceTiming = {
  displayedBarIndex: number | undefined;
  currentQuantizedBeat: number | undefined;
  originalBar: StructBar | undefined;
  originalBarIndex: number | undefined;
  originalQuantizedBeat: number | undefined;
  originalQuantizedSec: number | undefined;
  snapBeats: number | undefined;
  snapSec: number | undefined;
  snapMs: number | undefined;
  finalSec: number | undefined;
  drumOffsetBeats: number | undefined;
  drumOffsetSec: number | undefined;
  anchorDriftSec: number | undefined;
  anchorDriftBeats: number | undefined;
  envelopeRefineSec: number | undefined;
  rawModelSec: number | null | undefined;
  coarseAlignSec: number | null;
  fineAlignSec: number | null;
  hasAlignSplit: boolean;
  combinedAlignSec: number;
  quantisePasses: QuantPass[];
  hasPerPassQuantise: boolean;
  fallbackQuantSec: number | undefined;
  fallbackQuantSlots: number | undefined;
  barGridFrameShiftSec: number | undefined;
  unknownDriftSec: number | undefined;
  unknownDriftBeats: number | undefined;
  unknownDriftMs: number | undefined;
  /** ts-beats of the original bar from audio seconds; `undefined` when
   * the bar's tempo can't be resolved. */
  secToOrigBeats: (sec: number) => number | undefined;
  /** Integer slot count → audio seconds in the original bar's frame. */
  slotsToOrigSec: (slots: number) => number | undefined;
  /** ts-beats → slot count, using `gridDivision / unit` slots-per-ts-beat. */
  origBeatsToSlots: (beats: number | undefined) => number | undefined;
};

type ComputeInput = {
  entry: NoteProvenanceEntry;
  rendered: RenderedNoteContext | undefined;
  barTimings: ReadonlyMap<number, BarTiming> | null;
  structural: StructuralPresenter | null;
  songLeadInSec: number;
  gridDivision: number;
  slotsPerQuarterNote: number;
};

/**
 * Compute the full detected → final timing decomposition for one onset.
 * Behaviour-preserving lift of the timing-math block formerly inlined in
 * {@link NoteProvenanceDetails}; see that component's inline comments
 * (preserved below) for the reasoning behind each stage.
 */
export function computeProvenanceTiming({
  entry,
  rendered,
  barTimings,
  structural,
  songLeadInSec,
  gridDivision,
  slotsPerQuarterNote,
}: ComputeInput): ProvenanceTiming {
  // Two coordinate frames are tracked here:
  //
  //   1. ORIGINAL, the bar the detector placed the onset in
  //      (`entry.bar + 1` in jot's 1-indexed convention) and the
  //      post-MIDI-snap beat position inside it. Immutable for a given
  //      note: derived purely from `entry.tick` (the unsnapped MIDI
  //      tick preserved through `from_midi.ts`) and the 1/48 grid
  //      snap, neither of which depend on the Beat-offset slider.
  //   2. CURRENT, `rendered.bar` and the note's current quantized
  //      position inside it, after `applyDrumOffsetStructure` has
  //      re-bucketed the note under the user's slider value. This is
  //      what the score is drawing right now and what the playback
  //      scheduler will fire.
  //
  // The Snap-delta row reads ORIGINAL (so dragging the Beat slider
  // doesn't change it, it's a property of the MIDI quantization, not
  // the user's view); the Final-position row reads CURRENT (so the
  // operator sees where the note actually plays after the slider);
  // the Drum-offset row is the transition between them.
  let displayedBarIndex: number | undefined;
  let currentQuantizedBeat: number | undefined;
  let currentSecPerQuarterNote: number | undefined;
  let originalBar: StructBar | undefined;
  let originalBarIndex: number | undefined;
  let originalSecPerQuarterNote: number | undefined;
  let originalQuantizedBeat: number | undefined;
  let originalQuantizedSec: number | undefined;
  let snapBeats: number | undefined;
  let snapSec: number | undefined;
  let snapMs: number | undefined;
  let finalSec: number | undefined;
  let drumOffsetBeats: number | undefined;
  let drumOffsetSec: number | undefined;
  let anchorDriftSec: number | undefined;
  let anchorDriftBeats: number | undefined;

  if (rendered) {
    currentQuantizedBeat = renderedBeatInBar(rendered.note, rendered.bar);
    displayedBarIndex = rendered.bar.index;

    const currentBarTiming = barTimings?.get(rendered.bar.index);
    if (currentBarTiming && rendered.bar.beats > 0) {
      currentSecPerQuarterNote = currentBarTiming.durationSec / rendered.bar.beats;
    }

    // The ORIGINAL bar lives in the rendered jot's structure at array
    // position `provenance.leadBars + entry.bar` (see the docstring on
    // `NoteProvenanceContextValue.leadBars` and `note_provenance.py`:
    // the MIDI lays `lead_bars` empty bar-0-sized blocks before
    // transcriber bar 0). We look it up by *array position*, NOT by
    // `bar.index === entry.bar + 1`, because the naïve index mapping
    // breaks whenever `from_midi.ts` counts more leading all-rest bars
    // than the transcriber's `lead_bars` did; e.g. when the filter
    // LLM kept no onsets in transcriber bars 0..N-1, those bars come
    // through the MIDI as all-rest and `from_midi`'s leading-rest walk
    // folds them into the lead-in, inflating `jot.globalMetadata.leadBars`
    // past `provenance.leadBars` and shifting every drum bar's
    // `bar.index` left by the difference. Array position survives that:
    // bar 0 of the transcriber's structure is always at array index
    // `provenance.leadBars`, with or without the inflation. `applyDrumOffsetStructure`
    // shallow-clones bars and preserves both array order and `index`,
    // so this lookup is also drum-offset-invariant.
    const structBars = structural?.layers[0]?.bars;
    const originalBarArrayPos = rendered.provenance.leadBars + entry.bar;
    originalBar =
      structBars && originalBarArrayPos >= 0 && originalBarArrayPos < structBars.length
        ? structBars[originalBarArrayPos]
        : undefined;
    originalBarIndex = originalBar?.index;
    const originalBarTiming =
      originalBarIndex !== undefined ? barTimings?.get(originalBarIndex) : undefined;
    if (originalBar && originalBarTiming && originalBar.beats > 0) {
      originalSecPerQuarterNote = originalBarTiming.durationSec / originalBar.beats;
    }

    // Snap delta is the audio-time displacement from the detected onset
    // to where the 1/48 MIDI grid landed; both expressed in the
    // ORIGINAL bar's tempo frame. Derived from `entry.tick`
    // (post-`onsets_midi.py` rounding to integer ticks, pre-`from_midi`
    // grid snap), so the value is a fixed property of the MIDI and
    // doesn't move when the Beat-offset slider re-buckets the rendered
    // note into a different bar.
    if (
      originalBar &&
      originalBarTiming &&
      originalSecPerQuarterNote !== undefined &&
      entry.tick !== null
    ) {
      const gridTicks = midiGridTicks(gridDivision);
      const postSnapTick = Math.round(entry.tick / gridTicks) * gridTicks;
      const snapDeltaQn = (postSnapTick - entry.tick) / TICKS_PER_BEAT;
      // qn → ts-beats: 1 ts-beat = 4/unit qn, so 1 qn = unit/4 ts-beats.
      snapBeats = snapDeltaQn * (originalBar.tsUnit / 4);
      snapSec = snapDeltaQn * originalSecPerQuarterNote;
      snapMs = snapSec * 1000;
      // Post-Python-quantise position, sourced from the provenance
      // sidecar's `quantised_time_sec` (audio time, set by every pass
      // that placed this onset). The earlier `entry.beat_in_bar +
      // snapBeats` formula combined the *detector* beat with only the
      // JS-side 1/48 snap, it ignored the Python geometric / envelope
      // / grid / LLM shifts, so the "Quantized to" readout could print
      // a value past the bar's last slot (e.g. "49/48") while the MIDI
      // bytes carried the true clamped position. Falls back to the
      // detected time when quantise didn't shift this onset; that
      // matches what the MIDI tick was emitted from (see
      // `onsets_to_midi_bytes`).
      const quantSourceSec = entry.quantised_time_sec ?? entry.detected_time_sec;
      const originalBarAudioStart = originalBarTiming.startSec - songLeadInSec;
      const intraBarQn =
        (quantSourceSec - originalBarAudioStart) / originalSecPerQuarterNote;
      originalQuantizedBeat = 1 + intraBarQn * (originalBar.tsUnit / 4);
      originalQuantizedSec = quantSourceSec;
    }

    // Final position uses the CURRENT bar's timing; where the note
    // plays now after the slider has applied its shift.
    if (currentBarTiming && currentSecPerQuarterNote !== undefined) {
      const intra =
        (currentQuantizedBeat - 1) * (4 / rendered.bar.tsUnit) * currentSecPerQuarterNote;
      finalSec = currentBarTiming.startSec + intra - songLeadInSec;
    }

    // Bar-anchor drift: the difference between where the JOT places
    // the note's original bar in audio time and where the detector's
    // post-alignment view of beat_in_bar implies it should be. In a
    // perfect round-trip these match exactly; in practice the most
    // common source of mismatch is `transcriber/app/pipeline/onsets_midi.py`'s
    // `compute_bar_tick_grid` rounding `lead_bars = round(lead_in_secs *
    // initial_tempo_ticks_per_sec / bar0_ticks)` to zero when the
    // pre-roll is shorter than ~half a bar. The MIDI then carries no
    // lead-in, `from_midi.ts` reconstructs `songLeadIn = 0`, and every
    // bar is anchored `lead_in_secs` early in audio time. Secondary
    // sources (per-bar bpm attribution drift, integer-tick rounding)
    // can also contribute small amounts but are dominated by lead-in
    // rounding when it triggers.
    if (
      originalBar &&
      originalBarTiming &&
      originalSecPerQuarterNote !== undefined &&
      originalSecPerQuarterNote > 0
    ) {
      // Where the detector says `originalBar` starts in audio time:
      // back out the intra-bar offset from the detected onset.
      const intraSecFromDetected =
        (entry.beat_in_bar - 1) * (4 / originalBar.tsUnit) * originalSecPerQuarterNote;
      const transcriberBarAudioTime = entry.detected_time_sec - intraSecFromDetected;
      // Where the JOT places that same bar in audio time.
      const jotBarAudioTime = originalBarTiming.startSec - songLeadInSec;
      const drift = jotBarAudioTime - transcriberBarAudioTime;
      if (Math.abs(drift) > 1e-6) {
        anchorDriftSec = drift;
        anchorDriftBeats = (drift / originalSecPerQuarterNote) * (originalBar.tsUnit / 4);
      }
    }

    // Manual drum-offset slider (`effectiveDrumOffsetBeats` is in
    // quarter notes by convention, `applyDrumOffset` shifts each
    // `note.beat` by this amount, and `note.beat` itself is in quarter
    // notes from bar start). The audio-time displacement uses the
    // CURRENT (destination) bar's tempo, since that's the tempo the
    // note's new intra-bar position is interpreted under. Cross-bar
    // shifts under a per-bar bpm change are approximate; the residual
    // surfaces in the Unknown-drift row.
    const offsetQn = structural?.effectiveDrumOffsetBeats;
    if (typeof offsetQn === 'number' && Math.abs(offsetQn) > 1e-9) {
      drumOffsetBeats = offsetQn;
      if (currentSecPerQuarterNote !== undefined) {
        drumOffsetSec = offsetQn * currentSecPerQuarterNote;
      }
    }
  }

  // ─── New per-stage shifts (provenance format v3) ───
  //
  // Envelope refine inside the `onsets` stage: shift the ADTOF model
  // peak time picked up when `_refine_peak_times_audio` snapped to the
  // audio envelope's local max. `null`/`undefined` on legacy bundles
  // or non-ADTOF detection paths (none in production today).
  const rawModelSec = entry.raw_model_time_sec;
  const envelopeRefineSec =
    rawModelSec !== null && rawModelSec !== undefined
      ? entry.detected_time_sec - rawModelSec
      : undefined;

  // Beat-grid alignment split (v3). When the bundle predates the
  // split, both fields are null; we fall back to surfacing the
  // combined `beatAlignmentOffsetSec` as a single "Beat alignment"
  // stage. Both halves can be negative; only suppress when literally
  // 0.0 (or the bundle didn't surface them).
  const coarseAlignSec = rendered?.provenance.beatAlignCoarseOffsetSec ?? null;
  const fineAlignSec = rendered?.provenance.beatAlignFineOffsetSec ?? null;
  const hasAlignSplit = coarseAlignSec !== null || fineAlignSec !== null;
  const combinedAlignSec = rendered?.provenance.beatAlignmentOffsetSec ?? 0;

  // Per-pass quantise contributions (v3). `null`/`undefined` means
  // the pass didn't run for this onset (off-grid for any pass after
  // geometric; envelope pass skipped because no envelope was
  // available; grid/LLM toggled off; LLM cancelled/errored). `0`
  // means the pass ran but didn't shift (or its shift was rejected
  // by the monotonic-injective guard).
  const quantisePasses: QuantPass[] = [
    {
      key: 'q-geo',
      label: 'Quantise · geometric snap',
      slots: entry.geometric_shift_slots,
    },
    {
      key: 'q-env',
      label: 'Quantise · envelope re-snap',
      slots: entry.envelope_shift_slots,
    },
    {
      key: 'q-grid',
      label: 'Quantise · musical grid',
      slots: entry.grid_shift_slots,
    },
    {
      key: 'q-llm',
      label: 'Quantise · LLM residual',
      slots: entry.llm_shift_slots,
    },
  ];
  const hasPerPassQuantise = quantisePasses.some(
    (p) => p.slots !== null && p.slots !== undefined
  );

  // Legacy v1/v2 fallback: a single combined quantise row using the
  // summed `quantised_shift_slots`. Only emitted when the per-pass
  // split is absent so we never double-count.
  const fallbackQuantSec =
    !hasPerPassQuantise &&
    entry.quantised_time_sec !== null &&
    entry.quantised_time_sec !== undefined
      ? entry.quantised_time_sec - entry.detected_time_sec
      : undefined;
  const fallbackQuantSlots =
    !hasPerPassQuantise &&
    entry.quantised_shift_slots !== null &&
    entry.quantised_shift_slots !== undefined
      ? entry.quantised_shift_slots
      : undefined;

  // ─── Unit conversion helpers (shared by chain build + dl render) ───
  //
  // ts-beats of the original bar from audio seconds. Returns
  // `undefined` when the bar's tempo can't be resolved.
  const secToOrigBeats = (sec: number): number | undefined =>
    originalBar !== undefined &&
    originalSecPerQuarterNote !== undefined &&
    originalSecPerQuarterNote > 0
      ? (sec / originalSecPerQuarterNote) * (originalBar.tsUnit / 4)
      : undefined;
  // Integer slot count → audio seconds in the original bar's frame.
  // 1 slot = 1/(gridDivision/4) qn, so `slots / slotsPerQn × secPerQn`.
  const slotsToOrigSec = (slots: number): number | undefined =>
    originalSecPerQuarterNote !== undefined && originalSecPerQuarterNote > 0
      ? (slots / slotsPerQuarterNote) * originalSecPerQuarterNote
      : undefined;
  // ts-beats → slot count, using `gridDivision / unit` slots-per-ts-beat.
  const origBeatsToSlots = (beats: number | undefined): number | undefined =>
    beats !== undefined && originalBar !== undefined
      ? (beats * gridDivision) / originalBar.tsUnit
      : undefined;

  // ─── Bar-grid frame shift attribution ───
  //
  // Two values that LOOK like they should advance the onset along the
  // detected → final chain actually don't, they shift the *frame*
  // we're measuring against, not the onset itself:
  //
  //   - **Envelope refine** is pre-canonical. `entry.detected_time_sec`
  //     is set in `note_provenance.py` to `float(c.time)`, and `c.time`
  //     was already snapped to the audio-envelope local-max by
  //     `_refine_peak_times_audio` *before* being stored on the
  //     candidate. The chain starts AT detected_time_sec, so adding
  //     `envelopeRefineSec = detected − raw_model` on top would
  //     double-count the move from raw model peak to canonical time.
  //   - **Beat-grid alignment** shifts every beat's `time` (and thus
  //     each bar's `start_time`) by `align_offset_sec`, but the
  //     onset's audio time `c.time` is unchanged. The post-alignment
  //     bar positions ARE reflected in the JOT's bar audio anchors
  //     (via the transcriber's per-bar `set_tempo` map +
  //     `from_midi.ts`'s `songLeadIn` reconstruction), so
  //     `finalSec = currentBarTiming.startSec + intra + songLeadIn`
  //     is already in the post-alignment frame.
  //
  // Their combined value `(envelopeRefineSec ?? 0) + combinedAlignSec`
  // is what previously surfaced unlabeled as "Unknown drift" on every
  // onset of a track with a non-zero net frame shift, the chain
  // included those shifts in `accountedSec` while `finalSec` (rightly)
  // did not, leaving the gap as a confusing residual of magnitude
  // `|envelopeRefineSec + combinedAlignSec|` (often 50-80 ms once
  // both contribute). We now keep them OUT of `accountedSec` (so the
  // residual collapses to just the genuinely-unknown rounding noise)
  // and surface their sum as its own labeled "Bar-grid frame shift"
  // row in the debug details below.
  const barGridFrameShiftSec: number | undefined =
    envelopeRefineSec !== undefined || Math.abs(combinedAlignSec) > 1e-9
      ? (envelopeRefineSec ?? 0) + combinedAlignSec
      : undefined;

  // ─── Unknown drift residual ───
  //
  // `finalSec` minus everything the named-stages chain accounts for.
  // After the bar-grid frame-shift fix above this should sit at ~0
  // for well-formed bundles; any remaining gap is a genuine drift
  // source we haven't enumerated (per-bar BPM attribution mismatches
  // between the transcriber and from_midi, cross-bar drum-offset
  // re-bucketing under varying tempo, etc.).
  let unknownDriftSec: number | undefined;
  let unknownDriftBeats: number | undefined;
  let unknownDriftMs: number | undefined;
  if (finalSec !== undefined && rendered) {
    let accountedSec = entry.detected_time_sec;
    // `envelopeRefineSec` NOT added, see the `barGridFrameShiftSec`
    // comment above. The chain starts at `entry.detected_time_sec`,
    // which is already the post-envelope-refine audio time.
    if (hasPerPassQuantise) {
      for (const p of quantisePasses) {
        if (p.slots === null || p.slots === undefined || p.slots === 0) continue;
        const sec = slotsToOrigSec(p.slots);
        if (sec !== undefined) accountedSec += sec;
      }
    } else if (fallbackQuantSec !== undefined) {
      accountedSec += fallbackQuantSec;
    }
    if (snapSec !== undefined) accountedSec += snapSec;
    // `combinedAlignSec` NOT added, see the `barGridFrameShiftSec`
    // comment above. The alignment shifts the beat grid (and
    // thereby `bar.start_time`), not the onset's audio time; the
    // post-alignment bar positions are already reflected in
    // `originalBarTiming.startSec + songLeadIn`.
    if (anchorDriftSec !== undefined) accountedSec += anchorDriftSec;
    if (drumOffsetSec !== undefined) accountedSec += drumOffsetSec;
    const residual = finalSec - accountedSec;
    if (Math.abs(residual) > 1e-6) {
      unknownDriftSec = residual;
      unknownDriftMs = residual * 1000;
      unknownDriftBeats = secToOrigBeats(residual);
    }
  }

  return {
    displayedBarIndex,
    currentQuantizedBeat,
    originalBar,
    originalBarIndex,
    originalQuantizedBeat,
    originalQuantizedSec,
    snapBeats,
    snapSec,
    snapMs,
    finalSec,
    drumOffsetBeats,
    drumOffsetSec,
    anchorDriftSec,
    anchorDriftBeats,
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
    barGridFrameShiftSec,
    unknownDriftSec,
    unknownDriftBeats,
    unknownDriftMs,
    secToOrigBeats,
    slotsToOrigSec,
    origBeatsToSlots,
  };
}
