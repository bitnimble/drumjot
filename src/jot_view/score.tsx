import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { NoteProvenanceEntry } from 'src/debug_zip';
import { Instrument, Modifier, Sticking } from 'src/dsl';
import {
  RenderedJot,
  StructuralBar,
  StructuralNote,
  StructuralPatternSpan,
  StructuralTupletSpan,
  ViewConfig,
} from 'src/jot';
import { buildTimeline, computeWindowPeaks, jotPlayer } from 'src/playback';
import sharedStyles from '../jot_view.module.css';
import { GutterResizeHandle } from './components/gutter_resize_handle';
import {
  BarTimingsContext,
  NoteProvenanceContext,
  NoteProvenanceContextValue,
  SelectionContext,
} from './contexts';
import { Playhead } from './playback';
import styles from './score.module.css';

/**
 * Shared click-to-seek handler for the score bars row and the audio-track
 * waveforms. Bails on clicks that originated on a note, pattern label,
 * or anything else tagged `data-noseek` so those keep their own
 * behaviour. `e.currentTarget` is the bars-row element whose left edge
 * is x=0 in `bar.x` space, so `clientX - rect.left` is the bars-row-
 * local pixel regardless of horizontal scroll.
 */
export function seekFromClick(
  e: React.MouseEvent<HTMLDivElement>,
  onSeek: (x: number) => void
): void {
  if (e.button !== 0) return;
  if ((e.target as HTMLElement).closest('[data-noseek]')) return;
  const rect = e.currentTarget.getBoundingClientRect();
  onSeek(e.clientX - rect.left);
}

/**
 * True when `beat` falls inside any tuplet bracket on this bar. The
 * upper bound is inclusive because `endBeat` is now the last slot's
 * onset (see jot.ts) — the final tuplet note sits exactly on it and is
 * still covered by the bracket.
 */
function coveredByTuplet(bar: StructuralBar, beat: number): boolean {
  const eps = 1e-6;
  return bar.tupletSpans.some(
    (s) => beat >= s.startBeat - eps && beat <= s.endBeat + eps
  );
}

export function formatSubtitle(jot: RenderedJot): string {
  const parts: string[] = [];
  const { bpm, time, vol } = jot.globalMetadata;
  if (typeof bpm === 'number') parts.push(`${bpm} bpm`);
  else if (bpm) parts.push(`${bpm.start ?? '?'}-${bpm.end} bpm`);
  if (time) parts.push(`${time.count}/${time.unit}`);
  if (typeof vol === 'string') parts.push(vol);
  else if (vol) parts.push(`${vol.start ?? '?'} -> ${vol.end}`);
  return parts.join('  -  ');
}

export const Legend = observer(({ jot }: { jot: RenderedJot }) => {
  // Aggregate unique pitches across all voices, in first-seen order.
  // Reads `jot.structure` (zoom-invariant) so the legend doesn't
  // re-render every time the zoom slider moves.
  const seen = new Map<string, { color: string; name?: string }>();
  for (const voice of jot.structure.voices) {
    for (const bar of voice.bars) {
      for (const pitch of Object.keys(bar.tracks)) {
        if (!seen.has(pitch)) {
          const track = bar.tracks[pitch];
          seen.set(pitch, { color: track.color, name: track.instrument.name });
        }
      }
    }
  }
  if (seen.size === 0) return null;
  return (
    <div className={sharedStyles.legend}>
      {Array.from(seen.entries()).map(([pitch, info]) => (
        <span key={pitch} className={sharedStyles.legendChip}>
          <span className={sharedStyles.legendSwatch} style={{ background: info.color }} />
          <strong>{pitch}</strong>
          {info.name ? <span>{info.name}</span> : null}
        </span>
      ))}
    </div>
  );
});

/**
 * Sticky-gutter header above the audio tracks / score that labels each
 * bar boundary with its 1-based bar number and the playback time at that
 * boundary (mm:ss). Tick marks sit on the same `bar.x` line as the
 * score's barlines below so the header reads as a ruler over the
 * timeline. Click-to-seek mirrors the score and audio-track rows.
 *
 * Per-bar timings come from the live playback timeline whenever it
 * matches the current jot (so tempo overrides and the lead-in offset
 * stay in sync with the playhead); otherwise we build a one-shot
 * timeline so the header still labels everything correctly before the
 * user hits Play.
 */
export const TimelineHeader = observer(
  ({
    jot,
    onSeek,
    onResizeGutterStart,
  }: {
    jot: RenderedJot;
    onSeek: (x: number) => void;
    /** Pointer-down handler for the gutter resize affordance rendered
     * on the right edge of this header's gutter. */
    onResizeGutterStart: (e: React.PointerEvent<HTMLDivElement>) => void;
  }) => {
    // Reading the structural cache (not `jot.resolved`) keeps this
    // header stable across zoom — the per-tick `--bar-start-beat` is
    // set inline, and CSS calc() multiplies by the score-root's
    // `--px-per-beat` to get the final pixel position. Without this
    // the header re-rendered every wheel tick, re-creating 100+ tick
    // marks just to reposition each by one calc-arithmetic step.
    const voice = jot.structure.voices[0];
    if (!voice || voice.bars.length === 0) return null;

    const liveTimeline = jotPlayer.timeline;
    const timeline =
      liveTimeline.bars.length > 0 && liveTimeline.rendered === jot
        ? liveTimeline
        : buildTimeline(jot);

    const leadInBeats = voice.drumsLeadInSec * (voice.leadInBpm / 60);
    let voiceBeats = leadInBeats;
    for (const b of voice.bars) voiceBeats += b.beats;

    let cumBeats = leadInBeats;
    return (
      <div className={styles.timelineHeader}>
        <div className={styles.timelineHeaderGutter}>
          <span className={styles.timelineHeaderLabel}>Bar / Time</span>
          <GutterResizeHandle onResizeStart={onResizeGutterStart} />
        </div>
        <div
          className={styles.timelineHeaderBarsRow}
          style={
            { ['--voice-beats' as string]: voiceBeats } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          {voice.bars.map((bar, i) => {
            const timing = timeline.bars[i];
            const timeSec = timing?.startSec ?? 0;
            const startBeat = cumBeats;
            cumBeats += bar.beats;
            return (
              <div
                key={i}
                className={styles.timelineHeaderTick}
                style={
                  {
                    ['--bar-start-beat' as string]: startBeat,
                  } as React.CSSProperties
                }
              >
                <span className={styles.timelineHeaderBar}>{bar.index}</span>
                <span className={styles.timelineHeaderTime}>{formatTime(timeSec)}</span>
              </div>
            );
          })}
          <Playhead showLabel onSeek={onSeek} />
        </div>
      </div>
    );
  }
);

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const min = Math.floor(total / 60);
  const sec = total % 60;
  return `${min}:${String(sec).padStart(2, '0')}`;
}

export const BarView = observer(
  ({
    bar,
    pitches,
    config,
    isAnacrusis,
    highlightedPattern,
    onPatternClick,
    isPitchAudible,
    showBrackets = true,
    rowPitch,
    pitchOrder,
  }: {
    bar: StructuralBar;
    pitches: string[];
    config: ViewConfig;
    isAnacrusis: boolean;
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    isPitchAudible: (pitch: string) => boolean;
    /**
     * Whether to draw bar chrome that belongs to the score as a whole
     * (tuplet brackets and lead-in label). Pattern brackets are drawn
     * per-row instead — see {@link rowPitch} — so this flag doesn't
     * gate them.
     */
    showBrackets?: boolean;
    /**
     * In the unified mixer, the DSL pitch this BarView's row represents.
     * Pattern brackets only render when this pitch is in the span's
     * `pitches` set — rows for pitches the pattern doesn't play get no
     * bracket on that span, so the outline visually "skips" them.
     * Undefined falls back to drawing every span (label always shown).
     */
    rowPitch?: string;
    /**
     * Drum pitches in mixer-row order. Used together with {@link rowPitch}
     * to decide which row is the topmost / bottommost contributor for a
     * given span — the topmost shows the pattern label and the top edge
     * of the bracket, the bottommost shows the bottom edge, middles show
     * only the left/right sides so the outline reads as one connected
     * box across all participating rows.
     */
    pitchOrder?: readonly string[];
  }) => {
    const beatCount = bar.time.count;
    // Beat spacing inside the bar, in quarter notes. Each beat divider
    // is `i × beatSpacingBeats` quarter-notes into the bar, scaled to
    // pixels by the score-root's `--px-per-beat`. Stable per bar.
    const beatSpacingBeats = bar.beats / beatCount;
    // Inline style carries only zoom-invariant data so React's prop
    // diff sees no change on a zoom tick: `--bar-beats` is the bar's
    // length in quarter notes. `minHeight` is config-derived; the bar
    // also `align-self: stretch`es to the barsRow's full height so the
    // right-edge barline doesn't stop short of the row separator (the
    // pitch gutter is taller than a single lane). When stretched, the
    // flex-column on `.bar` keeps the lanes vertically centred; when
    // not stretched, the `minHeight` preserves the natural lane-stack
    // size.
    const barStyle = {
      ['--bar-beats' as string]: bar.beats,
      minHeight: pitches.length * (config.trackHeight as number),
    } as React.CSSProperties;
    return (
      <div
        className={classNames(styles.bar, isAnacrusis && styles.barAnacrusis)}
        style={barStyle}
      >
        {/* One dashed line directly under each beat's notehead — the
            same x the renderer places that beat's note at, computed
            in CSS from `--divider-beat × --px-per-beat`. */}
        {Array.from({ length: beatCount }, (_, i) => (
          <div
            key={`beat-${i + 1}`}
            className={styles.beatDivider}
            style={{ ['--divider-beat' as string]: i * beatSpacingBeats } as React.CSSProperties}
          />
        ))}
        {pitches.map((pitch) => {
          const track = bar.tracks[pitch];
          const dim = !isPitchAudible(pitch);
          return (
            <div
              key={pitch}
              className={classNames(styles.lane, dim && styles.laneDim)}
              style={{ height: config.trackHeight }}
            >
              {track?.notes.map((note, i) => (
                <NoteView
                  key={i}
                  note={note}
                  bar={bar}
                  color={track.color}
                  config={config}
                  instrument={track.instrument}
                  // A non-straight note already inside a tuplet bracket
                  // is explained by that bracket, so only flag the
                  // strays (e.g. an off-grid note not authored as a
                  // group) individually.
                  offGrid={!note.straight && !coveredByTuplet(bar, note.beat)}
                />
              ))}
            </div>
          );
        })}
        {bar.patternSpans.map((span, i) => {
          const position = bracketPositionForRow(span, rowPitch, pitchOrder);
          if (position === 'hidden') return null;
          return (
            <PatternBracket
              key={i}
              span={span}
              highlighted={highlightedPattern === span.name}
              onClick={onPatternClick}
              position={position}
            />
          );
        })}
        {showBrackets &&
          bar.tupletSpans.map((span, i) => <TupletBracket key={i} span={span} />)}
      </div>
    );
  }
);

/**
 * Where this row's slice of a pattern bracket sits within the
 * top-to-bottom span across all contributing rows.
 *
 *   - `single`: this is the only contributing row — render a full box.
 *   - `top`:    this is the topmost contributor — render top edge + sides + label.
 *   - `middle`: a contributor between top and bottom — render sides only;
 *               the bracket reads as continuous across stacked rows.
 *   - `bottom`: the bottommost contributor — render bottom edge + sides.
 *   - `hidden`: this row's pitch isn't in the pattern — render nothing.
 */
type BracketPosition = 'single' | 'top' | 'middle' | 'bottom' | 'hidden';

function bracketPositionForRow(
  span: StructuralPatternSpan,
  rowPitch: string | undefined,
  pitchOrder: readonly string[] | undefined,
): BracketPosition {
  // No row context (non-mixer caller) → render as a self-contained box,
  // same as the pre-mixer behaviour.
  if (rowPitch === undefined || !pitchOrder) return 'single';
  if (!span.pitches.has(rowPitch)) return 'hidden';
  let firstIdx = -1;
  let lastIdx = -1;
  for (let i = 0; i < pitchOrder.length; i++) {
    if (span.pitches.has(pitchOrder[i])) {
      if (firstIdx === -1) firstIdx = i;
      lastIdx = i;
    }
  }
  // The pattern body could in principle include pitches the mixer doesn't
  // surface (e.g. a brand-new pitch type that hasn't been added to the
  // row order yet). Treat the row as a single contributor in that case
  // rather than silently producing an open-ended bracket.
  if (firstIdx === -1 || lastIdx === -1) return 'single';
  if (firstIdx === lastIdx) return 'single';
  const myIdx = pitchOrder.indexOf(rowPitch);
  if (myIdx === firstIdx) return 'top';
  if (myIdx === lastIdx) return 'bottom';
  return 'middle';
}

/**
 * Pattern bracket colors, cycled in `colorIndex` order. Each entry
 * resolves at runtime to the matching `--color-pattern-N` token in
 * src/design_tokens.css, so updating a shade only requires editing the
 * token — this array just names the slots. Length intentionally
 * matches the token list; if it ever shrinks, the modulo wrap below
 * silently recycles colors.
 */
const PATTERN_COLOR_VARS: readonly string[] = [
  'var(--color-pattern-1)',
  'var(--color-pattern-2)',
  'var(--color-pattern-3)',
  'var(--color-pattern-4)',
  'var(--color-pattern-5)',
  'var(--color-pattern-6)',
  'var(--color-pattern-7)',
  'var(--color-pattern-8)',
  'var(--color-pattern-9)',
  'var(--color-pattern-10)',
  'var(--color-pattern-11)',
  'var(--color-pattern-12)',
];

const PatternBracket = observer(
  ({
    span,
    highlighted,
    onClick,
    position,
  }: {
    span: StructuralPatternSpan;
    highlighted: boolean;
    onClick: (name: string) => void;
    position: Exclude<BracketPosition, 'hidden'>;
  }) => {
    const color =
      PATTERN_COLOR_VARS[span.colorIndex % PATTERN_COLOR_VARS.length];
    return (
      <div
        className={classNames(
          styles.patternBracket,
          (position === 'top' || position === 'single') && styles.patternBracketTop,
          (position === 'bottom' || position === 'single') && styles.patternBracketBottom,
          highlighted && styles.patternBracketHighlight
        )}
        style={
          {
            ['--span-start-beat' as string]: span.startBeat,
            ['--span-end-beat' as string]: span.endBeat,
            ['--pattern-color' as string]: color,
          } as React.CSSProperties
        }
      >
        {/* Label sits with the bracket's top edge — only render on the
            topmost contributing row so a multi-row bracket doesn't stack
            duplicate labels down the score. */}
        {(position === 'top' || position === 'single') && (
          <button
            type="button"
            data-noseek="true"
            className={classNames(
              styles.patternLabel,
              highlighted && styles.patternLabelHighlight
            )}
            onClick={(e) => {
              e.stopPropagation();
              onClick(span.name);
            }}
            onMouseDown={(e) => e.stopPropagation()}
            title={`Pattern usage: ${span.name} (click to highlight other usages)`}
          >
            {span.name}
          </button>
        )}
      </div>
    );
  }
);

/**
 * Classic engraved tuplet bracket: a thin line over the grouped notes
 * with the slot count (3 = triplet, 5 = quintuplet, ...) on it. Purely
 * decorative — no interaction, unlike the pattern bracket.
 */
const TupletBracket = observer(({ span }: { span: StructuralTupletSpan }) => (
  <div
    className={styles.tupletBracket}
    style={
      {
        ['--span-start-beat' as string]: span.startBeat,
        ['--span-end-beat' as string]: span.endBeat,
      } as React.CSSProperties
    }
    title={`${span.count}-tuplet (not a straight subdivision)`}
  >
    <span className={styles.tupletNumber}>{span.count}</span>
  </div>
));

const NoteView = observer(
  ({
    note,
    bar,
    color,
    config,
    instrument,
    offGrid,
  }: {
    note: StructuralNote;
    /**
     * The bar this note lives in. Carried through so the `Debug details`
     * panel can compute the rendered (post-quantization) beat position
     * — `note.beat` is in quarter-notes from bar start, and the bar's
     * time signature is needed to convert into the 1-indexed
     * `beat_in_bar` convention the provenance entry uses.
     */
    bar: StructuralBar;
    color: string;
    config: ViewConfig;
    instrument: Instrument;
    offGrid: boolean;
  }) => {
    const isAccent = note.modifiers.has('a');
    const isGhost = note.modifiers.has('g');
    const isFlam = note.modifiers.has('fl');
    const isDrag = note.modifiers.has('dr');
    const isCross = note.modifiers.has('x');
    const badge = pickBadge(note);
    const selection = React.useContext(SelectionContext);
    const selected = selection?.selectedNote === note;
    const [hovered, setHovered] = React.useState(false);
    const showLabel = selected || hovered;
    const description = offGrid
      ? `${describeNote(note, instrument)} — off the straight grid (triplet/tuplet)`
      : describeNote(note, instrument);
    // Per-note debug provenance, when a filter-mode debug bundle is
    // loaded. Keyed by the original MIDI tick preserved through
    // `from_midi.ts`. Falls back to `undefined` for notes that didn't
    // round-trip through MIDI (e.g. examples, hand-loaded jots) or
    // when no bundle is loaded — the `Debug details` section is hidden
    // in those cases.
    const provenance = React.useContext(NoteProvenanceContext);
    const sourceMeta = note.source.metadata as
      | { midi?: { tick?: number } }
      | undefined;
    const tick = sourceMeta?.midi?.tick;
    const provenanceEntry =
      provenance && typeof tick === 'number'
        ? provenance.byTick.get(`${note.pitch}:${tick}`)
        : undefined;

    return (
      <div
        // Notes opt out of click-to-seek so clicking a note keeps its
        // own meaning (selection / hover label) instead of moving the
        // playhead.
        data-noseek="true"
        className={classNames(
          styles.note,
          isAccent && styles.accent,
          isGhost && styles.ghost,
          isCross && styles.cross,
          note.roll && styles.roll,
          offGrid && styles.offGrid,
          selected && styles.noteSelected,
          showLabel && styles.noteShowingLabel,
          hovered && styles.noteHovered
        )}
        style={
          {
            // Beat is stable per note (set inline); the CSS rule on
            // `.note` derives `left` from `padLeft + beat × pxPerBeat`,
            // so a zoom tick changes one root var instead of mutating
            // this element's style.
            ['--note-beat' as string]: note.beat,
            top: (config.trackHeight as number) / 2,
            width: config.noteDiameter,
            height: config.noteDiameter,
            background: isCross ? 'var(--color-bg-base)' : color,
            color,
            borderStyle: isCross ? 'solid' : undefined,
            border: isCross ? `2px solid ${color}` : undefined,
          } as React.CSSProperties
        }
        // Suppress the container's mousedown handler — it begins a
        // marquee selection that clears the existing state on every
        // press, which would wipe this note's selection before the
        // click ever fires.
        onMouseDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation();
          selection?.selectNote(note);
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {isFlam && <FlamGrace color={color} config={config} />}
        {isDrag && <DragGrace color={color} config={config} />}
        {badge && <span className={styles.modifierBadge}>{badge}</span>}
        {note.sticking && <span className={styles.stickingBadge}>{note.sticking.toUpperCase()}</span>}
        {showLabel && (
          <div className={styles.noteLabel}>
            <div className={styles.noteLabelText}>{description}</div>
            {provenanceEntry && (
              <NoteProvenanceDetails
                entry={provenanceEntry}
                rendered={{ note, bar, provenance: provenance! }}
                startOpen
              />
            )}
          </div>
        )}
      </div>
    );
  }
);

function FlamGrace({ color, config }: { color: string; config: ViewConfig }) {
  const size = (config.noteDiameter as number) * 0.55;
  return (
    <span
      style={{
        position: 'absolute',
        left: -size - 2,
        top: '50%',
        transform: 'translateY(-50%)',
        width: size,
        height: size,
        background: color,
        borderRadius: '50%',
        opacity: 0.7,
      }}
    />
  );
}

function DragGrace({ color, config }: { color: string; config: ViewConfig }) {
  const size = (config.noteDiameter as number) * 0.45;
  return (
    <>
      {[0, 1].map((i) => (
        <span
          key={i}
          style={{
            position: 'absolute',
            left: -((size + 2) * (i + 1)),
            top: '50%',
            transform: 'translateY(-50%)',
            width: size,
            height: size,
            background: color,
            borderRadius: '50%',
            opacity: 0.6,
          }}
        />
      ))}
    </>
  );
}

/**
 * Optional rendering context for {@link NoteProvenanceDetails}.
 *
 * Present only when the panel is hosted by a kept note (i.e. by
 * NoteView). Lets the panel compare the detector's view of the onset
 * against where the score actually drew it after all post-detection
 * processing — beat-tracker alignment and the MIDI→Jot quantization
 * pass in {@link from_midi.ts} that snaps every onset to the 16th
 * grid. FilteredOnsetView omits this prop (a rejected onset has no
 * rendered note) and the panel skips the rendered/snap rows.
 */
type RenderedNoteContext = {
  note: StructuralNote;
  bar: StructuralBar;
  provenance: NoteProvenanceContextValue;
};

/**
 * Computes the rendered (post-quantization) beat position in the same
 * 1-indexed `beat_in_bar` convention the provenance entry uses, so the
 * "Detected beat" → "Quantized to" comparison reads natively.
 *
 * `note.beat` is in quarter-notes from bar start (0-indexed). The
 * provenance's `beat_in_bar` counts beats of the bar's time
 * signature (1-indexed, where downbeat = 1.0). The conversion is
 * `1 + note.beat × (time.count / bar.beats)` — equals
 * `1 + note.beat` in 4/4, scales correctly for 6/8, 7/8, etc.
 */
function renderedBeatInBar(note: StructuralNote, bar: StructuralBar): number {
  if (bar.beats <= 0) return 1;
  return 1 + (note.beat / bar.beats) * bar.time.count;
}

/**
 * Approximate seconds-equivalent of `beats` in a given bar, using the
 * bar's tempo. `bar.beats` is in quarter notes; the time signature
 * relates beats-of-the-signature to quarter notes
 * (one beat = 4/unit quarter notes). Returns `null` when the bar has
 * no resolvable tempo so the panel can fall back to beats-only.
 */
function beatsToSecondsInBar(
  beats: number,
  bar: StructuralBar,
  fallbackBpm: number | undefined,
): number | null {
  const meta = bar.source.metadata;
  const inlineBpm =
    meta && typeof meta.bpm === 'number'
      ? meta.bpm
      : meta && meta.bpm && typeof meta.bpm === 'object'
        ? meta.bpm.start ?? meta.bpm.end
        : undefined;
  const bpm = inlineBpm ?? fallbackBpm;
  if (typeof bpm !== 'number' || bpm <= 0) return null;
  const quarterNotesPerBeat = 4 / bar.time.unit;
  const quarterNotes = beats * quarterNotesPerBeat;
  return (quarterNotes * 60) / bpm;
}

/** Inverse of {@link beatsToSecondsInBar}; same fallback semantics. */
function secondsToBeatsInBar(
  seconds: number,
  bar: StructuralBar,
  fallbackBpm: number | undefined,
): number | null {
  const meta = bar.source.metadata;
  const inlineBpm =
    meta && typeof meta.bpm === 'number'
      ? meta.bpm
      : meta && meta.bpm && typeof meta.bpm === 'object'
        ? meta.bpm.start ?? meta.bpm.end
        : undefined;
  const bpm = inlineBpm ?? fallbackBpm;
  if (typeof bpm !== 'number' || bpm <= 0) return null;
  const quarterNotes = (seconds * bpm) / 60;
  const quarterNotesPerBeat = 4 / bar.time.unit;
  return quarterNotes / quarterNotesPerBeat;
}

/** Total pixel width of the timing-visualization panel — wide enough to
 * resolve a ±150 ms window without crowding the labels in the diff
 * rows, narrow enough that the parent label popover doesn't run off
 * the score for selections near the right edge. */
const TIMING_VIZ_WIDTH = 320;
const TIMING_VIZ_WAVE_HEIGHT = 40;
const TIMING_VIZ_ROW_HEIGHT = 18;
/** Minimum half-window so a near-zero snap/alignment still shows
 * waveform context around the detected onset; otherwise the window
 * collapses to a single pixel. Wider than the snap deltas typically
 * encountered so the snippet reads as "where in the song are we"
 * rather than a tight zoom on the transient alone. */
const TIMING_VIZ_MIN_HALF_WINDOW_SEC = 0.3;

/**
 * Per-onset timing diagram, rendered above the textual {@link
 * NoteProvenanceDetails} grid. Layers, top to bottom:
 *
 *   1. Header — bar number + the audio-time window the snippet spans.
 *   2. Waveform canvas (drawn from the first loaded audio track) with
 *      overlay vertical lines for each bar boundary inside the window,
 *      the original detected onset, and the final-position landing
 *      (post-quantization, post-alignment).
 *   3. One or more diff rows: each row's bar starts at the end of the
 *      previous row's bar (the detected onset for row 1) and extends
 *      by that stage's delta in seconds. Bar colour encodes the stage;
 *      label inside reads stage + delta in beats and ms.
 *
 * Skips the waveform row when no audio is loaded so the panel still
 * conveys the deltas; if no rendering context (filtered onsets) is
 * provided the parent suppresses this component entirely.
 */
const OnsetTimingVisualization = observer(({
  entry,
  rendered,
  snapSec,
  snapBeats,
  alignmentBeats,
  finalSec,
  displayedBarIndex,
}: {
  entry: NoteProvenanceEntry;
  rendered: RenderedNoteContext;
  /** Snap delta in audio-time seconds (post-quantization minus
   * detected). `undefined` when the bar has no resolvable tempo. */
  snapSec: number | undefined;
  snapBeats: number | undefined;
  alignmentBeats: number | undefined;
  /** Final-position audio time (post-quantization, post-alignment).
   * `undefined` when the playback timeline can't resolve the bar's
   * start. */
  finalSec: number | undefined;
  displayedBarIndex: number | undefined;
}) => {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  const detectedSec = entry.detected_time_sec;
  const quantizedSec =
    snapSec !== undefined ? detectedSec + snapSec : undefined;
  const alignmentSec = rendered.provenance.beatAlignmentOffsetSec;

  // Window: span the detected, quantized, and final positions plus a
  // small symmetric pad, clamped to a minimum half-window so a zero-
  // delta onset still gets visible waveform context.
  const stagePositions: number[] = [detectedSec];
  if (quantizedSec !== undefined) stagePositions.push(quantizedSec);
  if (finalSec !== undefined) stagePositions.push(finalSec);
  const minPos = Math.min(...stagePositions);
  const maxPos = Math.max(...stagePositions);
  const span = maxPos - minPos;
  const halfWindow = Math.max(TIMING_VIZ_MIN_HALF_WINDOW_SEC, span * 1.2);
  const center = (minPos + maxPos) / 2;
  const windowStart = center - halfWindow;
  const windowEnd = center + halfWindow;
  const windowDur = windowEnd - windowStart;

  const timeToX = (t: number): number =>
    ((t - windowStart) / windowDur) * TIMING_VIZ_WIDTH;

  // Pick the audio track most likely to expose this onset clearly.
  // The debug bundle's manifest carries an authoritative pitch →
  // audio-filename map (set up server-side in
  // `transcriber/app/debug_bundle.py`); the isolated stem for the
  // note's pitch is the right source — it isolates the drum we're
  // inspecting from the rest of the kit. Fallback chain when the
  // mapping doesn't resolve (legacy bundle, manual file load, etc.):
  //   1. Any other mapped stem — still a per-pitch isolated source.
  //   2. Any loaded track other than `no_drums.mp3` — `no_drums` by
  //      definition has the drum content removed and shows nothing.
  //   3. Whatever's loaded.
  // `undefined` when no audio is loaded; the row collapses to a
  // "(no audio loaded)" placeholder in that case.
  const audioTrack = (() => {
    const tracks = Array.from(jotPlayer.audioTracks.values());
    if (tracks.length === 0) return undefined;
    const mapping = rendered.provenance.audioFilenameByPitch;
    const wantedFilename = mapping.get(entry.pitch);
    if (wantedFilename) {
      const exact = tracks.find(
        (t) => t.filename.toLowerCase() === wantedFilename.toLowerCase(),
      );
      if (exact) return exact;
    }
    // Any other mapped per-pitch stem — skip `no_drums`, which is the
    // backing track and won't show drum hits.
    const mappedStemFilenames = new Set(
      Array.from(mapping.entries())
        .filter(([key]) => key !== 'no_drums')
        .map(([, filename]) => filename.toLowerCase()),
    );
    const anyMappedStem = tracks.find((t) =>
      mappedStemFilenames.has(t.filename.toLowerCase()),
    );
    if (anyMappedStem) return anyMappedStem;
    const noDrumsName = mapping.get('no_drums')?.toLowerCase();
    const nonNoDrums = tracks.find(
      (t) => t.filename.toLowerCase() !== noDrumsName,
    );
    if (nonNoDrums) return nonNoDrums;
    return tracks[0];
  })();

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioTrack) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(TIMING_VIZ_WIDTH * dpr));
    canvas.height = Math.max(1, Math.floor(TIMING_VIZ_WAVE_HEIGHT * dpr));
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, TIMING_VIZ_WIDTH, TIMING_VIZ_WAVE_HEIGHT);
    const peaks = computeWindowPeaks(
      audioTrack.buffer,
      windowStart,
      windowDur,
      TIMING_VIZ_WIDTH,
    );
    const mid = TIMING_VIZ_WAVE_HEIGHT / 2;
    const scale = mid * 0.9;
    // Canvas doesn't resolve CSS vars in fillStyle strings; read the
    // waveform's blue off the element's computed style so the snippet
    // matches the main mixer waveform's colour.
    const computed = window.getComputedStyle(canvas);
    const wave = computed.getPropertyValue('--color-pattern-2').trim();
    ctx.fillStyle = wave || '#5BA8E8';
    // Always paint at least a 1 px centerline per column (no skip-zero
    // shortcut) so silent ranges render as a continuous ground line
    // instead of empty gaps — in the long mixer waveform the skip is
    // fine because silent regions fade into the chrome around them,
    // but in the snippet it reads as broken rendering.
    for (let p = 0; p < TIMING_VIZ_WIDTH; p++) {
      const mn = peaks[p * 2];
      const mx = peaks[p * 2 + 1];
      const y0 = mid - mx * scale;
      const y1 = mid - mn * scale;
      ctx.fillRect(p, y0, 1, Math.max(1, y1 - y0));
    }
  }, [audioTrack, windowStart, windowDur]);

  // Bar boundaries visible inside the window. Walk the timeline once;
  // boundaries sit at each bar's audio-time start. We label each one
  // with its rendered bar index so the operator can orient on the
  // snippet without consulting the score above.
  const timeline = jotPlayer.timeline;
  const drumsT0Sec = jotPlayer.drumsT0Sec;
  const barBoundaries: { x: number; label: number | null }[] = [];
  if (timeline.rendered) {
    const renderedBars = timeline.rendered.structure.voices[0]?.bars ?? [];
    for (let i = 0; i < timeline.bars.length; i++) {
      const t = timeline.bars[i]!.startSec + drumsT0Sec;
      if (t >= windowStart && t <= windowEnd) {
        barBoundaries.push({ x: timeToX(t), label: renderedBars[i]?.index ?? null });
      }
    }
  }

  const detectedX = timeToX(detectedSec);
  const finalX = finalSec !== undefined ? timeToX(finalSec) : undefined;

  // Each diff row: a coloured bar from `anchorX` to `endX` (always
  // drawn left-to-right via min/abs), with an inline label. The anchor
  // chains: quantization starts at the detected line; alignment starts
  // where quantization left off. This makes the row stack read as a
  // sequence of transformations, not a parallel set.
  type DiffRow = {
    key: string;
    label: string;
    deltaBeats: number | undefined;
    deltaSec: number;
    anchorX: number;
    endX: number;
    className: string;
  };
  const diffRows: DiffRow[] = [];
  if (snapSec !== undefined && Math.abs(snapSec) > 1e-9) {
    diffRows.push({
      key: 'quant',
      label: 'Quantization',
      deltaBeats: snapBeats,
      deltaSec: snapSec,
      anchorX: detectedX,
      endX: timeToX(detectedSec + snapSec),
      className: styles.timingVizDiffBarQuant,
    });
  }
  if (
    alignmentSec !== null &&
    Math.abs(alignmentSec) > 1e-9 &&
    quantizedSec !== undefined &&
    finalSec !== undefined
  ) {
    const alignDelta = finalSec - quantizedSec;
    diffRows.push({
      key: 'align',
      label: 'Beat alignment',
      deltaBeats: alignmentBeats,
      deltaSec: alignDelta,
      anchorX: timeToX(quantizedSec),
      endX: finalX!,
      className: styles.timingVizDiffBarAlign,
    });
  }

  const renderSignedBeats = (b: number) =>
    `${b >= 0 ? '+' : ''}${b.toFixed(3)} beats`;
  const renderSignedMs = (ms: number) =>
    `${ms >= 0 ? '+' : ''}${ms.toFixed(1)} ms`;

  return (
    <div
      className={styles.timingViz}
      style={{ width: TIMING_VIZ_WIDTH } as React.CSSProperties}
    >
      <div className={styles.timingVizHeader}>
        {displayedBarIndex !== undefined && (
          <span className={styles.timingVizHeaderBar}>
            bar {displayedBarIndex}
          </span>
        )}
        <span className={styles.timingVizHeaderRange}>
          {windowStart.toFixed(3)}s ─ {windowEnd.toFixed(3)}s
        </span>
      </div>
      <div
        className={styles.timingVizWaveformRow}
        style={{ height: TIMING_VIZ_WAVE_HEIGHT } as React.CSSProperties}
      >
        {audioTrack ? (
          <canvas
            ref={canvasRef}
            className={styles.timingVizCanvas}
            style={{
              width: TIMING_VIZ_WIDTH,
              height: TIMING_VIZ_WAVE_HEIGHT,
            }}
          />
        ) : (
          <div className={styles.timingVizNoAudio}>(no audio loaded)</div>
        )}
        {barBoundaries.map((b, i) => (
          <React.Fragment key={`bar-${i}`}>
            <div
              className={styles.timingVizBarLine}
              style={{ left: b.x }}
            />
            {b.label !== null && (
              <div
                className={styles.timingVizBarLabel}
                style={{ left: b.x }}
              >
                {b.label}
              </div>
            )}
          </React.Fragment>
        ))}
        <div
          className={styles.timingVizDetectedLine}
          style={{ left: detectedX }}
          title={`Detected · ${detectedSec.toFixed(3)}s`}
        />
        {finalX !== undefined && (
          <div
            className={styles.timingVizFinalLine}
            style={{ left: finalX }}
            title={`Final · ${finalSec!.toFixed(3)}s`}
          />
        )}
      </div>
      {diffRows.map((row) => {
        const left = Math.min(row.anchorX, row.endX);
        const width = Math.abs(row.endX - row.anchorX);
        const beatsPart =
          row.deltaBeats !== undefined
            ? `· ${renderSignedBeats(row.deltaBeats)} `
            : '';
        const fullText = `${row.label} ${beatsPart}· ${renderSignedMs(row.deltaSec * 1000)}`;
        return (
          <div
            key={row.key}
            className={styles.timingVizDiffRow}
            style={{ height: TIMING_VIZ_ROW_HEIGHT } as React.CSSProperties}
          >
            <div
              className={`${styles.timingVizDiffBar} ${row.className}`}
              style={{ left, width }}
              title={fullText}
            >
              <span className={styles.timingVizDiffLabel}>{fullText}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
});

/**
 * Collapsible "Debug details" block that surfaces a single onset's full
 * provenance (detected time, strength, beat-tracker placement, MIDI
 * quantization, filter decision, MIDI tick, …) inside the selection
 * label. Shared between {@link NoteView} (where it appears under the
 * human-readable description with the post-quantization comparison
 * filled in) and {@link FilteredOnsetView} (where it IS the label —
 * filtered onsets have no rendered counterpart so the rendered/snap
 * rows are hidden).
 *
 * Toggle state is local to this component, so it resets every time the
 * label remounts. Closing the popover (de-selecting / un-hovering the
 * note) collapses the details again next time — acceptable for v1
 * since the block is short.
 */
const NoteProvenanceDetails = observer(({
  entry,
  rendered,
  startOpen = false,
}: {
  entry: NoteProvenanceEntry;
  /** Present when hosted by a kept note; absent for filtered ghosts. */
  rendered?: RenderedNoteContext;
  /** Open by default (used by FilteredOnsetView so the user immediately
   * sees why the onset was rejected — for kept notes the toggle is
   * collapsed by default so it doesn't crowd the basic description). */
  startOpen?: boolean;
}) => {
  const [open, setOpen] = React.useState(startOpen);
  // Stop the container's mousedown handler so clicks on the toggle don't
  // begin a marquee selection (which would clear the surrounding note's
  // selection and immediately unmount this component).
  const stop = (e: React.MouseEvent) => e.stopPropagation();

  // Eager per-bar timings table provided once at JotView. Lets the
  // "Final position" row resolve a bar's absolute jot-time start without
  // waiting for the player's timeline to be built — pre-Play the
  // player's timeline is `EMPTY_TIMELINE`, but the math doesn't actually
  // need any playback state.
  const barTimings = React.useContext(BarTimingsContext);

  // Post-quantization position + snap delta — only meaningful when the
  // panel is hosted by a kept note. The conversion to seconds is
  // best-effort: it only fires when the bar has a resolvable tempo
  // (per-bar override or jot global), otherwise the row stays
  // beats-only.
  let quantizedBeat: number | undefined;
  let snapBeats: number | undefined;
  let snapSec: number | undefined;
  let snapMs: number | undefined;
  let displayedBarIndex: number | undefined;
  let quantizedSec: number | undefined;
  let alignmentBeats: number | undefined;
  let finalSec: number | undefined;
  if (rendered) {
    quantizedBeat = renderedBeatInBar(rendered.note, rendered.bar);
    displayedBarIndex = rendered.bar.index;

    // Final position in audio time = bar's absolute jot-time start (from
    // the BarTimings table) + the note's intra-bar beat offset +
    // `drumsT0Sec` (jot t=0 sits at audio t=drumsT0Sec). Lives in the
    // same coordinate frame as `detected_time_sec`, so the row reads
    // natively next to it and the visualization can plot detected,
    // quantized, and final on the same audio-time axis. Keyed by
    // `bar.index` because the rendering chain shallow-clones bars
    // (PitchRow rewrites `tracks` for its lane), so the original
    // reference is gone.
    const barStartSec = barTimings?.get(rendered.bar.index)?.startSec;
    if (barStartSec !== undefined) {
      const intra = beatsToSecondsInBar(
        quantizedBeat - 1,
        rendered.bar,
        undefined,
      );
      if (intra !== null) finalSec = barStartSec + intra + jotPlayer.drumsT0Sec;
    }

    // Old bundles emit `null` when alignment didn't apply; new bundles
    // always carry a numeric value (0.0 on rejection). Coerce to 0 so
    // the displayed alignment row reads consistently as "+0.000s"
    // either way instead of disappearing on legacy bundles.
    const offsetSec = rendered.provenance.beatAlignmentOffsetSec ?? 0;
    const beats = secondsToBeatsInBar(offsetSec, rendered.bar, undefined);
    if (beats !== null) alignmentBeats = beats;

    // Snap delta is derived from beat math (`quantizedBeat −
    // entry.beat_in_bar` × `sec_per_beat`), independently of
    // `finalSec`'s timeline-based derivation above. Keeping the two
    // sides independent is deliberate: if the transcriber's bpm
    // drifts from the rendered jot's per-bar bpm, the snap-bar's
    // right edge stops landing on the final-position marker, and the
    // visible gap surfaces the bug instead of hiding it behind a
    // formula that ties them together. The alignment row already
    // covers the case where the gap is the global beat-grid offset;
    // anything beyond that is a calculation drift worth seeing.
    snapBeats = quantizedBeat - entry.beat_in_bar;
    const snapInSec = beatsToSecondsInBar(snapBeats, rendered.bar, undefined);
    if (snapInSec !== null) {
      snapSec = snapInSec;
      snapMs = snapInSec * 1000;
      quantizedSec = entry.detected_time_sec + snapInSec;
    }
  }

  const renderSignedMs = (ms: number) =>
    `${ms >= 0 ? '+' : ''}${ms.toFixed(1)} ms`;
  const renderSignedBeats = (b: number) =>
    `${b >= 0 ? '+' : ''}${b.toFixed(3)}`;
  const renderSignedSec = (s: number) =>
    `${s >= 0 ? '+' : ''}${s.toFixed(3)}s`;

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
        {open ? '▾' : '▸'} Debug details
      </button>
      {open && rendered && (
        <OnsetTimingVisualization
          entry={entry}
          rendered={rendered}
          snapSec={snapSec}
          snapBeats={snapBeats}
          alignmentBeats={alignmentBeats}
          finalSec={finalSec}
          displayedBarIndex={displayedBarIndex}
        />
      )}
      {open && (
        <dl className={styles.debugDetailsList}>
          <dt>Detected beat</dt>
          <dd>
            {/* entry.bar is 0-indexed in the transcriber's BeatStructure
                where bar 0 = the first audio drum bar; the rendered jot
                anchors bar 1 to drums-t0 (with negative indices for any
                pre-drum lead-in bars), so the two reads natively under
                the same convention once we shift by +1. */}
            bar {entry.bar + 1} · {entry.beat_in_bar.toFixed(3)} ·{' '}
            {entry.detected_time_sec.toFixed(3)}s
          </dd>
          <dt>Strength</dt>
          <dd>{entry.strength.toFixed(3)}</dd>
          {quantizedBeat !== undefined && (
            <>
              <dt>Quantized to</dt>
              <dd>
                bar {displayedBarIndex} · {quantizedBeat.toFixed(3)}
                {quantizedSec !== undefined &&
                  ` · ${quantizedSec.toFixed(3)}s`}
              </dd>
              <dt>Snap delta</dt>
              <dd>
                {renderSignedBeats(snapBeats!)} beats
                {snapMs !== undefined && ` (${renderSignedMs(snapMs)})`}
              </dd>
            </>
          )}
          {rendered && (
            <>
              <dt>Global beat alignment</dt>
              <dd>
                {alignmentBeats !== undefined &&
                  `${renderSignedBeats(alignmentBeats)} beats `}
                ({renderSignedSec(
                  rendered.provenance.beatAlignmentOffsetSec ?? 0,
                )})
              </dd>
            </>
          )}
          {finalSec !== undefined && quantizedBeat !== undefined && (
            <>
              <dt>Final position</dt>
              <dd>
                bar {displayedBarIndex} · {quantizedBeat.toFixed(3)} ·{' '}
                {finalSec.toFixed(3)}s
              </dd>
            </>
          )}
          {entry.midi_note !== null && (
            <>
              <dt>MIDI note</dt>
              <dd>{entry.midi_note}</dd>
            </>
          )}
          {entry.tick !== null && (
            <>
              <dt>MIDI tick</dt>
              <dd>{entry.tick}</dd>
            </>
          )}
          {(() => {
            // Raw MIDI velocity is preserved on `note.metadata.midi.velocity`
            // by from_midi.ts; the same file's [A7] step maps it into the
            // `:a` (≥100) / `:g` (<40) modifiers visible in the description
            // above. Surface the raw value so the operator can see exactly
            // why a note picked up (or didn't) the accent/ghost decoration.
            const velocity = (
              rendered?.note.source.metadata as
                | { midi?: { velocity?: number } }
                | undefined
            )?.midi?.velocity;
            if (typeof velocity !== 'number') return null;
            return (
              <>
                <dt>Velocity</dt>
                <dd>{velocity}</dd>
              </>
            );
          })()}
          <dt>Filter</dt>
          <dd>
            {entry.kept
              ? 'kept'
              : entry.rejected_by ?? (entry.out_of_range ? 'out of range' : 'rejected')}
          </dd>
        </dl>
      )}
    </div>
  );
});

/**
 * Renders one rejected onset as a dashed ghost circle at its detected
 * `(bar, beat_in_bar)` position inside a `PitchRow`'s bars row.
 * Absolutely positioned via the same `--note-pad-px` / `--px-per-beat`
 * CSS vars the real notes use, but with `--filtered-beat` = the
 * onset's cumulative beat offset from the start of the row (lead-in +
 * prior bars + intra-bar offset) so it lands at the right absolute x
 * without needing per-bar ResolvedBar geometry.
 *
 * Click toggles a stuck-open detail popover (independent of the
 * SelectionStore — a filtered onset is not a real note); hover shows
 * the same popover transiently.
 */
export const FilteredOnsetView = ({
  entry,
  beatOffset,
  color,
  trackHeight,
}: {
  entry: NoteProvenanceEntry;
  /** Total beat offset from the start of the bars row (leadInBeats +
   * cumulative bar beats + (beat_in_bar - 1)). The CSS calc derives
   * the pixel `left` from this and the score-root's `--px-per-beat`. */
  beatOffset: number;
  /** Pitch lane colour. Mirrors what the real notes use; falls back to
   * a neutral grey for filtered-only pitches with no rendered notes. */
  color: string;
  trackHeight: number;
}) => {
  const [hovered, setHovered] = React.useState(false);
  const [clicked, setClicked] = React.useState(false);
  const show = hovered || clicked;
  const stop = (e: React.MouseEvent) => e.stopPropagation();
  return (
    <div
      // Same opt-out as real notes so a click on the ghost doesn't move
      // the playhead via the bars-row seek handler.
      data-noseek="true"
      className={classNames(
        styles.filteredOnset,
        show && styles.filteredOnsetShowingLabel,
      )}
      style={
        {
          ['--filtered-beat' as string]: beatOffset,
          top: trackHeight / 2,
          color,
        } as React.CSSProperties
      }
      onMouseDown={stop}
      onClick={(e) => {
        stop(e);
        setClicked((c) => !c);
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={`Filtered onset · pitch ${entry.pitch} · bar ${entry.bar} beat ${entry.beat_in_bar.toFixed(2)}`}
    >
      {show && (
        <div className={styles.filteredOnsetLabel}>
          <NoteProvenanceDetails entry={entry} startOpen />
        </div>
      )}
    </div>
  );
};

function pickBadge(note: StructuralNote): string | undefined {
  const m = note.modifiers;
  if (m.has('c')) return 'C';
  if (m.has('o')) return 'O';
  if (m.has('h')) return 'H';
  if (m.has('f')) return 'F';
  if (m.has('s')) return 'S';
  if (m.has('r')) return 'R';
  if (m.has('z')) return 'Z';
  if (m.has('k')) return 'K';
  if (m.has('m')) return 'M';
  if (m.has('l')) return 'L';
  if (m.has('rf')) return 'Ruff';
  return undefined;
}

/**
 * Human-readable tooltip text for a note. Combines the resolved instrument
 * name with friendly modifier / sticking / roll labels.
 *
 * Examples:
 *   `s:a`       -> "Snare (accented)"
 *   `s:fl@l`    -> "Snare (flam, left hand)"
 *   `h:c`       -> "Hi-Hat (closed)"
 *   `c~_8:o`    -> "Crash (open, roll)"
 */
function describeNote(note: StructuralNote, instrument: Instrument): string {
  const name = instrument.name ?? `Pitch ${note.pitch}`;
  const qualifiers: string[] = [];
  for (const mod of note.modifiers) {
    qualifiers.push(MODIFIER_LABELS[mod] ?? mod);
  }
  if (note.roll) qualifiers.push('roll');
  if (note.sticking) qualifiers.push(STICKING_LABELS[note.sticking]);
  return qualifiers.length > 0 ? `${name} (${qualifiers.join(', ')})` : name;
}

const MODIFIER_LABELS: Partial<Record<Modifier, string>> = {
  a: 'accented',
  g: 'ghost',
  c: 'closed',
  h: 'half-open',
  o: 'open',
  f: 'foot',
  s: 'splash',
  r: 'rim shot',
  x: 'cross-stick',
  z: 'buzz',
  k: 'choke',
  m: 'mute',
  l: 'let ring',
  fl: 'flam',
  dr: 'drag',
  rf: 'ruff',
};

const STICKING_LABELS: Record<Sticking, string> = {
  r: 'right hand',
  l: 'left hand',
  rf: 'right foot',
  lf: 'left foot',
};
