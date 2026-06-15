import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Instrument, Modifier, Sticking } from 'src/schema/dsl/dsl';
import type {
  StructBar,
  StructNote,
  StructPatternSpan,
  StructTupletSpan,
} from 'src/jot_view/structure/structure_store';
import { ViewConfig } from 'src/jot_view/viewport/view_config';
import { msOffsetToBeats } from 'src/schema/dsl/tempo';
import { BarTimingsContext } from '../jot_view_contexts';
import { SelectionContext } from 'src/jot_view/selection/selection';
import { NoteProvenanceContext } from '../provenance/provenance_contexts';
import styles from './score.module.css';
import { NoteProvenanceDetails } from './note_provenance_details';
import { PopoverPortal } from './popover_portal';

// Bars-row rendering: a single bar (BarView) and everything it draws, 
// note glyphs (NoteView + flam/drag grace), pattern/tuplet brackets, and
// the note-description helpers behind the selection-label tooltip. Split
// out of score.tsx; consumed by the mixer’s InstrumentRow.

/**
 * True when `beat` falls inside any tuplet bracket on this bar. The
 * upper bound is inclusive because `endBeat` is now the last slot's
 * onset (see jot.ts); the final tuplet note sits exactly on it and is
 * still covered by the bracket.
 */
function coveredByTuplet(bar: StructBar, beat: number): boolean {
  const eps = 1e-6;
  return bar.tupletSpans.some((s) => beat >= s.startBeat - eps && beat <= s.endBeat + eps);
}

export const BarView = observer(
  ({
    bar,
    barStartBeat,
    pitches,
    config,
    isAnacrusis,
    highlightedPattern,
    onPatternClick,
    isPitchAudible,
    showBrackets = true,
    rowPitch,
    pitchOrder,
    colorForPitch,
    instrumentForPitch,
  }: {
    bar: StructBar;
    /** Cumulative quarter-note position of this bar's left edge within
     *  the voice (sum of `beats` for every bar before this one). Drives
     *  the bar's absolute CSS left; see `.bar` in score.module.css. */
    barStartBeat: number;
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
     * given span; the topmost shows the pattern label and the top edge
     * of the bracket, the bottommost shows the bottom edge, middles show
     * only the left/right sides so the outline reads as one connected
     * box across all participating rows.
     */
    pitchOrder?: readonly string[];
    /**
     * Optional per-pitch colour override. The unified mixer uses this to
     * layer the user's per-instrument-track colour pick on top of the
     * jot's palette default (undefined / empty falls through to a neutral).
     */
    colorForPitch?: (pitch: string) => string | undefined;
    /** Per-pitch instrument (from the global mapping), for the note glyph's
     *  hover/selection tooltip. */
    instrumentForPitch: (pitch: string) => Instrument;
  }) => {
    // Inline style carries only zoom-invariant data so React's prop
    // diff sees no change on a zoom tick: `--bar-start-beat` /
    // `--bar-beats` are the bar's cumulative left edge and length in
    // quarter notes. The bar is absolutely positioned within the bars
    // row (see `.bar` in score.module.css) rather than relying on flex
    // accumulation; that gives each barline a sub-pixel-precise left
    // edge driven directly by `--px-per-beat`, matching the timeline
    // ticks, waveform chunks, and lyric words. `minHeight` is
    // config-derived; the bar stretches top:0/bottom:0 to the bars
    // row's full height so the right-edge barline reaches the row
    // separator (the pitch gutter is taller than a single lane).
    const barStyle = {
      ['--bar-start-beat' as string]: barStartBeat,
      ['--bar-beats' as string]: bar.beats,
      minHeight: pitches.length * (config.trackHeight as number),
    } as React.CSSProperties;
    const isLeadIn = bar.index < 0;
    // `bar.index === -1` is the last lead-in bar (lead-in indices count
    // -leadBars..-1 inclusive). Tag it so its right border draws the
    // dashed lead-in→music boundary; the rest of the lead-in bars have
    // border-right suppressed so they merge into one visual block.
    const isLastLeadIn = bar.index === -1;
    // Each lead-in bar paints a 45° hatch as its own background, so the
    // pattern's phase resets at every row boundary; and the rows don't
    // stack at a fixed multiple of the pattern period (1px row borders,
    // gutter-driven row heights). Walk `offsetParent` up to the document
    // and expose the bar's accumulated top as `--leadin-y`; the CSS
    // shifts `background-position-y` by it so every lead-in's hatch
    // reads as a slice of one global gradient anchored at the document
    // origin, and diagonals run seamlessly through the row separators.
    const barRef = React.useRef<HTMLDivElement>(null);
    React.useLayoutEffect(() => {
      if (!isLeadIn) return;
      const el = barRef.current;
      if (!el) return;
      let y = 0;
      let node: HTMLElement | null = el;
      while (node) {
        y += node.offsetTop;
        node = node.offsetParent as HTMLElement | null;
      }
      el.style.setProperty('--leadin-y', `${y}px`);
    });
    return (
      <div
        ref={isLeadIn ? barRef : undefined}
        className={classNames(
          styles.bar,
          isAnacrusis && styles.barAnacrusis,
          isLeadIn && styles.barLeadIn,
          isLastLeadIn && styles.barLeadInLast
        )}
        style={barStyle}
      >
        {/* Four grid-line overlay layers (main beat / 16th / triplet /
            48th). Always present in the DOM so toggling a grid is a
            single `setProperty` on the score root with zero per-bar
            React work; visibility flows in through
            `display: var(--grid-display-<family>, none)`, which
            `GridLineVars` writes when the corresponding toggle is on
            and removes when it's off. Each layer paints a single
            `background-color` clipped to a dashed/dotted vertical-line
            pattern via two intersecting masks (see `.gridLayer*` in the
            CSS), so we keep the original dashed/dotted look without
            the per-divider DOM spam that made playback laggy at 48ths.
            Skipped on lead-in bars (the hatched intro chrome reads
            cleaner without a grid on top). */}
        {!isLeadIn && (
          <>
            <div className={styles.gridLayerMain} aria-hidden="true" />
            <div className={styles.gridLayerSubBeat16} aria-hidden="true" />
            <div className={styles.gridLayerSubBeatQuarterTriplet} aria-hidden="true" />
            <div className={styles.gridLayerSubBeatTriplet} aria-hidden="true" />
            <div className={styles.gridLayerSubBeat48} aria-hidden="true" />
          </>
        )}
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
                  color={colorForPitch?.(pitch) ?? 'var(--color-text-faint-strong)'}
                  config={config}
                  instrument={instrumentForPitch(pitch)}
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
        {showBrackets && bar.tupletSpans.map((span, i) => <TupletBracket key={i} span={span} />)}
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
  span: StructPatternSpan,
  rowPitch: string | undefined,
  pitchOrder: readonly string[] | undefined
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
    span: StructPatternSpan;
    highlighted: boolean;
    onClick: (name: string) => void;
    position: Exclude<BracketPosition, 'hidden'>;
  }) => {
    const color = PATTERN_COLOR_VARS[span.colorIndex % PATTERN_COLOR_VARS.length];
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
            className={classNames(styles.patternLabel, highlighted && styles.patternLabelHighlight)}
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
const TupletBracket = observer(({ span }: { span: StructTupletSpan }) => (
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
    note: StructNote;
    /**
     * The bar this note lives in. Carried through so the `Debug details`
     * panel can compute the rendered (post-quantization) beat position
     * — `note.beat` is in quarter-notes from bar start, and the bar's
     * time signature is needed to convert into the 1-indexed
     * `beat_in_bar` convention the provenance entry uses.
     */
    bar: StructBar;
    color: string;
    config: ViewConfig;
    instrument: Instrument;
    offGrid: boolean;
  }) => {
    const isAccent = note.modifiers.includes('a');
    const isGhost = note.modifiers.includes('g');
    const isFlam = note.modifiers.includes('fl');
    const isDrag = note.modifiers.includes('dr');
    const isCross = note.modifiers.includes('x');
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
    // Sub-slot timing offset: shift the glyph from its slot to where the
    // note actually plays. The score's x-axis is notational beats, so the
    // ms offset is converted via the bar's local sec-per-beat (from the
    // eager per-bar timings, keyed by the clone-stable `bar.index`).
    const barTimings = React.useContext(BarTimingsContext);
    const offsetMs = note.offsetMs;
    let offsetBeats = 0;
    if (offsetMs !== undefined) {
      const timing = barTimings?.get(bar.index);
      if (timing && bar.beats > 0) {
        offsetBeats = msOffsetToBeats(offsetMs, timing.durationSec / bar.beats);
      }
    }
    const tick = note.midiTick;
    const provenanceEntry =
      provenance && typeof tick === 'number'
        ? provenance.byTick.get(`${note.pitch}:${tick}`)
        : undefined;

    const noteRef = React.useRef<HTMLDivElement>(null);

    return (
      <div
        ref={noteRef}
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
            // this element's style. `offsetBeats` nudges an off-grid note
            // to its true sub-slot position (0 for on-grid notes).
            ['--note-beat' as string]: note.beat + offsetBeats,
            top: (config.trackHeight as number) / 2,
            width: config.noteDiameter,
            height: config.noteDiameter,
            background: isCross ? 'var(--color-bg-base)' : color,
            color,
            borderStyle: isCross ? 'solid' : undefined,
            border: isCross ? `2px solid ${color}` : undefined,
          } as React.CSSProperties
        }
        // Suppress the container's mousedown handler, it begins a
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
        {note.sticking && (
          <span className={styles.stickingBadge}>{note.sticking.toUpperCase()}</span>
        )}
        <PopoverPortal
          anchorRef={noteRef}
          show={showLabel}
          className={styles.noteLabel}
        >
          <div className={styles.noteLabelText}>{description}</div>
          {provenanceEntry && (
            <NoteProvenanceDetails
              entry={provenanceEntry}
              rendered={{ note, bar, provenance: provenance! }}
              startOpen
            />
          )}
        </PopoverPortal>
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

function pickBadge(note: StructNote): string | undefined {
  const m = note.modifiers;
  if (m.includes('c')) return 'C';
  if (m.includes('o')) return 'O';
  if (m.includes('h')) return 'H';
  if (m.includes('f')) return 'F';
  if (m.includes('s')) return 'S';
  if (m.includes('r')) return 'R';
  if (m.includes('z')) return 'Z';
  if (m.includes('k')) return 'K';
  if (m.includes('m')) return 'M';
  if (m.includes('l')) return 'L';
  if (m.includes('rf')) return 'Ruff';
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
function describeNote(note: StructNote, instrument: Instrument): string {
  const name = instrument.name ?? `Pitch ${note.pitch}`;
  const qualifiers: string[] = [];
  for (const mod of note.modifiers) {
    qualifiers.push(MODIFIER_LABELS[mod as Modifier] ?? mod);
  }
  if (note.roll) qualifiers.push('roll');
  if (note.sticking) qualifiers.push(STICKING_LABELS[note.sticking as Sticking]);
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
