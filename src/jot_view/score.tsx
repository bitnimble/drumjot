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
import { buildTimeline, jotPlayer } from 'src/playback';
import sharedStyles from '../jot_view.module.css';
import {
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
  ({ jot, onSeek }: { jot: RenderedJot; onSeek: (x: number) => void }) => {
    const voice = jot.resolved.voices[0];
    if (!voice || voice.bars.length === 0) return null;

    const liveTimeline = jotPlayer.timeline;
    const timeline =
      liveTimeline.bars.length > 0 && liveTimeline.rendered === jot
        ? liveTimeline
        : buildTimeline(jot);

    return (
      <div className={styles.timelineHeader}>
        <div className={styles.timelineHeaderGutter}>
          <span className={styles.timelineHeaderLabel}>Bar / Time</span>
        </div>
        <div
          className={styles.timelineHeaderBarsRow}
          style={{ width: voice.width }}
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          {voice.bars.map((bar, i) => {
            const timing = timeline.bars[i];
            const timeSec = timing?.startSec ?? 0;
            return (
              <div
                key={i}
                className={styles.timelineHeaderTick}
                style={{ left: bar.x }}
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
  }: {
    bar: StructuralBar;
    pitches: string[];
    config: ViewConfig;
    isAnacrusis: boolean;
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    isPitchAudible: (pitch: string) => boolean;
    /**
     * Whether to draw the bar's pattern + tuplet brackets on top of the
     * lane. In the unified mixer the brackets describe the score
     * structure (not any one instrument), so only the topmost
     * drum-pitch row of each contiguous drum block renders them — every
     * other pitch row sets this `false` to avoid duplicating the same
     * overlay on every lane.
     */
    showBrackets?: boolean;
  }) => {
    const beatCount = bar.time.count;
    // Beat spacing inside the bar, in quarter notes. Each beat divider
    // is `i × beatSpacingBeats` quarter-notes into the bar, scaled to
    // pixels by the score-root's `--px-per-beat`. Stable per bar.
    const beatSpacingBeats = bar.beats / beatCount;
    // Inline style carries only zoom-invariant data so React's prop
    // diff sees no change on a zoom tick: `--bar-beats` is the bar's
    // length in quarter notes, `height` is config-derived.
    const barStyle = {
      ['--bar-beats' as string]: bar.beats,
      height: pitches.length * (config.trackHeight as number),
    } as React.CSSProperties;
    return (
      <div
        className={classNames(styles.bar, isAnacrusis && styles.barAnacrusis)}
        style={barStyle}
        title={`Bar ${bar.index} - ${bar.time.count}/${bar.time.unit}`}
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
        {showBrackets &&
          bar.patternSpans.map((span, i) => (
            <PatternBracket
              key={i}
              span={span}
              highlighted={highlightedPattern === span.name}
              onClick={onPatternClick}
            />
          ))}
        {showBrackets &&
          bar.tupletSpans.map((span, i) => <TupletBracket key={i} span={span} />)}
      </div>
    );
  }
);

const PatternBracket = observer(
  ({
    span,
    highlighted,
    onClick,
  }: {
    span: StructuralPatternSpan;
    highlighted: boolean;
    onClick: (name: string) => void;
  }) => {
    return (
      <div
        className={classNames(
          styles.patternBracket,
          highlighted && styles.patternBracketHighlight
        )}
        style={
          {
            ['--span-start-beat' as string]: span.startBeat,
            ['--span-end-beat' as string]: span.endBeat,
          } as React.CSSProperties
        }
      >
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
const NoteProvenanceDetails = ({
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

  // Post-quantization position + snap delta — only meaningful when the
  // panel is hosted by a kept note. The conversion to seconds is
  // best-effort: it only fires when the bar has a resolvable tempo
  // (per-bar override or jot global), otherwise the row stays
  // beats-only.
  let quantizedBeat: number | undefined;
  let snapBeats: number | undefined;
  let snapMs: number | undefined;
  let displayedBarIndex: number | undefined;
  if (rendered) {
    quantizedBeat = renderedBeatInBar(rendered.note, rendered.bar);
    snapBeats = quantizedBeat - entry.beat_in_bar;
    displayedBarIndex = rendered.bar.index;
    const seconds = beatsToSecondsInBar(
      snapBeats,
      rendered.bar,
      undefined,
    );
    if (seconds !== null) snapMs = seconds * 1000;
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
      {open && (
        <dl className={styles.debugDetailsList}>
          <dt>Detected at</dt>
          <dd>{entry.detected_time_sec.toFixed(3)}s</dd>
          <dt>Strength</dt>
          <dd>{entry.strength.toFixed(3)}</dd>
          <dt>Detected beat</dt>
          <dd>
            bar {entry.bar} · {entry.beat_in_bar.toFixed(3)}
          </dd>
          {rendered && rendered.provenance.beatAlignmentOffsetSec !== null && (
            <>
              <dt>Grid align</dt>
              <dd>
                {renderSignedSec(
                  rendered.provenance.beatAlignmentOffsetSec,
                )}
              </dd>
            </>
          )}
          {quantizedBeat !== undefined && (
            <>
              <dt>Quantized to</dt>
              <dd>
                bar {displayedBarIndex} · {quantizedBeat.toFixed(3)}
              </dd>
              <dt>Snap delta</dt>
              <dd>
                {renderSignedBeats(snapBeats!)} beats
                {snapMs !== undefined && ` (${renderSignedMs(snapMs)})`}
              </dd>
            </>
          )}
          <dt>Backend</dt>
          <dd>{entry.detection_backend}</dd>
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
};

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
