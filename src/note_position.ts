import type { TimeSignature } from 'src/dsl';

/**
 * A symbolic moment inside a Jot, with one internal data representation
 * and many stringification options.
 *
 * The bar index and the float `beatInBar` are the canonical
 * coordinate; the bar's time signature is carried so the 48th-note
 * stringifier can compute the bar's slot count. Audio time and MIDI
 * tick are optional anchors that consumers can attach when they
 * happen to know them (e.g. the detected-onset position has a
 * `detected_time_sec`, the quantized position has both a tick and a
 * second).
 *
 * Stringifiers return `null` when the requested facet wasn't supplied,
 * so callers can `.filter(Boolean).join(' · ')` to build the debug
 * panel's dense single-line readout without conditionals at the call
 * site.
 */

export type NotePositionInput = {
  /** 1-indexed; matches `StructuralBar.index` (anacrusis = 0). */
  barIndex: number;
  /** 1-indexed within the bar, float. Downbeat = 1.0. */
  beatInBar: number;
  /**
   * Slots per quarter-note beat of the producing jot's grid (e.g. 12 for
   * the 1/48 default). Supply via `slotsPerQuarter(jot)` from `src/grid.ts`
   * so the slot readouts track the jot's actual grid density rather than
   * assuming 48.
   */
  slotsPerQuarter: number;
  /**
   * The bar's time signature. Omit when the structural bar isn't
   * resolved (e.g. fallback rendering of a filtered onset whose bar
   * couldn't be looked up); 48th-of-bar stringification then returns
   * `null` rather than guessing a meter.
   */
  timeSig?: TimeSignature;
  /** Absolute audio time in seconds, if known. */
  audioSec?: number;
  /** Absolute MIDI tick, if known. */
  midiTick?: number;
  /** Sub-slot timing offset in ms (the note's `offset`), if any. */
  offsetMs?: number;
};

export class NotePosition {
  readonly barIndex: number;
  readonly beatInBar: number;
  readonly slotsPerQuarter: number;
  readonly timeSig: TimeSignature | undefined;
  readonly audioSec: number | undefined;
  readonly midiTick: number | undefined;
  readonly offsetMs: number | undefined;

  constructor(input: NotePositionInput) {
    this.barIndex = input.barIndex;
    this.beatInBar = input.beatInBar;
    this.slotsPerQuarter = input.slotsPerQuarter;
    this.timeSig = input.timeSig;
    this.audioSec = input.audioSec;
    this.midiTick = input.midiTick;
    this.offsetMs = input.offsetMs;
  }

  /** Total slots in this bar at the jot's grid (e.g. 48 for 4/4, 36 for
   * 3/4 or 6/8 at the default 1/48 grid); `null` when no time signature
   * was supplied. */
  get slotsPerBar(): number | null {
    if (!this.timeSig) return null;
    const { count, unit } = this.timeSig;
    return Math.round((count * this.slotsPerQuarter * 4) / unit);
  }

  /** 1-indexed slot of this position within its bar, or `null` when no
   * time signature was supplied. */
  get slotIndex(): number | null {
    const slots = this.slotsPerBar;
    if (slots === null || !this.timeSig) return null;
    return Math.round((this.beatInBar - 1) * (slots / this.timeSig.count)) + 1;
  }

  /** "bar 3 · 2.500" */
  formatBarBeat(decimals = 3): string {
    return `bar ${this.barIndex} · ${this.beatInBar.toFixed(decimals)}`;
  }

  /** "13/48" (1-indexed; denominator is the bar's 48th-note count).
   * Returns `null` when no time signature was supplied. */
  formatBarBeat48ths(): string | null {
    const slot = this.slotIndex;
    const slots = this.slotsPerBar;
    if (slot === null || slots === null) return null;
    return `${slot}/${slots}`;
  }

  /** "1.234s" or `null` if no audio anchor is attached. */
  formatSeconds(decimals = 3): string | null {
    return this.audioSec === undefined
      ? null
      : `${this.audioSec.toFixed(decimals)}s`;
  }

  /** "480 t" or `null` if no MIDI tick anchor is attached. */
  formatMidiTicks(): string | null {
    return this.midiTick === undefined ? null : `${this.midiTick} t`;
  }

  /** "+12.3 ms" sub-slot offset, or `null` when the note is on its slot. */
  formatOffset(): string | null {
    if (this.offsetMs === undefined) return null;
    return `${this.offsetMs >= 0 ? '+' : ''}${this.offsetMs.toFixed(1)} ms`;
  }

  /**
   * The default dense single-line readout used in the debug details
   * view: bar/beat float, 48ths-of-bar, audio seconds (when present),
   * MIDI tick (when present); joined with " · ".
   */
  toString(): string {
    return [
      this.formatBarBeat(),
      this.formatBarBeat48ths(),
      this.formatSeconds(),
      this.formatMidiTicks(),
      this.formatOffset(),
    ]
      .filter((s): s is string => s !== null)
      .join(' · ');
  }
}
