/**
 * Deterministic recomposition of per-instrument monophonic Drumjot
 * fragments into one Jot.
 *
 * The transcriber makes one LLM call per drum instrument, each emitting
 * a single monophonic line (one pitch letter or `.` per position — no
 * `+`, no `||`, no metadata block). This module merges those lines back
 * into one Jot:
 *
 *   - **Hands** (everything that isn't a foot instrument) form the
 *     first voice. Concurrent hand hits on the same grid slot become an
 *     onset-aligned simultaneity `a+b`. When two hand instruments use
 *     *different* subdivisions in the same bar (genuine polyrhythm —
 *     e.g. 16th hats over triplet ride), the bar is emitted directly as
 *     `+`-joined groups `(<A>) + (<B>)`, which the DSL supports
 *     first-class (see SPEC.md "Notes, rests, sequencing" / example 4).
 *     No LCM rest-grid, no quantization reconciliation.
 *   - **Feet** (kick — the MDX23C separator has no hi-hat-pedal stem)
 *     become the second `||` voice.
 *
 * The merge works in `(bar, beat)` numeric space: each fragment is
 * parsed by the canonical parser and its resolved note positions are
 * recovered, so the subdivision is re-derived deterministically from
 * exact positions rather than by string-splicing bar text. This lives
 * in TypeScript (next to the parser) so DSL logic stays single-sourced;
 * the Python transcriber calls it through `tools/recompose_jot.ts`.
 *
 * The shared bar/beat/tempo/time-signature frame comes from the
 * transcriber's beat tracker and is passed in as `structure`; every
 * per-instrument call was given that same frame, so recomposition is
 * pure slotting against it.
 */
import { RenderedJot } from './jot';
import { parse } from './parser';

export type RecomposeBar = {
  index: number;
  timeSig: [number, number];
  tempoBpm: number;
};

export type RecomposeStructure = {
  initialTempo: number;
  initialTimeSig: [number, number];
  hasTempoChanges: boolean;
  hasTimeSigChanges: boolean;
  bars: RecomposeBar[];
};

export type RecomposeInput = {
  /** pitch letter -> that instrument's monophonic DSL fragment. */
  lines: Record<string, string>;
  structure: RecomposeStructure;
  /** Pitches routed to the second `||` (feet) voice, e.g. `["k"]`. */
  feetPitches: string[];
  /** pitch letter -> display name for the `instrumentMapping`. */
  instrumentNames: Record<string, string>;
};

export type RecomposeResult = {
  dsl: string;
  /** Pitches whose fragment failed to parse and were dropped. */
  dropped: string[];
};

// Subdivisions (slots per quarter-note beat) tried in increasing
// slot-count order so the *coarsest* exact fit wins. Covers binary
// (2/4/8/16), ternary (3/6/12) and their common combination
// (24 = lcm(8,3)). The last entry is the hard cap: a stray onset that
// fits nothing coarser is snapped to the 24-grid.
const CANDIDATE_Q = [1, 2, 3, 4, 6, 8, 12, 16, 24];
const MAX_Q = CANDIDATE_Q[CANDIDATE_Q.length - 1];

// A position fits subdivision `q` when `pos*q` is within this of an
// integer. Parser positions are exact, so the residual is ~1e-8;
// 0.02 is comfortably above the noise and below half a slot at q=24.
const FIT_TOL = 0.02;

type BarOnset = { beat: number; modifiers: string[] };

/** Bar length in quarter notes, matching `ResolvedNote.beat` units. */
function barLengthQuarters(timeSig: [number, number]): number {
  const [count, unit] = timeSig;
  return (count * 4) / unit;
}

function fragmentToBars(
  pitch: string,
  fragment: string,
  numBars: number
): Map<number, BarOnset[]> | null {
  let jot;
  try {
    jot = parse(fragment);
  } catch {
    return null;
  }
  const resolved = new RenderedJot(jot).resolved;
  const byBar = new Map<number, BarOnset[]>();
  for (const voice of resolved.voices) {
    let barIndex = 0;
    for (const bar of voice.bars) {
      if (barIndex < numBars) {
        const track = bar.tracks[pitch];
        if (track) {
          for (const note of track.notes) {
            const list = byBar.get(barIndex) ?? [];
            list.push({
              beat: note.beat,
              modifiers: Array.from(
                note.modifiers as ReadonlySet<string>
              ).sort(),
            });
            byBar.set(barIndex, list);
          }
        }
      }
      barIndex += 1;
    }
  }
  return byBar;
}

/** Does `q` place every beat on a distinct in-range integer slot? */
function fitsQ(beats: number[], q: number, barLen: number): boolean {
  const slots = beats.map((b) => Math.round(b * q));
  const n = Math.round(barLen * q);
  return (
    beats.every((b, i) => Math.abs(b * q - slots[i]) <= FIT_TOL) &&
    new Set(slots).size === slots.length &&
    slots.every((s) => s >= 0 && s < n)
  );
}

/** Coarsest subdivision that fits one pitch's onsets alone. */
function chooseSubdivision(beats: number[], barLen: number): number {
  if (beats.length === 0) return 1;
  for (const q of CANDIDATE_Q) {
    if (fitsQ(beats, q, barLen)) return q;
  }
  return MAX_Q;
}

/**
 * Smallest subdivision on which *every* pitch's onsets land (cross-pitch
 * collisions are allowed — they become `+` chords), or null if none in
 * `CANDIDATE_Q` works for all. This is the minimal common grid; it is
 * always ≥ each pitch's own natural subdivision.
 */
function chooseCommonSubdivision(
  active: [string, BarOnset[]][],
  barLen: number
): number | null {
  for (const q of CANDIDATE_Q) {
    if (active.every(([, os]) => fitsQ(os.map((o) => o.beat), q, barLen))) {
      return q;
    }
  }
  return null;
}

function noteToken(pitch: string, modifiers: string[]): string {
  if (modifiers.length === 0) return pitch;
  return pitch + modifiers.map((m) => `:${m}`).join('');
}

function nearestFree(grid: (string | null)[], slot: number): number | null {
  const n = grid.length;
  for (let d = 1; d < n; d++) {
    for (const cand of [slot - d, slot + d]) {
      if (cand >= 0 && cand < n && grid[cand] === null) return cand;
    }
  }
  return null;
}

function slotTokens(
  onsets: BarOnset[],
  pitch: string,
  q: number,
  barLen: number
): (string | null)[] {
  const n = Math.round(barLen * q);
  const grid: (string | null)[] = new Array(n).fill(null);
  const sorted = [...onsets].sort((a, b) => a.beat - b.beat);
  for (const o of sorted) {
    let slot = Math.round(o.beat * q);
    if (slot < 0) slot = 0;
    if (slot >= n) slot = n - 1;
    if (grid[slot] !== null) {
      const free = nearestFree(grid, slot);
      if (free === null) continue; // grid full; never happens in practice
      slot = free;
    }
    grid[slot] = noteToken(pitch, o.modifiers);
  }
  return grid;
}

function renderBar(
  pitches: string[],
  onsets: Map<string, Map<number, BarOnset[]>>,
  barIndex: number,
  barLen: number
): string {
  const active: [string, BarOnset[]][] = [];
  for (const p of pitches) {
    const bo = onsets.get(p)?.get(barIndex);
    if (bo && bo.length > 0) active.push([p, bo]);
  }
  if (active.length === 0) return '.';

  const perPitchQ = new Map<string, number>();
  for (const [p, os] of active) {
    perPitchQ.set(
      p,
      chooseSubdivision(
        os.map((o) => o.beat),
        barLen
      )
    );
  }
  const maxNatural = Math.max(...perPitchQ.values());
  const commonQ = chooseCommonSubdivision(active, barLen);

  // Merge into one sequence when the instruments are grid-compatible:
  // a common grid exists and it's no finer than the finest instrument
  // already needs (the coarser parts just sit on a subset of slots).
  // Only when the sole common grid is an LCM blow-up finer than every
  // part (genuinely different subdivision families — e.g. straight
  // 8ths vs triplets) do we emit the polyrhythm as `+`-joined groups.
  if (commonQ !== null && commonQ === maxNatural) {
    const q = commonQ;
    const n = Math.round(barLen * q);
    const merged: string[][] = Array.from({ length: n }, () => []);
    for (const [p, os] of active) {
      const grid = slotTokens(os, p, q, barLen);
      grid.forEach((tok, i) => {
        if (tok !== null) merged[i].push(tok);
      });
    }
    return merged
      .map((cell) => (cell.length > 0 ? cell.join('+') : '.'))
      .join(' ');
  }

  // Genuine polyrhythm: distinct subdivisions in the same bar. Emit
  // each pitch's line as its own bar-spanning group, joined with `+`.
  const groups: string[] = [];
  for (const [p, os] of active) {
    const q = perPitchQ.get(p) as number;
    const grid = slotTokens(os, p, q, barLen);
    const seq = grid.map((t) => (t !== null ? t : '.')).join(' ');
    groups.push(`(${seq})`);
  }
  return groups.join(' + ');
}

function renderVoice(
  pitches: string[],
  onsets: Map<string, Map<number, BarOnset[]>>,
  structure: RecomposeStructure,
  inlineMeta: boolean
): string {
  const out: string[] = ['|'];
  let prevSig: [number, number] | null = null;
  let prevBpm: number | null = null;
  for (const bar of structure.bars) {
    const sig = bar.timeSig;
    const bpm = bar.tempoBpm;
    if (inlineMeta && prevSig !== null) {
      if (
        structure.hasTimeSigChanges &&
        (sig[0] !== prevSig[0] || sig[1] !== prevSig[1])
      ) {
        out.push(` {{ time: "${sig[0]}/${sig[1]}" }} |`);
      }
      if (
        structure.hasTempoChanges &&
        prevBpm !== null &&
        Math.abs(bpm - prevBpm) > 1e-6
      ) {
        out.push(` {{ bpm: ${bpm.toFixed(2)} }} |`);
      }
    }
    const content = renderBar(
      pitches,
      onsets,
      bar.index,
      barLengthQuarters(sig)
    );
    out.push(` ${content} |`);
    prevSig = sig;
    prevBpm = bpm;
  }
  return out.join('');
}

function globalBlock(
  structure: RecomposeStructure,
  present: string[],
  names: Record<string, string>
): string {
  const [count, unit] = structure.initialTimeSig;
  const entries = present
    .map((p) => `${p}: { name: "${names[p] ?? p}" }`)
    .join(', ');
  const mapping = entries ? `, instrumentMapping: { ${entries} }` : '';
  return `{{ bpm: ${structure.initialTempo.toFixed(
    2
  )}, time: "${count}/${unit}"${mapping} }}`;
}

export function recompose(input: RecomposeInput): RecomposeResult {
  const { lines, structure, feetPitches, instrumentNames } = input;
  const numBars = structure.bars.length;
  const feet = new Set(feetPitches);
  const dropped: string[] = [];

  if (numBars === 0) {
    return {
      dsl: globalBlock(
        structure,
        Object.keys(lines).sort(),
        instrumentNames
      ),
      dropped,
    };
  }

  const onsets = new Map<string, Map<number, BarOnset[]>>();
  for (const [pitch, fragment] of Object.entries(lines)) {
    const byBar = fragmentToBars(pitch, fragment, numBars);
    if (byBar === null) {
      dropped.push(pitch);
      continue;
    }
    onsets.set(pitch, byBar);
  }

  const present = [...onsets.keys()].sort();
  if (present.length === 0) {
    return { dsl: globalBlock(structure, [], instrumentNames), dropped };
  }

  const handPitches = present.filter((p) => !feet.has(p));
  const footPitches = present.filter((p) => feet.has(p));

  const parts = [globalBlock(structure, present, instrumentNames)];
  if (handPitches.length > 0) {
    parts.push(renderVoice(handPitches, onsets, structure, true));
    if (footPitches.length > 0) {
      parts.push('||');
      parts.push(renderVoice(footPitches, onsets, structure, false));
    }
  } else {
    // Feet-only (only a kick line survived): single voice, no `||`.
    parts.push(renderVoice(footPitches, onsets, structure, true));
  }

  return { dsl: parts.join('\n'), dropped };
}
