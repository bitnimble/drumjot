/**
 * Performance-tier error: two notes at the same onset assigned to the
 * same explicit hand (or same foot) via `@l` / `@r` / `@lf` / `@rf`.
 *
 * For notes WITHOUT explicit sticking the linter runs a coarse inference
 * (`assignPreferredHands`): per voice, assign each unique hand instrument
 * a "preferred hand" based on overall note count (densest → right hand).
 * Notes without `@` then get the preferred hand of their instrument. Two
 * notes at one onset that resolve to the same hand → error.
 *
 * The inferrer is intentionally simple — it covers the common "hi-hat
 * ostinato + snare backbeat" case correctly without trying to solve the
 * full sticking-assignment problem. Edge cases that the simple model
 * mis-classifies are recoverable by explicit `@l`/`@r` annotation, which
 * the inferrer always honours.
 */
import { Sticking } from 'src/dsl/dsl';
import { DrumInstrumentKind } from 'src/instruments/instruments';
import { LintDiagnostic } from '../diagnostics';
import { Rule, ResolvedNote } from '../rule';

type HandSide = 'l' | 'r';
type FootSide = 'lf' | 'rf';
type LimbResolution =
  | { kind: 'hand'; side: HandSide; explicit: boolean }
  | { kind: 'foot'; side: FootSide; explicit: boolean }
  | { kind: 'unknown' };

function stickingToResolution(stk: Sticking | undefined): LimbResolution | null {
  if (!stk) return null;
  if (stk === 'r' || stk === 'l') return { kind: 'hand', side: stk, explicit: true };
  return { kind: 'foot', side: stk, explicit: true };
}

/**
 * Assign each hand-instrument kind in a voice a "preferred hand" based on
 * count. Densest plays right; sparsest plays left. Ties resolve
 * alphabetically by kind for determinism.
 */
function assignPreferredHands(notes: ResolvedNote[]): Map<DrumInstrumentKind, HandSide> {
  const handCounts = new Map<DrumInstrumentKind, number>();
  for (const n of notes) {
    if (n.limbCategory !== 'hand' && n.limbCategory !== 'either') continue;
    handCounts.set(n.kind, (handCounts.get(n.kind) ?? 0) + 1);
  }
  const sorted = Array.from(handCounts.entries()).sort((a, b) => {
    if (a[1] !== b[1]) return b[1] - a[1]; // descending by count
    return a[0].localeCompare(b[0]);        // tiebreak: alphabetical
  });
  const out = new Map<DrumInstrumentKind, HandSide>();
  if (sorted.length === 0) return out;
  // Densest gets right; second-densest gets left; subsequent kinds
  // share with the existing assignments (rare — typically caught by the
  // too-many-hands rule before this matters).
  out.set(sorted[0][0], 'r');
  if (sorted.length > 1) out.set(sorted[1][0], 'l');
  // Beyond 2 kinds: fall back to right for the rest. Conservative.
  for (let i = 2; i < sorted.length; i++) out.set(sorted[i][0], 'r');
  return out;
}

function resolveLimb(
  note: ResolvedNote,
  preferred: Map<DrumInstrumentKind, HandSide>
): LimbResolution {
  // Explicit sticking always wins.
  const fromStk = stickingToResolution(note.note.sticking);
  if (fromStk) return fromStk;
  // Instrument mapping limb (instrumentMapping[pitch].limb) is the next
  // strongest signal — but it's already only set on the per-pitch
  // mapping, not on individual notes. Look it up via the note's pitch.
  // For now we just consult preferred-hand inference; the mapping limb
  // is consulted via ResolvedNote.limbCategory.
  if (note.limbCategory === 'foot') {
    // Default to right foot (kicks) unless the note is hi-hat (left foot).
    if (note.kind === 'hihat') return { kind: 'foot', side: 'lf', explicit: false };
    return { kind: 'foot', side: 'rf', explicit: false };
  }
  if (note.limbCategory === 'hand') {
    const side = preferred.get(note.kind);
    if (side) return { kind: 'hand', side, explicit: false };
    // Single isolated hand note with no preferred hand inferred — fall back to right.
    return { kind: 'hand', side: 'r', explicit: false };
  }
  return { kind: 'unknown' };
}

export const sameHandConflictRule: Rule = {
  id: 'performance/same-hand-conflict',
  defaultSeverity: 'error',
  kind: 'performance',
  description: 'Two simultaneous notes resolve to the same hand or foot.',
  check: (ctx, severity) => {
    const out: LintDiagnostic[] = [];
    // Assign preferred hands per voice. Voice-scoping matters: ||-split
    // jots typically have hands in one voice and feet in another, and we
    // don't want a global tally to confuse the inferrer.
    const notesByVoice = new Map<number, ResolvedNote[]>();
    for (const n of ctx.notes) {
      const arr = notesByVoice.get(n.voiceIndex);
      if (arr) arr.push(n);
      else notesByVoice.set(n.voiceIndex, [n]);
    }
    const preferredByVoice = new Map<number, Map<DrumInstrumentKind, HandSide>>();
    for (const [v, ns] of notesByVoice) {
      preferredByVoice.set(v, assignPreferredHands(ns));
    }

    for (const [, notes] of ctx.notesBySimul) {
      if (notes.length < 2) continue;
      const seen = new Map<string, ResolvedNote>();
      for (const n of notes) {
        const preferred =
          preferredByVoice.get(n.voiceIndex) ?? new Map<DrumInstrumentKind, HandSide>();
        const res = resolveLimb(n, preferred);
        if (res.kind === 'unknown') continue;
        const key = res.kind === 'hand' ? `hand:${res.side}` : `foot:${res.side}`;
        const prior = seen.get(key);
        if (prior) {
          // Only flag if at least one of the two was explicit, OR both are
          // on the same hand instrument with the inferrer picking the same
          // hand for both (an actually-physical conflict).
          const explicitConflict =
            prior.note.sticking !== undefined || n.note.sticking !== undefined;
          const inferredOnSameKind = prior.kind === n.kind;
          if (!explicitConflict && !inferredOnSameKind) continue;
          out.push({
            ruleId: sameHandConflictRule.id,
            severity,
            kind: 'performance',
            message:
              `Two simultaneous notes assigned to the same ${res.kind} ` +
              `(${res.side}): pitch '${prior.pitch}' and '${n.pitch}'. ` +
              `Change one's sticking (@l / @r) or remove the conflict.`,
            range: n.range,
            barIndex: n.barIndex,
            voiceIndex: n.voiceIndex,
          });
        } else {
          seen.set(key, n);
        }
      }
    }
    return out;
  },
};
