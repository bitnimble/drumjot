/**
 * Performance-tier error: more than two hand-only notes at a single onset.
 *
 * Humans have two hands. Anything claiming three simultaneous hand strokes
 * (e.g. crash + snare + ride at the same instant) is physically impossible
 * absent stick tricks. Foot instruments (kick, hi-hat with :f / :s) are
 * excluded from the count — a kick + ride + crash + hi-hat-pedal is two
 * hands plus two feet, which is fine.
 */
import { LintDiagnostic } from '../diagnostics';
import { Rule, ResolvedNote } from '../rule';

function isHandStroke(note: ResolvedNote): boolean {
  return note.limbCategory === 'hand' || note.limbCategory === 'either';
}

export const tooManyHandsRule: Rule = {
  id: 'performance/too-many-hands',
  defaultSeverity: 'error',
  kind: 'performance',
  description: 'More than two simultaneous hand-instrument hits are impossible.',
  check: (ctx, severity) => {
    const out: LintDiagnostic[] = [];
    for (const [simulId, notes] of ctx.notesBySimul) {
      const hands = notes.filter(isHandStroke);
      if (hands.length <= 2) continue;
      const lanes = hands.map((n) => `${n.lane}(${n.kind})`).join(', ');
      // Prefer the first note's range for the diagnostic anchor — it's
      // the most useful "click here" target for an editor.
      out.push({
        ruleId: tooManyHandsRule.id,
        severity,
        kind: 'performance',
        message:
          `${hands.length} hand-instrument hits stacked at one onset: ${lanes}. ` +
          `Humans have two hands; drop the least likely hit or move one to a foot ` +
          `instrument (e.g. hi-hat with ':f' / ':s').`,
        range: hands[0].range,
        barIndex: hands[0].barIndex,
        layerIndex: hands[0].layerIndex,
      });
      void simulId;
    }
    return out;
  },
};
