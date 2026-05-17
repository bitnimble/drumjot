/**
 * Instrument-tier warning: a modifier that's technically valid for the
 * note's instrument but uncommon enough that we flag it for review.
 *
 * Currently covers `:o` (open) on crash / china — a real-world hit, but
 * the transcriber's LLM over-emits it when it can't decide between crash
 * and china. Confirming intent is cheap; ignoring a real misclassification
 * is expensive.
 */
import { INSTRUMENT_METADATA, isWarningModifier } from 'src/instruments';
import { LintDiagnostic } from '../diagnostics';
import { Rule } from '../rule';

export const discouragedModifierRule: Rule = {
  id: 'instrument/discouraged-modifier',
  defaultSeverity: 'warning',
  kind: 'instrument',
  description: 'Modifier is valid but uncommon for this instrument.',
  check: (ctx, severity) => {
    const out: LintDiagnostic[] = [];
    for (const note of ctx.notes) {
      if (note.kind === 'custom') continue;
      const meta = INSTRUMENT_METADATA[note.kind];
      for (const mod of note.modifiers) {
        if (!isWarningModifier(note.kind, mod)) continue;
        out.push({
          ruleId: discouragedModifierRule.id,
          severity,
          kind: 'instrument',
          message:
            `':${mod}' on ${meta.label.toLowerCase()} is technically valid ` +
            `but uncommon — confirm this is intentional.`,
          range: note.range,
          barIndex: note.barIndex,
          voiceIndex: note.voiceIndex,
        });
      }
    }
    return out;
  },
};
