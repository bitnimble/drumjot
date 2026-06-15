/**
 * Instrument-tier rule: flag modifiers that aren't valid for the note's
 * instrument kind. E.g. `:o` (open) on a kick, `:r` (rim shot) on a ride.
 *
 * `custom` instruments skip this rule entirely (their valid-modifier set
 * is null = unrestricted) — that's the point of the catch-all kind.
 */
import { INSTRUMENT_METADATA, isValidModifier } from 'src/instruments/instruments';
import { LintDiagnostic } from '../diagnostics';
import { Rule } from '../rule';

export const invalidModifierRule: Rule = {
  id: 'instrument/invalid-modifier',
  defaultSeverity: 'error',
  kind: 'instrument',
  description: "Modifier is not valid for the note's instrument kind.",
  check: (ctx, severity) => {
    const out: LintDiagnostic[] = [];
    for (const note of ctx.notes) {
      if (note.kind === 'custom') continue;
      const meta = INSTRUMENT_METADATA[note.kind];
      for (const mod of note.modifiers) {
        if (isValidModifier(note.kind, mod)) continue;
        out.push({
          ruleId: invalidModifierRule.id,
          severity,
          kind: 'instrument',
          message:
            `':${mod}' is not a valid modifier for ${meta.label.toLowerCase()} ` +
            `(lane '${note.lane}', kind '${note.kind}'). ` +
            `Remove the modifier or change the note's instrument kind.`,
          range: note.range,
          barIndex: note.barIndex,
          layerIndex: note.layerIndex,
          suggestedFix: `Remove ':${mod}' from this note.`,
        });
      }
    }
    return out;
  },
};
