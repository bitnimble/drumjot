/**
 * Performance-tier warning: roll (`~`) on a kick drum.
 *
 * Double-pedal players genuinely roll kicks (cf. extreme metal), so this
 * isn't a hard impossibility, but the transcriber's LLM tends to over-emit
 * kick rolls when it can't account for fast 32nd-note patterns. Flag as
 * a warning so refinement can correct false positives without blocking
 * legitimate double-pedal charts.
 */
import { LintDiagnostic } from '../diagnostics';
import { Rule } from '../rule';

export const rollOnKickRule: Rule = {
  id: 'performance/roll-on-kick',
  defaultSeverity: 'warning',
  kind: 'performance',
  description: 'Roll on a kick is rare outside double-pedal contexts.',
  check: (ctx, severity) => {
    const out: LintDiagnostic[] = [];
    for (const note of ctx.notes) {
      if (note.kind !== 'kick') continue;
      if (!note.note.roll) continue;
      out.push({
        ruleId: rollOnKickRule.id,
        severity,
        kind: 'performance',
        message:
          `Roll ('~') on a kick — typically only playable with a double pedal. ` +
          `Confirm the pattern actually rolls, or replace with discrete kick hits.`,
        range: note.range,
        barIndex: note.barIndex,
        layerIndex: note.layerIndex,
      });
    }
    return out;
  },
};
