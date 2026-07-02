/**
 * Performance-tier error: roll (`~`) on a group that spans more than one
 * hand-instrument simultaneously.
 *
 * A roll is a single sustained bounce stream on one drum — a roll across
 * "snare + ride" isn't a thing humans can produce. Foot instruments are
 * excluded because they aren't part of the rolling hand pattern.
 */
import { effectiveLimbCategory } from 'src/instruments/instruments';
import { LintDiagnostic } from '../diagnostics';
import { Rule } from '../rule';

export const rollOnMultiInstrumentRule: Rule = {
  id: 'performance/roll-on-multi-instrument',
  defaultSeverity: 'error',
  kind: 'performance',
  description: 'A roll cannot span multiple hand instruments simultaneously.',
  check: (ctx, severity) => {
    const out: LintDiagnostic[] = [];
    for (const g of ctx.groups) {
      if (!g.group.roll) continue;
      // Use the pre-computed childKinds; filter to hand-only kinds for
      // this rule's specific concern.
      const handKinds = new Set<string>();
      for (const k of g.childKinds) {
        // foot kicks etc. don't participate in the hand-rolling pattern.
        if (k === 'kick') continue;
        // Hi-hat with :f / :s globally on the group is foot — we approximate
        // by treating any hi-hat as a hand instrument here; the modifier-aware
        // limb assignment is left for the dedicated sticking rules.
        if (effectiveLimbCategory(k, new Set()) === 'foot') continue;
        handKinds.add(k);
      }
      if (handKinds.size > 1) {
        out.push({
          ruleId: rollOnMultiInstrumentRule.id,
          severity,
          kind: 'performance',
          message:
            `Roll spans ${handKinds.size} hand instruments (${Array.from(handKinds)
              .sort()
              .join(', ')}). Rolls happen on a single drum at a time — ` +
            `split this into per-instrument rolls or replace with discrete hits.`,
          range: g.range,
          barIndex: g.barIndex,
          layerIndex: g.layerIndex,
        });
      }
    }
    return out;
  },
};
