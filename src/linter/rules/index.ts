/**
 * Rule registry. Adding a new rule means importing it here and including it
 * in `ALL_RULES`. Each rule's `id` is its disable/configure handle.
 *
 * Rules are deliberately kept small and isolated (ESLint-style) so that:
 *   - They can be individually disabled via lint config.
 *   - The set is easy to grow over time.
 *   - Each one is independently testable.
 */
import { Rule } from '../rule';
import { discouragedModifierRule } from './discouraged_modifier';
import { invalidModifierRule } from './invalid_modifier';
import { rollOnKickRule } from './roll_on_kick';
import { rollOnMultiInstrumentRule } from './roll_on_multi_instrument';
import { sameHandConflictRule } from './same_hand_conflict';
import { tooManyHandsRule } from './too_many_hands';

export const ALL_RULES: readonly Rule[] = [
  invalidModifierRule,
  discouragedModifierRule,
  rollOnKickRule,
  rollOnMultiInstrumentRule,
  sameHandConflictRule,
  tooManyHandsRule,
];

export {
  discouragedModifierRule,
  invalidModifierRule,
  rollOnKickRule,
  rollOnMultiInstrumentRule,
  sameHandConflictRule,
  tooManyHandsRule,
};
