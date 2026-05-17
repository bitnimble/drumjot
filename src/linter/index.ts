/**
 * Public entry points for the Jot linter.
 *
 * Importing `lint` and the diagnostic types from `src/linter` is the
 * intended surface; rule internals live under `./rules/` and shouldn't
 * be reached for directly.
 */
export { lint, ALL_RULES } from './lint';
export type { LintConfig, Rule } from './lint';
export type {
  LintDiagnostic,
  LintKind,
  LintResult,
  LintSeverity,
} from './diagnostics';
export { hasErrors, countBySeverity } from './diagnostics';
export type { LintContext, ResolvedNote, ResolvedGroup } from './rule';
