/**
 * Diagnostic data model emitted by linter rules.
 *
 * Severity / kind are kept as small string-literal unions rather than enums
 * so they serialise straight to JSON without extra encoding work — the
 * Python refinement loop consumes these via the bun bridge.
 */
import { SourceRange } from 'src/schema/dsl/dsl';

/** Hard-vs-soft severity. Errors block; warnings nudge. */
export type LintSeverity = 'error' | 'warning';

/**
 * Categorisation of the rule's domain. Language-tier parse errors are NOT
 * represented here — those flow through `ParseError` and bubble out of the
 * parser itself before the linter ever runs. Refactoring suggestions are
 * a separate planned phase and would get their own kind when added.
 */
export type LintKind = 'instrument' | 'performance';

/**
 * A single diagnostic. Carries enough position info for the LLM to either
 * (a) target a specific span via `extractSnippet`, or (b) ingest the
 * `(line, column)` description alongside the full Jot text. `ruleId` is
 * the lint rule's id, so per-rule disable lists and rule-specific prompts
 * are straightforward to build.
 */
export type LintDiagnostic = {
  ruleId: string;
  severity: LintSeverity;
  kind: LintKind;
  message: string;
  /**
   * Source range pointing at the offending element. Undefined when the
   * lint is whole-Jot (e.g. a global metadata problem) or when the
   * underlying AST node was hand-built without a `range`.
   */
  range?: SourceRange;
  /**
   * Jot bar index the offending element sits in. Populated by rules that
   * fire on a specific note / group; allows the refinement loop to look
   * up the audio time range for that bar and pull the corresponding
   * onset candidates back into the prompt. Patterns / anacrusis emit
   * negative bar indices — callers should skip those when mapping to
   * audio time.
   */
  barIndex?: number;
  /** Jot voice index the offending element sits in. */
  voiceIndex?: number;
  /**
   * Optional suggested fix as a free-form string. Surfaced verbatim to
   * the LLM during refinement; not applied automatically.
   */
  suggestedFix?: string;
};

/**
 * Result of running the linter — diagnostics plus the source text they
 * reference. Bundling the two means downstream consumers (the bun bridge,
 * the LSP server, the refinement prompt builder) don't have to thread the
 * source text separately.
 */
export type LintResult = {
  source: string;
  diagnostics: LintDiagnostic[];
};

export function hasErrors(result: LintResult): boolean {
  return result.diagnostics.some((d) => d.severity === 'error');
}

export function countBySeverity(
  result: LintResult,
  severity: LintSeverity
): number {
  let n = 0;
  for (const d of result.diagnostics) if (d.severity === severity) n++;
  return n;
}
