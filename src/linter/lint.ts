/**
 * Public linter entry point.
 *
 * `lint(jot, source)` runs every enabled rule and returns a `LintResult`.
 * Configuration lets you flip individual rules off or change their
 * severity. The default config enables every rule at its declared
 * default severity.
 */
import { Jot } from 'src/dsl';
import { LintDiagnostic, LintResult, LintSeverity } from './diagnostics';
import { Rule, buildLintContext } from './rule';
import { ALL_RULES } from './rules';

/**
 * Per-rule overrides. Missing entries fall back to the rule's declared
 * `defaultSeverity` and `enabled = true`.
 */
export type LintConfig = {
  rules?: Record<string, { enabled?: boolean; severity?: LintSeverity }>;
};

export function lint(
  jot: Jot,
  source: string = '',
  config: LintConfig = {}
): LintResult {
  const ctx = buildLintContext(jot, source);
  const diagnostics: LintDiagnostic[] = [];
  for (const rule of ALL_RULES) {
    const cfg = config.rules?.[rule.id];
    if (cfg?.enabled === false) continue;
    const severity = cfg?.severity ?? rule.defaultSeverity;
    try {
      diagnostics.push(...rule.check(ctx, severity));
    } catch (err) {
      // A rule throwing shouldn't take down the whole linter; emit a
      // self-diagnostic so the failure is visible.
      diagnostics.push({
        ruleId: 'linter/internal-error',
        severity: 'error',
        kind: 'instrument',
        message: `Rule '${rule.id}' threw: ${
          err instanceof Error ? err.message : String(err)
        }`,
      });
    }
  }
  // Stable sort: by source position (when known), then rule id. Diagnostics
  // without ranges sink to the end so the LLM sees the positioned ones first.
  diagnostics.sort((a, b) => {
    const ar = a.range?.start ?? Number.MAX_SAFE_INTEGER;
    const br = b.range?.start ?? Number.MAX_SAFE_INTEGER;
    if (ar !== br) return ar - br;
    return a.ruleId.localeCompare(b.ruleId);
  });
  return { source, diagnostics };
}

// Re-export the rule registry so callers can introspect / build CLI flags.
export { ALL_RULES };
export type { Rule };
