#!/usr/bin/env bun
/**
 * Run the Drumjot linter over a DSL string. Used by the Python refinement
 * loop to surface deterministic instrument/performance diagnostics back to
 * the LLM for targeted fixups.
 *
 * stdin:  Drumjot DSL text
 * stdout: JSON {
 *           "diagnostics": [
 *             {
 *               "ruleId": "instrument/invalid-modifier",
 *               "severity": "error",
 *               "kind": "instrument",
 *               "message": "...",
 *               "range": { "start": 12, "end": 18 },
 *               "line": 3,
 *               "column": 5,
 *               "endLine": 3,
 *               "endColumn": 11,
 *               "snippet": "..."   // optional, when range exists
 *             }
 *           ],
 *           "errors": 1,
 *           "warnings": 0
 *         }
 *
 * Parse errors are reported on stderr (with PARSE_ERROR: prefix) and the
 * process exits with code 2 — same convention as `jot_to_onsets.ts`. That
 * matches the existing Python bridge so callers can reuse the same
 * error-handling path.
 */
import { parse } from 'src/parser';
import { lint } from 'src/linter';
import {
  buildLineIndex,
  extractSnippet,
  lookupOffset,
} from 'src/parser/positions';

async function main() {
  const dsl = await Bun.stdin.text();
  let jot;
  try {
    jot = parse(dsl);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    process.stderr.write(`PARSE_ERROR: ${msg}\n`);
    process.exit(2);
  }

  const result = lint(jot, dsl);
  const idx = buildLineIndex(dsl);

  type Out = {
    ruleId: string;
    severity: 'error' | 'warning';
    kind: 'instrument' | 'performance';
    message: string;
    range?: { start: number; end: number };
    line?: number;
    column?: number;
    endLine?: number;
    endColumn?: number;
    snippet?: string;
    barIndex?: number;
    voiceIndex?: number;
    suggestedFix?: string;
  };

  let errors = 0;
  let warnings = 0;
  const diagnostics: Out[] = result.diagnostics.map((d) => {
    if (d.severity === 'error') errors++;
    else warnings++;
    const out: Out = {
      ruleId: d.ruleId,
      severity: d.severity,
      kind: d.kind,
      message: d.message,
    };
    if (d.range) {
      const start = lookupOffset(idx, d.range.start);
      const end = lookupOffset(idx, d.range.end);
      out.range = d.range;
      out.line = start.line;
      out.column = start.column;
      out.endLine = end.line;
      out.endColumn = end.column;
      out.snippet = extractSnippet(dsl, d.range, 1).snippet;
    }
    if (typeof d.barIndex === 'number') out.barIndex = d.barIndex;
    if (typeof d.voiceIndex === 'number') out.voiceIndex = d.voiceIndex;
    if (d.suggestedFix) out.suggestedFix = d.suggestedFix;
    return out;
  });

  // Per-voice bar ranges so the Python side can slice the DSL into
  // segments for surgical patching. Outer index = voice; inner = bar.
  // Bars without a recorded range (hand-built jots, etc.) emit
  // { start: 0, end: 0 } and callers should treat that as "skip".
  const bars: Array<Array<{ start: number; end: number }>> = jot.voices.map(
    (v) => v.bars.map((b) => b.range ?? { start: 0, end: 0 })
  );

  process.stdout.write(
    JSON.stringify({ diagnostics, errors, warnings, bars }, null, 2)
  );
}

main();
