#!/usr/bin/env bun
/**
 * Pretty-print a Drumjot DSL string via the canonical TS formatter
 * (`src/format.ts`). Used by the Python pipeline so every Jot written to
 * disk (`initial.jot` / `final.jot`) and returned in the HTTP response is
 * consistently formatted (one bar per line, pattern defs on their own
 * line, etc.) rather than however `recompose` / the LLM happened to emit
 * it.
 *
 * stdin:  Drumjot DSL text
 * stdout: formatted Drumjot DSL text
 *
 * Parse errors are reported on stderr (with PARSE_ERROR: prefix) and the
 * process exits with code 2 — same convention as `lint_jot.ts` /
 * `jot_to_onsets.ts`. The Python wrapper treats any failure as
 * "pass the original text through unchanged" so a formatter hiccup can
 * never corrupt or drop a transcription.
 */
import { formatJot } from 'src/format';
import { parse } from 'src/parser';

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
  process.stdout.write(formatJot(jot));
}

main();
