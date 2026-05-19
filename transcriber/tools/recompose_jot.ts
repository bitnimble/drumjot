#!/usr/bin/env bun
/**
 * Recompose per-instrument monophonic Drumjot fragments into one Jot.
 *
 * Bridge to the canonical TypeScript recomposition in `src/recompose`,
 * so the merge logic (subdivision detection, `+` chords, polyrhythm
 * groups, `||` feet voice) lives next to the parser and isn't
 * reimplemented in Python.
 *
 * stdin:  JSON {
 *           "lines": { "h": "<dsl>", "k": "<dsl>", ... },
 *           "structure": {
 *             "initialTempo": number,
 *             "initialTimeSig": [count, unit],
 *             "hasTempoChanges": boolean,
 *             "hasTimeSigChanges": boolean,
 *             "bars": [{ "index": n, "timeSig": [c,u], "tempoBpm": n }]
 *           },
 *           "feetPitches": ["k"],
 *           "instrumentNames": { "k": "Kick", ... }
 *         }
 * stdout: JSON { "dsl": "<merged DSL>", "dropped": ["<pitch>", ...] }
 *
 * Malformed input (bad JSON / missing fields) writes to stderr and
 * exits 2, matching jot_to_onsets.ts. Individual fragments that fail to
 * parse are not fatal — they're reported in `dropped`.
 */
import { RecomposeInput, recompose } from 'src/recompose';

async function main() {
  const raw = await Bun.stdin.text();
  let input: RecomposeInput;
  try {
    input = JSON.parse(raw) as RecomposeInput;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    process.stderr.write(`INPUT_ERROR: invalid JSON: ${msg}\n`);
    process.exit(2);
  }

  if (
    !input ||
    typeof input.lines !== 'object' ||
    typeof input.structure !== 'object' ||
    !Array.isArray(input.structure.bars)
  ) {
    process.stderr.write(
      'INPUT_ERROR: expected { lines, structure: { bars }, ... }\n'
    );
    process.exit(2);
  }

  const result = recompose({
    lines: input.lines,
    structure: input.structure,
    feetPitches: input.feetPitches ?? ['k'],
    instrumentNames: input.instrumentNames ?? {},
  });

  process.stdout.write(JSON.stringify(result));
}

main();
