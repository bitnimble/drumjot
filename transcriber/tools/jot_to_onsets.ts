#!/usr/bin/env bun
/**
 * Extract per-pitch onset times + predicted velocities from a Drumjot DSL
 * string. Used by the Python refinement pipeline as a bridge to the
 * canonical TypeScript parser, so we don't have to maintain a parallel
 * Python parser.
 *
 * stdin:  Drumjot DSL text
 * stdout: JSON {
 *           "bpm": number,
 *           "timeSignature": { "count": number, "unit": number },
 *           "title": string,
 *           "onsets": {
 *             "k": [{"time": 0.500, "velocity": 80, "modifiers": ["a"]}],
 *             "s": [...]
 *           }
 *         }
 * Parse errors are written to stderr and the process exits with code 2.
 */
import { parse } from 'src/parser';
import { RenderedJot } from 'src/jot';

const VOL_MAP: Record<string, number> = {
  pp: 16,
  p: 33,
  mp: 49,
  mf: 64,
  f: 80,
  ff: 96,
};

function resolveVelocity(metadata: unknown, modifiers: ReadonlySet<string>): number {
  const meta = metadata as
    | {
        midi?: { velocity?: number };
        vol?: string | { start?: string; end?: string };
      }
    | undefined;
  if (typeof meta?.midi?.velocity === 'number') return clamp(Math.round(meta.midi.velocity));
  let baseline = 80;
  const vol = meta?.vol;
  if (typeof vol === 'string') baseline = VOL_MAP[vol] ?? baseline;
  else if (vol && typeof vol === 'object') {
    const v = vol.start ?? vol.end;
    if (typeof v === 'string') baseline = VOL_MAP[v] ?? baseline;
  }
  if (modifiers.has('a')) baseline += 24;
  if (modifiers.has('g')) baseline -= 32;
  return clamp(Math.round(baseline));
}

function clamp(v: number): number {
  if (!Number.isFinite(v)) return 1;
  if (v < 1) return 1;
  if (v > 127) return 127;
  return v;
}

function resolveBpm<T>(raw: unknown, fallback: T): number | T {
  if (typeof raw === 'number' && Number.isFinite(raw) && raw > 0) return raw;
  if (raw && typeof raw === 'object') {
    const obj = raw as { start?: number; end?: number };
    if (typeof obj.start === 'number') return obj.start;
    if (typeof obj.end === 'number') return obj.end;
  }
  return fallback;
}

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

  const rendered = new RenderedJot(jot);
  const resolved = rendered.resolved;

  const globalBpm = resolveBpm(jot.globalMetadata.bpm, 120);

  type OnsetOut = { time: number; velocity: number; modifiers: string[] };
  const onsets: Record<string, OnsetOut[]> = {};

  // Walk voices/bars carrying running "active" bpm so inline {{bpm:...}}
  // changes mid-track affect subsequent bar timings. Per-bar
  // `bar.metadata.bpm` (populated by the parser whenever the active value
  // changed) is the source of truth; otherwise fall back to global.
  for (let vi = 0; vi < resolved.voices.length; vi++) {
    const voice = resolved.voices[vi];
    let activeBpm = globalBpm;
    let barOffsetSeconds = 0;
    for (const bar of voice.bars) {
      const barMetaBpm = resolveBpm((bar.source as { metadata?: { bpm?: unknown } })?.metadata?.bpm, undefined);
      if (barMetaBpm !== undefined) activeBpm = barMetaBpm;
      const secsPerBeat = 60 / activeBpm;
      const barDuration = bar.beats * secsPerBeat;
      for (const pitch of voice.pitches) {
        const track = bar.tracks[pitch];
        if (!track) continue;
        for (const note of track.notes) {
          const t = barOffsetSeconds + note.beat * secsPerBeat;
          const modifierSet = note.modifiers as ReadonlySet<string>;
          (onsets[pitch] ??= []).push({
            time: Number(t.toFixed(4)),
            velocity: resolveVelocity(note.source.metadata, modifierSet),
            modifiers: Array.from(modifierSet),
          });
        }
      }
      barOffsetSeconds += barDuration;
    }
  }

  for (const pitch of Object.keys(onsets)) {
    onsets[pitch].sort((a, b) => a.time - b.time);
  }

  process.stdout.write(
    JSON.stringify({
      bpm: globalBpm,
      timeSignature: jot.globalMetadata.time ?? { count: 4, unit: 4 },
      title: jot.title,
      onsets,
    })
  );
}

main();
