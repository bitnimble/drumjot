/**
 * Collision-aware allocation of DSL letters for RLRR drum classes that are
 * not present in `CLASS_TO_DRUM`. The same problem and shape as
 * `allocatePitchesForMidi` in src/midi/gm.ts: the converter has to map
 * arbitrary instrument-instance names to single letters without colliding
 * with the canonical kit mapping or with each other.
 *
 * Strategy: deterministic per-song allocation. A hash of the instrument
 * name picks a starting letter (so identical RLRR files map to the same
 * letters across runs); we then walk down the alphabet looking for the
 * first free slot, skipping anything already claimed.
 */
import { CLASS_TO_DRUM, instanceNameToClass } from './drums';

/**
 * Build a per-song `instanceName -> pitch` map. Drum classes already
 * present in `CLASS_TO_DRUM` use their canonical pitch; everything else
 * is allocated a unique fallback letter.
 *
 * Returns a `Map` (rather than a Record) so callers can distinguish "no
 * entry" from "entry with value undefined".
 */
export function allocateFallbackLetters(
  instanceNames: Iterable<string>
): Map<string, string> {
  const out = new Map<string, string>();
  const claimed = new Set<string>();
  const uniqueNames = Array.from(new Set(instanceNames)).sort();

  for (const name of uniqueNames) {
    const cls = instanceNameToClass(name);
    if (cls) {
      const descriptor = CLASS_TO_DRUM[cls];
      if (descriptor) {
        out.set(name, descriptor.pitch);
        claimed.add(descriptor.pitch);
      }
    }
  }
  for (const name of uniqueNames) {
    if (out.has(name)) continue;
    const letter = pickFreeLetter(name, claimed);
    out.set(name, letter);
    claimed.add(letter);
  }
  return out;
}

function pickFreeLetter(seed: string, claimed: ReadonlySet<string>): string {
  let h = 0;
  for (let i = 0; i < seed.length; i++) {
    h = (h * 31 + seed.charCodeAt(i)) | 0;
  }
  const hint = String.fromCharCode('z'.charCodeAt(0) - (Math.abs(h) % 26));
  if (!claimed.has(hint)) return hint;
  for (let i = 25; i >= 0; i--) {
    const c = String.fromCharCode('a'.charCodeAt(0) + i);
    if (!claimed.has(c)) return c;
  }
  // More than 26 distinct unknown drum classes in one song: accept the
  // collision and let later writes overwrite. In practice unreachable.
  return 'z';
}
