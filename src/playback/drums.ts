/**
 * Map GM percussion note numbers onto smplr `DrumMachine` sample group names.
 *
 * smplr ships several drum machines (TR-808, TR-909, ...) whose group names
 * differ between kits (`bass` vs `kick`, `hihat-closed` vs `chh`, etc.), so
 * the mapping is two-stage:
 *
 *   1. Classify the MIDI note into an abstract `DrumRole` (kick / snare /
 *      hihat-closed / ride / ...).
 *   2. Resolve that role against the kit's actual `getGroupNames()` output
 *      using a preference list of common aliases.
 *
 * This keeps role classification (which depends only on GM, a published
 * standard) decoupled from kit-specific naming (which is smplr's affair and
 * may change across versions).
 */

export type DrumRole =
  | 'kick'
  | 'snare'
  | 'rim'
  | 'clap'
  | 'hihat_closed'
  | 'hihat_pedal'
  | 'hihat_open'
  | 'tom_low'
  | 'tom_mid'
  | 'tom_hi'
  | 'crash'
  | 'ride'
  | 'cowbell'
  | 'tambourine';

const MIDI_TO_ROLE: Record<number, DrumRole> = {
  35: 'kick', // Acoustic Bass Drum
  36: 'kick', // Kick
  37: 'rim', // Side Stick / cross-stick
  38: 'snare',
  39: 'clap',
  40: 'snare', // Electric Snare
  41: 'tom_low', // Low Floor Tom
  42: 'hihat_closed',
  43: 'tom_low', // High Floor Tom
  44: 'hihat_pedal',
  45: 'tom_mid', // Low Tom
  46: 'hihat_open',
  47: 'tom_mid', // Low-Mid Tom
  48: 'tom_hi', // Hi-Mid Tom
  49: 'crash',
  50: 'tom_hi', // High Tom
  51: 'ride',
  52: 'crash', // Chinese
  53: 'ride', // Ride Bell
  54: 'tambourine',
  55: 'crash', // Splash
  56: 'cowbell',
  57: 'crash', // Crash 2
  59: 'ride', // Ride 2
};

export function midiNoteToRole(midiNote: number): DrumRole | undefined {
  return MIDI_TO_ROLE[midiNote];
}

// Preference lists are ordered most-specific to most-generic. Each kit's
// own naming style is the leading entry where known (TR-808 uses
// `hihat-close`, `mid-tom`, `rimshot`, etc.; CR-8000 / LM-2 / MFB-512
// follow similar but not identical conventions). Substring matching in
// `resolveGroupForRole` picks up close variants we haven't enumerated.
//
// TR-808 has no distinct `ride`; we fall back to `cymbal` so jots with
// ride hits still produce audible sound on that kit.
const ROLE_PREFERENCES: Record<DrumRole, string[]> = {
  kick: ['kick', 'bass', 'bassdrum', 'bd', 'kk'],
  snare: ['snare', 'sd', 'sn'],
  rim: ['rimshot', 'rim', 'sidestick', 'side-stick', 'rs'],
  clap: ['clap', 'handclap', 'cp'],
  hihat_closed: ['hihat-close', 'hihat-closed', 'closedhat', 'chh', 'hat-closed', 'closed', 'hihat', 'hat', 'ch'],
  hihat_pedal: ['hihat-pedal', 'pedalhat', 'phh', 'hihat-foot', 'foot-hat', 'hihat-close', 'hihat-closed', 'hihat'],
  hihat_open: ['hihat-open', 'openhat', 'ohh', 'hat-open', 'open', 'hihat', 'hat', 'oh'],
  tom_low: ['tom-low', 'low-tom', 'lt', 'tom-3', 'tom3', 'low-conga', 'conga-low', 'tom'],
  tom_mid: ['mid-tom', 'tom-mid', 'mt', 'tom-2', 'tom2', 'mid-conga', 'conga-mid', 'tom'],
  tom_hi: ['tom-hi', 'hi-tom', 'high-tom', 'ht', 'tom-1', 'tom1', 'high-conga', 'conga-hi', 'tom'],
  crash: ['crash', 'cymbal-crash', 'cymbal', 'cy', 'cr'],
  ride: ['ride', 'ride-cymbal', 'rd', 'cymbal'],
  cowbell: ['cowbell', 'bell', 'cb'],
  tambourine: ['tambourine', 'tamb', 'shaker', 'maraca', 'maracas'],
};

/**
 * Find the best matching group name in `availableGroups` for the given role.
 *
 * Strategy:
 *   1. Exact (case-insensitive) match on any alias in the preference list.
 *   2. Substring match in either direction (kit names like `kick-1` should
 *      satisfy a `kick` preference, and a preference `hihat-closed` should
 *      satisfy a kit name `hihat` if nothing more specific is available).
 *
 * Returns `undefined` when no group looks plausible; callers should then
 * skip the event rather than triggering an arbitrary drum.
 */
export function resolveGroupForRole(
  role: DrumRole,
  availableGroups: readonly string[]
): string | undefined {
  if (availableGroups.length === 0) return undefined;
  const lowered = availableGroups.map((g) => g.toLowerCase());
  const prefs = ROLE_PREFERENCES[role];

  for (const pref of prefs) {
    const idx = lowered.indexOf(pref);
    if (idx >= 0) return availableGroups[idx];
  }
  for (const pref of prefs) {
    for (let i = 0; i < lowered.length; i++) {
      if (lowered[i].includes(pref) || pref.includes(lowered[i])) {
        return availableGroups[i];
      }
    }
  }
  return undefined;
}
