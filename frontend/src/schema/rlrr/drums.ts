/**
 * Bidirectional mapping between Paradiddle drum classes (`BP_<Class>_C`),
 * Drumjot DSL lanes, and General-MIDI percussion note numbers.
 *
 * Drum class names come from
 * https://github.com/emretanirgan/ParadiddleUtilities/blob/master/PDUtilities/drum_sets/defaultset.rlrr
 * and the upstream `midi_mapping.yaml`. We pick a stable single letter per
 * class to satisfy the DSL constraint that lanes are `a`-`z`.
 *
 * Where a single DSL letter collapses several classes (e.g. all crashes ->
 * `c`), the converter preserves the original drum class on the produced
 * `Note.metadata.rlrr.name` so a round trip is lossless.
 */
import { Modifier } from 'src/schema/dsl/dsl';

export type DrumDescriptor = {
  /** Single-letter lane for the DSL. */
  lane: string;
  /** Modifiers to attach when materialising a note for this class. */
  modifiers?: Modifier[];
  /** Default GM percussion MIDI note for this class. */
  midi: number;
  /** Human-readable name used as `Instrument.name` in the DSL mapping. */
  name: string;
};

export const CLASS_TO_DRUM: Readonly<Record<string, DrumDescriptor>> = {
  BP_HiHat_C: { lane: 'h', modifiers: ['c'], midi: 42, name: 'Hi-Hat' },
  BP_Snare_C: { lane: 's', midi: 38, name: 'Snare' },
  BP_Kick_C: { lane: 'k', midi: 36, name: 'Kick' },
  BP_Crash13_C: { lane: 'c', midi: 49, name: 'Crash 13"' },
  BP_Crash15_C: { lane: 'c', midi: 49, name: 'Crash 15"' },
  BP_Crash17_C: { lane: 'c', midi: 57, name: 'Crash 17"' },
  BP_China15_C: { lane: 'c', midi: 52, name: 'China 15"' },
  BP_FloorTom_C: { lane: 'f', midi: 41, name: 'Floor Tom' },
  BP_Ride17_C: { lane: 'd', midi: 51, name: 'Ride 17"' },
  BP_Ride20_C: { lane: 'd', midi: 59, name: 'Ride 20"' },
  BP_Tom1_C: { lane: 't', midi: 48, name: 'Tom 1' },
  BP_Tom2_C: { lane: 't', midi: 47, name: 'Tom 2' },
  BP_Timpani1_C: { lane: 'i', midi: 47, name: 'Timpani 1' },
  BP_Timpani2_C: { lane: 'i', midi: 48, name: 'Timpani 2' },
  BP_Timpani3_C: { lane: 'i', midi: 50, name: 'Timpani 3' },
  BP_Triangle_C: { lane: 'n', midi: 81, name: 'Triangle' },
  BP_BongoH_C: { lane: 'n', midi: 60, name: 'Bongo (High)' },
  BP_BongoL_C: { lane: 'n', midi: 61, name: 'Bongo (Low)' },
  BP_Xylophone_C: { lane: 'y', midi: 72, name: 'Xylophone' },
  BP_Marimba_C: { lane: 'y', midi: 72, name: 'Marimba' },
  BP_Glockenspiel_C: { lane: 'e', midi: 84, name: 'Glockenspiel' },
  BP_Gong_C: { lane: 'q', midi: 52, name: 'Gong' },
  BP_Tambourine1_C: { lane: 'b', midi: 54, name: 'Tambourine 1' },
  BP_Tambourine2_C: { lane: 'b', midi: 54, name: 'Tambourine 2' },
  BP_Cowbell_C: { lane: 'b', midi: 56, name: 'Cowbell' },
};

/** Default-difficulty MIDI note -> drum class map, ported from midi_mapping.yaml. */
export const DEFAULT_NOTE_TO_CLASS: Readonly<Record<number, string>> = {
  35: 'BP_Kick_C',
  36: 'BP_Kick_C',
  38: 'BP_Snare_C',
  40: 'BP_Snare_C',
  41: 'BP_FloorTom_C',
  42: 'BP_HiHat_C',
  43: 'BP_FloorTom_C',
  46: 'BP_HiHat_C',
  47: 'BP_Tom2_C',
  48: 'BP_Tom1_C',
  49: 'BP_Crash15_C',
  50: 'BP_Tom1_C',
  51: 'BP_Ride17_C',
  53: 'BP_Ride17_C',
  57: 'BP_Crash17_C',
  59: 'BP_Ride20_C',
};

/** Extract the class name from an instrument instance name. */
export function instanceNameToClass(name: string): string | undefined {
  const m = /^(BP_.+_C)_\d+$/.exec(name);
  return m ? m[1] : undefined;
}

/** Look up a drum descriptor for an instrument instance name. */
export function describeDrum(instanceName: string): DrumDescriptor | undefined {
  const cls = instanceNameToClass(instanceName);
  return cls ? CLASS_TO_DRUM[cls] : undefined;
}

/**
 * Inverse of `CLASS_TO_DRUM`: pick a Paradiddle class for a `(lane, modifiers)`
 * combination. Returns `undefined` if there is no reasonable mapping; callers
 * should then drop the note or apply their own fallback.
 *
 * Notes on choices:
 *  - `h:o` (open) maps to a separate "drum class" only via the MIDI note
 *    (46 vs 42); Paradiddle uses the same class so we can't actually
 *    disambiguate at the RLRR layer. The MIDI round-trip uses the `event.midi`
 *    extension to preserve the original note number.
 */
export function laneToClass(lane: string, mods: ReadonlySet<Modifier>): string | undefined {
  switch (lane) {
    case 'k':
      return 'BP_Kick_C';
    case 's':
      return 'BP_Snare_C';
    case 'h':
      return 'BP_HiHat_C';
    case 'c':
      if (mods.has('o')) return 'BP_Crash17_C';
      if (mods.has('h')) return 'BP_Crash13_C';
      return 'BP_Crash15_C';
    case 'd':
      return 'BP_Ride17_C';
    case 't':
      return 'BP_Tom1_C';
    case 'f':
      return 'BP_FloorTom_C';
    case 'b':
      return 'BP_Tambourine1_C';
    case 'n':
      return 'BP_Triangle_C';
    case 'i':
      return 'BP_Timpani1_C';
    case 'y':
      return 'BP_Xylophone_C';
    case 'e':
      return 'BP_Glockenspiel_C';
    case 'q':
      return 'BP_Gong_C';
    default:
      return undefined;
  }
}
