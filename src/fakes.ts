import { Jot, bar, group, note, patternRef, rest, simul } from 'src/dsl';

/** A named example jot shown in the web UI's example picker. */
export type ExampleJot = {
  id: string;
  label: string;
  jot: Jot;
};

/**
 * Simple rock loop, adapted from SPEC.md example 1. Authored as two voices
 * joined by `||`, split by limb group:
 *
 *   - Voice 1 ("Hands"): hi-hat eighths plus snare backbeats.
 *   - Voice 2 ("Feet"):  kick pattern.
 *
 * Roughly:
 * ```
 * | h:c h:c h:c+s:a h:c h:c h:c h:c+s:a s:fl@r |
 * | h:c h:c h:c+s:a h:c h:c h:c h:c+s@l h:o+s@l |
 * ||
 * | k . . . k . . . |
 * | k k:g . . k . . . |
 * ```
 *
 * Bar 1 ends with a snare-flam pickup (the right hand briefly leaves the
 * hi-hat). Bar 2 ends with an open hi-hat + snare on the left hand, an
 * idiomatic lead-in back to bar 1.
 *
 * Within each voice, lane order comes from
 * `globalMetadata.instrumentMapping` declaration order ({ h, s, k }), so
 * the hands staff stacks hi-hat on top of snare and the feet staff renders
 * the kick on its own lane below.
 */
export const rockJot: Jot = {
  title: 'Simple rock loop',
  globalMetadata: {
    bpm: 120,
    time: { count: 4, unit: 4 },
    instrumentMapping: {
      h: { name: 'HiHat', limb: 'rh' },
      s: { name: 'Snare', limb: 'lh' },
      k: { name: 'Kick', limb: 'rf' },
    },
  },
  voices: [
    {
      name: 'Hands',
      bars: [
        bar(
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          simul(note('h', { modifiers: ['c'] }), note('s', { modifiers: ['a'] })),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          simul(note('h', { modifiers: ['c'] }), note('s', { modifiers: ['a'] })),
          // Right hand jumps off the hi-hat for a snare flam pickup into bar 2.
          // Left hand plays the grace stroke immediately before the right.
          note('s', { modifiers: ['fl'], sticking: 'r' })
        ),
        bar(
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          simul(note('h', { modifiers: ['c'] }), note('s', { modifiers: ['a'] })),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          simul(note('h', { modifiers: ['c'] }), note('s', { sticking: 'l' })),
          // Open hi-hat lead-in to bar 1's downbeat. Right hand opens the
          // hat, left hand plays the snare on the and-of-4.
          simul(note('h', { modifiers: ['o'] }), note('s', { sticking: 'l' }))
        ),
      ],
    },
    {
      name: 'Feet',
      bars: [
        bar(
          note('k'),
          rest(),
          rest(),
          rest(),
          note('k'),
          rest(),
          rest(),
          rest()
        ),
        bar(
          note('k'),
          note('k', { modifiers: ['g'] }),
          rest(),
          rest(),
          note('k'),
          rest(),
          rest(),
          rest()
        ),
      ],
    },
  ],
};

/**
 * Showcase of triplets, simultaneity, and pattern reuse.
 *
 * The `Groove` pattern is defined once and replayed across three of the four
 * bars (AABA form: groove, groove, fill, groove). Bar 3 is a half-bar
 * triplet fill, demonstrating how custom bars sit alongside pattern refs.
 *
 * Roughly:
 * ```
 * [Groove=(k.s.kks.)]
 * | [Groove] | [Groove] | k . s . (k+s k+s k+s)_4 | [Groove] |
 * ```
 *
 * Instrument-mapping order puts snare above kick.
 */
export const tripletJot: Jot = {
  title: 'Triplet showcase',
  globalMetadata: {
    bpm: 110,
    time: { count: 4, unit: 4 },
    instrumentMapping: {
      s: { name: 'Snare', limb: 'lh' },
      k: { name: 'Kick', limb: 'rf' },
    },
  },
  patterns: {
    Groove: {
      name: 'Groove',
      silent: false,
      elements: [
        group([
          note('k'),
          rest(),
          note('s'),
          rest(),
          note('k'),
          note('k'),
          note('s'),
          rest(),
        ]),
      ],
    },
  },
  voices: [
    {
      bars: [
        bar(patternRef('Groove')),
        bar(patternRef('Groove')),
        bar(
          note('k'),
          rest(),
          note('s'),
          rest(),
          group(
            [
              simul(note('k'), note('s')),
              simul(note('k'), note('s')),
              simul(note('k'), note('s')),
            ],
            { weight: 4 }
          )
        ),
        bar(patternRef('Groove')),
      ],
    },
  ],
};

/** Registry of example jots offered by the web UI picker. */
export const EXAMPLE_JOTS: readonly ExampleJot[] = [
  { id: 'rock', label: 'Simple rock loop', jot: rockJot },
  { id: 'triplet', label: 'Triplet showcase', jot: tripletJot },
];
