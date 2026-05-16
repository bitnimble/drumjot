import { Jot, bar, group, note, patternRef, rest, simul } from 'src/dsl';

/**
 * Simple rock loop, matching SPEC.md example 1:
 *
 * ```
 * {{ bpm: 120, time: "4/4",
 *    mapping: { k:{name:"Kick"}, s:{name:"Snare"}, h:{name:"HiHat"} } }}
 * | h:c h:c h:c h:c h:c h:c h:c h:c |
 * ||
 * | k . s . k . s . |
 * ```
 *
 * Authored as a two-voice jot: the hi-hat eighths run in parallel with the
 * kick/snare backbeat.
 */
export const rockJot: Jot = {
  title: 'Simple rock loop',
  globalMetadata: {
    bpm: 120,
    time: { count: 4, unit: 4 },
    mapping: {
      k: { name: 'Kick', limb: 'rf' },
      s: { name: 'Snare', limb: 'lh' },
      h: { name: 'HiHat', limb: 'rh' },
    },
  },
  voices: [
    {
      bars: [
        bar(
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] })
        ),
        bar(
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['c'] }),
          note('h', { modifiers: ['o'] })
        ),
      ],
    },
    {
      bars: [
        bar(
          note('k'),
          rest(),
          note('s', { modifiers: ['a'] }),
          rest(),
          note('k'),
          rest(),
          note('s', { modifiers: ['a'] }),
          rest()
        ),
        bar(
          note('k'),
          note('k', { modifiers: ['g'] }),
          note('s', { modifiers: ['a'] }),
          rest(),
          note('k'),
          rest(),
          note('s', { sticking: 'l' }),
          note('s', { modifiers: ['fl'], sticking: 'r' })
        ),
      ],
    },
  ],
};

/**
 * Showcase of triplets, simultaneity and a pattern with manipulation.
 *
 * Roughly:
 * ```
 * [Groove=(k.s.kks.)]
 * | [Groove] |
 * | k . s . (k+s k+s k+s)_4 |
 * ```
 */
export const tripletJot: Jot = {
  title: 'Triplet showcase',
  globalMetadata: {
    bpm: 110,
    time: { count: 4, unit: 4 },
    mapping: {
      k: { name: 'Kick', limb: 'rf' },
      s: { name: 'Snare', limb: 'lh' },
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
      ],
    },
  ],
};
