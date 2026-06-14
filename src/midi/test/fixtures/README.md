# MIDI round-trip fixtures

Drop `.mid` or `.midi` files into this folder. The test harness in
`../midi.test.ts` will discover them automatically and run a round-trip
check (`fromMidi` -> `toMidi` -> re-parse) for each one.

Each fixture is checked against the following invariants:

- The number of drum note-ons on the drum channel survives the round trip
  (within a small tolerance that accounts for quantization).
- The set of MIDI note numbers used is preserved.
- The first tempo and time signature are preserved.

If the folder is empty (or only contains this README), the fixture suite
is skipped and the synthetic round-trip tests remain.
