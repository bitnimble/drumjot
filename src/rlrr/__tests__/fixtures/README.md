# RLRR round-trip fixtures

Drop `.rlrr` files into this folder. The harness in `../round_trip.test.ts`
discovers them automatically and runs three checks per fixture:

- `parses without throwing` - JSON-decodes and runs through `rlrrToJot`.
- `Jot round trip preserves note count` - `rlrrToJot` then `jotToRlrr`
  yields the same number of events.
- `MIDI round trip preserves event count` - `rlrrToMidi` then re-parse
  yields the same number of drum note-ons.

If the folder is empty (or only contains this README), the suite is
skipped and the synthetic baseline tests still run.
