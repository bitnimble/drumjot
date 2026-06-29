# RLRR round-trip fixtures

Drop `.rlrr` files into this folder. The harness in `../round_trip.test.ts`
discovers them automatically and runs two checks per fixture:

- `parses without throwing` - JSON-decodes and runs through `parseRlrr`.
- `Jot round trip preserves note count` - `parseRlrr` then `writeRlrr`
  yields the same number of events.

If the folder is empty (or only contains this README), the suite is
skipped and the synthetic baseline tests still run.
