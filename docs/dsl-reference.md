# DSL quick reference

The authoritative grammar is [SPEC.md](../SPEC.md). The most
authoritative *examples* are the parser tests:
[src/parser/__tests__/parser.test.ts](../src/parser/__tests__/parser.test.ts).
This page is the at-a-glance summary.

Single-letter pitches `a`–`z` resolve to instruments via
`globalMetadata.instrumentMapping`. Suffixes attach to a primary element
tightly (no whitespace is canonical, but whitespace is allowed):

| Suffix | Meaning |
|---|---|
| `:mod` | Modifier (a/g/c/h/o/f/s/r/x/z/k/m/l, or multi: fl/dr/rf). Chain `:a:r`. |
| `@stick` | Sticking (r/l/rf/lf). Only on notes. |
| `_N` | Weight (relative duration in a sequence). |
| `*N` | Repeat the element N times in place. |
| `~` | Roll/buzz. |
| `{...}` | Note/group metadata. |

Top-level structure:

```
{{ globalMetadata }}
[Pattern=(...)] (silent definitions; play via [Pattern] references)
| bar1 elements | bar2 elements |
||
| voice 2 bar 1 | voice 2 bar 2 |
```

Fiddly things that matter:

- **Macros vs patterns**: `[$name=...]` is a textual preprocessor
  substitution; `[Name=...]` (no `$`) is a parsed pattern with
  position-aware substitutions (`[Name#3=(x)]`, `[Name#5-8=...]`).
- **Pattern definitions never play at their position.** `[Name=(...)]`
  only declares; the body plays exclusively through `[Name]` references.
  To play at the definition site, write the reference explicitly:
  `[Name=(...)][Name]`. (The older `?`-prefixed silent form is removed.)
- **The `time` key** in metadata is written as a string (`"4/4"`) in DSL
  but normalises to `{ count, unit }` in the AST.
- **Voice == one side of `||`.** Nothing else is a "voice".
- **Per-bar metadata** lives in `Bar.metadata` (parser snapshot logic);
  consumers fall back to `globalMetadata`. If you add a `Metadata` field
  that needs per-bar propagation (beyond `time` + `bpm`), update the
  `BarMeta` type in `src/parser/parser.ts`.
