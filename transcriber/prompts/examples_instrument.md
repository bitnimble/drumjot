Each example is one instrument's monophonic line: only that pitch
letter and rests, no `+`, no `||`, no metadata block, one bar per
input bar.

### Example 1: steady 1/8 hi-hat, 4/4

Input feel `straight8`, hits on every 1/8:

```
| h h h h h h h h | h h h h h h h h |
```

### Example 2: kick on 1 and 3, with a pickup into bar 2

Coarsest subdivision that fits — quarter-note grid, so 4 elements:

```
| k . k . | k . k k |
```

### Example 3: snare backbeat with ghost notes

Accented snare on beats 2 and 4, quiet ghosts between (tagged `:g`),
1/8 grid:

```
| . . s:a . s:g . s:a . | . . s:a s:g s:a . s:a . |
```

### Example 4: triplet ride feel (free subdivision)

This instrument is in triplets even if other instruments are straight —
transcribe what it played. Each beat is a triplet group:

```
| (d d d) (d d d) (d d d) (d d d) |
```

### Example 5: a bar this instrument doesn't play

A bar with no hits for this instrument is a single rest:

```
| k . k . | . | k . k . |
```
