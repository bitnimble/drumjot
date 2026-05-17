### Example 1: simple 4/4 rock backbeat (two voices, hands + feet)

```
{{ bpm: 120, time: "4/4",
   instrumentMapping: { h: { name: "HiHat" },
                        s: { name: "Snare" },
                        k: { name: "Kick" } } }}
| h:c h:c h:c+s:a h:c h:c h:c h:c+s:a h:c |
| h:c h:c h:c+s:a h:c h:c h:c h:c+s h:o+s |
||
| k . . . k . . . |
| k k:g . . k . . . |
```

### Example 2: pattern reuse + triplet fill

```
{{ bpm: 110, time: "4/4",
   instrumentMapping: { s: { name: "Snare" }, k: { name: "Kick" } } }}
[Groove=(k.s.kks.)]
| [Groove] | [Groove] | k . s . (k+s k+s k+s)_4 | [Groove] |
```

### Example 3: anacrusis + accents

```
{{ bpm: 96, time: "4/4",
   instrumentMapping: { h: { name: "HiHat" }, s: { name: "Snare" }, k: { name: "Kick" } } }}
k k k | s:a . k . s:a . k . | s:a . k . s:a . k . |
```
