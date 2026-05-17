# MDB Drums — paste files here

MDB Drums is split across two sources:

1. **Annotations**: <https://github.com/CarlSouthall/MDBDrums> (MIT). A
   plain `git clone` works — no auth required.
2. **Audio**: comes from MedleyDB v1
   (<https://medleydb.weebly.com>), which requires registering for
   access. There is no unauthenticated direct-download URL.

After you have both, arrange them like this:

```
benchmarks/datasets/mdb-drums/
├── annotations/
│   ├── class/
│   │   ├── MusicDelta_80sRock_class.txt
│   │   ├── MusicDelta_Beatles_class.txt
│   │   └── ...
│   ├── subclass/        (optional — not used by the 3-class harness)
│   └── beats_and_subdivisions/  (optional)
└── audio/
    ├── MusicDelta_80sRock_MIX.wav
    ├── MusicDelta_Beatles_MIX.wav
    └── ...
```

The loader pairs each `<track>_class.txt` annotation file with
`audio/<track>_MIX.wav`. If the full mix isn't available, it falls
back to `audio/<track>_Drum.wav` (MDB Drums also ships drum-stem-only
mixes) and logs a warning — full-mix evaluation matches N2N's protocol
but stem-only is useful for sanity checks.

Ground truth format: tab-separated `<onsetSec>\t<label>` lines. Labels
used by the 3-class evaluation:

| Class | Annotation labels |
|---|---|
| KD | `KD` |
| SD | `SD` (`SDD`, `SDB` are also folded in if present) |
| HH | `HH`, `OH` (closed + open) |

`TT`, `CY`, `CB`, ... are ignored for the 3-class metric.
