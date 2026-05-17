# E-GMD — paste files here

The harness expects the **extracted** contents of `e-gmd-v1.0.0.zip`
(from <https://magenta.tensorflow.org/datasets/e-gmd>) to live directly
in this folder.

After extraction the layout should look like:

```
benchmarks/datasets/e-gmd/
├── e-gmd-v1.0.0.csv
├── drummer1/
│   └── session1/
│       ├── 1_funk-groove1_138_beat_4-4.wav
│       ├── 1_funk-groove1_138_beat_4-4.midi
│       └── ...
├── drummer2/
└── ...
```

The loader keys off `e-gmd-v1.0.0.csv` — specifically the `audio_filename`,
`midi_filename`, and `split` columns. Only rows whose `split` matches
`--split` (default `test`) are evaluated, so the train/validation rows
are ignored even though they sit in the same folder.

Ground truth: the MIDI files use General MIDI percussion. The loader
maps to the 3-class evaluation set as:

| Class | GM pitches |
|---|---|
| KD (kick) | 35, 36 |
| SD (snare) | 37, 38, 40 |
| HH (hi-hat, closed + open) | 42, 44, 46 |

All other percussion (toms, ride, crash, ...) is ignored for the 3-class
metric. This matches the convention N2N and most ADT papers report.

Notes:

- The full archive is ~135 GB (44.1 kHz stereo WAV). If disk is tight,
  drop in just a subset (e.g. one drummer / one session) and pass
  `--limit` / `--sample-ratio` to keep the run small.
- `e-gmd-v1.0.0-midi.zip` (MIDI-only) is **not enough** — the harness
  needs the audio because it benchmarks audio-in, DSL-out.
