# IDMT-SMT-Drums — paste files here

Available from Fraunhofer IDMT at
<https://www.idmt.fraunhofer.de/en/publications/datasets/drums.html>.
Requires accepting a license agreement on their site; download is a
single archive (`IDMT-SMT-DRUMS-V2.zip` or similar, ~7 GB).

After extracting, the layout should look like:

```
benchmarks/datasets/idmt-smt-drums/
├── audio/
│   ├── RealDrum01_00#HH.wav
│   ├── WaveDrum02_60#MIX.wav
│   └── ...
├── annotation_xml/
│   ├── RealDrum01_00#HH.xml
│   ├── WaveDrum02_60#MIX.xml
│   └── ...
└── (optional) annotation_svl/, audio_drumset_only/, ...
```

The loader pairs each `audio/<id>.wav` with `annotation_xml/<id>.xml`.

The XML format the loader expects:

```xml
<events>
  <event>
    <onsetSec>0.123</onsetSec>
    <instrument>KD</instrument>
  </event>
  ...
</events>
```

Instrument labels used by the 3-class evaluation:

| Class | XML `<instrument>` values |
|---|---|
| KD | `KD` |
| SD | `SD` |
| HH | `HH` (IDMT only annotates closed HH) |

The dataset has three subsets — WaveDrum (synth), RealDrum (acoustic),
TechnoDrum (electronic). The standard convention reports a single
combined score across all three, which is what this harness produces.
Use `--limit` / `--sample-ratio` if you want to scope a run to a
subset (filename prefix filtering can be added if you need it).
