# Third-party licenses & acknowledgements

Drumjot bundles, or downloads at runtime, the third-party components below.
**Drumjot is distributed for non-commercial use.** Several of the ML models it
relies on are released under non-commercial licenses (CC-BY-NC / CC-BY-NC-SA);
those are honoured by Drumjot's non-commercial distribution. Using them in a
**commercial** product would require replacing them or obtaining separate
commercial licenses, see the "Non-commercial" flags below.

How each component reaches the user:
- **bundled**, shipped inside the installer / portable archive.
- **vendored wheel**; built into a wheel and shipped in the installer.
- **downloaded**, fetched from its origin onto the user's machine on first use
  (Drumjot does not redistribute these; the user obtains them from the source).

---

## Permissive (redistributable, incl. commercially)

### Beat-Transformer, beat/downbeat model (bundled code + downloaded checkpoint)
- License: **MIT**. Copyright (c) 2022 Zhao Jingwei.
- Source: https://github.com/zhaojw1998/Beat-Transformer
- The `DilatedTransformer` model code is vendored under `transcriber/app/vendor/beat_transformer/`; the `fold_N` checkpoint is fetched at runtime.

### signalsmith-stretch, time-stretch (bundled, frontend WASM/worklet)
- License: **MIT**. Copyright (c) 2022 Geraint Luff / Signalsmith Audio Ltd.
- Source: https://github.com/Signalsmith-Audio/signalsmith-stretch

### uv, Python package manager (bundled binary)
- License: **MIT OR Apache-2.0** (at your option). Copyright (c) Astral Software Inc.
- Source: https://github.com/astral-sh/uv

### madmom, audio signal processing (vendored wheel)
- **Code**: BSD 2-Clause. Copyright (c) Institute of Computational Perception,
  Johannes Kepler University Linz.
- ⚠️ **madmom's bundled model files are CC-BY-NC-SA 4.0 (non-commercial).** Drumjot
  uses Beat-Transformer for beat tracking (only madmom's *code*, the DBN
  post-processor + DSP utils), not madmom's pretrained models. Verify the
  vendored wheel does not ship `madmom/models/*` if distributing.
- Source: https://github.com/CPJKU/madmom

### ctc-forced-aligner, lyric forced alignment (vendored wheel)
- **Code**: BSD. Source: https://github.com/MahmoudAshraf97/ctc-forced-aligner
- ⚠️ Its **default alignment model is non-commercial**, see below.

### kuromoji.js, Japanese tokenizer (bundled)
- **Code**: Apache-2.0. Source: https://github.com/takuyaa/kuromoji.js
- **Dictionary** (`mecab-ipadic-2.7.0`): NAIST license. Copyright (c) 2000-2003
  Nara Institute of Science and Technology. Redistribution permitted; the NAIST
  copyright + no-warranty notice must be preserved (reproduced below).

---

## Attribution / share-alike (redistributable; obligations attached)

### JMdict / JmdictFurigana, furigana data (bundled)
- License: **CC-BY-SA** (same as JMdict). © James William Breen and the
  Electronic Dictionary Research and Development Group (EDRDG).
- Sources: https://github.com/Doublevil/JmdictFurigana , https://www.edrdg.org/
- Obligations: visible **attribution** (an Acknowledgements screen, Settings →
  About), ship/link the license, no copyright claim over the data, and
  **share-alike** on any modified JMdict-derived data. License:
  https://creativecommons.org/licenses/by-sa/4.0/

---

## Non-commercial (used under Drumjot's non-commercial distribution only)

### ADTOF, drum onset detector (vendored wheel + bundled weights). DEFAULT onsets
- License: **CC-BY-NC-SA 4.0** (the ADTOF project + its models/dataset). The
  PyTorch port (xavriley/ADTOF-pytorch) states no separate license.
- ⚠️ **Non-commercial.** Sources: https://github.com/xavriley/ADTOF-pytorch ,
  https://github.com/MZehren/ADTOF

### MERT, music audio encoder (downloaded); learned onsets (opt-in)
- License: **CC-BY-NC-4.0**. ⚠️ Non-commercial.
- Source: https://huggingface.co/m-a-p/MERT-v1-95M
- Only used when the learned onset backend is explicitly enabled.

### MMS forced-aligner model (downloaded). DEFAULT lyric alignment
- License: **CC-BY-NC-4.0** (Meta MMS). ⚠️ Non-commercial. ctc-forced-aligner
  supports swapping to a permissively-licensed model for commercial use.
- Source: https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner

### Separation models (downloaded). BS-Roformer SW, MDX23C DrumSep, UVR vocals
- License: **not formally stated** (UVR / community weights). Treat as
  research/personal use; redistribution + commercial terms are unclear.
- Sources: https://huggingface.co/jarredou/BS-ROFO-SW-Fixed ,
  https://github.com/jarredou/models ,
  https://github.com/Anjok07/ultimatevocalremovergui

---

## Runtime Python dependencies

The transcribe/separation/lyrics capabilities install their Python stack (torch,
audio-separator, transformers, demucs, onnxruntime, etc.) from PyPI on the user's
machine at install time; Drumjot does not redistribute them. Each is governed by
its own license (predominantly BSD / Apache-2.0 / MIT). The non-commercial
restriction above applies to the specific *model weights*, not these libraries.

---

## License texts

### MIT License
(applies to: Beat-Transformer, signalsmith-stretch, uv [MIT option], and the
permissive frontend dependencies)

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### BSD 2-Clause License
(applies to: madmom [code], ctc-forced-aligner [code])

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES ... ARE DISCLAIMED. (full text per the
upstream LICENSE file.)
```

### Apache License 2.0
(applies to: kuromoji.js [code], uv [Apache option], and Apache-licensed
runtime deps), full text: https://www.apache.org/licenses/LICENSE-2.0

### NAIST license, mecab-ipadic dictionary
```
Copyright (c) 2000-2003 Nara Institute of Science and Technology.
All Rights Reserved.

Use, reproduction, and distribution of this software is permitted. Any copy of
this software, whether in its original form or modified, must include both the
above copyright notice and the following paragraphs.

Nara Institute of Science and Technology (NAIST) disclaims any and all
warranties ... In no event shall NAIST be liable for any special, indirect or
consequential damages ... (full text per the kuromoji.js NOTICE.md).
```

### Creative Commons
- CC-BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/legalcode
- CC-BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/legalcode
- CC-BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
</content>
