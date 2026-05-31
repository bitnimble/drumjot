# Browser-side lyrics alignment (research)

Status: design exploration. Not scoped, not scheduled. This doc
captures what would be involved in moving the lyrics-alignment
pipeline from the Python service to the browser, so a future
decision is grounded.

Sibling code: `transcriber/app/pipeline/lyrics_align.py` (the
server-side pipeline this would replace) and the upstream vocals
separation in `transcriber/app/pipeline/separate.py`.

## 1. Why consider this

Lyrics alignment is structurally different from the rest of the
transcribe pipeline: the user already has the audio on their machine,
the LLM-cost story doesn't apply (no Anthropic calls in this stage),
and "ship vocals to a server" is the part most users would find
objectionable for licensed music. If browser inference is viable
anywhere in Drumjot today, lyrics alignment is the first candidate
because the privacy/legal win is real and the accuracy ceiling is set
by an off-the-shelf wav2vec2 head we can ship as static weights.

This doc is purely a feasibility / sizing exercise; it does not
recommend doing the port.

## 2. Pipeline stages, with browser-feasibility verdict

Mapped 1:1 from `lyrics_align.py::realign_text`. "Trivial" = no model
download, no GPU, straight TS/WASM port. "Yes" = needs a model file
and WebGPU but maps onto an existing browser ML runtime.

| Stage | Today (server) | Browser path |
|---|---|---|
| Vocals separation (upstream) | BS-Roformer / MDX-class | **Yes**, MDX23C exports today via ORT Web + WebGPU; BS-Roformer needs more op coverage. Long pole, same as elsewhere. |
| `load_audio` + 16 kHz resample | `ctc_forced_aligner.load_audio` (libsoundfile + torchaudio) | **Trivial.** `decodeAudioData` → `OfflineAudioContext` at 16 kHz, or `WebCodecs` + polyphase. |
| `_detect_language_from_text` | Python codepoint counting | **Trivial.** Direct TS port, ~200 LOC. |
| Whisper fallback (text-free input) | faster-whisper | **Yes.** Transformers.js ships Whisper-tiny / -small with WebGPU. ~75 MB at fp16 for tiny. |
| Japanese morphological + romaji (`jp_romaji`) | `fugashi` + `cutlet` + unidic | **Yes**, with substitution. `kuromoji.js` + `kuroshiro` give equivalent kana→romaji. Dictionary is ~12 MB (IPADIC), cacheable in OPFS. |
| uroman text preprocessing | `ctc_forced_aligner.preprocess_text` | **Trivial.** Rule-based tables + rewrite engine; ship as JSON. |
| CTC acoustic model (`load_alignment_model` + `generate_emissions`) | wav2vec2-large-robust (en) or MMS-300m (other) | **Yes.** Best-supported audio model shape in browser ML; see §3. |
| `get_alignments` (CTC Viterbi) | C++ CUDA kernel in `ctc-forced-aligner` | **Trivial.** O(T·N) trellis; ~36M ops for a 3-min song. JS or WASM-SIMD. The package's own `align.py` has a NumPy reference to port from. |
| `get_spans` | Pure logic | **Trivial.** Direct TS port. |
| `_repair_low_score_words` | Re-runs Viterbi on emissions slices | **Trivial.** Same Viterbi as above on a sub-trellis. |
| `postprocess_results` + `_partition_words_by_line` + `_stitch_lines` | Pure Python data manipulation | **Trivial.** Direct TS port. |

**Net:** every stage is physically possible in a browser today. Two
stages are non-trivial: the upstream vocals separator (which is the
same long pole as the rest of Drumjot's transcribe pipeline and may
already be browser-side if the wider browser-pipeline effort happens)
and the acoustic model download (300-600 MB one-time per language
family, depending on quantization).

## 3. The acoustic model, the part that actually matters

Two checkpoints are in play today (see `_pick_alignment_model` in
`lyrics_align.py`):

| Checkpoint | Params | fp32 size | fp16 size | int8 weight-only |
|---|---|---|---|---|
| `facebook/wav2vec2-large-robust-ft-libri-960h` (English) | ~317M | ~1.27 GB | ~634 MB | ~317 MB |
| MMS-300m (`MahmoudAshraf/mms-300m-1130-forced-aligner`, all other languages) | ~315M | ~1.26 GB | (avoid, see §3.2) | ~315 MB |

Both are wav2vec2-family CTC heads and run today in Transformers.js
v3 with the WebGPU backend; the MMS-300m export specifically is the
one Transformers.js ships.

### 3.1 Realistic inference time

For a 3-minute song's vocals stem on a 4070-class GPU:

- Native CUDA fp16: ~2-4 s for `generate_emissions`.
- Browser WebGPU fp16: ~6-15 s.
- CTC Viterbi (`get_alignments` + `get_spans` + repair): <1 s in
  WASM-SIMD on either platform.

The MMS-300m fp32 path is ~2-3× slower in the browser than fp16
because WebGPU FP32 matmul is further from CUDA's tensor-core path
than fp16 is. Mixed precision (conv frontend fp16, transformer encoder
int8) is the realistic ship config.

### 3.2 Int8 quantization, what to expect

**For forced alignment, int8 is much safer than for ASR.** No beam
search; Viterbi against known text. Per-frame log-prob errors rarely
flip which token wins; CTC blank states smooth boundaries further.

- Word-boundary drift vs fp32: **~5-15 ms typical**. Below the
  wav2vec2 80 ms frame stride, so most word boundaries land on the
  same frame.
- ASR WER analog (for context only): 0.5-2 % degradation.

Three int8 variants to distinguish:

| Scheme | What | Notes |
|---|---|---|
| **W8A32 / W8A16** (weight-only) | Weights stored int8, dequantized in-shader. Saves disk + bandwidth, not compute. | Safe default. ORT Web WebGPU EP, Transformers.js. |
| **W8A8** (full int8) | Both weights and activations int8; int8 matmul. Speed win, more accuracy risk. | Needs calibration data; in-browser support uneven. |
| **Mixed precision** | Conv frontend fp16, transformer encoder int8. | Best for wav2vec2, costs ~5 MB to keep conv fp16, removes the most quantization-sensitive risk. |

Specific failure modes to watch:

- **MMS-300m's fp16 LayerNorm overflow** (documented in
  `lyrics_align.py`): an fp16 dynamic-range problem, not a
  precision-in-general problem. Per-tensor int8 scale factors absorb
  the magnitude, so int8 PTQ doesn't reproduce it, provided you
  calibrate on **real vocal-stem audio**, not whatever default
  calibration set the toolchain ships with.
- **Wav2vec2 conv feature extractor** is the most
  quantization-sensitive part (small receptive field, stacked convs).
  Mixed precision keeps the conv fp16. ~5 MB cost, removes most
  remaining accuracy risk.
- **Tonal languages** (zh, vi, th) take a slightly bigger int8 hit
  because per-phoneme distinctions are subtler. Not relevant for the
  English + Japanese-via-romaji mix Drumjot actually sees, but worth
  flagging if non-Latin/CJK use grows.

**Recommended target shapes for ship:**

- wav2vec2-large-robust at W4A16 ("q4" in Transformers.js):
  **~80-100 MB on disk**. Accuracy loss in the noise for forced
  alignment.
- MMS-300m at W8A32: **~315 MB**. No W8A8 risk.
- Both together at aggressive quant: ~400 MB. At safer int8 W8A32
  across the board: ~600 MB.

### 3.3 Compression

`.safetensors` and `.onnx` files are essentially raw weight tensors
with a small header, **zero compression** by default. Weights compress
poorly with generic algorithms because float bit-patterns look
near-random, but the more quantized the weights, the better they
compress:

| Format | gzip savings | zstd-19 savings |
|---|---|---|
| fp16 weights | ~5-10 % | ~10-15 % |
| int8 weights | ~10-20 % | ~15-25 % |
| int4 grouped | ~25-40 % | ~30-50 % |

Browser-available decompressors:

- **`DecompressionStream`** (native) supports gzip + deflate
  everywhere, and **zstd** in Chrome 123+, Firefox 126+, Safari TP.
  Streams the decode, no peak-memory bloat.
- WASM fallbacks: `brotli-wasm`, `@bokuweb/zstd-wasm`, `lz4js`.

HuggingFace serves `.safetensors` with `Content-Encoding: gzip` on
many endpoints; for tighter wire size we'd pre-compress the
checkpoint server-side with zstd-19 and let
`DecompressionStream('deflate-raw')` / `zstd` handle decode.

Realistic best-case wire size for the English head:

- fp16 safetensors: ~634 MB disk, ~580 MB gzipped.
- int8 ONNX: ~317 MB disk, ~270 MB gzipped, ~250 MB zstd-19.
- q4_k_m-equivalent: **~85-100 MB on the wire**, but this requires a
  GGUF-aware loader (llama.cpp WASM), not the standard
  Transformers.js path.

## 4. Storage strategy

Total ship footprint at ship-recommended quant: **~400 MB** for both
acoustic heads + the kuromoji IPADIC dictionary + Whisper-tiny (if we
keep the audio-based language fallback).

### 4.1 Per-origin quota by browser

| Browser | Per-origin quota | Eviction |
|---|---|---|
| Chrome / Edge / Brave | Up to ~60 % of disk; multiple GB on any normal desktop. | LRU under storage pressure unless `persist()` granted. |
| Firefox | ~50 % of disk per origin, ~10 % per eTLD+1 group. | LRU; `persist()` prompts user. |
| Safari (macOS) | ~1 GB initial soft cap, prompts above. | 7-day ITP eviction unless `persist()` and meaningful engagement. |
| Safari (iOS) | ~1-2 GB typical hard ceiling. | Same 7-day eviction. |

Query at runtime via `navigator.storage.estimate()` → `{ quota,
usage }`. 400 MB fits everywhere comfortably on desktop; iOS Safari
is the worst case and the one we'd build the prompt + fallback flow
around.

### 4.2 Where to put the files, **OPFS, not Cache API or IDB**

The Origin Private File System gives us:

- Files that can be opened as `FileSystemSyncAccessHandle` in a worker
  and read directly into a GPU buffer with zero main-thread copies.
- Random access (read a chunk of the weights without
  decompressing the rest).
- No `Request`/`Response` overhead.
- First-class support in Transformers.js v3's caching layer.

Cache API and IDB both force deserialization through `arrayBuffer()`
which copies through main thread and can stall the UI on slow disks.
For ~400 MB of weights that's a real cost.

Storage APIs to avoid for this:

- **`localStorage`**: 5-10 MB, string-only, synchronous. Useless.
- **`sessionStorage`**: same constraints, per-tab. Useless.
- **Cookies**: ~4 KB per cookie. Useless.

### 4.3 Persistence

Call `await navigator.storage.persist()` after the first download.
Chrome auto-grants for engaged sites; Firefox prompts; Safari only
grants after meaningful engagement (multiple visits, home-screen
install, notification permission). Without it, Safari will evict the
checkpoints after 7 days of inactivity and the user re-pays the
download cost. The download flow should treat a missing checkpoint as
"re-fetch quietly," not as an error.

## 5. Architecture sketch (if we built this)

```
┌─────────────────────────────────────────────────────────────────┐
│ Main thread (React)                                             │
│   - Upload audio, kick off alignment                            │
│   - Receive LyricLine[] back, render                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Comlink / postMessage
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Worker (lyrics-align.worker.ts)                                 │
│   1. Vocals separation (WebGPU via ORT Web)        ← optional   │
│   2. Resample to 16 kHz (OfflineAudioContext)                   │
│   3. Language detection (TS port of _detect_language_from_text) │
│   4. Optional: kuromoji + kuroshiro romaji rewrite              │
│   5. uroman preprocess (TS port)                                │
│   6. wav2vec2/MMS emissions (WebGPU via Transformers.js)        │
│   7. CTC Viterbi forced alignment (WASM-SIMD)                   │
│   8. _repair_low_score_words (WASM-SIMD, same Viterbi)          │
│   9. postprocess + stitch (TS port)                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │ OPFS SyncAccessHandle
                               ▼
                  Model weights, kuromoji dict
                  (~400 MB total at ship-quant)
```

The worker boundary matters: forced alignment can take 10-20s on a
3-minute song, and we don't want it blocking the main thread or the
animation loop (per `frame-budget-120hz` constraint). Transferable
ArrayBuffers across the worker boundary handle the audio and
emissions tensors without copies.

## 6. What this doc does *not* establish

- **Whether to build this.** The privacy/legal angle is real but the
  user-visible wins are smaller than the engineering cost suggests
  (one-time 400 MB download is friction; cold-start latency is
  worse than the server path; iOS Safari is a real worst case). A
  decision needs a use-case driver, e.g. "user can drop in a song
  without uploading it" as a public-facing feature, or batch alignment
  of a local library that's too big to upload.
- **Vocals separation.** Treated as upstream-shared with the wider
  browser-pipeline question. If the rest of Drumjot stays
  server-side, lyrics alignment can still run browser-side against
  a server-returned vocals stem, at which point the privacy
  argument weakens (audio still leaves the box). The clean version of
  this design only makes sense if separation also moves browser-side.
- **MMS-300m's CC-BY-NC licence.** Same blocker as on the server path.
  Browser-side doesn't help. Tracked separately in the docstring.

## 7. Open questions for a future decision

1. **Vocals separation co-design.** Is separation moving browser-side
   on its own track? If not, does the privacy story still motivate
   browser-side alignment alone?
2. **Quant strategy.** Ship `q4_k_m` via a GGUF loader (smaller, off
   the Transformers.js path) or W8A32 via Transformers.js (bigger,
   on the supported path)?
3. **Cold-start UX.** First-run is a one-time ~400 MB download
   plus model warm-up. Acceptable as a one-time prompt, or
   needs progressive download / "align on server while warming up
   locally" hybrid?
4. **Whisper for language detection.** Currently the audio-based
   fallback for text-free input. Worth shipping ~75 MB more for an
   edge case, or fall back to "default to English and ask the user"
   if the text-based detector returns `None`?
5. **iOS Safari.** Is it a supported target for this feature or
   gracefully degraded to "upload to server"? The 7-day eviction +
   1 GB cap makes it the hardest browser to ship to.

## 8. Cross-references

- `transcriber/app/pipeline/lyrics_align.py`, the pipeline this would
  port. Read the module docstring + `realign_text` first.
- `transcriber/app/pipeline/jp_romaji.py`, the Japanese morphological
  + romaji pass that would be swapped to kuromoji.js + kuroshiro.
- `transcriber/app/pipeline/separate.py`, the upstream vocals
  separation. Shares the browser-feasibility story with the rest of
  the transcribe pipeline.
- Transformers.js v3 docs (Wav2Vec2ForCTC + WebGPU backend), the
  recommended runtime if we built this.
- `ctc-forced-aligner` (MahmoudAshraf97); the Python package being
  ported; `align.py` in that repo has the CPU CTC Viterbi reference
  to port from.
