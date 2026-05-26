# Waveform zoom-rerender delay analysis

Working notes from a diagnosis pass on the "few hundred ms before the
canvas waveform rerenders in high quality after a zoom scroll" symptom.

## TL;DR

The delay is **not** the worker. The worker round-trip is ~0.3 ms per
chunk. The delay comes from the bucket-transition machinery in
`src/jot_view/mixer.tsx`; specifically the `IntersectionObserver`
fire latency on newly-mounted chunks after a bucket boundary is
crossed.

---

## Hot-path WASM ROI (the broader question)

Realistic gain from migrating frontend hot paths to Rust + WASM: 0–15 %
on user-visible perf, weeks of work. Not worth pursuing. Reasoning:

- The only file shaped like a WASM candidate is
  `src/playback/waveform_compute.ts`; tight monomorphic loops over
  `Float32Array`, mono/stereo fast paths, already in a worker. V8's
  JIT compiles this close to native scalar speed.
- Bigger files (`jot.ts` layout, `score.tsx`, `player.ts`) are either
  one-shot per mutation (not per frame), or bound on DOM / AudioContext
  (WASM can't help).
- The boundary cost (wasm-bindgen, marshalling) often eats the gains
  for short calls.

If you ever wanted to try: port `computeWaveformPeaksFromChannels` +
`computeTrackAmpScale`, keep the worker shape, benchmark against a
5-minute stereo track. Don't expand if you don't see ≥3× on a hot run.

---

## Worker round-trip cost (per chunk)

What crosses the wire per `computePeaks` call:

| Direction | Payload | Cost |
|---|---|---|
| Main → worker | `{kind, id, bars: BarSlice[], totalWidthPx, drumsT0Sec, reqId}`; ~300 B, structured-cloned | ~5-15 µs clone + ~30-150 µs queue/pickup |
| Worker → main | `peaks: Float32Array`, 2×widthPx (24 KB at zoom 1×, also ~24 KB after bucket subdivision), **transferred** via `peaks.buffer` | ~5-10 µs pointer swap + ~30-150 µs queue/pickup |
| **Total** | | **~0.1-0.3 ms per chunk** |

p99 stretches to 1-2 ms under main-thread contention. Per zoom tick
with 2-6 visible chunks per track × 1-3 tracks → 4-18 chunks →
**~1-6 ms of worker overhead per frame**. Negligible against the 16.6 ms
budget.

The pipeline is already rAF-coalesced in `mixer.tsx:1451-1457`. Each
`livePxPerBeat` change cancels any pending draw and queues a fresh one,
so at most one worker call per chunk per displayed frame. There is no
time-based debounce.

Canvas paint is the dominant per-frame cost (~3-15 ms): per-column
`fillRect` × `widthPx` × N chunks. If that ever becomes the limit,
`putImageData` of a manually-filled `Uint8ClampedArray` is typically
3-5× faster for dense vertical-line workloads.

---

## What's actually causing the "few hundred ms" delay

Bucket transitions, not worker latency.

### Within a bucket (e.g. 1.0× → 1.9×, both bucket = 2)

`chunkLayout` is memoed on `[jot, zoomBucket]` (`mixer.tsx:1106-1109`).
Same reference → same chunk component identities → existing
`AudioTrackWaveformChunk` instances stay mounted. Their draw effect
re-runs on each `livePxPerBeat` tick, rAF-coalesced. **~16 ms to high
quality. No delay.**

### Across a bucket boundary (e.g. 1.95× → 2.05×, bucket flips 2 → 4)

1. `zoomBucket` flips. `buildChunkLayout` rebuilds with smaller
   jot-time chunks.
2. New chunk keys (`${jotStart}-${jotEnd}` at `mixer.tsx:1190,1194`),
   so every `AudioTrackWaveformChunk` **unmounts**. New ones mount with
   `isVisible=false`, `renderedPxPerBeat=null` (`mixer.tsx:1281-1282`).
   Their canvas is 0×0; blank.
3. The `HOLDOVER_MS = 1500` machinery (`mixer.tsx:1140`) keeps the
   **old** chunks in the DOM behind the new ones. Their bitmaps are
   intact, but their CSS box is `--px-per-beat × beats` at the **new**
   zoom value; so the cached bitmap is stretched, and
   `image-rendering: pixelated` nearest-neighbour upscales it. **This
   is the "low quality" stretch the user sees.**
4. The new chunks then have to:
   - **Wait for `IntersectionObserver` to fire** (`mixer.tsx:1311-1323`).
     IO callbacks are not synchronous with mount. Best case ~16-50 ms;
     during a big DOM churn (many new elements appearing in a scroll
     container that's also resizing) Chrome batches them and **100-300 ms
     to first fire is common**.
   - Run the draw effect; `renderedPxPerBeat === null` branch →
     immediate `void draw()` at `mixer.tsx:1438-1439`.
   - Worker round-trip (~0.3 ms) + canvas paint (~5-15 ms).

The IO delay is the gating factor by an order of magnitude. Even at
zero worker cost, the user-visible delay is dominated by step 4a.

---

## Fixes, in order of impact

1. **Stop gating the initial draw on `IntersectionObserver`.** The
   parent already knows which chunks exist in the current viewport
   range (it just rendered them); trust that for the initial draw and
   only use IO for off-screen un-mounting later. Eliminates the
   50-300 ms IO-fire wait on bucket transitions. Cheapest experiment;
   likely drops the delay to ~30 ms (one rAF + worker + paint).

2. **Stabilise chunk identity across bucket transitions.** The bucket
   scheme exists to dodge the 16 384 px canvas backing-store cap at
   high zoom (`mixer.tsx:1386`). The unmount/remount is what makes
   transitions visible. Two options:
   - Drop bucketing entirely. At zoom 8× with `pxPerBeat = 112` and
     `DPR = 2`, a 30 s @ 120 BPM chunk = ~10 752 backing px; already
     under the cap. May not need buckets at all.
   - Key chunks by a transition-stable anchor (e.g.
     `floor(jotStart / FINEST_CHUNK_SECONDS)`), so the same React
     element survives a bucket flip and only its `widthPx` changes.

3. **Replace IO with scroll-driven `visibleChunks`** computed in the
   parent from `scroller.scrollLeft + clientWidth`. Synchronous, no
   fire latency, no batching surprises. Also skips per-chunk IO
   overhead (an IO per chunk per track adds up with 30+ chunks).

4. **Drop `HOLDOVER_MS` once 1-3 land.** Holdover only exists to mask
   the unmount/remount blank period. Without that gap, it stops being
   useful.

Recommended sequence: start with (1) alone; smallest diff, biggest
win for the observed symptom.

---

## File / line references

- `src/playback/waveform_compute.ts`; pure peak compute
  (`computeWaveformPeaksFromChannels`, `computeWindowPeaksFromChannels`,
  `computeTrackAmpScale`).
- `src/playback/waveform_worker.ts`; worker that owns per-track PCM
  copies and computes peaks. Replies with the `Float32Array` as a
  transferable (line 90, 105).
- `src/playback/waveform_worker_client.ts`; singleton main-thread
  client. Channel arrays are transferred at register time
  (lines 107-117); peak responses are transferable.
- `src/jot_view/waveform_chunks.ts`; `buildChunkLayout`: structural
  per-bar `BarBeat[]` + `WaveformChunk[]` slicing. Zoom-invariant
  output (memo-stable across wheel ticks within a bucket).
- `src/jot_view/mixer.tsx`:
  - `1095-1109`; `zoomBucket` derivation, layout memo.
  - `1140-1172`; `HOLDOVER_MS` holdover machinery.
  - `1190, 1194`; chunk key (`${jotStart}-${jotEnd}`).
  - `1281-1282`; per-chunk initial state (`renderedPxPerBeat=null`,
    `isVisible=false`).
  - `1311-1323`; `IntersectionObserver` setup.
  - `1333-1457`; draw effect; `1438-1439` immediate-draw path,
    `1451-1457` rAF-coalesced path.
  - `1413-1419`, per-column `fillRect` paint loop.
