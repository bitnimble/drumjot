/**
 * smplr `Storage` that caches fetched sample files in the browser Cache
 * API and (optionally) reports byte-level download progress.
 *
 * smplr ships `CacheStorage`, but it gives no progress hook, so we do the
 * Cache API read/write ourselves and stream the response body. After the
 * first successful load each file is served from the cache on every later
 * session — no network, near-instant.
 *
 * The byte-progress callback (`loaded`/`total` against `Content-Length`)
 * is meant for a single large download — here the ~30 MB GeneralUser GS
 * `.sf2`, which is exactly one `fetch`, so the bar tracks the real wait.
 * A cache hit fires the callback once with `fromCache: true`.
 *
 * Failures here must never break playback: a missing Cache API (insecure
 * origin / old browser) or a `cache.put` quota error degrades to a plain
 * network fetch — the samples still load, we just lose cross-session
 * caching.
 */
import { Storage, StorageResponse } from 'smplr';

export type SampleLoadProgress = {
  /** Bytes received so far. */
  loaded: number;
  /** Total bytes expected from `Content-Length` while streaming; 0 when
   * the server didn't send it. On the *final* progress event the
   * storage layer always sets `total = loaded` so callers can detect
   * "download complete" with `loaded === total && total > 0` even when
   * `Content-Length` was missing mid-stream. */
  total: number;
  /** True when served from the Cache API (no network; effectively instant). */
  fromCache: boolean;
};

export class ProgressCacheStorage implements Storage {
  constructor(
    private readonly cacheName: string,
    private readonly onProgress: (p: SampleLoadProgress) => void,
  ) {}

  async fetch(url: string): Promise<StorageResponse> {
    const cache = await this.openCache();
    if (cache) {
      const hit = await cache.match(url);
      if (hit) {
        // Signal "cache hit" BEFORE reading the body. `arrayBuffer()` on
        // a ~30 MB Cache API entry isn't instant (disk read + copy into
        // an ArrayBuffer can take hundreds of ms to seconds), and during
        // that window the UI would otherwise still say "waiting for
        // server…"; wrong on a cache hit and surprising to the user.
        // The final tick below carries the real byte count.
        this.onProgress({ loaded: 0, total: 0, fromCache: true });
        const buf = await hit.arrayBuffer();
        this.onProgress({ loaded: buf.byteLength, total: buf.byteLength, fromCache: true });
        return bufferResponse(buf, 200);
      }
    }

    const res = await fetch(url);
    // No streamable body (error status, or a runtime without
    // ReadableStream on Response): fall back to a one-shot read so the
    // load still completes; we just can't show incremental progress.
    if (!res.ok || !res.body) {
      const buf = await res.arrayBuffer();
      this.onProgress({ loaded: buf.byteLength, total: buf.byteLength, fromCache: false });
      await this.put(cache, url, buf, res);
      return bufferResponse(buf, res.status);
    }

    const total = Number(res.headers.get('content-length') ?? 0);
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      this.onProgress({ loaded, total, fromCache: false });
    }
    // Final tick with `total = loaded` so the caller can recognise
    // "download complete" even when `Content-Length` was missing — see
    // the `total` doc on `SampleLoadProgress`.
    this.onProgress({ loaded, total: loaded, fromCache: false });

    const buf = concat(chunks, loaded);
    await this.put(cache, url, buf, res);
    return bufferResponse(buf, res.status);
  }

  /** Open the named cache, or `undefined` if the Cache API is unusable
   * here (insecure origin / unsupported) — caller then skips caching. */
  private async openCache(): Promise<Cache | undefined> {
    if (typeof caches === 'undefined') return undefined;
    try {
      return await caches.open(this.cacheName);
    } catch {
      return undefined;
    }
  }

  private async put(
    cache: Cache | undefined,
    url: string,
    buf: ArrayBuffer,
    res: Response,
  ): Promise<void> {
    if (!cache) return;
    try {
      await cache.put(
        url,
        new Response(buf, {
          status: 200,
          headers: {
            'content-type':
              res.headers.get('content-type') ?? 'application/javascript',
          },
        }),
      );
    } catch (err) {
      // Quota exceeded / opaque response / private-mode restrictions:
      // the samples are already in memory for this session, so just log
      // and move on without the persistent cache.
      console.warn('[samples] cache write failed (continuing uncached):', err);
    }
  }
}

function concat(chunks: Uint8Array[], total: number): ArrayBuffer {
  const out = new Uint8Array(new ArrayBuffer(total));
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return out.buffer;
}

/** Wrap raw bytes in the minimal `StorageResponse` smplr consumes. */
function bufferResponse(buf: ArrayBuffer, status: number): StorageResponse {
  return {
    status,
    arrayBuffer: async () => buf,
    text: async () => new TextDecoder().decode(buf),
    json: async () => JSON.parse(new TextDecoder().decode(buf)),
  };
}
