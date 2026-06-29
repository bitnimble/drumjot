/**
 * Browser dictionary loader for @sglkc/kuromoji.
 *
 * Why not kuromoji's own `builder({ dicPath })`: its BrowserDictionaryLoader
 * fetches each `*.dat.gz` file and *unconditionally* runs `fflate.gunzipSync`
 * on the bytes. That's only correct when the response body is still gzip-
 * compressed - and whether it is depends on the host:
 *
 *   - Vite's static server (sirv, behind both `vite` dev and `vite preview`)
 *     serves any `.gz` file with `Content-Encoding: gzip`, so the browser
 *     transparently inflates the body before JS sees it. kuromoji then
 *     gunzips the already-plain bytes and throws "invalid gzip data"; the
 *     failure is swallowed in furigana.ts and furigana silently never loads.
 *   - Caddy's `file_server` (docker/Caddyfile.frontend, production) serves
 *     the same files with no Content-Encoding, so there the body really is
 *     gzip and kuromoji's gunzip is correct.
 *
 * Two hosts, two truths, and the public API gives us no seam (it accepts
 * only a `dicPath`, never a custom loader or pre-decoded buffers). So we own
 * the fetch + decode: subclass kuromoji's base DictionaryLoader (which still
 * drives the file list + DynamicDictionaries assembly) and override only the
 * per-file byte fetch, gunzipping *only* when the bytes actually start with
 * the gzip magic. That's exactly the prod (raw `.gz`) case and never the dev
 * (already-inflated) case - server-agnostic, no dev-server middleware, no
 * build-time recompression, and it drops the `fflate` dependency in favour
 * of the platform `DecompressionStream`.
 */
import type {
  IpadicFeatures,
  Tokenizer as TokenizerInstance,
} from '@sglkc/kuromoji';
import DictionaryLoader from '@sglkc/kuromoji/src/loader/DictionaryLoader.js';
import Tokenizer from '@sglkc/kuromoji/src/Tokenizer.js';

/** gzip magic (RFC 1952): every member starts with 0x1f 0x8b. */
function looksGzipped(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

/** Decompress `buf` when (and only when) it's actually gzip. A `.dat` file
 *  the host already inflated falls straight through; a raw `.dat.gz` is
 *  expanded via the platform `DecompressionStream`. The try/catch covers the
 *  vanishing case of already-inflated dictionary bytes that happen to open
 *  with the two magic bytes: fall back to the raw buffer rather than fail. */
async function maybeGunzip(buf: ArrayBuffer): Promise<ArrayBuffer> {
  if (!looksGzipped(new Uint8Array(buf))) return buf;
  try {
    const inflated = new Response(buf).body!.pipeThrough(
      new DecompressionStream('gzip'),
    );
    return await new Response(inflated).arrayBuffer();
  } catch {
    return buf;
  }
}

/** kuromoji loader that fetches over HTTP and decodes any gzip itself (see
 *  the module header). Reuses the base loader's `load()` for the file list
 *  and dictionary assembly; only the per-file fetch is ours. */
class FetchDictionaryLoader extends DictionaryLoader {
  loadArrayBuffer(
    url: string,
    callback: (err: unknown, buffer: ArrayBuffer | null) => void,
  ): void {
    fetch(url)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`${res.status} ${res.statusText}: ${url}`);
        }
        return res.arrayBuffer();
      })
      .then(maybeGunzip)
      .then((buf) => callback(null, buf))
      .catch((err) => callback(err, null));
  }
}

/** Build a kuromoji tokenizer whose dictionary loads from `dicPath` (a URL
 *  base; the loader appends `base.dat.gz`, `cc.dat.gz`, … to it). */
export function buildBrowserTokenizer(
  dicPath: string,
): Promise<TokenizerInstance<IpadicFeatures>> {
  return new Promise((resolve, reject) => {
    new FetchDictionaryLoader(dicPath).load((err, dic) => {
      if (err) {
        reject(err instanceof Error ? err : new Error(String(err)));
        return;
      }
      // The runtime object is a genuine kuromoji Tokenizer; bridge the
      // minimal internal type to the public interface furigana.ts uses.
      resolve(new Tokenizer(dic) as unknown as TokenizerInstance<IpadicFeatures>);
    });
  });
}
