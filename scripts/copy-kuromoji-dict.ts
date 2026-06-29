// Copies @sglkc/kuromoji's compressed dictionary out of node_modules into
// frontend/public/ so Vite serves it at /kuromoji-dict/ (and `vite build` emits
// it to dist/). The browser tokenizer in frontend/src/lyrics/furigana.ts fetches
// it lazily, only for songs whose lyrics contain kanji.
//
// Wired as a postinstall hook so a fresh `bun install` provisions the asset;
// idempotent, and a no-op (clean exit 0) when the optional dep isn't installed
// yet. Run with bun.
import { access, cp, mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'node_modules', '@sglkc', 'kuromoji', 'dict');
const dest = join(root, 'frontend', 'public', 'kuromoji-dict');

try {
  await access(src);
} catch {
  console.log('[kuromoji-dict] @sglkc/kuromoji not installed yet; skipping.');
  process.exit(0);
}

await mkdir(dest, { recursive: true });
await cp(src, dest, { recursive: true });
console.log(`[kuromoji-dict] copied dictionary -> ${dest}`);
