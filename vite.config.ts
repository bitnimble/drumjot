import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { patchCssModules } from 'vite-css-modules';
import path from 'node:path';

// We want Vite's file watcher + module-graph invalidation pipeline to
// run on every save (otherwise the dev server keeps serving stale
// transforms across a hard reload, and you end up restarting the
// process to pick up changes). We do NOT want JS/TS edits to hot-patch
// the running page - the page is full of stateful audio (AudioWorklet,
// decoded buffers, transport position) AND of module-singleton MobX
// stores that a JS HMR cycle would re-execute, dropping their state.
// Compromise: suppress `full-reload` outright; allow `update` ONLY for
// CSS-only batches (CSS module class-map swaps are state-preserving -
// Vite re-imports the styles map, React re-renders with new class
// names, no module re-execution). Anything else still requires a
// manual browser reload to pick up fresh transforms from the (now-
// invalidated) module graph.
const noHmrPushPlugin: Plugin = {
  name: 'drumjot:no-hmr-push',
  configureServer(server) {
    const originalSend = server.ws.send.bind(server.ws);
    // `ws.send` is overloaded: an HMRPayload (object with `type`) for
    // built-in events, or `(event, payload)` for custom listeners. We
    // filter the built-in `update` / `full-reload` payloads; custom
    // event sends pass through untouched.
    server.ws.send = ((...args: unknown[]) => {
      const first = args[0];
      if (first && typeof first === 'object') {
        const type = (first as { type?: string }).type;
        if (type === 'full-reload') return;
        if (type === 'update') {
          // `acceptedPath` is the file that actually changed (or its
          // accepting boundary); `path` is the boundary module Vite
          // will tell the client to re-evaluate. For a CSS module
          // edit, acceptedPath is the .css file but path is the JS
          // consumer that accepted it - so checking acceptedPath is
          // what tells us "was the underlying edit a CSS edit?".
          const updates =
            (first as { updates?: Array<{ acceptedPath?: string; path?: string }> }).updates ?? [];
          const allCss = updates.every((u) => {
            const p = u.acceptedPath ?? u.path ?? '';
            return /\.css(\?|$)/.test(p);
          });
          if (!allCss) return;
        }
      }
      return (originalSend as (...a: unknown[]) => void)(...args);
    }) as typeof server.ws.send;
  },
};

// Note: the dev server intentionally has NO `/api` proxy. The browser
// bundle talks to `/api` on its own origin, and routing `/api/*` onward
// to the transcriber is handled by the Caddy edge proxy that fronts this
// dev server (see Caddyfile.dev / docker-compose.dev.yml). Vite's own
// `server.proxy` used to do it, but the dev server runs under Bun, whose
// node:http layer can't relay the transcriber's chunked NDJSON streaming
// responses (it hangs - oven-sh/bun#5737, #28396). So the proxy lives in
// Caddy and Vite never touches /api. Running Vite standalone (without the
// Caddy front) therefore won't have /api routing - that's intentional.

// Signalsmith Stretch installs its AudioWorklet processor by
// stringifying a function (via `${fn}`) and dropping it into a
// `URL.createObjectURL(new Blob([...]))`-backed `audioWorklet.addModule`
// call. If esbuild downlevels class fields to a `__publicField` helper,
// the helper exists in the main-bundle scope but the worklet's blob
// scope can't see it; the worklet fails to parse with
// `ReferenceError: __publicField is not defined`. Keep both the
// dep-prebundle path (dev) and the prod build at a target that emits
// class fields natively (`es2022`+) so the stringified processor is
// self-contained.
const ESBUILD_TARGET = 'es2022';

export default defineConfig({
  // `patchCssModules()` replaces Vite's built-in PostCSS-based CSS-modules
  // handling with one that routes `composes: … from` through Vite's own
  // module resolver. Without it, a file consumed ONLY via `composes`
  // (e.g. components/button.module.css, which nothing imports directly)
  // gets no module-graph edge to its consumers, so editing it never
  // invalidates them - HMR shows nothing and even a hard reload serves a
  // stale transform until the dev server restarts. This is unfixed in Vite
  // core (vitejs/vite#16074; the core fix PR #16018 was closed in favour
  // of this plugin). Bonus: it also de-duplicates composed styles instead
  // of inlining button.module.css into all ~20 consumers. Needs build
  // target es2022, which we already set below.
  plugins: [patchCssModules(), react(), noHmrPushPlugin],
  resolve: {
    alias: {
      src: path.resolve(__dirname, 'src'),
    },
  },
  esbuild: {
    target: ESBUILD_TARGET,
  },
  build: {
    target: ESBUILD_TARGET,
  },
  optimizeDeps: {
    // Dev dep-prebundle also goes through esbuild; without aligning the
    // target here, prebundled `signalsmith-stretch` would still emit
    // `__publicField` in dev even though the prod build is clean.
    esbuildOptions: {
      target: ESBUILD_TARGET,
    },
  },
  server: {
    host: true,
    port: 5173,
    allowedHosts: ['drumjot.kumo.dev'],
  },
  preview: {
    host: true,
    port: 5173,
    allowedHosts: ['drumjot.kumo.dev'],
  },
});
