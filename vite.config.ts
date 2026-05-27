import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

// We want Vite's file watcher + module-graph invalidation pipeline to
// run on every save (otherwise the dev server keeps serving stale
// transforms across a hard reload, and you end up restarting the
// process to pick up changes). We do NOT want the browser to
// auto-reload or hot-patch - the page is full of stateful audio
// (AudioWorklet, decoded buffers, transport position) that a surprise
// reload nukes mid-iteration. Compromise: keep HMR fully enabled
// server-side, but suppress the websocket pushes that would otherwise
// drive the client. A manual browser reload then fetches the fresh
// modules from the (now-invalidated) module graph.
const noHmrPushPlugin: Plugin = {
  name: 'drumjot:no-hmr-push',
  configureServer(server) {
    const originalSend = server.ws.send.bind(server.ws);
    // `ws.send` is overloaded: an HMRPayload (object with `type`) for
    // built-in events, or `(event, payload)` for custom listeners. We
    // only suppress the built-in `update` / `full-reload` payloads.
    server.ws.send = ((...args: unknown[]) => {
      const first = args[0];
      if (first && typeof first === 'object') {
        const type = (first as { type?: string }).type;
        if (type === 'update' || type === 'full-reload') return;
      }
      return (originalSend as (...a: unknown[]) => void)(...args);
    }) as typeof server.ws.send;
  },
};

// The transcriber service runs on port 8001 (see transcriber/docker-compose.yml).
//
// `TRANSCRIBER_URL` is read server-side only (node `process.env`) and sets
// the proxy target for both `vite dev` and `vite preview`. It is NOT
// prefixed with `VITE_`, so it is never exposed to the browser bundle;
// the bundle always talks to `/api` on its own origin and lets this
// process proxy the call onward. That keeps the browser CORS-safe when
// the frontend and the transcriber are on different origins (e.g.
// drumjot.kumo.dev vs. a LAN GPU box).
const TRANSCRIBER_URL = process.env.TRANSCRIBER_URL ?? 'http://localhost:8001';

const apiProxy = {
  '/api': {
    target: TRANSCRIBER_URL,
    changeOrigin: true,
    rewrite: (p: string) => p.replace(/^\/api/, ''),
  },
};

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
  plugins: [react(), noHmrPushPlugin],
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
    allowedHosts: ["drumjot.kumo.dev"],
    proxy: apiProxy,
  },
  preview: {
    host: true,
    port: 5173,
    allowedHosts: ["drumjot.kumo.dev"],
    proxy: apiProxy,
  },
});
