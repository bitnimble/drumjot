import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

// The transcriber service runs on port 8001 (see transcriber/docker-compose.yml).
//
// `VITE_TRANSCRIBER_URL` is read in two places by design:
//   - here (build-server time, node `process.env`): sets the dev proxy
//     target. Defaults to the docker-compose localhost address.
//   - in the browser bundle (`src/transcriber.ts`, via `import.meta.env`):
//     same name, but defaults to `/api` (i.e. routes through this proxy
//     in dev). When set at build time, Vite bakes the value into the
//     bundle so production deploys talk to the configured service URL.
//
// Using the same name in both places means a single env var configures
// both dev and prod paths.
const TRANSCRIBER_URL = process.env.VITE_TRANSCRIBER_URL ?? 'http://localhost:8001';

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
  plugins: [react()],
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
    hmr: false,
    host: true,
    port: 5173,
    allowedHosts: ["drumjot.kumo.dev"],
    proxy: {
      '/api': {
        target: TRANSCRIBER_URL,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
});
