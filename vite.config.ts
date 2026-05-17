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

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      src: path.resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: TRANSCRIBER_URL,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
});
