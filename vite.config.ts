import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

// The transcriber service runs on port 8001 (see transcriber/docker-compose.yml).
// In dev, Vite proxies /api/* there so the frontend can use same-origin URLs.
// In production, set VITE_TRANSCRIBER_URL to the deployed service's base URL.
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
