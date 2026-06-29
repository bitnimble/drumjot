import type { StorybookConfig } from '@storybook/react-vite';
import { patchCssModules } from 'vite-css-modules';
import path from 'node:path';

/**
 * Storybook (React + Vite). Stories live in per-feature `stories/`
 * subfolders (mirroring the `tests/` convention), matched by the glob
 * below.
 *
 * `viteFinal` re-applies the two pieces of the app's `vite.config.ts`
 * that stories depend on but Storybook's stock Vite config doesn't carry:
 *   - `patchCssModules()`, the CSS-module files use `composes: … from`,
 *     which Vite core mishandles; the app patches it and so must SB.
 *   - the `src` import alias, the codebase imports via `src/…`.
 * The `es2022` esbuild target matches the app (keeps class fields native;
 * see the note in vite.config.ts about the signalsmith-stretch worklet).
 */
const config: StorybookConfig = {
  stories: ['../frontend/src/**/stories/*.stories.@(ts|tsx)'],
  framework: { name: '@storybook/react-vite', options: {} },
  // Serve the app's `public/` at the iframe root so absolute asset URLs
  // resolve the same as in the app, the Logo `<img src="/favicon.svg">`
  // would otherwise fall through to Storybook's own default favicon.
  staticDirs: ['../frontend/public'],
  viteFinal: async (cfg) => {
    cfg.plugins = cfg.plugins ?? [];
    // `unshift` so the CSS-modules patch wraps Vite's handling before the
    // React plugin, matching the app's plugin order.
    cfg.plugins.unshift(patchCssModules());
    cfg.resolve = cfg.resolve ?? {};
    cfg.resolve.alias = {
      ...(cfg.resolve.alias as Record<string, string> | undefined),
      // Storybook compiles this config to CJS, so `__dirname` is the
      // reliable way to anchor the alias (import.meta is empty under cjs).
      src: path.resolve(__dirname, '../frontend/src'),
    };
    cfg.esbuild = { ...(cfg.esbuild || {}), target: 'es2022' };
    return cfg;
  },
};
export default config;
