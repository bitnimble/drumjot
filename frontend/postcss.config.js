import autoprefixer from 'autoprefixer';

// Autoprefixer reads `browserslist` in package.json (evergreen / last-2-years).
// It supplies the vendor prefixes the OS webviews still need that our source
// CSS omits, e.g. `-webkit-backdrop-filter` for WebKit (Safari/WebKitGTK under
// Tauri). The `patchCssModules()` Vite plugin routes each `composes: … from`
// file through Vite's module pipeline, so composed files get prefixed too.
export default {
  plugins: [autoprefixer()],
};
