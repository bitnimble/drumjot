import * as fs from 'fs';
import * as path from 'path';
import { htmlPlugin } from '@craftamap/esbuild-plugin-html';
import esbuild from 'esbuild';
// @ts-ignore
import postcssModulesValuesReplace from 'postcss-modules-values-replace';
import postcssUrl from 'postcss-url';
import postcssCalc from 'postcss-calc';
// @ts-ignore
import postcssColorFunction from 'postcss-color-function';
import autoprefixer from 'autoprefixer';
// @ts-ignore
import * as postcssPlugin from 'esbuild-plugin-postcss2';


const htmlTemplate = fs.readFileSync(path.resolve(__dirname, './index_template.html'),  'utf-8');
const entryPoint = path.resolve(__dirname, 'src/index.tsx');
const options: esbuild.BuildOptions = {
  entryPoints: [entryPoint],
  bundle: true,
  metafile: true,
  outdir: path.resolve(__dirname, 'dist'),
  entryNames: 'index.[hash]',
  assetNames: '[hash]',
  sourcemap: true,
  plugins: [
    postcssPlugin.default({
      modules: true,
      writeToFile: true,
      fileIsModule: (filepath: string) => !filepath.endsWith('.global.css'),
      plugins: [
        postcssModulesValuesReplace(),
        postcssUrl(),
        postcssCalc({}),
        postcssColorFunction(),
        autoprefixer(),
      ]
    }),
    htmlPlugin({
      files: [
        {
          entryPoints: [path.relative(__dirname, entryPoint)],
          filename: 'index.html',
          htmlTemplate,
        }
      ]
    }),
  ]
};

const watch = process.env.WATCH === 'true';

(async () => {
  if (watch) {
    let ctx = await esbuild.context(options);
    await ctx.watch();
  } else {
    await esbuild.build(options);
  }
})();
