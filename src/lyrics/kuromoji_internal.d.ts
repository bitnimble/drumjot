// Ambient types for the two @sglkc/kuromoji internals we import directly in
// kuromoji_loader.ts to supply our own dictionary loader. The package only
// ships types for its main entry (`declare module "@sglkc/kuromoji"`), so
// these deep CJS modules are otherwise untyped. Kept minimal: just the
// surface we actually touch.
declare module '@sglkc/kuromoji/src/loader/DictionaryLoader.js' {
  import type { DynamicDictionaries } from '@sglkc/kuromoji';

  /** Base loader. `load()` knows the dict file list and assembles the
   *  DynamicDictionaries; it delegates each file's byte fetch to the
   *  abstract `loadArrayBuffer`, which subclasses override. */
  class DictionaryLoader {
    constructor(dicPath: string);
    loadArrayBuffer(
      url: string,
      callback: (err: unknown, buffer: ArrayBuffer | null) => void,
    ): void;
    load(callback: (err: unknown, dic: DynamicDictionaries) => void): void;
  }
  export = DictionaryLoader;
}

declare module '@sglkc/kuromoji/src/Tokenizer.js' {
  import type { DynamicDictionaries, IpadicFeatures } from '@sglkc/kuromoji';

  class Tokenizer {
    constructor(dic: DynamicDictionaries);
    tokenize(text: string): IpadicFeatures[];
  }
  export = Tokenizer;
}
