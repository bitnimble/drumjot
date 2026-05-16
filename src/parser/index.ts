/**
 * Public API for the Drumjot DSL parser.
 *
 * Usage:
 *
 *   import { parse } from 'src/parser';
 *   const jot = parse(dslSource);
 */
export { parse } from './parser';
export { ParseError } from './errors';
export { preprocessMacros } from './preprocess';
export type { Macros, PreprocessResult } from './preprocess';
