/**
 * Convert byte offsets (the shape source positions take throughout the
 * parser and the AST `range` fields) into human-readable `(line, column)`
 * pairs. Both line and column are 1-indexed to match what editors display.
 */

export type LineColumn = {
  line: number;
  column: number;
};

/**
 * Map a 0-indexed byte offset into the source into a 1-indexed (line, column).
 * If `offset` is outside `[0, src.length]` it's clamped — callers don't have
 * to special-case EOF themselves.
 *
 * For repeated lookups against the same source, prefer `buildLineIndex` +
 * `lookupOffset` — O(log N) per query versus O(N) here.
 */
export function offsetToLineCol(src: string, offset: number): LineColumn {
  const idx = buildLineIndex(src);
  return lookupOffset(idx, offset);
}

/**
 * Precomputed sorted array of line-start offsets. `lineStarts[i]` is the
 * byte offset of the first character of line `i + 1`. Construction is O(N);
 * each lookup against the index is O(log N).
 */
export type LineIndex = {
  src: string;
  lineStarts: number[];
};

export function buildLineIndex(src: string): LineIndex {
  const lineStarts: number[] = [0];
  for (let i = 0; i < src.length; i++) {
    if (src[i] === '\n') lineStarts.push(i + 1);
  }
  return { src, lineStarts };
}

export function lookupOffset(idx: LineIndex, offset: number): LineColumn {
  const clamped = Math.max(0, Math.min(offset, idx.src.length));
  // Binary search for the largest lineStarts entry <= clamped.
  let lo = 0;
  let hi = idx.lineStarts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (idx.lineStarts[mid] <= clamped) lo = mid;
    else hi = mid - 1;
  }
  return {
    line: lo + 1,
    column: clamped - idx.lineStarts[lo] + 1,
  };
}

/**
 * Extract a snippet of `src` covering `range` with up to `contextLines` lines
 * of surrounding context. Useful for showing the LLM the offending region of
 * a Jot rather than the whole document.
 */
export function extractSnippet(
  src: string,
  range: { start: number; end: number },
  contextLines: number = 1
): { snippet: string; startLine: number } {
  const idx = buildLineIndex(src);
  const startLc = lookupOffset(idx, range.start);
  const endLc = lookupOffset(idx, range.end);
  const firstLine = Math.max(1, startLc.line - contextLines);
  const lastLine = Math.min(idx.lineStarts.length, endLc.line + contextLines);
  const startOffset = idx.lineStarts[firstLine - 1];
  const endOffset =
    lastLine < idx.lineStarts.length
      ? idx.lineStarts[lastLine] - 1 // strip the trailing newline
      : src.length;
  return {
    snippet: src.slice(startOffset, endOffset),
    startLine: firstLine,
  };
}
