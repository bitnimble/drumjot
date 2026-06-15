/**
 * Error type thrown by the Drumjot DSL parser. Carries source-position info
 * so the message includes a useful `(line N, col M)` suffix.
 */
export class ParseError extends Error {
  readonly pos: number;
  readonly line: number;
  readonly col: number;

  constructor(message: string, src: string, pos: number) {
    let line = 1;
    let col = 1;
    for (let i = 0; i < pos && i < src.length; i++) {
      if (src[i] === '\n') {
        line++;
        col = 1;
      } else {
        col++;
      }
    }
    super(`${message} (line ${line}, col ${col})`);
    this.pos = pos;
    this.line = line;
    this.col = col;
    this.name = 'ParseError';
  }
}
