import { ParseError } from './errors';

/**
 * Mutable text cursor used by the recursive-descent parser. Tracks a string
 * source and a byte position, with small helpers for peeking, matching, and
 * skipping whitespace.
 */
export class Cursor {
  pos: number;

  constructor(public readonly src: string, pos: number = 0) {
    this.pos = pos;
  }

  /** Return the character at offset `offset` from the current position, or '' at EOF. */
  peek(offset: number = 0): string {
    return this.src[this.pos + offset] ?? '';
  }

  eof(): boolean {
    return this.pos >= this.src.length;
  }

  advance(n: number = 1): void {
    this.pos += n;
  }

  /** True if `s` matches the source starting at the current position. */
  match(s: string): boolean {
    return this.src.startsWith(s, this.pos);
  }

  /** Require `s` at the current position and advance past it; otherwise throw. */
  consume(s: string): void {
    if (!this.match(s)) {
      throw new ParseError(`Expected '${s}'`, this.src, this.pos);
    }
    this.pos += s.length;
  }

  skipWs(): void {
    while (this.pos < this.src.length && isWs(this.src[this.pos])) {
      this.pos++;
    }
  }
}

function isWs(ch: string): boolean {
  return ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r' || ch === '\f' || ch === '\v';
}
