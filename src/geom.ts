export class Point {
  constructor(readonly x: number, readonly y: number) {}

  translate(x: number, y: number) {
    return new Point(this.x + x, this.y + y);
  }
}

export class Box {
  constructor(
    readonly x: number,
    readonly y: number,
    readonly width: number,
    readonly height: number
  ) {}

  get x1() {
    return this.x;
  }
  get x2() {
    return this.x + this.width;
  }
  get y1() {
    return this.y;
  }
  get y2() {
    return this.y + this.height;
  }

  encloses(p: Point): boolean {
    return p.x > this.x1 && p.x < this.x2 && p.y > this.y1 && p.y < this.y2;
  }

  static create(p1: Point, p2: Point) {
    const tl = new Point(Math.min(p1.x, p2.x), Math.min(p1.y, p2.y));
    const br = new Point(Math.max(p1.x, p2.x), Math.max(p1.y, p2.y));

    return new Box(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
  }
}
