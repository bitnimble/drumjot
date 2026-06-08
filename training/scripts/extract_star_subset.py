"""Extract a SUBSET of STAR Drums directly from the split parts.

Reads the `STAR_Drums_full.zip.part-a?` chunks as one virtual seekable file
and uses zipfile's selective extraction, so only the bytes for the chosen
members are read, no 181 GB reassembly needed (fast over a slow NAS link).
Each "song" = one mix annotation (.txt) + its mix audio (audio/mix/*.flac);
the data/<split>/... layout is preserved so `star.index(out_dir)` works.

For the FULL dataset, reassemble + unzip on a fast disk instead (see the
copy-and-extract commands in the README / fetch_star.sh).

Usage: extract_star_subset.py <parts_dir> <out_dir> [train_songs] [val_songs]
"""
from __future__ import annotations

import os
import sys
import zipfile


class MultiPartFile:
    """Seekable read-only view over ordered file parts as one byte stream."""

    name = "<multipart>"

    def __init__(self, paths: list[str]):
        self._fhs = [open(p, "rb") for p in paths]  # noqa: SIM115 (closed in close())
        self._sizes = [os.path.getsize(p) for p in paths]
        self._starts: list[int] = []
        acc = 0
        for s in self._sizes:
            self._starts.append(acc)
            acc += s
        self._total = acc
        self._pos = 0

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._total + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._total - self._pos
        out = bytearray()
        while n > 0 and self._pos < self._total:
            i = 0
            while i + 1 < len(self._starts) and self._starts[i + 1] <= self._pos:
                i += 1
            local = self._pos - self._starts[i]
            take = min(n, self._sizes[i] - local)
            self._fhs[i].seek(local)
            chunk = self._fhs[i].read(take)
            if not chunk:
                break
            out += chunk
            self._pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def close(self) -> None:
        for f in self._fhs:
            f.close()


def _split_of(name: str) -> str:
    low = name.lower()
    for s in ("training", "validation", "test"):
        if f"/{s}/" in low:
            return s
    return "?"


def main() -> None:
    parts_dir, out_dir = sys.argv[1], sys.argv[2]
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    n_val = int(sys.argv[4]) if len(sys.argv) > 4 else 80

    parts = sorted(
        os.path.join(parts_dir, f) for f in os.listdir(parts_dir) if ".zip.part-" in f
    )
    print("parts:", [os.path.basename(p) for p in parts], flush=True)
    mpf = MultiPartFile(parts)
    z = zipfile.ZipFile(mpf)
    names = z.namelist()
    name_set = set(names)
    print("total members in archive:", len(names), flush=True)

    by_split: dict[str, list[str]] = {"training": [], "validation": [], "test": []}
    for a in names:
        if "/annotation/" in a and a.endswith(".txt"):
            s = _split_of(a)
            if s in by_split:
                by_split[s].append(a)
    for v in by_split.values():
        v.sort()
    print("annotations/split:", {s: len(v) for s, v in by_split.items()}, flush=True)

    chosen = by_split["training"][:n_train]
    chosen += (by_split["validation"] or by_split["test"])[:n_val]

    members: set[str] = set()
    for a in chosen:
        members.add(a)
        parent, fname = a.rsplit("/annotation/", 1)
        mix = f"{parent}/audio/mix/{fname[:-4]}.flac"
        if mix in name_set:
            members.add(mix)
        else:
            print("  WARN no mix for", a, flush=True)
    members.update(n for n in names if n.endswith("class_mappings.py"))

    print(f"extracting {len(members)} members -> {out_dir}", flush=True)
    for i, m in enumerate(sorted(members)):
        z.extract(m, out_dir)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(members)}", flush=True)
    z.close()
    mpf.close()
    print("DONE. subset root:", out_dir, flush=True)


if __name__ == "__main__":
    main()
