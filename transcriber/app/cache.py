"""Disk-backed LRU blob cache for the /lyrics/align pipeline.

Two independent caches live under `settings.cache_dir`:

  - `vocals/`; opus-encoded separated vocals stems
  - `alignment/`; CTC forced-alignment output JSON

Both are content-addressed: callers pass an opaque filename key that
already encodes the input hash and the model identifiers that produced
the entry. Stale entries left behind by a model swap simply never get
hit; LRU eventually reclaims them.

LRU bookkeeping is in-memory only. On construction we walk the cache
directory once to populate the entry table from `os.stat`; thereafter
`get()` / `put_*()` update the in-memory access time themselves
(filesystem atime is unreliable with `noatime`/`relatime` mounts so we
don't depend on it).
"""
from __future__ import annotations

import contextlib
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class _Entry:
    name: str
    size: int
    atime: float


class BlobCache:
    """LRU-bounded content-addressed cache backed by a single directory.

    Thread-safe within one process via `self._lock`. Across processes
    (Caddy fans `/lyrics/align` to the pipeline worker), each worker
    holds its own in-memory index. The on-disk files stay consistent
    because writes go through `os.replace`; the per-process indexes
    drift but self-heal via the `get()` hydrate branch and the
    eviction-side unlink fallback.
    """

    def __init__(self, cache_dir: Path, cap_bytes: int) -> None:
        self._dir = cache_dir
        self._cap_bytes = cap_bytes
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._total_bytes: int = 0
        self._dir.mkdir(parents=True, exist_ok=True)
        self._rebuild()

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def get(self, key: str) -> Path | None:
        """Return the cached path if present, marking it as freshly used.
        Returns None on miss. Hydrates the in-memory index on demand for
        files written by another worker process since the last rebuild.

        `.part` keys are reserved for in-flight writes by `put_*` and
        never returned, so a crashed put can't surface a half-written
        file via a coincidentally-shaped lookup.
        """
        if key.endswith(".part"):
            return None
        path = self._dir / key
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                if not path.is_file():
                    return None
                try:
                    size = path.stat().st_size
                except OSError:
                    return None
                entry = _Entry(name=key, size=size, atime=time.time())
                self._entries[key] = entry
                self._total_bytes += size
            elif not path.is_file():
                # Indexed but gone from disk (operator pruned the file, or
                # another worker evicted it behind our back). Drop the stale
                # entry and report a miss, so the caller recomputes down its
                # fresh pathway instead of being handed a dead path that
                # 500s the moment it's read.
                self._total_bytes -= entry.size
                del self._entries[key]
                return None
            entry.atime = time.time()
        return path

    def put_path(self, key: str, src: Path) -> Path:
        """Copy `src` into the cache as `key` and run LRU eviction.

        Caller retains ownership of `src` (we copy, not move) so this is
        safe across filesystems. The copy lands in a sibling `.part`
        file inside the cache dir and is atomically renamed into place,
        so concurrent readers never see a half-written file.
        """
        dest = self._dir / key
        tmp = dest.with_suffix(dest.suffix + ".part")
        shutil.copyfile(str(src), str(tmp))
        os.replace(str(tmp), str(dest))
        self._record(key, dest)
        return dest

    def put_bytes(self, key: str, data: bytes) -> Path:
        """Write `data` into the cache as `key` and run LRU eviction."""
        dest = self._dir / key
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        os.replace(str(tmp), str(dest))
        self._record(key, dest)
        return dest

    def _record(self, key: str, path: Path) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            return
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._total_bytes -= old.size
            self._entries[key] = _Entry(name=key, size=size, atime=time.time())
            self._total_bytes += size
            self._evict()

    def _evict(self) -> None:
        """Drop LRU entries until total bytes <= cap. Lock-held."""
        if self._total_bytes <= self._cap_bytes:
            return
        ordered = sorted(self._entries.values(), key=lambda e: e.atime)
        for entry in ordered:
            if self._total_bytes <= self._cap_bytes:
                break
            with contextlib.suppress(FileNotFoundError):
                (self._dir / entry.name).unlink()
            self._entries.pop(entry.name, None)
            self._total_bytes -= entry.size
            log.info(
                "cache(%s): evicted %s (%d bytes); total=%d/%d",
                self._dir.name, entry.name, entry.size,
                self._total_bytes, self._cap_bytes,
            )

    def _rebuild(self) -> None:
        """Populate _entries from a one-shot directory walk. Skips .part
        files (partial writes). Initial access time defaults to file
        mtime, so the LRU order on first run is creation order.
        """
        for path in self._dir.iterdir():
            if not path.is_file() or path.name.endswith(".part"):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            self._entries[path.name] = _Entry(
                name=path.name, size=stat.st_size, atime=stat.st_mtime,
            )
            self._total_bytes += stat.st_size
