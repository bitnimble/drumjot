"""Tests for the LRU-bounded blob cache used by /lyrics/align.

Exercises `BlobCache.get` / `put_bytes` / `put_path`, the eviction
sweep, and the rebuild-from-disk path. Keeps imports narrow so the
module doesn't transitively pull in fastapi / ctc-forced-aligner.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.cache import BlobCache


def test_put_bytes_and_get_roundtrip(tmp_path: Path) -> None:
    cache = BlobCache(tmp_path, cap_bytes=10_000)
    path = cache.put_bytes("alpha.bin", b"hello world")
    assert path.exists()
    assert path.read_bytes() == b"hello world"

    fetched = cache.get("alpha.bin")
    assert fetched is not None
    assert fetched == path
    assert cache.total_bytes == len(b"hello world")


def test_get_returns_none_on_miss(tmp_path: Path) -> None:
    cache = BlobCache(tmp_path, cap_bytes=1000)
    assert cache.get("missing.bin") is None


def test_put_path_copies_src(tmp_path: Path) -> None:
    """`put_path` must copy (not move) so the caller's tempdir survives
    and is safe to rmtree afterwards. The cross-device case is handled
    by the same copyfile path."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "input.bin"
    src.write_bytes(b"payload")

    cache_dir = tmp_path / "cache"
    cache = BlobCache(cache_dir, cap_bytes=1000)
    dest = cache.put_path("entry.bin", src)

    assert dest.exists()
    assert dest.read_bytes() == b"payload"
    # Source still intact for caller cleanup.
    assert src.exists()


def test_evicts_lru_when_over_cap(tmp_path: Path) -> None:
    cache = BlobCache(tmp_path, cap_bytes=20)
    # Three 10-byte entries; the second put crosses the cap and should
    # evict the oldest.
    cache.put_bytes("a.bin", b"0123456789")
    # Stagger timestamps so LRU order is deterministic; time.time has
    # ~1us resolution but on some FSes the in-memory atime updates can
    # collide otherwise.
    time.sleep(0.01)
    cache.put_bytes("b.bin", b"0123456789")
    time.sleep(0.01)
    cache.put_bytes("c.bin", b"0123456789")

    # Total = 30 bytes, cap = 20 -> oldest (a.bin) must be gone.
    assert not (tmp_path / "a.bin").exists()
    assert (tmp_path / "b.bin").exists()
    assert (tmp_path / "c.bin").exists()
    assert cache.get("a.bin") is None
    assert cache.total_bytes == 20


def test_get_refreshes_lru_position(tmp_path: Path) -> None:
    """A `get()` must mark the entry as recently used so a later
    eviction sweep doesn't pick it."""
    cache = BlobCache(tmp_path, cap_bytes=20)
    cache.put_bytes("a.bin", b"0123456789")
    time.sleep(0.01)
    cache.put_bytes("b.bin", b"0123456789")
    time.sleep(0.01)
    # Touch `a.bin` so it's now newer than `b.bin`.
    assert cache.get("a.bin") is not None
    time.sleep(0.01)
    # Adding c.bin should now evict b.bin (oldest), not a.bin.
    cache.put_bytes("c.bin", b"0123456789")

    assert (tmp_path / "a.bin").exists()
    assert not (tmp_path / "b.bin").exists()
    assert (tmp_path / "c.bin").exists()


def test_put_overwrite_updates_size(tmp_path: Path) -> None:
    """Re-putting the same key replaces the old entry and updates the
    running byte total."""
    cache = BlobCache(tmp_path, cap_bytes=1000)
    cache.put_bytes("k.bin", b"short")
    assert cache.total_bytes == len(b"short")
    cache.put_bytes("k.bin", b"a much longer payload here")
    assert cache.total_bytes == len(b"a much longer payload here")
    fetched = cache.get("k.bin")
    assert fetched is not None
    assert fetched.read_bytes() == b"a much longer payload here"


def test_rebuild_picks_up_existing_files(tmp_path: Path) -> None:
    """A fresh BlobCache pointed at a populated directory must rebuild
    its in-memory index from disk so existing entries are visible."""
    (tmp_path / "x.bin").write_bytes(b"previously written")
    (tmp_path / "y.bin").write_bytes(b"another file")

    cache = BlobCache(tmp_path, cap_bytes=1000)
    assert cache.total_bytes == len(b"previously written") + len(b"another file")
    assert cache.get("x.bin") is not None
    assert cache.get("y.bin") is not None


def test_rebuild_ignores_part_files(tmp_path: Path) -> None:
    """Half-written .part files (left behind by a crashed put) must not
    be counted against the cap or surfaced via get()."""
    (tmp_path / "good.bin").write_bytes(b"01234")
    (tmp_path / "stale.bin.part").write_bytes(b"abcdefg")

    cache = BlobCache(tmp_path, cap_bytes=1000)
    assert cache.total_bytes == 5
    assert cache.get("stale.bin.part") is None


def test_get_hydrates_externally_written_file(tmp_path: Path) -> None:
    """A second worker process (or operator copying a file in by hand)
    may write into the cache directory after construction. `get()`
    must notice the file and adopt it into the in-memory index."""
    cache = BlobCache(tmp_path, cap_bytes=1000)
    assert cache.get("late.bin") is None

    (tmp_path / "late.bin").write_bytes(b"injected")

    fetched = cache.get("late.bin")
    assert fetched is not None
    assert fetched.read_bytes() == b"injected"
    assert cache.total_bytes == len(b"injected")


def test_evicted_file_disappears_from_disk(tmp_path: Path) -> None:
    cache = BlobCache(tmp_path, cap_bytes=5)
    cache.put_bytes("a.bin", b"01234")
    time.sleep(0.01)
    cache.put_bytes("b.bin", b"56789")
    assert not (tmp_path / "a.bin").exists()
    # Calling get on the evicted key must return None (not point at a
    # stale path).
    assert cache.get("a.bin") is None


def test_get_after_external_delete_returns_none(tmp_path: Path) -> None:
    """The cache file is gone (operator pruned it) but the in-memory
    index still thinks it's there. `get()` must notice the missing file,
    drop the stale entry, and report a miss so the caller recomputes down
    its fresh pathway instead of being handed a dead path that 500s the
    moment it's read."""
    cache = BlobCache(tmp_path, cap_bytes=1000)
    cache.put_bytes("k.bin", b"data")
    assert cache.total_bytes == len(b"data")
    # Hidden delete behind the cache's back.
    (tmp_path / "k.bin").unlink()
    assert cache.get("k.bin") is None
    # The stale entry self-heals out of the index (size released), so a
    # later re-put accounts cleanly rather than double-counting.
    assert cache.total_bytes == 0
    assert cache.get("k.bin") is None


@pytest.mark.parametrize("data", [b"", b"a", b"\x00" * 4096])
def test_various_payload_sizes(tmp_path: Path, data: bytes) -> None:
    cache = BlobCache(tmp_path, cap_bytes=1 << 20)
    cache.put_bytes("e.bin", data)
    fetched = cache.get("e.bin")
    assert fetched is not None
    assert fetched.read_bytes() == data
    assert cache.total_bytes == len(data)
