"""Tests for the /lyrics/align disk-cache key helpers (vocals + result).

The helpers live in `app.main` as module-level private functions.
Importing them is cheap because the lyrics_align module lazy-loads
ctc-forced-aligner / torch inside its methods, not at import time.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import app.main as main
from app.config import settings
from app.pipeline.lyrics_align import InputLine


def _lines() -> list[InputLine]:
    return [
        InputLine(start_sec=0.0, text="hello"),
        InputLine(start_sec=1.5, text="world"),
    ]


def test_sanitize_id_replaces_unsafe_chars() -> None:
    """Anything outside [A-Za-z0-9._-] becomes `_` so the id is safe to
    drop into a filename across both POSIX and Windows."""
    assert main._sanitize_id("Kim_Vocal_2.onnx") == "Kim_Vocal_2.onnx"
    assert main._sanitize_id("model:v3 (beta)") == "model_v3__beta_"
    assert main._sanitize_id("path/with/slashes") == "path_with_slashes"


def test_hash_bytes_matches_known_sha256() -> None:
    """SHA-256 of the empty input is the well-known constant below."""
    empty = main._hash_bytes(b"")
    assert empty == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert main._hash_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_vocals_cache_key_changes_with_separator_model(monkeypatch) -> None:
    h = "2" * 64
    monkeypatch.setattr(settings, "demucs_model", "model_bs_roformer_sw.ckpt")
    a = main._vocals_cache_key(h)
    monkeypatch.setattr(settings, "demucs_model", "some_other_model.ckpt")
    b = main._vocals_cache_key(h)
    assert a != b


def test_alignment_cache_key_format() -> None:
    """Key is `<audio_hash>__align-<version>-<lyrics_hash>.json`."""
    h = "a" * 64
    key = main._alignment_cache_key(_lines(), "en", h)
    assert key.startswith(f"{h}__align-")
    assert key.endswith(".json")


def test_alignment_cache_key_stable_for_identical_input() -> None:
    h = "a" * 64
    assert main._alignment_cache_key(_lines(), "en", h) == main._alignment_cache_key(
        _lines(), "en", h
    )


def test_alignment_cache_key_changes_with_lyrics_text() -> None:
    h = "a" * 64
    edited = [
        InputLine(start_sec=0.0, text="HELLO"),
        InputLine(start_sec=1.5, text="world"),
    ]
    assert main._alignment_cache_key(_lines(), "en", h) != main._alignment_cache_key(
        edited, "en", h
    )


def test_alignment_cache_key_changes_with_start_time() -> None:
    h = "a" * 64
    nudged = [
        InputLine(start_sec=0.0, text="hello"),
        InputLine(start_sec=1.6, text="world"),
    ]
    assert main._alignment_cache_key(_lines(), "en", h) != main._alignment_cache_key(
        nudged, "en", h
    )


def test_alignment_cache_key_changes_with_language() -> None:
    h = "a" * 64
    assert main._alignment_cache_key(_lines(), "en", h) != main._alignment_cache_key(
        _lines(), "ja", h
    )


def test_alignment_cache_key_treats_none_and_empty_language_alike() -> None:
    """`language or ""` folds the no-hint cases together so an empty form
    field and an omitted one hit the same entry."""
    h = "a" * 64
    assert main._alignment_cache_key(_lines(), None, h) == main._alignment_cache_key(
        _lines(), "", h
    )


def test_alignment_cache_key_changes_with_audio_hash() -> None:
    assert main._alignment_cache_key(
        _lines(), "en", "a" * 64
    ) != main._alignment_cache_key(_lines(), "en", "b" * 64)


def test_alignment_cache_key_changes_with_version(monkeypatch) -> None:
    """An aligner-version bump auto-invalidates every cached result."""
    h = "a" * 64
    before = main._alignment_cache_key(_lines(), "en", h)
    monkeypatch.setattr(main, "_ALIGN_CACHE_VERSION", "totally-different-v9")
    assert before != main._alignment_cache_key(_lines(), "en", h)


def test_alignment_cache_key_rounds_start_to_ms() -> None:
    """Sub-millisecond jitter on a start time collapses to one key, so a
    JSON round-trip's float noise doesn't fragment the cache."""
    h = "a" * 64
    a = main._alignment_cache_key([InputLine(start_sec=1.0, text="x")], "en", h)
    b = main._alignment_cache_key([InputLine(start_sec=1.00004, text="x")], "en", h)
    assert a == b


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch):
    """Re-point the vocals + alignment cache singletons at a per-test tmp
    dir so writes don't leak across tests or into the dev box's real
    /cache."""
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    monkeypatch.setattr(main, "_vocals_cache", None)
    monkeypatch.setattr(main, "_alignment_cache", None)
    yield tmp_path
    # Singletons get reset on the next test via the monkeypatch teardown.


def test_isolated_cache_singleton_uses_fresh_dir(isolated_cache) -> None:
    """Smoke check that the fixture's `_vocals_cache = None` reset
    actually re-points the singleton at the new tmp dir; otherwise the
    rest of the cache assertions are meaningless (and could write into
    the dev box's real /cache)."""
    vc = main._vocals_cache_instance()
    assert isolated_cache in vc.dir.parents


def test_alignment_cache_round_trip(isolated_cache) -> None:
    """A miss returns None; after `put_bytes`, the same key returns the
    stored payload byte-for-byte."""
    cache = main._alignment_cache_instance()
    assert isolated_cache in cache.dir.parents
    key = main._alignment_cache_key(_lines(), "en", "f" * 64)
    payload = b'[{"startSec":0.0,"text":"hello","words":[]}]'
    assert cache.get(key) is None
    cache.put_bytes(key, payload)
    got = cache.get(key)
    assert got is not None
    assert got.read_bytes() == payload
