"""Tests for the /lyrics/align vocals-cache key helpers.

The helpers live in `app.main` as module-level private functions.
Importing them is cheap because the lyrics_align module lazy-loads
whisperx / torch inside its methods, not at import time.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import app.main as main
from app.config import settings


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
    monkeypatch.setattr(settings, "vocals_model", "Kim_Vocal_2.onnx")
    a = main._vocals_cache_key(h)
    monkeypatch.setattr(settings, "vocals_model", "Other_Model.onnx")
    b = main._vocals_cache_key(h)
    assert a != b


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch):
    """Re-point the vocals-cache singleton at a per-test tmp dir so
    writes don't leak across tests or into the dev box's real /cache."""
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    monkeypatch.setattr(main, "_vocals_cache", None)
    yield tmp_path
    # Singleton gets reset on the next test via the monkeypatch teardown.


def test_isolated_cache_singleton_uses_fresh_dir(isolated_cache) -> None:
    """Smoke check that the fixture's `_vocals_cache = None` reset
    actually re-points the singleton at the new tmp dir; otherwise the
    rest of the cache assertions are meaningless (and could write into
    the dev box's real /cache)."""
    vc = main._vocals_cache_instance()
    assert isolated_cache in vc.dir.parents
