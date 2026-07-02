"""Guard `_download`'s truncation check: a transfer cut short at EOF without a
protocol error (proxy/CDN) must NOT be renamed to the real filename as if valid.

`_download` calls `httpx.stream(...)` directly (not via a client), so a
`MockTransport` can't intercept it; monkeypatch `httpx.stream` with a small fake
streaming context manager exposing the surface `_download` touches.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest

from app.pipeline import provision


class _FakeStream:
    """Stand-in for httpx's streaming Response. `num_bytes_downloaded` is the
    RAW/encoded byte count (what the real object tracks); set independently of the
    yielded body so a truncated transfer can be simulated."""

    def __init__(self, body: bytes, *, content_length: int | None, num_bytes: int) -> None:
        self._body = body
        self.headers = httpx.Headers(
            {"content-length": str(content_length)} if content_length is not None else {}
        )
        self.num_bytes_downloaded = num_bytes

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


def _patch_stream(monkeypatch: pytest.MonkeyPatch, stream: _FakeStream) -> None:
    @contextmanager
    def fake_stream(*_args: object, **_kwargs: object) -> Iterator[_FakeStream]:
        yield stream

    monkeypatch.setattr(httpx, "stream", fake_stream)


def test_truncated_body_raises_and_leaves_no_dest(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "weights.onnx"
    # Content-Length says 100, but only 40 bytes actually arrived.
    _patch_stream(monkeypatch, _FakeStream(b"x" * 40, content_length=100, num_bytes=40))
    with pytest.raises(OSError):
        provision._download("http://example/weights.onnx", dest)
    assert not dest.exists()
    assert not (tmp_path / "weights.onnx.part").exists()


def test_complete_body_succeeds(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "weights.onnx"
    body = b"x" * 100
    _patch_stream(monkeypatch, _FakeStream(body, content_length=100, num_bytes=100))
    provision._download("http://example/weights.onnx", dest)
    assert dest.read_bytes() == body
    assert not (tmp_path / "weights.onnx.part").exists()


def test_no_content_length_succeeds(tmp_path, monkeypatch) -> None:
    # Chunked / unknown-length response: no Content-Length header, so no check
    # runs and the download must not spuriously fail.
    dest = tmp_path / "weights.onnx"
    body = b"y" * 37
    _patch_stream(monkeypatch, _FakeStream(body, content_length=None, num_bytes=37))
    provision._download("http://example/weights.onnx", dest)
    assert dest.read_bytes() == body


def test_gzip_decoded_body_shorter_than_content_length_succeeds(tmp_path, monkeypatch) -> None:
    # Content-Length is the ENCODED size; iter_bytes yields DECODED (larger) bytes.
    # The check must compare against num_bytes_downloaded (encoded), so a body whose
    # decoded length differs from Content-Length must still succeed.
    dest = tmp_path / "weights.onnx"
    decoded = b"z" * 500  # decoded payload the caller writes
    _patch_stream(monkeypatch, _FakeStream(decoded, content_length=120, num_bytes=120))
    provision._download("http://example/weights.onnx", dest)
    assert dest.read_bytes() == decoded
