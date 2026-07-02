"""Tests for the NDJSON streaming primitives behind the streaming
endpoints: `_serialized_gpu_stream` (GPU-lock sequencing) and
`_pump_with_heartbeat` (the shared heartbeat pump the three streamers use).

`_serialized_gpu_stream` wraps a unit of GPU work behind the process-wide
GPU lock and emits a `{"type": "queued"}` envelope when the lock is already
held, so a client blocked behind another in-flight request can show a wait
state instead of a silent hang. Once it owns the lock it emits
`{"type": "running"}` and then forwards whatever the job yields.

`_pump_with_heartbeat` drives a next-item factory to exhaustion, emitting a
`{"type": "heartbeat"}` NDJSON line during silent gaps so an idle-timeout
proxy doesn't drop the connection.

Imports `app.main` (cheap; lyrics_align lazy-loads torch inside its
methods). The helpers are pure asyncio, so these drive the async generators
directly via `asyncio.run` rather than pulling in pytest-asyncio.
"""
from __future__ import annotations

import asyncio
import contextlib
import json

import app.main as main


def test_uncontended_emits_running_then_job_without_queued() -> None:
    """When the GPU lock is free, the stream skips `queued` entirely: it
    acquires immediately, emits `running`, then forwards the job's
    envelopes. The lock is released once the stream completes."""

    async def scenario():
        lock = asyncio.Lock()

        async def job():
            yield {"type": "result", "data": {"lines": []}}

        out = [env async for env in main._serialized_gpu_stream(lock, job)]
        return out, lock.locked()

    out, still_locked = asyncio.run(scenario())
    assert out == [
        {"type": "running"},
        {"type": "result", "data": {"lines": []}},
    ]
    assert still_locked is False


def test_contended_emits_queued_before_running() -> None:
    """When another holder owns the lock, the first envelope is `queued`
    and arrives WITHOUT blocking (it's yielded before the helper awaits
    the lock). After the holder releases, the stream proceeds to
    `running` + the job's output, and releases the lock at the end."""

    async def scenario():
        lock = asyncio.Lock()
        await lock.acquire()  # stand in for another in-flight GPU request

        async def job():
            yield {"type": "result", "data": {"lines": []}}

        gen = main._serialized_gpu_stream(lock, job)
        first = await gen.__anext__()
        # Hand the lock back so the helper can acquire and finish.
        lock.release()
        rest = [env async for env in gen]
        return first, rest, lock.locked()

    first, rest, still_locked = asyncio.run(scenario())
    assert first == {"type": "queued"}
    assert rest == [
        {"type": "running"},
        {"type": "result", "data": {"lines": []}},
    ]
    assert still_locked is False


def test_releases_lock_when_job_raises() -> None:
    """A job that raises mid-stream must still release the lock, otherwise
    one failed request would wedge every later one. The exception
    propagates so the caller's error handling still runs."""

    async def scenario():
        lock = asyncio.Lock()

        async def job():
            if True:
                raise RuntimeError("boom")
            yield {}  # unreachable; makes `job` an async generator

        gen = main._serialized_gpu_stream(lock, job)
        error: Exception | None = None
        try:
            async for _ in gen:
                pass
        except RuntimeError as exc:
            error = exc
        return error, lock.locked()

    error, still_locked = asyncio.run(scenario())
    assert isinstance(error, RuntimeError)
    assert still_locked is False


# ---------- _pump_with_heartbeat ----------------------------------------


def _iter_next(values):
    """A next-item factory over `values`: each call returns a fresh
    awaitable yielding the next value; raises StopAsyncIteration when
    exhausted, matching an async iterator's `__anext__`."""
    it = iter(values)

    async def next_item():
        try:
            return next(it)
        except StopIteration:
            raise StopAsyncIteration from None

    return next_item


def test_pump_forwards_items_then_ends_without_heartbeat() -> None:
    """With items always ready, no gap ever hits the heartbeat timeout, so
    the output is exactly the encoded items and nothing else."""

    async def scenario():
        return [
            chunk
            async for chunk in main._pump_with_heartbeat(
                _iter_next([{"type": "running"}, {"type": "result", "data": 1}]),
                main._encode_envelope,
            )
        ]

    out = asyncio.run(scenario())
    assert out == [
        (json.dumps({"type": "running"}) + "\n").encode("utf-8"),
        (json.dumps({"type": "result", "data": 1}) + "\n").encode("utf-8"),
    ]


def test_pump_emits_heartbeat_during_silent_gap_without_dropping_item(
    monkeypatch,
) -> None:
    """A silent gap longer than the heartbeat interval yields a heartbeat
    line; the awaited item is held (shielded) across the timeout and still
    delivered afterwards, never dropped."""
    monkeypatch.setattr(main, "HEARTBEAT_INTERVAL_SECONDS", 0.01)

    async def scenario():
        delivered = asyncio.Event()

        async def next_item():
            if delivered.is_set():
                raise StopAsyncIteration
            await asyncio.sleep(0.05)  # > heartbeat interval
            delivered.set()
            return {"type": "result", "data": 2}

        return [
            chunk
            async for chunk in main._pump_with_heartbeat(
                next_item, main._encode_envelope
            )
        ]

    out = asyncio.run(scenario())
    heartbeat = (json.dumps({"type": "heartbeat"}) + "\n").encode("utf-8")
    assert out[-1] == (json.dumps({"type": "result", "data": 2}) + "\n").encode("utf-8")
    assert heartbeat in out
    assert out.count(heartbeat) >= 1


def test_pump_encode_none_consumes_item_silently() -> None:
    """An `encode` that returns None (internal sentinel like `_done`)
    consumes the item without emitting a line; other items still flow."""

    def encode(value):
        if value.get("type") == "_done":
            return None
        return main._encode_envelope(value)

    async def scenario():
        return [
            chunk
            async for chunk in main._pump_with_heartbeat(
                _iter_next(
                    [{"type": "stage", "n": 1}, {"type": "_done"}, {"type": "result"}]
                ),
                encode,
            )
        ]

    out = asyncio.run(scenario())
    assert out == [
        (json.dumps({"type": "stage", "n": 1}) + "\n").encode("utf-8"),
        (json.dumps({"type": "result"}) + "\n").encode("utf-8"),
    ]


def test_pump_cancels_pending_on_unwind() -> None:
    """If the consumer stops iterating mid-wait (client disconnect), the
    in-flight pending await is cancelled so it doesn't leak. Cancelling the
    outstanding `__anext__()` pull propagates into the generator's `finally`,
    which cancels the shielded pending."""

    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def next_item():
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {"type": "result"}

        gen = main._pump_with_heartbeat(next_item, main._encode_envelope)
        pull = asyncio.ensure_future(gen.__anext__())
        await started.wait()
        pull.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pull
        # Let the cancelled pending settle.
        await asyncio.sleep(0)
        return cancelled.is_set()

    assert asyncio.run(scenario()) is True
