"""Tests for `_serialized_gpu_stream`: the NDJSON sequencing primitive
behind the streaming `/lyrics/align` endpoint.

The helper wraps a unit of GPU work behind the process-wide GPU lock and
emits a `{"type": "queued"}` envelope when the lock is already held, so a
client blocked behind another in-flight request can show a wait state
instead of a silent hang. Once it owns the lock it emits
`{"type": "running"}` and then forwards whatever the job yields.

Imports `app.main` (cheap; lyrics_align lazy-loads torch inside its
methods). The helper is pure asyncio, so these drive the async generator
directly via `asyncio.run` rather than pulling in pytest-asyncio.
"""
from __future__ import annotations

import asyncio

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
