"""Per-request log + stage-timing capture for the debug bundle.

A `RunLog` is attached to the request at the top of `/transcribe` and
`/transcribe/resume` and collects:

  - every log record emitted by `app.*` loggers (with timestamps), so
    the debug bundle reproduces what `docker compose logs` would have
    printed for that run;
  - per-stage start / end / elapsed times, so the operator can see at a
    glance which stage dominated wall-clock.

The capture is wired through a `ContextVar` (matching `debug.DebugSink`'s
pattern) so deep callees can call `current_run_log()` without having the
sink threaded through every signature. The runner records stage timings
explicitly via `record_stage`.

Everything here is best-effort: a malformed log record or a stage that
forgot to call `record_stage` must never fail the actual transcription.
"""
from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime

# The logger every per-request consumer attaches to. All transcriber
# modules use `logging.getLogger(__name__)` with `__name__` starting at
# `app.*`, so attaching to the `app` parent captures the whole tree
# without duplicating the existing root-logger output.
APP_LOGGER_NAME = "app"


@dataclass
class LogEntry:
    """One captured log record, JSON-serialisable verbatim."""

    timestamp: str          # ISO 8601 with millisecond precision
    elapsed_seconds: float  # monotonic seconds since RunLog construction
    level: str
    logger: str
    message: str


@dataclass
class StageTiming:
    """One pipeline stage's start/end/elapsed wall-clock timing."""

    stage: str
    start_timestamp: str
    end_timestamp: str
    elapsed_seconds: float


class _RunLogHandler(logging.Handler):
    """In-memory logging handler that appends formatted records to a RunLog."""

    def __init__(self, run_log: RunLog) -> None:
        super().__init__(level=logging.INFO)
        self._run_log = run_log

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception as exc:  # pragma: no cover - defensive
            message = f"<log-format failed: {exc}>"
        self._run_log._append(  # noqa: SLF001 - intentional internal call
            LogEntry(
                timestamp=_iso_utc(record.created),
                elapsed_seconds=round(record.created - self._run_log.started, 3),
                level=record.levelname,
                logger=record.name,
                message=message,
            )
        )


class RunLog:
    """Captures log records + stage timings for one /transcribe request.

    Construct one per request, call `install()` to attach the handler (and
    `uninstall()` in a `finally`), pump stage timings via `record_stage()`,
    and serialise via `to_dict()` when building the debug bundle.

    Thread-safe enough for FastAPI + a stage running in `asyncio.to_thread`:
    `logging.Handler.emit` is the only path that races with `to_dict`,
    and both take the same lock.
    """

    def __init__(self) -> None:
        self.started = time.time()
        self.started_monotonic = time.perf_counter()
        self.entries: list[LogEntry] = []
        self.stages: list[StageTiming] = []
        self._lock = threading.Lock()
        self._handler: _RunLogHandler | None = None

    # -------------------------------------------------------------- install

    def install(self) -> None:
        """Attach the handler to the `app` logger tree."""
        if self._handler is not None:
            return
        self._handler = _RunLogHandler(self)
        logging.getLogger(APP_LOGGER_NAME).addHandler(self._handler)

    def uninstall(self) -> None:
        """Detach the handler. Safe to call multiple times."""
        if self._handler is None:
            return
        logging.getLogger(APP_LOGGER_NAME).removeHandler(self._handler)
        self._handler = None

    # ------------------------------------------------------------- recording

    def _append(self, entry: LogEntry) -> None:
        with self._lock:
            self.entries.append(entry)

    def record_stage(
        self,
        stage: str,
        start_wall: float,
        end_wall: float,
    ) -> None:
        """Record one stage's wall-clock timings.

        `start_wall` / `end_wall` are `time.time()` epoch seconds; the
        runner measures these around each `_run_stage` call so the
        bundle's `stage_timings` lists every stage that actually ran in
        the order it ran.
        """
        with self._lock:
            self.stages.append(
                StageTiming(
                    stage=stage,
                    start_timestamp=_iso_utc(start_wall),
                    end_timestamp=_iso_utc(end_wall),
                    elapsed_seconds=round(end_wall - start_wall, 3),
                )
            )

    # ------------------------------------------------------------ serialize

    def to_dict(self) -> dict[str, object]:
        with self._lock:
            return {
                "started_at": _iso_utc(self.started),
                "elapsed_seconds": round(time.perf_counter() - self.started_monotonic, 3),
                "stage_timings": [
                    {
                        "stage": s.stage,
                        "start": s.start_timestamp,
                        "end": s.end_timestamp,
                        "elapsed_seconds": s.elapsed_seconds,
                    }
                    for s in self.stages
                ],
                "logs": [
                    {
                        "timestamp": e.timestamp,
                        "elapsed_seconds": e.elapsed_seconds,
                        "level": e.level,
                        "logger": e.logger,
                        "message": e.message,
                    }
                    for e in self.entries
                ],
            }


def _iso_utc(epoch_seconds: float) -> str:
    """`time.time()` / `record.created` → ISO 8601 UTC string, ms precision."""
    return (
        datetime.fromtimestamp(epoch_seconds, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


# Request-scoped current RunLog. Mirrors `debug._CURRENT_DEBUG_SINK` so
# deep callees can record stage timings without the sink threaded
# through their signatures.
_CURRENT_RUN_LOG: ContextVar[RunLog | None] = ContextVar(
    "drumjot_run_log", default=None
)


def current_run_log() -> RunLog | None:
    return _CURRENT_RUN_LOG.get()


def set_current_run_log(run_log: RunLog | None) -> Token:
    return _CURRENT_RUN_LOG.set(run_log)


def reset_current_run_log(token: Token) -> None:
    _CURRENT_RUN_LOG.reset(token)
