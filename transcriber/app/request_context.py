"""Per-request id propagation for log correlation.

Every streaming endpoint (`/transcribe`, `/transcribe/resume`,
`/lyrics/align`) mints a short request id at the top of the handler and
stashes it in a `ContextVar`. The logging format string then renders
that id on every record, so an operator can `grep` a single id and see
the whole request's log trail interleaved across the many loggers the
pipeline touches (`app.pipeline.filter_llm`, `app.pipeline.quantise`,
separation, etc.).

Why a ContextVar (and not, say, a thread-local or an explicit arg):

  * The pipeline runs under `asyncio.to_thread(run_pipeline, ...)`, which
    snapshots the *calling* context with `contextvars.copy_context()` and
    runs the worker inside it. So as long as we `set_request_id(...)` on
    the event-loop side before the `to_thread` hop, the worker thread sees
    the same id with zero plumbing. The same mechanism already carries the
    debug-sink / run-log contextvars (see `app.debug`, `app.run_log`).

  * The LLM fan-out stages (`filter_llm`, `quantise`) submit work onto
    their own `ThreadPoolExecutor`s. Executors do NOT copy contextvars
    into their worker threads, so those log lines would otherwise drop
    back to the `"-"` default. Each of those call sites submits through
    `contextvars.copy_context().run(...)` so the id (and the existing
    debug-sink / run-log contextvars) ride along, see the comments at
    each submit site.

The `RequestIdLogFilter` is what bridges the contextvar to the log
record: it stamps `record.request_id` on its way through the single
StreamHandler installed in `main.py`. Because that handler's format
string references `%(request_id)s`, the filter MUST be attached to that
handler (a record lacking the attribute would raise during formatting);
the filter always sets it, so it never does.
"""
from __future__ import annotations

import contextvars
import logging
import uuid

# Default `"-"` (rather than empty) so a log line emitted outside any
# request context (startup, health probes, the api-role worker) renders
# as `[-]` instead of a confusing blank, and so the format string never
# trips over a missing value.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def set_request_id(request_id: str) -> None:
    """Bind `request_id` to the current context.

    Intentionally fire-and-forget (no reset token returned): each request
    runs in its own task / `copy_context()` snapshot, so there's nothing
    to restore, the binding dies with the context.
    """
    request_id_var.set(request_id)


def get_request_id() -> str:
    """Return the request id bound to the current context (or `"-"`)."""
    return request_id_var.get()


def new_request_id() -> str:
    """Mint a fresh short request id (8 hex chars). Short enough to read
    in a log line at a glance, wide enough (32 bits) that collisions
    between concurrent in-flight requests are vanishingly unlikely."""
    return uuid.uuid4().hex[:8]


class RequestIdLogFilter(logging.Filter):
    """Stamp the current request id onto every record passing through.

    Attached to the StreamHandler in `main.py` whose format string
    references `%(request_id)s`. Always sets the attribute (and returns
    True to keep the record), so the formatter never sees a record missing
    it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True
