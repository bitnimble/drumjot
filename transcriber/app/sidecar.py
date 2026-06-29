"""`python -m app.sidecar` -- the stdio backend the Tauri broker spawns.

stdout carries the control protocol; everything else (logging, library chatter)
goes to stderr so it can't corrupt the frame stream.
"""
from __future__ import annotations

import asyncio
import logging
import sys

from app.comms.runners import build_registry
from app.comms.stdio_adapter import StdioAdapter


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    # The control protocol owns the real stdout. Hand the adapter that stream,
    # then repoint sys.stdout at stderr so a stray print() in any dependency
    # (e.g. adtof_pytorch's weight-load message) can't inject a non-JSON line
    # into the frame stream. The adapter keeps its own reference, so its writes
    # still go to the real stdout.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    adapter = StdioAdapter(build_registry(), stdin=sys.stdin, stdout=protocol_out)
    asyncio.run(adapter.run())


if __name__ == "__main__":
    main()
