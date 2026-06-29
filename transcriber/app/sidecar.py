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
    adapter = StdioAdapter(build_registry(), stdin=sys.stdin, stdout=sys.stdout)
    asyncio.run(adapter.run())


if __name__ == "__main__":
    main()
