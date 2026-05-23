"""Shared test setup.

Empty for now — earlier versions pointed the bun-bridge env vars at the
repo copy of the now-removed `transcriber/tools/` so the DSL-pathway
tests could exercise the TS parser outside Docker. With the DSL pathway
and bun bridges gone, there is no setup left to do here. Keeping the
file (rather than deleting it) is a pytest convention — auto-discovered
`conftest.py` is the natural extension point for future shared fixtures.
"""
from __future__ import annotations
