"""Smoke test: the /score route is registered and app.main imports cleanly.

The handler logic is covered by test_score_map.py; this just guards the
wiring (import errors, route registration)."""
from __future__ import annotations


def test_score_route_registered() -> None:
    from app.main import app

    paths = {route.path for route in app.routes}
    assert "/score" in paths
