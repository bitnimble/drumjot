"""Shared test setup.

Re-points the Docker-defaulted volume paths (`/outputs`, `/cache`,
`/models`) at tmp dirs BEFORE any test module imports them. Pydantic-
settings resolves these at `Settings()` construction time, which fires
on the first `from app.config import settings`; setting env vars here
(inside conftest, which pytest loads before collecting siblings) makes
sure that resolution lands on writable paths on the dev box.

Importing `app.main` also runs `settings.outputs_dir.mkdir(...)` at
module load, so without these overrides the mere act of collecting an
endpoint test triggers a PermissionError on /outputs.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="drumjot_test_"))
os.environ.setdefault("OUTPUTS_DIR", str(_TEST_ROOT / "outputs"))
os.environ.setdefault("CACHE_DIR", str(_TEST_ROOT / "cache"))
os.environ.setdefault("MODELS_DIR", str(_TEST_ROOT / "models"))
os.environ.setdefault("DEBUG_DIR", str(_TEST_ROOT / "debug"))
