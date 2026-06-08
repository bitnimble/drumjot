"""Resolve dataset locations from an env var or a local config file.

The data owner is often AFK; datasets land under `/codebox-workspace`
later. Paths come from (priority order) an explicit env var
`DRUMJOT_<KEY>`, then a local TOML config (`training/data_paths.toml`,
git-ignored), so the owner just fills the file in. Nothing here assumes a
dataset is present; it only resolves where it *would* be.

Example `training/data_paths.toml`:

    egmd = "/codebox-workspace/datasets/e-gmd-v1.0.0"
"""
from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path

# Default config file location (repo-relative: training/data_paths.toml).
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "data_paths.toml"


def resolve_path(
    key: str,
    env: Mapping[str, str],
    config: Mapping[str, object],
) -> Path:
    """Resolve dataset `key` to a Path. `DRUMJOT_<KEY>` env var wins, then
    `config[key]`. Raises KeyError (with a fill-in hint) if neither is set."""
    env_name = f"DRUMJOT_{key.upper()}"
    if env_name in env:
        return Path(env[env_name])
    if key in config:
        return Path(str(config[key]))
    raise KeyError(
        f"dataset path {key!r} not set: export {env_name}=... or add "
        f'`{key} = "/path"` to {DEFAULT_CONFIG}'
    )


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Parse the TOML config file, or return {} if it doesn't exist yet."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def dataset_path(key: str) -> Path:
    """Resolve `key` against the real environment + config file."""
    return resolve_path(key, os.environ, load_config())
