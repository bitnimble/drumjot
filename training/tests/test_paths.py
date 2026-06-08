from pathlib import Path

import pytest

import drumjot_training.paths as paths


def test_env_var_overrides_config():
    p = paths.resolve_path("egmd", env={"DRUMJOT_EGMD": "/x"}, config={"egmd": "/y"})
    assert p == Path("/x")


def test_config_used_when_no_env():
    p = paths.resolve_path("egmd", env={}, config={"egmd": "/y"})
    assert p == Path("/y")


def test_missing_path_raises_with_hint():
    with pytest.raises(KeyError):
        paths.resolve_path("egmd", env={}, config={})


def test_load_config_reads_toml(tmp_path):
    f = tmp_path / "data_paths.toml"
    f.write_text('egmd = "/data/egmd"\n')
    cfg = paths.load_config(f)
    assert cfg["egmd"] == "/data/egmd"


def test_load_config_missing_file_is_empty(tmp_path):
    assert paths.load_config(tmp_path / "nope.toml") == {}
