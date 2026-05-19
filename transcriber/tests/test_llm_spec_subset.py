"""The per-instrument prompt must never expose multi-voice syntax.

`_load_spec_subset` strips the `||` ("Global simultaneity") section
from the canonical SPEC.md at load time so the single-instrument model
can't learn or emit `||`. SPEC.md itself stays untouched on disk (it's
shared with the frontend renderer + bun bridge).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.pipeline import llm

_REPO_SPEC = Path(__file__).resolve().parents[2] / "SPEC.md"


@pytest.fixture
def _real_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the loader at the repo's real SPEC.md (the default
    `/app/SPEC.md` only exists inside the container)."""
    monkeypatch.setattr(settings, "spec_path", _REPO_SPEC)


def test_spec_subset_removes_global_simultaneity(_real_spec: None) -> None:
    subset = llm._load_spec_subset()
    assert "## Global simultaneity" not in subset
    assert not any(line.strip() == "||" for line in subset.split("\n"))
    assert "| `\\|\\|` | Global simultaneity |" not in subset


def test_spec_subset_keeps_core_grammar(_real_spec: None) -> None:
    subset = llm._load_spec_subset()
    # Groups, weights, polyrhythm-via-`+`-groups, patterns all survive —
    # only the multi-voice operator is removed.
    assert "## Groups" in subset
    assert "## Duration weight" in subset
    assert "(a a a)_4 + (b b b b)_4" in subset
    assert "## Patterns" in subset


def test_instrument_prompt_forbids_multivoice_tokens() -> None:
    template = (
        Path(llm.PROMPT_DIR) / "transcribe_instrument.md"
    ).read_text(encoding="utf-8")
    # The template must explicitly forbid the multi-voice / chord /
    # metadata constructs and carry the per-instrument placeholders.
    assert "{PITCH}" in template
    assert "{INSTRUMENT_NAME}" in template
    assert "`||`" in template  # mentioned only to forbid it
    assert "monophonic" in template


def test_instrument_examples_are_monophonic() -> None:
    examples = (
        Path(llm.PROMPT_DIR) / "examples_instrument.md"
    ).read_text(encoding="utf-8")
    assert "||" not in examples
    # No global metadata block in the few-shot outputs.
    assert "{{" not in examples
