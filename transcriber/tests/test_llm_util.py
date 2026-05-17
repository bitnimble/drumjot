"""Tests for the shared LLM utility helpers."""
from __future__ import annotations

from app.pipeline.llm_util import strip_code_fence


def test_returns_plain_text_unchanged() -> None:
    assert strip_code_fence("hello world") == "hello world"


def test_strips_leading_and_trailing_fence_with_lang_tag() -> None:
    text = "```drumjot\n| k . s . |\n```"
    assert strip_code_fence(text) == "| k . s . |"


def test_handles_dashed_language_tag() -> None:
    # The old implementation only recognised a fixed set of tags
    # ('dsl', 'drumjot', 'text', 'json'). 'drumjot-dsl' slipped through.
    text = "```drumjot-dsl\n| k . s . |\n```"
    assert strip_code_fence(text) == "| k . s . |"


def test_tolerates_leading_whitespace_before_fence() -> None:
    text = "   \n```\nbody\n```"
    assert strip_code_fence(text) == "body"


def test_tolerates_trailing_whitespace_after_fence() -> None:
    text = "```\nbody\n```   \n"
    assert strip_code_fence(text) == "body"


def test_strips_json_fence() -> None:
    text = '```json\n[{"x": 1}]\n```'
    assert strip_code_fence(text) == '[{"x": 1}]'


def test_no_fence_just_strips() -> None:
    # Old implementations only returned `text.strip()` when no opening
    # fence was present. Preserve that behaviour.
    assert strip_code_fence("  body  ") == "body"
