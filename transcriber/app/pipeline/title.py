"""Inject a `title:` field into the Drumjot DSL's global metadata block.

The few-shot examples in `prompts/examples.md` deliberately don't include
a title - LLM-invented titles occasionally tripped Anthropic's output
content filter on benign drum audio. Instead we derive the title
deterministically from the uploaded audio filename in this
post-processing step, after all LLM work is done.

The global metadata block is the first `{{ ... }}` at the top of the
DSL. We insert (or replace) a `title: "..."` field inside that block.
If no `{{ ... }}` block exists - shouldn't happen given the prompt rules
but worth surviving rather than crashing - we return the DSL unchanged
on the assumption that downstream parsing will fail anyway and the user
will see a clearer error.
"""
from __future__ import annotations

import re
from pathlib import Path

# `(.*?)` is non-greedy so the first `}}` ends the block. Inline metadata
# blocks (per-bar `{{ bpm: ... }}` / `{{ time: ... }}` overrides) use the
# same `{{ ... }}` delimiter; we only touch the first match.
_METADATA_BLOCK = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)
# Strings in the DSL allow `\"` and `\\` escapes, so the body of an
# existing title may contain escaped quotes. Match accordingly.
_EXISTING_TITLE = re.compile(r'\btitle\s*:\s*"(?:[^"\\]|\\.)*"\s*,?\s*')


def title_from_filename(filename: str | None) -> str | None:
    """Strip the extension off `filename` and clean it up for use as a title.

    Returns None when `filename` is None/empty or yields an empty stem so
    callers can skip injection entirely rather than emit `title: ""`.
    """
    if not filename:
        return None
    stem = Path(filename).stem.strip()
    return stem or None


def inject_title(dsl: str, title: str | None) -> str:
    """Insert a `title: "..."` field into the first `{{...}}` metadata block.

    If a `title:` field already exists in that block (e.g. the model
    emitted one despite the few-shot examples no longer showing one),
    the existing field is replaced. Passing `title=None` returns `dsl`
    unchanged.
    """
    if title is None:
        return dsl

    match = _METADATA_BLOCK.search(dsl)
    if not match:
        return dsl

    inner = _EXISTING_TITLE.sub("", match.group(1))
    escaped = title.replace("\\", "\\\\").replace('"', '\\"')
    new_inner = f' title: "{escaped}", {inner.lstrip()}'
    return dsl[: match.start()] + "{{" + new_inner + "}}" + dsl[match.end() :]
