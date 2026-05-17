"""Small utilities shared across LLM-facing pipeline modules.

Currently just `strip_code_fence`, which used to live (with subtle
variations) in `llm.py`, `refine.py` and `critic.py`. Keeping one
implementation here means any future improvement to fence handling
benefits every caller.
"""
from __future__ import annotations

import re

# Recognise an opening triple-backtick fence with an optional language tag
# like "drumjot", "drumjot-dsl", "ts", "json", etc. The language tag is
# any run of letters / digits / dashes / underscores, ended by newline.
_OPENING_FENCE = re.compile(r"^\s*```[A-Za-z0-9_\-]*\s*\n?")
_CLOSING_FENCE = re.compile(r"\n?\s*```\s*$")


def strip_code_fence(text: str) -> str:
    """Remove a single leading and trailing triple-backtick fence if present.

    This is forgiving:
    - Leading whitespace before the opening fence is allowed.
    - Any language tag (`drumjot`, `drumjot-dsl`, `ts`, `json`, ...) is
      consumed along with the optional trailing newline.
    - Trailing whitespace after the closing fence is allowed.

    No-op if there's no opening fence; in that case the returned string is
    `text.strip()`.
    """
    body = _OPENING_FENCE.sub("", text, count=1)
    body = _CLOSING_FENCE.sub("", body, count=1)
    return body.strip()
