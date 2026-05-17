"""Small utilities shared across LLM-facing pipeline modules.

Originally just `strip_code_fence`; now also hosts the content-filter
refusal-retry wrapper, which every Anthropic call in the pipeline
should funnel through so a single block doesn't crash the whole
request.
"""
from __future__ import annotations

import logging
import re

import anthropic

from app.debug import current_debug_sink

log = logging.getLogger(__name__)

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


# Hardened directive appended verbatim to the user prompt on retry when
# the first call hit Anthropic's content-filter / refusal stop reason.
# The goal is to push the model toward emitting ONLY the DSL syntax (or
# the requested non-Jot artefact for the critic path), with no titles /
# proper nouns / narrative names that could trip output filters on
# benign drum audio. We've observed this in practice — see the now-
# removed `pipeline/title.py` for the original workaround that pushed
# title generation entirely off the LLM.
REFUSAL_RETRY_DIRECTIVE = (
    "\n\nIMPORTANT — your previous attempt was blocked by the content "
    "filter. Your output here is purely structured: drum-notation DSL, "
    "JSON, or short structured text. It contains no narrative content. "
    "Do NOT emit a `title:` field, pattern names with real-world / "
    "proper-noun content, comments, or any creative naming. Use only "
    "neutral identifiers (Groove / Verse / Chorus / FillA). Produce "
    "ONLY the structured output requested — no commentary, no markdown, "
    "no titles."
)


def call_messages_with_refusal_retry(
    client: anthropic.Anthropic,
    create_kwargs: dict,
    *,
    base_prompt: str,
    purpose: str | None = None,
) -> anthropic.types.Message:
    """Wrap `messages.create` with a single retry on output-filter refusal.

    Anthropic surfaces a content-policy block in two ways depending on
    the SDK and model version: as a `BadRequestError` whose message
    contains "content filter" / "Output blocked", or as a 200 response
    whose `stop_reason == "refusal"`. Handle both, retrying once with
    `REFUSAL_RETRY_DIRECTIVE` bolted onto the prompt.

    `create_kwargs` must be a single-message-create kwargs dict. The
    retry rebuilds the messages array with the directive appended to
    the original user prompt.

    When a `purpose` is supplied AND a request-scoped DebugSink is
    active, the full hydrated prompt is dumped to the debug folder
    before the call (and again for the retry, with a `__refusal_retry`
    suffix). Skip the dump by passing `purpose=None`.
    """
    _dump_prompt(purpose, create_kwargs, base_prompt)
    try:
        response = client.messages.create(**create_kwargs)
    except anthropic.BadRequestError as exc:
        if not _looks_like_refusal(str(exc)):
            raise
        log.warning(
            "LLM call blocked by content filter on first try; retrying with "
            "hardened directive (request_id=%s)",
            _request_id_from_error(exc),
        )
        return _retry_with_directive(client, create_kwargs, base_prompt, purpose)

    if getattr(response, "stop_reason", None) == "refusal":
        log.warning(
            "LLM call returned stop_reason='refusal'; retrying with hardened "
            "directive (id=%s)",
            getattr(response, "id", "?"),
        )
        return _retry_with_directive(client, create_kwargs, base_prompt, purpose)

    return response


def _retry_with_directive(
    client: anthropic.Anthropic,
    create_kwargs: dict,
    base_prompt: str,
    purpose: str | None,
) -> anthropic.types.Message:
    retry_prompt = base_prompt + REFUSAL_RETRY_DIRECTIVE
    retry_kwargs = dict(create_kwargs)
    retry_kwargs["messages"] = [
        {"role": "user", "content": retry_prompt}
    ]
    retry_purpose = f"{purpose}__refusal_retry" if purpose else None
    _dump_prompt(retry_purpose, retry_kwargs, retry_prompt)
    return client.messages.create(**retry_kwargs)


def _dump_prompt(
    purpose: str | None,
    create_kwargs: dict,
    prompt: str,
) -> None:
    """Best-effort dump of the hydrated prompt to the active DebugSink.

    No-op when `purpose` is None (caller opted out) or when no sink is
    installed (debug persistence disabled). Failures inside the sink
    are already swallowed by `write_text`, so this can't take down the
    LLM call.
    """
    if not purpose:
        return
    sink = current_debug_sink()
    if sink is None:
        return
    extra = {
        "max_tokens": create_kwargs.get("max_tokens"),
        "temperature": create_kwargs.get("temperature"),
    }
    sink.write_llm_prompt(
        purpose=purpose,
        model=str(create_kwargs.get("model", "?")),
        prompt=prompt,
        extra=extra,
    )


def _looks_like_refusal(message: str) -> bool:
    lower = message.lower()
    return (
        "content filter" in lower
        or "output blocked" in lower
        or "content_filter" in lower
        or "refusal" in lower
    )


def _request_id_from_error(exc: Exception) -> str:
    body = getattr(exc, "body", None) or {}
    if isinstance(body, dict):
        rid = body.get("request_id")
        if rid:
            return str(rid)
    return getattr(exc, "request_id", None) or "?"
