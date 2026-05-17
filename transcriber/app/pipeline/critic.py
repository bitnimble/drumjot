"""Cheap LLM that triages the deterministic issue list, groups related
issues and adds musical context for the generator pass.

This is the "second-LLM critic" pattern: a small/cheap model (Claude
Haiku by default) ranks and filters issues so the expensive generator
(Opus) only has to consider the most likely-real, highest-impact ones.

The critic uses Anthropic's tool-use / structured-output channel rather
than free-form JSON text: we declare a `report_triaged_issues` tool with
a JSON Schema for the result shape and force the model to call it via
`tool_choice`. Anthropic validates the arguments against the schema
server-side, so we never see malformed JSON (trailing commas, missing
delimiters, truncations, etc. — all common Haiku failure modes that
used to drop us into the deterministic fallback).

If no `critic_model` is configured we deterministically rank by
confidence and return the top-K — the loop still works, just without
the musical-context summarisation step.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anthropic

from app.config import settings
from app.pipeline.diff import Issue
from app.pipeline.llm_util import call_messages_with_refusal_retry

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# Hard cap on issues sent to the critic. On dense tracks `diff_onsets`
# can emit thousands of mismatches; pasting all of them blows
# Anthropic's 200K-token context window. The critic's job is musical
# re-ranking + grouping of the highest-impact items, so the tail (low
# confidence) is unlikely to influence the kept set anyway — feeding
# only the top-200 by confidence preserves the signal at a fraction of
# the token cost. The critic still filters this down to `max_issues`
# (default 25).
MAX_CRITIC_INPUT_ISSUES = 200

# Char-budget safety net for the assembled critic prompt. Anthropic's
# context is 200K tokens ≈ 800K chars at the canonical 4-chars/token
# heuristic; we cap at 600K so the system prompt, tool schema, and the
# model's own response all fit comfortably alongside our input. If
# `MAX_CRITIC_INPUT_ISSUES` issues still blow this (extremely long
# `notes` fields, etc.), we drop the lowest-confidence ones until
# under the limit.
MAX_CRITIC_PROMPT_CHARS = 600_000

# Cap on a single issue's `notes` field to keep one chatty diff entry
# from monopolising the prompt. Drops are marked with an ellipsis so
# the critic can see the truncation happened.
MAX_NOTE_CHARS = 240

# Issue.type's allowed values. Kept in sync with `IssueType` in `diff.py`;
# the enum constraint here is what lets Anthropic reject hallucinated
# values like "missing_kick" server-side before they reach our parser.
_ISSUE_TYPES = [
    "missing_onset",
    "extra_onset",
    "velocity_mismatch",
    "tempo_mismatch",
    "time_sig_mismatch",
    "structure_refactor",
]

# JSON Schema for a single triaged issue. Anthropic enforces required-
# field presence and `enum` membership; optional fields (e.g. `time`,
# `expected_bpm`) are simply omitted by the model when not applicable.
_ISSUE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": _ISSUE_TYPES,
            "description": "Issue category.",
        },
        "pitch": {
            "type": "string",
            "description": "Single-letter Drumjot pitch (k, s, h, c, d, t, ...).",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "How confident you are this issue is real and worth fixing. "
                "Lower for grouped / less-certain issues."
            ),
        },
        "notes": {
            "type": "string",
            "description": (
                "Short, generator-facing note. When grouping multiple onsets, "
                "preserve their `(bar, beat)` positions as a comma-separated list."
            ),
        },
        "time": {
            "type": "number",
            "description": "Optional absolute audio-time anchor in seconds.",
        },
        "expected_velocity": {"type": "integer"},
        "current_velocity": {"type": "integer"},
        "expected_bpm": {"type": "number"},
        "current_bpm": {"type": "number"},
    },
    "required": ["type", "pitch", "confidence", "notes"],
    "additionalProperties": False,
}

_TRIAGE_TOOL: dict[str, Any] = {
    "name": "report_triaged_issues",
    "description": (
        "Return the prioritised subset of issues that should be fixed at "
        "this refinement level. Drop false positives, group related issues, "
        "rank by musical importance. Keep at most max_issues items."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": _ISSUE_INPUT_SCHEMA,
                "description": (
                    "Triaged issues in priority order (highest-impact first)."
                ),
            },
        },
        "required": ["issues"],
        "additionalProperties": False,
    },
}


def triage_issues(
    issues: list[Issue],
    level: str,
    max_issues: int = 25,
) -> list[Issue]:
    if not issues:
        return []
    if not settings.critic_model or not settings.anthropic_api_key:
        return _deterministic_topk(issues, max_issues)

    # 1) Confidence-based count cap. Sorts and truncates deterministically
    #    so the critic always sees the same top-N for the same input.
    if len(issues) > MAX_CRITIC_INPUT_ISSUES:
        log.info(
            "Critic input truncated %d -> %d issues by confidence at level=%s "
            "(cap = MAX_CRITIC_INPUT_ISSUES)",
            len(issues), MAX_CRITIC_INPUT_ISSUES, level,
        )
        issues_for_critic = _deterministic_topk(issues, MAX_CRITIC_INPUT_ISSUES)
    else:
        issues_for_critic = sorted(issues, key=lambda i: -i.confidence)

    payload = [_issue_to_dict(i) for i in issues_for_critic]
    template = (PROMPT_DIR / "critic.md").read_text(encoding="utf-8")

    def assemble(p: list[dict[str, Any]]) -> str:
        return (
            template
            .replace("{LEVEL}", level)
            .replace("{MAX_ISSUES}", str(max_issues))
            .replace("{ISSUES_JSON}", _format_issues_for_prompt(p))
        )

    prompt = assemble(payload)

    # 2) Char-budget safety net. If `notes` fields are exceptionally long,
    #    the count cap alone might not be enough. Drop the lowest-
    #    confidence tail until under the budget. payload is already
    #    confidence-sorted desc so list[:-k] = drop lowest k.
    while len(prompt) > MAX_CRITIC_PROMPT_CHARS and len(payload) > 5:
        # Drop ~10% of the tail per iteration; converges quickly even
        # when over-budget by a large factor.
        drop = max(1, len(payload) // 10)
        payload = payload[: max(5, len(payload) - drop)]
        prompt = assemble(payload)

    if len(payload) < len(issues_for_critic):
        log.info(
            "Critic prompt char budget %d trimmed to %d issues at level=%s",
            MAX_CRITIC_PROMPT_CHARS, len(payload), level,
        )

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = call_messages_with_refusal_retry(
            client,
            {
                "model": settings.critic_model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [_TRIAGE_TOOL],
                # Force the model to call the tool rather than free-form
                # text reply. Eliminates the "model emitted prose around
                # the JSON" failure mode entirely.
                "tool_choice": {"type": "tool", "name": _TRIAGE_TOOL["name"]},
            },
            base_prompt=prompt,
            purpose=f"critic_{level}",
        )

        tool_block = _find_tool_use(response, _TRIAGE_TOOL["name"])
        if tool_block is None:
            log.warning(
                "Critic at level=%s returned no tool_use block; falling back",
                level,
            )
            return _deterministic_topk(issues, max_issues)

        triaged_raw = tool_block.input.get("issues", [])
        if not isinstance(triaged_raw, list):
            log.warning(
                "Critic tool_use returned non-list issues (%s); falling back",
                type(triaged_raw).__name__,
            )
            return _deterministic_topk(issues, max_issues)

        triaged = [_dict_to_issue(d) for d in triaged_raw][:max_issues]
        log.info(
            "Critic selected %d of %d candidate issues to forward to the "
            "generator at level=%s (the rest are deferred / dropped as "
            "lower-priority for this pass)",
            len(triaged), len(issues), level,
        )
        return triaged
    except Exception as exc:
        log.warning(
            "Critic call failed (%s); falling back to confidence ranking",
            exc,
        )
        return _deterministic_topk(issues, max_issues)


def _find_tool_use(
    response: anthropic.types.Message, tool_name: str
) -> Any | None:
    """Locate the first `tool_use` content block matching `tool_name`.

    Returns the SDK's tool-use block (with `.input` dict) or None when
    the model produced only text / no matching tool call. The shape of
    a tool_use block is `{ type: "tool_use", id, name, input }`.
    """
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != tool_name:
            continue
        return block
    return None


def _format_issues_for_prompt(payload: list[dict[str, Any]]) -> str:
    """Render the input issues as a readable bulleted list for the prompt.

    We previously dumped JSON here; switching to a flat key-value list
    keeps the prompt small and avoids any chance of the Haiku model
    echoing nested JSON back as its tool input (the tool-use channel
    handles structured output now — the prompt is just for context).
    Long `notes` fields are clipped at `MAX_NOTE_CHARS` with an ellipsis
    so a single chatty diff entry can't monopolise the prompt.
    """
    if not payload:
        return "(no issues)"
    lines: list[str] = []
    for i, item in enumerate(payload, start=1):
        parts = [f"#{i} type={item.get('type', '?')}"]
        for key in (
            "pitch",
            "confidence",
            "time",
            "expected_velocity",
            "current_velocity",
            "expected_bpm",
            "current_bpm",
        ):
            if key in item and item[key] is not None:
                parts.append(f"{key}={item[key]}")
        notes = item.get("notes")
        if notes:
            text = str(notes)
            if len(text) > MAX_NOTE_CHARS:
                text = text[: MAX_NOTE_CHARS - 1] + "…"
            parts.append(f"notes={text!r}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _deterministic_topk(issues: list[Issue], k: int) -> list[Issue]:
    return sorted(issues, key=lambda i: -i.confidence)[:k]


def _issue_to_dict(i: Issue) -> dict[str, Any]:
    d = asdict(i)
    # Drop None fields to keep the prompt short
    return {k: v for k, v in d.items() if v is not None}


def _dict_to_issue(d: dict[str, Any]) -> Issue:
    return Issue(
        type=d.get("type", "missing_onset"),
        pitch=d.get("pitch", ""),
        confidence=float(d.get("confidence", 0.5)),
        notes=d.get("notes", ""),
        time=d.get("time"),
        expected_velocity=d.get("expected_velocity"),
        current_velocity=d.get("current_velocity"),
        expected_bpm=d.get("expected_bpm"),
        current_bpm=d.get("current_bpm"),
    )
