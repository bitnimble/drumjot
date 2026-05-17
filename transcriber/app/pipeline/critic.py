"""Cheap LLM that triages the deterministic issue list, groups related
issues and adds musical context for the generator pass.

This is the "second-LLM critic" pattern: a small/cheap model (Claude
Haiku by default) ranks and filters issues so the expensive generator
(Opus) only has to consider the most likely-real, highest-impact ones.

If no `critic_model` is configured we deterministically rank by
confidence and return the top-K - the loop still works, just without the
musical-context summarisation step.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anthropic

from app.config import settings
from app.pipeline.diff import Issue
from app.pipeline.llm_util import strip_code_fence

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def triage_issues(
    issues: list[Issue],
    level: str,
    max_issues: int = 25,
) -> list[Issue]:
    if not issues:
        return []
    if not settings.critic_model or not settings.anthropic_api_key:
        return _deterministic_topk(issues, max_issues)

    payload = [_issue_to_dict(i) for i in issues]
    template = (PROMPT_DIR / "critic.md").read_text(encoding="utf-8")
    prompt = (
        template
        .replace("{LEVEL}", level)
        .replace("{MAX_ISSUES}", str(max_issues))
        .replace("{ISSUES_JSON}", json.dumps(payload, indent=2))
    )

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.critic_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text")).strip()
        text = strip_code_fence(text)
        triaged_raw = json.loads(text)
        triaged = [_dict_to_issue(d) for d in triaged_raw][:max_issues]
        log.info(
            "Critic kept %d/%d issues at level=%s",
            len(triaged), len(issues), level,
        )
        return triaged
    except Exception as exc:
        log.warning(
            "Critic call failed (%s); falling back to confidence ranking",
            exc,
        )
        return _deterministic_topk(issues, max_issues)


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
