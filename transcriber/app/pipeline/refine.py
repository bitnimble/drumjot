"""Multi-level convergence loop.

After the initial transcription produces a Jot DSL, this module:

  1. Macro pass     -> tempo / time signature corrections (1 iteration)
  2. Structure pass -> pattern factoring (audio-independent; 1 iteration)
  3. Onsets pass    -> missing / extra hits (up to 3 iterations)
  4. Velocity pass  -> dynamics matching (1 iteration)

Each pass:
  - Computes a typed issue list via `pipeline/diff.py`
  - Triages it with the cheap critic LLM (`pipeline/critic.py`)
  - Asks the expensive generator LLM to revise the Jot
  - Validates that the new Jot still parses (retry once on parse error)
  - Scores the new Jot against the source stems
  - Accepts only if score strictly improves

The whole loop is monotone-improving in expectation because of the
score-gated acceptance, and bounded in iterations to a small constant.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import anthropic
import numpy as np

from app.config import settings
from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.critic import triage_issues
from app.pipeline.diff import (
    Issue,
    diff_onsets,
    diff_tempo,
    diff_velocities,
    structure_refactor_hint,
)
from app.pipeline.jot_extract import (
    ExtractedJot,
    JotParseError,
    extract_jot,
)
from app.pipeline.llm_util import strip_code_fence
from app.pipeline.score import score_jot

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class RefineLevel(str, Enum):
    MACRO = "macro"
    STRUCTURE = "structure"
    ONSETS = "onsets"
    VELOCITY = "velocity"


LEVEL_ORDER = [
    RefineLevel.MACRO,
    RefineLevel.STRUCTURE,
    RefineLevel.ONSETS,
    RefineLevel.VELOCITY,
]

LEVEL_MAX_ITERATIONS: dict[RefineLevel, int] = {
    RefineLevel.MACRO: 1,
    RefineLevel.STRUCTURE: 1,
    RefineLevel.ONSETS: 3,
    RefineLevel.VELOCITY: 1,
}


@dataclass
class IterationLog:
    level: str
    iteration: int
    issues_detected: int
    issues_sent_to_llm: int
    score_before: float
    score_after: float
    accepted: bool
    note: str = ""


@dataclass
class RefinementLog:
    initial_score: float
    final_score: float
    elapsed_seconds: float
    iterations: list[IterationLog] = field(default_factory=list)


def refine_jot(
    initial_dsl: str,
    stem_onsets: dict[str, list[OnsetCandidate]],
    stem_audios: dict[str, tuple[np.ndarray, int]],
    structure: BeatStructure,
) -> tuple[str, RefinementLog]:
    """Run the full multi-level loop. Returns (best_dsl, log).

    `stem_audios[pitch] = (audio, sr)` - the per-instrument audio loaded
    once at the top of `/transcribe`, reused for velocity diff. The
    `structure` is the beat-tracker output; we use it for the global
    tempo reference and for `(bar, beat)` annotations in issue notes.
    """
    started = time.perf_counter()
    iterations: list[IterationLog] = []

    try:
        extracted = extract_jot(initial_dsl)
    except JotParseError as exc:
        log.error("Initial DSL failed to parse for refinement: %s", exc)
        return initial_dsl, RefinementLog(
            initial_score=0.0,
            final_score=0.0,
            elapsed_seconds=time.perf_counter() - started,
        )

    initial_score = score_jot(extracted, stem_onsets).onset_f1
    best_dsl = initial_dsl
    best_score = initial_score
    log.info("Refinement starting. Initial F1=%.4f", initial_score)

    for level in LEVEL_ORDER:
        max_iter = LEVEL_MAX_ITERATIONS[level]
        for iteration in range(max_iter):
            t_iter = time.perf_counter()
            try:
                extracted = extract_jot(best_dsl)
            except JotParseError as exc:
                log.warning("Re-parse failed entering %s/%d: %s", level.value, iteration, exc)
                break

            issues = _compute_issues_for_level(
                level, extracted, stem_onsets, stem_audios, structure
            )
            if not issues:
                log.info("No issues at %s/%d; advancing", level.value, iteration)
                break

            triaged = triage_issues(issues, level.value, max_issues=25)
            try:
                candidate_dsl = _generator_revise(best_dsl, triaged, level)
            except Exception as exc:
                log.warning("Generator failed at %s/%d: %s", level.value, iteration, exc)
                iterations.append(IterationLog(
                    level=level.value, iteration=iteration,
                    issues_detected=len(issues), issues_sent_to_llm=len(triaged),
                    score_before=best_score, score_after=best_score, accepted=False,
                    note=f"generator error: {exc}",
                ))
                break

            try:
                cand_extracted = extract_jot(candidate_dsl)
            except JotParseError as exc:
                log.info(
                    "Candidate DSL didn't parse at %s/%d: %s; retrying once",
                    level.value, iteration, exc,
                )
                try:
                    candidate_dsl = _generator_revise(
                        best_dsl, triaged, level, parse_error=str(exc)
                    )
                    cand_extracted = extract_jot(candidate_dsl)
                except Exception as exc2:
                    # Covers both another JotParseError on the retry and
                    # network / API errors from the generator call.
                    iterations.append(IterationLog(
                        level=level.value, iteration=iteration,
                        issues_detected=len(issues), issues_sent_to_llm=len(triaged),
                        score_before=best_score, score_after=best_score, accepted=False,
                        note=f"parse retry failed: {exc2}",
                    ))
                    continue

            cand_score = score_jot(cand_extracted, stem_onsets).onset_f1
            accept = cand_score > best_score
            iterations.append(IterationLog(
                level=level.value,
                iteration=iteration,
                issues_detected=len(issues),
                issues_sent_to_llm=len(triaged),
                score_before=best_score,
                score_after=cand_score,
                accepted=accept,
                note=f"{time.perf_counter() - t_iter:.1f}s",
            ))
            if accept:
                log.info(
                    "Accepted %s/%d: F1 %.4f -> %.4f",
                    level.value, iteration, best_score, cand_score,
                )
                best_dsl = candidate_dsl
                best_score = cand_score
            else:
                log.info(
                    "Rejected %s/%d: F1 %.4f -> %.4f (no improvement; stopping level)",
                    level.value, iteration, best_score, cand_score,
                )
                break  # stop this level; advance to the next

    elapsed = time.perf_counter() - started
    log.info(
        "Refinement complete in %.2fs. F1 %.4f -> %.4f over %d iterations",
        elapsed, initial_score, best_score, len(iterations),
    )
    return best_dsl, RefinementLog(
        initial_score=initial_score,
        final_score=best_score,
        elapsed_seconds=elapsed,
        iterations=iterations,
    )


def _compute_issues_for_level(
    level: RefineLevel,
    extracted: ExtractedJot,
    stem_onsets: dict[str, list[OnsetCandidate]],
    stem_audios: dict[str, tuple[np.ndarray, int]],
    structure: BeatStructure,
) -> list[Issue]:
    if level == RefineLevel.MACRO:
        return diff_tempo(extracted, structure.initial_tempo)
    if level == RefineLevel.STRUCTURE:
        return structure_refactor_hint()
    issues: list[Issue] = []
    if level == RefineLevel.ONSETS:
        for pitch in sorted(set(stem_onsets) | set(extracted.onsets_by_pitch)):
            predicted = extracted.onsets_by_pitch.get(pitch, [])
            actual = stem_onsets.get(pitch, [])
            if not predicted and not actual:
                continue
            issues.extend(diff_onsets(pitch, predicted, actual, structure))
        return issues
    if level == RefineLevel.VELOCITY:
        for pitch, predicted in extracted.onsets_by_pitch.items():
            audio_sr = stem_audios.get(pitch)
            if not audio_sr:
                continue
            audio, sr = audio_sr
            issues.extend(diff_velocities(pitch, predicted, audio, sr, structure))
        return issues
    return issues


def _generator_revise(
    current_dsl: str,
    issues: list[Issue],
    level: RefineLevel,
    parse_error: str | None = None,
) -> str:
    template_path = PROMPT_DIR / f"refine_{level.value}.md"
    template = template_path.read_text(encoding="utf-8")

    payload: list[dict[str, Any]] = []
    for i in issues:
        d: dict[str, Any] = {
            "type": i.type,
            "pitch": i.pitch,
            "confidence": round(i.confidence, 3),
            "notes": i.notes,
        }
        if i.time is not None:
            d["time"] = round(i.time, 3)
        if i.expected_velocity is not None:
            d["expected_velocity"] = i.expected_velocity
        if i.current_velocity is not None:
            d["current_velocity"] = i.current_velocity
        if i.expected_bpm is not None:
            d["expected_bpm"] = round(i.expected_bpm, 2)
        if i.current_bpm is not None:
            d["current_bpm"] = round(i.current_bpm, 2)
        payload.append(d)

    parse_hint = ""
    if parse_error:
        parse_hint = (
            "\nYour previous response failed to parse with: "
            f"{parse_error}\nProduce a syntactically valid Drumjot DSL this time.\n"
        )

    prompt = (
        template
        .replace("{CURRENT_JOT}", current_dsl)
        .replace("{ISSUES_JSON}", json.dumps(payload, indent=2))
        .replace("{PARSE_ERROR_HINT}", parse_hint)
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if hasattr(b, "text")).strip()
    return strip_code_fence(text)
