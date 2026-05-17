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
from enum import StrEnum
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
from app.pipeline.lint import (
    LintDiagnostic,
    LintError,
    LintResult,
    format_for_prompt,
    lint_jot,
)
from app.pipeline.llm_util import (
    call_messages_with_refusal_retry,
    strip_code_fence,
)
from app.pipeline.score import score_jot

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# Lint pass operates surgically on the bars containing each diagnostic.
# `LINT_CONTEXT_BARS` controls how many bars on either side of an affected
# bar are included in the segment sent to the LLM — context that lets the
# model see the surrounding groove without inflating per-call token cost.
LINT_CONTEXT_BARS = 1


class RefineLevel(StrEnum):
    LINT = "lint"
    MACRO = "macro"
    STRUCTURE = "structure"
    ONSETS = "onsets"
    VELOCITY = "velocity"


# LINT runs first: it fixes deterministic instrument/performance errors
# that would otherwise bias the F1-gated levels (e.g. an invalid `:o` on
# a kick distorts the kick's onset comparison). Subsequent passes assume
# they're operating on a chart that's at least musically well-formed.
LEVEL_ORDER = [
    RefineLevel.LINT,
    RefineLevel.MACRO,
    RefineLevel.STRUCTURE,
    RefineLevel.ONSETS,
    RefineLevel.VELOCITY,
]

LEVEL_MAX_ITERATIONS: dict[RefineLevel, int] = {
    RefineLevel.LINT: 2,
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
    levels: list[RefineLevel] | None = None,
) -> tuple[str, RefinementLog]:
    """Run the multi-level loop over the requested `levels`.

    `stem_audios[pitch] = (audio, sr)` - the per-instrument audio loaded
    once at the top of `/transcribe`, reused for velocity diff. The
    `structure` is the beat-tracker output; we use it for the global
    tempo reference and for `(bar, beat)` annotations in issue notes.

    `levels` is the subset of `LEVEL_ORDER` to run, evaluated in that
    canonical order regardless of input order. `None` (default) runs
    every level — equivalent to the pre-toggle behaviour. Callers that
    want fine-grained control (e.g. lint-only, refine-only) pass an
    explicit list.
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

    # Predicted onset times in `extracted` are bar-relative starting at
    # bar 0 = t=0. `stem_onsets` are absolute audio time. Anchor the two
    # by adding the audio's first-bar start time to every prediction
    # before comparison. Without this every onset misses by the pre-roll
    # and the F1 collapses to ~0.
    time_offset = (
        structure.bars[0].start_time if structure.bars else 0.0
    )

    initial = score_jot(
        extracted,
        stem_onsets,
        time_offset=time_offset,
        structure=structure,
        debug_tag="initial",
    )
    initial_score = initial.onset_f1
    best_dsl = initial_dsl
    best_score = initial_score
    # Cache the extracted form alongside best_dsl so we don't pay another
    # bun subprocess at the top of every iteration. Each accepted
    # candidate updates both fields together; rejection / failure leaves
    # both unchanged.
    best_extracted: ExtractedJot = extracted

    # Honour `levels` but always traverse in canonical LEVEL_ORDER so the
    # lint pass still runs before F1-gated levels regardless of input
    # order. Unknown / unrequested levels are silently dropped.
    if levels is None:
        active_levels = list(LEVEL_ORDER)
    else:
        requested = set(levels)
        active_levels = [lvl for lvl in LEVEL_ORDER if lvl in requested]

    if not active_levels:
        log.info("Refinement called with no active levels; returning initial DSL unchanged.")
        return best_dsl, RefinementLog(
            initial_score=initial_score,
            final_score=initial_score,
            elapsed_seconds=time.perf_counter() - started,
        )

    log.info(
        "Refinement starting. Initial F1=%.4f (per-pitch: %s), "
        "time_offset=%.3fs, levels=[%s]",
        initial_score,
        initial.per_pitch_summary(),
        time_offset,
        ", ".join(lvl.value for lvl in active_levels),
    )

    for level in active_levels:
        max_iter = LEVEL_MAX_ITERATIONS[level]
        for iteration in range(max_iter):
            t_iter = time.perf_counter()
            extracted = best_extracted

            # LINT is its own world: deterministic diagnostics, not the
            # diff/critic/F1 dance the other levels use. Acceptance is
            # gated on reducing the error count rather than on F1.
            if level == RefineLevel.LINT:
                outcome = _run_lint_iteration(
                    best_dsl=best_dsl,
                    best_extracted=best_extracted,
                    best_score=best_score,
                    stem_onsets=stem_onsets,
                    structure=structure,
                    time_offset=time_offset,
                    iteration=iteration,
                    t_iter=t_iter,
                )
                iterations.append(outcome.log)
                if outcome.accepted:
                    best_dsl = outcome.new_dsl
                    best_extracted = outcome.new_extracted
                    best_score = outcome.new_score
                if outcome.stop_level:
                    break
                continue

            issues = _compute_issues_for_level(
                level, extracted, stem_onsets, stem_audios, structure
            )
            if not issues:
                log.info("No issues at %s/%d; advancing", level.value, iteration)
                break

            triaged = triage_issues(issues, level.value, max_issues=25)
            generator_purpose = f"refine_{level.value}_iter{iteration}"
            try:
                candidate_dsl = _generator_revise(
                    best_dsl, triaged, level, debug_purpose=generator_purpose,
                )
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
                        best_dsl, triaged, level,
                        parse_error=str(exc),
                        debug_purpose=f"{generator_purpose}__parse_retry",
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

            cand_score_obj = score_jot(
                cand_extracted,
                stem_onsets,
                time_offset=time_offset,
                structure=structure,
                debug_tag=f"{level.value}_iter{iteration}",
            )
            cand_score = cand_score_obj.onset_f1
            accept = cand_score > best_score
            iterations.append(IterationLog(
                level=level.value,
                iteration=iteration,
                issues_detected=len(issues),
                issues_sent_to_llm=len(triaged),
                score_before=best_score,
                score_after=cand_score,
                accepted=accept,
                note=(
                    f"{time.perf_counter() - t_iter:.1f}s "
                    f"per-pitch: {cand_score_obj.per_pitch_summary()}"
                ),
            ))
            if accept:
                log.info(
                    "Accepted %s/%d: F1 %.4f -> %.4f",
                    level.value, iteration, best_score, cand_score,
                )
                best_dsl = candidate_dsl
                best_score = cand_score
                best_extracted = cand_extracted
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


@dataclass
class _LintIterationOutcome:
    """Bundled result of one LINT-level iteration so the caller can apply
    the new state atomically (or skip entirely on no-op / failure)."""

    log: IterationLog
    accepted: bool
    stop_level: bool
    new_dsl: str = ""
    new_extracted: ExtractedJot | None = None
    new_score: float = 0.0


@dataclass
class _LintSegment:
    """A patchable slice of the DSL.

    A segment is one or more adjacent bars in a single voice that contain
    at least one diagnostic, expanded by `LINT_CONTEXT_BARS` on either
    side for read-only context. Segments don't cross voice boundaries:
    `||`-split voices are patched independently.
    """

    voice_index: int
    # Inclusive bar-index range that actually carries diagnostics.
    first_bar: int
    last_bar: int
    # Inclusive context-expanded bar-index range (sent to the LLM).
    context_first: int
    context_last: int
    # Byte range into the source DSL covering `context_first..context_last`.
    byte_start: int
    byte_end: int
    diagnostics: list[LintDiagnostic]


def _run_lint_iteration(
    *,
    best_dsl: str,
    best_extracted: ExtractedJot,
    best_score: float,
    stem_onsets: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    time_offset: float,
    iteration: int,
    t_iter: float,
) -> _LintIterationOutcome:
    """Run one lint pass: lint -> patch each affected segment surgically.

    For each iteration we:
      1. Lint the current best DSL.
      2. Group diagnostics into segments (voice + bar window + context).
      3. Ask the LLM to rewrite each segment individually, sending only
         that segment's text + its diagnostics + the audio onsets in the
         affected bars. This bounds the blast radius and keeps per-call
         token cost low.
      4. Apply patches right-to-left so earlier byte offsets stay valid.
      5. Re-lint the patched DSL and accept if errors decreased.
    """

    def _bail(note: str) -> _LintIterationOutcome:
        return _LintIterationOutcome(
            log=IterationLog(
                level=RefineLevel.LINT.value,
                iteration=iteration,
                issues_detected=0,
                issues_sent_to_llm=0,
                score_before=best_score,
                score_after=best_score,
                accepted=False,
                note=note,
            ),
            accepted=False,
            stop_level=True,
        )

    try:
        before = lint_jot(best_dsl)
    except LintError as exc:
        log.warning("Lint bridge failed before lint/%d: %s", iteration, exc)
        return _bail(f"lint bridge failed: {exc}")

    if not before.has_any:
        log.info("No lint issues at lint/%d; advancing", iteration)
        return _bail("no diagnostics")

    log.info(
        "Lint pass iteration %d: %d errors, %d warnings",
        iteration, before.errors, before.warnings,
    )

    segments = _build_lint_segments(before, best_dsl)
    # Diagnostics without bar/voice info (e.g. emitted on hand-built jots)
    # don't fit the surgical model — log them but proceed with whatever
    # segments are addressable. If NONE are addressable, bail.
    unaddressable = [
        d for d in before.diagnostics
        if d.bar_index is None or d.voice_index is None
    ]
    if unaddressable:
        log.info(
            "Lint iteration %d: skipping %d diagnostic(s) without bar info",
            iteration, len(unaddressable),
        )
    if not segments:
        log.info("Lint iteration %d: no surgically-patchable diagnostics", iteration)
        return _bail("no patchable segments")

    # Apply per-segment patches right-to-left. Each segment's prompt sees
    # the ORIGINAL bar text (so it's not biased by upstream patches in
    # the same iteration); the cascade lands on the rolling output DSL.
    candidate_dsl = best_dsl
    sent_to_llm = 0
    last_exc: Exception | None = None
    for seg in segments:
        segment_text = best_dsl[seg.byte_start:seg.byte_end]
        # Newlines before the segment = line number of segment line 1
        # in the full DSL, minus one. Pass through so diagnostic
        # positions render segment-relative — otherwise the LLM sees
        # `at line 8:5` next to a snippet that starts at line 1.
        segment_line_offset = best_dsl.count("\n", 0, seg.byte_start)
        onset_context = _build_lint_onset_context_for_bars(
            stem_onsets, structure,
            voice_index=seg.voice_index,
            bar_indices=range(seg.first_bar, seg.last_bar + 1),
        )
        purpose = (
            f"lint_iter{iteration}_v{seg.voice_index}_b{seg.first_bar}-{seg.last_bar}"
        )
        try:
            patched_text = _lint_generator_revise_segment(
                segment_text=segment_text,
                diagnostics=seg.diagnostics,
                onset_context=onset_context,
                segment_line_offset=segment_line_offset,
                debug_purpose=purpose,
            )
        except Exception as exc:
            last_exc = exc
            log.warning(
                "Lint segment patch failed at iteration %d, voice %d bars %d-%d: %s",
                iteration, seg.voice_index, seg.first_bar, seg.last_bar, exc,
            )
            continue
        candidate_dsl = candidate_dsl[:seg.byte_start] + patched_text + candidate_dsl[seg.byte_end:]
        sent_to_llm += len(seg.diagnostics)

    if candidate_dsl == best_dsl:
        # Every per-segment call failed; nothing was actually changed.
        return _LintIterationOutcome(
            log=IterationLog(
                level=RefineLevel.LINT.value,
                iteration=iteration,
                issues_detected=len(before.diagnostics),
                issues_sent_to_llm=sent_to_llm,
                score_before=best_score,
                score_after=best_score,
                accepted=False,
                note=f"all segment patches failed (last: {last_exc})",
            ),
            accepted=False,
            stop_level=True,
        )

    # Validate the cascaded patches by re-parsing + re-linting the result.
    try:
        cand_extracted = extract_jot(candidate_dsl)
    except JotParseError as exc:
        log.info(
            "Lint candidate didn't parse at iteration %d: %s",
            iteration, exc,
        )
        return _LintIterationOutcome(
            log=IterationLog(
                level=RefineLevel.LINT.value,
                iteration=iteration,
                issues_detected=len(before.diagnostics),
                issues_sent_to_llm=sent_to_llm,
                score_before=best_score,
                score_after=best_score,
                accepted=False,
                note=f"parse failed: {exc}",
            ),
            accepted=False,
            stop_level=False,
        )

    try:
        after = lint_jot(candidate_dsl)
    except LintError as exc:
        log.warning(
            "Lint bridge failed on candidate at iteration %d: %s", iteration, exc,
        )
        return _LintIterationOutcome(
            log=IterationLog(
                level=RefineLevel.LINT.value,
                iteration=iteration,
                issues_detected=len(before.diagnostics),
                issues_sent_to_llm=sent_to_llm,
                score_before=best_score,
                score_after=best_score,
                accepted=False,
                note=f"lint candidate bridge failed: {exc}",
            ),
            accepted=False,
            stop_level=True,
        )

    accept = after.errors < before.errors
    cand_score_obj = score_jot(
        cand_extracted,
        stem_onsets,
        time_offset=time_offset,
        structure=structure,
        debug_tag=f"lint_iter{iteration}",
    )
    cand_score = cand_score_obj.onset_f1
    note = (
        f"{time.perf_counter() - t_iter:.1f}s; "
        f"{len(segments)} segments, errors {before.errors}->{after.errors}, "
        f"warnings {before.warnings}->{after.warnings}; "
        f"per-pitch: {cand_score_obj.per_pitch_summary()}"
    )
    stop = (not accept) or after.errors == 0
    return _LintIterationOutcome(
        log=IterationLog(
            level=RefineLevel.LINT.value,
            iteration=iteration,
            issues_detected=len(before.diagnostics),
            issues_sent_to_llm=sent_to_llm,
            score_before=best_score,
            score_after=cand_score if accept else best_score,
            accepted=accept,
            note=note,
        ),
        accepted=accept,
        stop_level=stop,
        new_dsl=candidate_dsl if accept else "",
        new_extracted=cand_extracted if accept else None,
        new_score=cand_score if accept else best_score,
    )


def _build_lint_segments(
    lint_result: LintResult,
    dsl: str,
) -> list[_LintSegment]:
    """Group diagnostics into per-voice patchable segments.

    Two diagnostics merge into one segment if their context-expanded
    bar ranges touch (within the same voice). Diagnostics with no
    voice/bar info, or whose voice/bar lookup misses, are silently
    skipped — they're reported separately by the caller.
    """
    # voice_index -> list of (bar_index, diagnostic).
    by_voice: dict[int, list[tuple[int, LintDiagnostic]]] = {}
    for d in lint_result.diagnostics:
        if d.voice_index is None or d.bar_index is None:
            continue
        if d.voice_index < 0 or d.bar_index < 0:
            continue
        rng = lint_result.bar_range(d.voice_index, d.bar_index)
        if rng is None:
            continue
        by_voice.setdefault(d.voice_index, []).append((d.bar_index, d))

    segments: list[_LintSegment] = []
    for voice_index, entries in by_voice.items():
        bars_for_voice = lint_result.bars[voice_index]
        # Sort by bar index; group adjacent (after context expansion).
        entries.sort(key=lambda e: e[0])
        cur_first = entries[0][0]
        cur_last = entries[0][0]
        cur_diags: list[LintDiagnostic] = [entries[0][1]]
        groups: list[tuple[int, int, list[LintDiagnostic]]] = []
        for bar_idx, diag in entries[1:]:
            # Merge if the next diagnostic's expanded window touches or
            # overlaps the current group's expanded window. The "+1"
            # captures touching windows (e.g. context-last of group A is
            # bar 4 and context-first of next diag is bar 5 — still worth
            # merging since the prompt would be near-identical).
            if bar_idx - LINT_CONTEXT_BARS <= cur_last + LINT_CONTEXT_BARS + 1:
                if bar_idx > cur_last:
                    cur_last = bar_idx
                cur_diags.append(diag)
            else:
                groups.append((cur_first, cur_last, cur_diags))
                cur_first = bar_idx
                cur_last = bar_idx
                cur_diags = [diag]
        groups.append((cur_first, cur_last, cur_diags))

        for first, last, diags in groups:
            context_first = max(0, first - LINT_CONTEXT_BARS)
            context_last = min(len(bars_for_voice) - 1, last + LINT_CONTEXT_BARS)
            byte_start = bars_for_voice[context_first].start
            byte_end = bars_for_voice[context_last].end
            if byte_end <= byte_start:
                # Skip degenerate (e.g. zero-length) ranges defensively.
                continue
            if byte_end > len(dsl):
                byte_end = len(dsl)
            segments.append(_LintSegment(
                voice_index=voice_index,
                first_bar=first,
                last_bar=last,
                context_first=context_first,
                context_last=context_last,
                byte_start=byte_start,
                byte_end=byte_end,
                diagnostics=diags,
            ))

    # Sort right-to-left so applying patches in order doesn't invalidate
    # byte offsets of segments we haven't touched yet.
    segments.sort(key=lambda s: s.byte_start, reverse=True)
    return segments


def _lint_generator_revise_segment(
    *,
    segment_text: str,
    diagnostics: list[LintDiagnostic],
    onset_context: str,
    segment_line_offset: int = 0,
    parse_error: str | None = None,
    debug_purpose: str = "lint_segment",
) -> str:
    """Ask the LLM to rewrite a single segment with the given diagnostics fixed.

    The model sees only the segment text plus the relevant diagnostics
    and audio onsets — not the whole Jot. Output is the corrected
    segment text (same bar count, same opening/closing structure),
    which the caller splices back into the full DSL.
    """
    template_path = PROMPT_DIR / "refine_lint_segment.md"
    template = template_path.read_text(encoding="utf-8")

    parse_hint = ""
    if parse_error:
        parse_hint = (
            "\nYour previous response failed to parse correctly: "
            f"{parse_error}\nProduce a valid Drumjot DSL fragment this time.\n"
        )

    # Reuse the multi-line diagnostic formatter but feed it only this
    # segment's diagnostics rather than the whole result.
    diag_payload = LintResult(
        diagnostics=diagnostics,
        errors=sum(1 for d in diagnostics if d.severity == "error"),
        warnings=sum(1 for d in diagnostics if d.severity == "warning"),
        bars=[],
    )

    prompt = (
        template
        .replace("{SEGMENT}", segment_text)
        .replace(
            "{LINT_DIAGNOSTICS}",
            format_for_prompt(diag_payload, line_offset=segment_line_offset),
        )
        .replace("{ONSET_CONTEXT}", onset_context or "(no audio context available)")
        .replace("{PARSE_ERROR_HINT}", parse_hint)
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = call_messages_with_refusal_retry(
        client,
        {
            "model": settings.llm_model,
            "max_tokens": settings.llm_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        base_prompt=prompt,
        purpose=debug_purpose,
    )
    text = "".join(b.text for b in response.content if hasattr(b, "text")).strip()
    return strip_code_fence(text)


def _build_lint_onset_context_for_bars(
    stem_onsets: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    *,
    voice_index: int,
    bar_indices,
) -> str:
    """Render audio onsets for a specific set of bars (single segment).

    Used by the per-segment patcher; only the affected bars are included,
    not the surrounding context bars (which the LLM sees in the segment
    text itself).
    """
    blocks: list[str] = []
    for bar_idx in bar_indices:
        if bar_idx < 0 or bar_idx >= len(structure.bars):
            continue
        bar = structure.bars[bar_idx]
        header = (
            f"### Bar {bar_idx} (time {bar.start_time:.2f}s - {bar.end_time:.2f}s, "
            f"{bar.time_signature[0]}/{bar.time_signature[1]}, "
            f"{bar.tempo_bpm:.1f} BPM, feel={bar.feel})"
        )
        lines = [header]
        any_onset = False
        for pitch in sorted(stem_onsets):
            in_bar = [
                c for c in stem_onsets[pitch]
                if bar.start_time <= c.time < bar.end_time
            ]
            if not in_bar:
                continue
            any_onset = True
            entries = " ".join(
                f"({c.beat_in_bar:.3f},{c.strength:.2f})" for c in in_bar
            )
            lines.append(f"  {pitch}: {entries}")
        if not any_onset:
            lines.append("  (no onsets detected in this bar)")
        blocks.append("\n".join(lines))
    if not blocks:
        return ""
    # voice_index left in scope so future cross-voice context can grow
    # naturally; currently it's a single-voice prompt.
    return "\n\n".join(blocks)


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
    debug_purpose: str = "refine",
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
    response = call_messages_with_refusal_retry(
        client,
        {
            "model": settings.llm_model,
            "max_tokens": settings.llm_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        base_prompt=prompt,
        purpose=debug_purpose,
    )
    text = "".join(b.text for b in response.content if hasattr(b, "text")).strip()
    return strip_code_fence(text)
