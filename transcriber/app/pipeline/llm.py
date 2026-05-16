"""Claude-based translation of candidate onsets to Drumjot DSL.

The LLM does the cognitive work this pipeline doesn't have a great
deterministic answer for: filtering false-positive onsets, detecting
repeating bar patterns, applying accent/ghost tags, choosing whether
to express a bar as a straight grid or a `(...)_N` triplet group, and
deciding when to emit inline `{{ bpm: ... }}` / `{{ time: ... }}`
metadata blocks for tempo / time-signature changes.

Input to the LLM is **per-bar**:
  - bar index, time signature, local tempo, detected feel
  - candidate onsets per pitch, each at a beat-relative position
    (e.g. `(2.333, 9.4)` = "beat 2.333 - one-third into beat 2,
    strength 9.4")

This means the LLM never has to do tempo math itself, and the per-bar
feel hint tells it when to emit triplet/swing notation rather than
straight 1/16.
"""
from __future__ import annotations

import logging
from pathlib import Path

import anthropic

from app.config import settings
from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure, BarInfo
from app.pipeline.jot_extract import JotParseError, extract_jot
from app.pipeline.score import score_jot

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def transcribe_to_jot(
    candidates_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    temperature: float = 0.0,
    parse_error_hint: str | None = None,
) -> str:
    """Single LLM call. Returns the DSL string. Retry / scoring is the
    caller's responsibility - use `transcribe_to_jot_with_self_consistency`
    for the production path.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot call the LLM. "
            "Configure it in transcriber/.env."
        )

    spec_text = _load_spec()
    example_text = _load_examples()
    prompt_template = _load_prompt_template()

    parse_hint = ""
    if parse_error_hint:
        parse_hint = (
            "\nIMPORTANT: A previous attempt failed to parse with this error:\n"
            f"   {parse_error_hint}\n"
            "Make sure the output you produce now parses cleanly.\n"
        )

    bar_blocks = _format_bars(candidates_by_pitch, structure)
    initial_tempo = structure.initial_tempo
    initial_sig = structure.initial_time_signature
    initial_sig_str = f"{initial_sig[0]}/{initial_sig[1]}"

    prompt = (
        prompt_template
        .replace("{SPEC}", spec_text)
        .replace("{EXAMPLES}", example_text)
        .replace("{INITIAL_TEMPO}", f"{initial_tempo:.2f}")
        .replace("{INITIAL_TIME_SIG}", initial_sig_str)
        .replace("{BAR_COUNT}", str(len(structure.bars)))
        .replace(
            "{TEMPO_CHANGES}",
            "yes" if structure.has_tempo_changes else "no",
        )
        .replace(
            "{TIME_SIG_CHANGES}",
            "yes" if structure.has_time_sig_changes else "no",
        )
        .replace("{BARS}", bar_blocks)
    ) + parse_hint

    log.info(
        "Calling LLM model=%s temperature=%.2f prompt_chars=%d bars=%d",
        settings.llm_model, temperature, len(prompt), len(structure.bars),
    )
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    text = "".join(parts).strip()
    return _strip_code_fence(text)


def transcribe_to_jot_with_self_consistency(
    candidates_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    samples: int = 1,
) -> tuple[str, list[float]]:
    """Run `samples` LLM calls at varied temperatures, score each by its
    onset F1 against the source stem onsets, and return the best.

    Returns `(best_dsl, per_sample_scores)`.
    """
    if samples < 1:
        samples = 1

    temperatures = _temperatures_for(samples)
    candidates_dsl: list[str] = []
    for i, temp in enumerate(temperatures):
        try:
            dsl = transcribe_to_jot(
                candidates_by_pitch=candidates_by_pitch,
                structure=structure,
                temperature=temp,
            )
        except Exception as exc:
            log.warning("Self-consistency sample %d/%d failed: %s", i + 1, samples, exc)
            continue
        candidates_dsl.append(dsl)

    if not candidates_dsl:
        raise RuntimeError("All self-consistency samples failed")

    scored: list[tuple[float, str]] = []
    per_sample_scores: list[float] = []
    for i, dsl in enumerate(candidates_dsl):
        try:
            extracted = extract_jot(dsl)
            score = score_jot(extracted, candidates_by_pitch).onset_f1
        except (JotParseError, Exception) as exc:
            log.info(
                "Self-consistency sample %d/%d unscoreable (%s); excluding",
                i + 1, samples, exc,
            )
            per_sample_scores.append(0.0)
            continue
        per_sample_scores.append(score)
        scored.append((score, dsl))
        log.info("Self-consistency sample %d/%d: F1=%.4f", i + 1, samples, score)

    if not scored:
        log.warning("No self-consistency sample parsed; falling back to first.")
        return candidates_dsl[0], per_sample_scores

    best_score, best_dsl = max(scored, key=lambda s: s[0])
    log.info(
        "Self-consistency picked best: F1=%.4f (range %.4f-%.4f over %d valid samples)",
        best_score,
        min(s for s, _ in scored),
        max(s for s, _ in scored),
        len(scored),
    )
    return best_dsl, per_sample_scores


def _temperatures_for(samples: int) -> list[float]:
    """Return the temperatures to use for `samples` candidates.

    First sample is greedy (0.0); subsequent samples add increasing
    diversity. Tuned empirically; values above ~0.8 produce too much
    structural variance for ADT.
    """
    if samples == 1:
        return [0.0]
    base = [0.0, 0.4, 0.7]
    if samples <= len(base):
        return base[:samples]
    return base + [0.7 + 0.05 * (i + 1) for i in range(samples - len(base))]


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        text = text[3:]
        if text.startswith(("dsl", "drumjot", "text", "json")):
            text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.strip("`\n ")
    if text.endswith("```"):
        text = text[: -3].strip()
    return text


def _format_bars(
    by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
) -> str:
    """Render the per-bar onset listing in a compact form.

    Each bar is a block:

        Bar 0 [4/4, 120.0 BPM, feel=straight16]:
          k: (1.000, 9.4) (3.000, 8.8)
          s: (2.000, 11.0) (4.000, 10.5)
          h: (1.000, 7.2) (1.500, 6.8) (2.000, 7.0) ...

    Candidates with `bar == -1` (i.e. outside the tracked range) are
    omitted - the LLM should treat them as silence/padding noise.
    """
    if not structure.bars:
        return "(no bars detected)"

    # Bucket candidates by bar
    by_bar: dict[int, dict[str, list[OnsetCandidate]]] = {}
    for pitch, cands in by_pitch.items():
        for c in cands:
            if c.bar < 0:
                continue
            by_bar.setdefault(c.bar, {}).setdefault(pitch, []).append(c)

    blocks: list[str] = []
    for bar in structure.bars:
        header = (
            f"Bar {bar.index} "
            f"[{bar.time_signature[0]}/{bar.time_signature[1]}, "
            f"{bar.tempo_bpm:.1f} BPM, feel={bar.feel}]:"
        )
        bar_cands = by_bar.get(bar.index, {})
        if not bar_cands:
            blocks.append(f"{header}\n  (no onsets in this bar)")
            continue
        lines = [header]
        for pitch in sorted(bar_cands.keys()):
            ps = sorted(bar_cands[pitch], key=lambda c: c.beat_in_bar)
            entries = " ".join(
                f"({c.beat_in_bar:.3f},{c.strength:.2f})" for c in ps
            )
            lines.append(f"  {pitch}: {entries}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _load_spec() -> str:
    """Load the canonical Drumjot DSL grammar.

    `settings.spec_path` points to the SPEC.md mounted into the container
    (default `/app/SPEC.md`, populated by docker-compose from `../SPEC.md`).
    """
    path = settings.spec_path
    if not path.exists():
        log.warning(
            "SPEC.md not found at %s - using minimal grammar block", path
        )
        return "(SPEC.md unavailable - fall back to grammar from few-shot examples below.)"
    return path.read_text(encoding="utf-8")


def _load_examples() -> str:
    examples_path = PROMPT_DIR / "examples.md"
    return examples_path.read_text(encoding="utf-8")


def _load_prompt_template() -> str:
    template_path = PROMPT_DIR / "transcribe.md"
    return template_path.read_text(encoding="utf-8")
