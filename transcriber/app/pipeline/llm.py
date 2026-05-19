"""Claude-based translation of candidate onsets to Drumjot DSL.

Transcription is **per instrument**: one LLM call per drum pitch, each
seeing only that instrument's candidate onsets plus the shared
bar/beat/tempo/time-signature frame, and emitting a single monophonic
line (one pitch letter or `.` per position — no `+`, no `||`, no
metadata block). The independent lines are merged back into one Jot
deterministically by `pipeline/recompose.py` (hands joined with `+`,
genuine polyrhythm as `+`-joined groups, kick as the second `||`
voice).

Why per-instrument:

- An autoregressive model maintains a single coherent monophonic line
  far more reliably than it interleaves several independent ones at
  every grid position; column-merging all limbs into one sequence is
  the worst case for accuracy and token cost.
- Each call's prompt is tiny, so calls run in parallel and best-of-K
  is applied per instrument (scored on that pitch's onset F1).
- Errors are isolated and independently re-scorable, which the
  refinement loop (already per-stem F1) exploits directly.

The LLM never decides the grid: tempo / time signature / per-bar feel
come from `beats.py` and are handed to every call as a fixed frame.
Only the subdivision *within* a beat is the model's call, per
instrument — which is what makes genuine polyrhythm transcribable.

Input to each call is **per-bar**:
  - bar index, time signature, local tempo, detected feel
  - this instrument's candidate onsets, each at a beat-relative
    position (e.g. `(2.333, 9.4)` = "beat 2.333 - one-third into beat
    2, strength 9.4")
"""
from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path

import anthropic

from app.config import settings
from app.debug import (
    current_debug_sink,
    reset_current_debug_sink,
    set_current_debug_sink,
)
from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.jot_extract import extract_jot
from app.pipeline.llm_util import (
    call_messages_with_refusal_retry,
    strip_code_fence,
)
from app.pipeline.recompose import PITCH_DISPLAY_NAMES
from app.pipeline.score import score_jot

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def transcribe_instrument_to_jot(
    pitch: str,
    instrument_name: str,
    candidates_for_pitch: list[OnsetCandidate],
    structure: BeatStructure,
    temperature: float | None = None,
    parse_error_hint: str | None = None,
    debug_purpose: str | None = None,
) -> str:
    """One LLM call transcribing a single instrument. Returns its
    monophonic DSL fragment (bars only, no metadata block).

    Retry / scoring is the caller's responsibility — use
    `transcribe_instrument_best_of_k` for the production path.

    `temperature` is silently ignored for models that don't accept it
    (Opus 4.7+ uses extended thinking and rejects any `temperature`
    value). For older models, pass a float to override the default.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot call the LLM. "
            "Configure it in transcriber/.env."
        )

    spec_text = _load_spec_subset()
    example_text = _load_instrument_examples()
    prompt_template = _load_instrument_prompt_template()

    parse_hint = ""
    if parse_error_hint:
        parse_hint = (
            "\nIMPORTANT: A previous attempt failed to parse with this error:\n"
            f"   {parse_error_hint}\n"
            "Make sure the output you produce now parses cleanly.\n"
        )

    # `_format_bars` already emits one labelled line per pitch per bar;
    # feeding it a single-key dict yields exactly this instrument's
    # per-bar listing under the shared bar headers.
    bar_blocks = _format_bars({pitch: candidates_for_pitch}, structure)
    initial_tempo = structure.initial_tempo
    initial_sig = structure.initial_time_signature
    initial_sig_str = f"{initial_sig[0]}/{initial_sig[1]}"

    prompt = (
        prompt_template
        .replace("{PITCH}", pitch)
        .replace("{INSTRUMENT_NAME}", instrument_name)
        .replace("{SPEC}", spec_text)
        .replace("{EXAMPLES}", example_text)
        .replace("{INITIAL_TEMPO}", f"{initial_tempo:.2f}")
        .replace("{INITIAL_TIME_SIG}", initial_sig_str)
        .replace("{BAR_COUNT}", str(len(structure.bars)))
        .replace("{BARS}", bar_blocks)
    ) + parse_hint

    purpose = debug_purpose or f"transcribe_{pitch}"
    log.info(
        "Calling LLM (instrument=%s) model=%s temperature=%s "
        "prompt_chars=%d bars=%d",
        pitch,
        settings.llm_model,
        f"{temperature:.2f}" if temperature is not None else "default",
        len(prompt),
        len(structure.bars),
    )
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    create_kwargs: dict = {
        "model": settings.llm_model,
        "max_tokens": settings.llm_max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        create_kwargs["temperature"] = temperature

    response = call_messages_with_refusal_retry(
        client, create_kwargs, base_prompt=prompt, purpose=purpose,
    )

    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    text = "".join(parts).strip()
    return strip_code_fence(text)


def transcribe_instrument_best_of_k(
    pitch: str,
    instrument_name: str,
    candidates_for_pitch: list[OnsetCandidate],
    structure: BeatStructure,
    samples: int = 1,
) -> tuple[str, list[float]]:
    """Run `samples` LLM calls for one instrument, score each by its
    own onset F1 against this instrument's source onsets, return the
    best fragment + per-sample scores.

    Best-of-K is per instrument: K candidates for this pitch, each
    scored on this pitch's F1 alone.
    """
    if samples < 1:
        samples = 1

    # Opus 4.7 ignores any temperature override (extended-thinking
    # models fix it internally), so best-of-K relies on the model's
    # intrinsic stochasticity at default temperature. `_temperatures_for`
    # is kept for older models; if the API rejects the override we retry
    # once with temperature=None.
    temperatures = _temperatures_for(samples)
    candidates_dsl: list[str] = []
    for i, temp in enumerate(temperatures):
        purpose = (
            f"transcribe_{pitch}_sample_{i + 1}of{samples}_temp_{temp:.2f}"
        )
        try:
            dsl = transcribe_instrument_to_jot(
                pitch=pitch,
                instrument_name=instrument_name,
                candidates_for_pitch=candidates_for_pitch,
                structure=structure,
                temperature=temp,
                debug_purpose=purpose,
            )
        except anthropic.BadRequestError as exc:
            if "temperature" in str(exc).lower():
                log.info(
                    "Best-of-K %s sample %d/%d: model rejected temperature, "
                    "retrying without", pitch, i + 1, samples,
                )
                try:
                    dsl = transcribe_instrument_to_jot(
                        pitch=pitch,
                        instrument_name=instrument_name,
                        candidates_for_pitch=candidates_for_pitch,
                        structure=structure,
                        temperature=None,
                        debug_purpose=f"{purpose}__no_temperature",
                    )
                except Exception as exc2:
                    log.warning(
                        "Best-of-K %s sample %d/%d failed on retry: %s",
                        pitch, i + 1, samples, exc2,
                    )
                    continue
            else:
                log.warning(
                    "Best-of-K %s sample %d/%d failed: %s",
                    pitch, i + 1, samples, exc,
                )
                continue
        except Exception as exc:
            log.warning(
                "Best-of-K %s sample %d/%d failed: %s",
                pitch, i + 1, samples, exc,
            )
            continue
        candidates_dsl.append(dsl)

    if not candidates_dsl:
        raise RuntimeError(f"All best-of-K samples failed for {pitch}")

    scored: list[tuple[float, str]] = []
    per_sample_scores: list[float] = []
    for i, dsl in enumerate(candidates_dsl):
        try:
            score = _score_instrument_fragment(
                dsl, pitch, candidates_for_pitch, structure,
                debug_tag=f"best_of_k_{pitch}_sample_{i + 1}of{samples}",
            )
        except Exception as exc:
            log.info(
                "Best-of-K %s sample %d/%d unscoreable (%s); excluding",
                pitch, i + 1, samples, exc,
            )
            per_sample_scores.append(0.0)
            continue
        per_sample_scores.append(score)
        scored.append((score, dsl))
        log.info(
            "Best-of-K %s sample %d/%d: F1=%.4f", pitch, i + 1, samples, score,
        )

    if not scored:
        log.warning(
            "No best-of-K sample scored for %s; falling back to first.", pitch,
        )
        return candidates_dsl[0], per_sample_scores

    best_score, best_dsl = max(scored, key=lambda s: s[0])
    log.info(
        "Best-of-K %s picked best: F1=%.4f (range %.4f-%.4f over %d valid)",
        pitch,
        best_score,
        min(s for s, _ in scored),
        max(s for s, _ in scored),
        len(scored),
    )
    return best_dsl, per_sample_scores


def transcribe_all_instruments(
    candidates_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    samples: int = 1,
    max_workers: int | None = None,
) -> tuple[dict[str, str], dict[str, list[float]]]:
    """Transcribe every instrument that has usable onsets, in parallel.

    Returns `(lines_by_pitch, scores_by_pitch)`. `scores_by_pitch` is
    populated only when `samples > 1` (per-instrument best-of-K
    per-sample F1s); empty otherwise. Instruments with no in-range
    candidate onsets are skipped entirely (they weren't played).
    """
    pitches = sorted(
        p for p, cands in candidates_by_pitch.items()
        if any(c.bar >= 0 for c in cands)
    )
    if not pitches:
        log.warning("transcribe: no instrument had any in-range onsets")
        return {}, {}

    workers = max(1, max_workers or settings.instrument_concurrency)
    # ContextVars don't cross thread boundaries automatically, so capture
    # the request-scoped debug sink here and re-install it inside each
    # worker — otherwise per-instrument prompt dumps + score JSON
    # (written via `current_debug_sink()`) silently vanish.
    sink = current_debug_sink()

    def work(pitch: str) -> tuple[str, str, list[float]]:
        token = set_current_debug_sink(sink)
        try:
            name = PITCH_DISPLAY_NAMES.get(pitch, pitch)
            cands = candidates_by_pitch.get(pitch, [])
            if samples > 1:
                frag, scores = transcribe_instrument_best_of_k(
                    pitch, name, cands, structure, samples,
                )
                return pitch, frag, scores
            frag = transcribe_instrument_to_jot(
                pitch, name, cands, structure,
                debug_purpose=f"transcribe_{pitch}",
            )
            return pitch, frag, []
        finally:
            reset_current_debug_sink(token)

    lines_by_pitch: dict[str, str] = {}
    scores_by_pitch: dict[str, list[float]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        futures = {executor.submit(work, p): p for p in pitches}
        for fut in concurrent.futures.as_completed(futures):
            pitch = futures[fut]
            try:
                p, frag, scores = fut.result()
            except Exception as exc:
                log.warning(
                    "transcribe: instrument %s failed entirely (%s); "
                    "dropping it", pitch, exc,
                )
                continue
            lines_by_pitch[p] = frag
            if scores:
                scores_by_pitch[p] = scores

    return lines_by_pitch, scores_by_pitch


def _score_instrument_fragment(
    fragment: str,
    pitch: str,
    candidates_for_pitch: list[OnsetCandidate],
    structure: BeatStructure,
    debug_tag: str | None = None,
) -> float:
    """Onset F1 of a single-instrument fragment against its source
    onsets. The fragment parses on its own (a bare bar sequence); the
    bun-bridge BPM is irrelevant because scoring is structure-anchored
    (re-times each predicted onset off the audio's bar boundaries —
    see `score_jot`)."""
    extracted = extract_jot(fragment)
    time_offset = structure.bars[0].start_time if structure.bars else 0.0
    return score_jot(
        extracted,
        {pitch: candidates_for_pitch},
        time_offset=time_offset,
        structure=structure,
        debug_tag=debug_tag,
    ).onset_f1


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


def _format_bars(
    by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
) -> str:
    """Render the per-bar onset listing in a compact form.

    Each bar is a block:

        Bar 0 [4/4, 120.0 BPM, feel=straight16]:
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


def _load_spec_subset() -> str:
    """The canonical grammar with multi-voice (`||`) removed.

    Per-instrument calls emit a single monophonic line and must never
    produce `||`. Rather than mutate the shared canonical `SPEC.md`
    (also consumed by the frontend renderer + bun bridge), we strip the
    `## Global simultaneity` section and any bare `||` operator lines
    (the `||` separator inside example code fences and the reserved-
    characters table row) at load time. SPEC.md stays untouched on disk.
    """
    raw = _load_spec()
    out: list[str] = []
    skip_section = False
    for line in raw.split("\n"):
        if line.startswith("## "):
            skip_section = line.strip() == "## Global simultaneity"
        if skip_section:
            continue
        stripped = line.strip()
        if stripped == "||":
            continue
        if stripped.startswith("| `\\|\\|`"):
            continue
        out.append(line)
    return "\n".join(out)


def _load_instrument_examples() -> str:
    return (PROMPT_DIR / "examples_instrument.md").read_text(encoding="utf-8")


def _load_instrument_prompt_template() -> str:
    return (PROMPT_DIR / "transcribe_instrument.md").read_text(
        encoding="utf-8"
    )
