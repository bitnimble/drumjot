"""Claude-based *filtering* of candidate onsets (no DSL, no Jot).

The `filter` transcribe pathway narrows the LLM's job to the one thing
only it can do well: deciding which detected onsets are real musical
hits vs. separation / detection artifacts. Everything else (timing,
MIDI assembly, scoring) is deterministic.

Per instrument, one LLM call sees:
  - that instrument's detected onsets, each printed with a stable
    integer index `#N` and its `(beat_in_bar, strength)`,
  - the fixed beat frame (per-bar tempo / time signature / feel),
  - a compact cross-instrument summary (other pitches' hits by beat
    position) so it can recognise bleed (an onset in this stem that
    only exists because a *louder* instrument hit at the same instant).

The model returns, via Anthropic's tool-use channel (schema-validated
server-side — no JSON-from-text parsing), the indices it judges to be
artifacts. We keep everything it didn't reject, with original onset
times verbatim. The LLM only ever *removes*; it never adds or moves an
onset, so recall is capped by `detect_onsets()` (tuned high-recall) by
design.

Onsets outside the beat-tracked range (`bar < 0`) are never indexed and
never kept — same convention as `llm.py::_format_bars`, which treats
them as padding noise.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic

from app.config import settings
from app.debug import (
    current_debug_sink,
    reset_current_debug_sink,
    set_current_debug_sink,
)
from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.llm_util import call_messages_with_refusal_retry
from app.pipeline.separate import PITCH_DISPLAY_NAMES

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# Short reason codes the filter LLM picks from. Kept tight on purpose:
# a small vocabulary keeps output tokens bounded (one short string per
# rejection instead of a free-form sentence) and lets the frontend group
# / colour rejections by reason later. `custom` is the escape hatch for
# the long tail; it REQUIRES `reason_text` so an opaque "custom" alone
# never makes it out.
REASON_CODES: frozenset[str] = frozenset({
    "bleed", "double_trigger", "noise", "custom",
})

_FILTER_TOOL: dict[str, Any] = {
    "name": "report_artifact_onsets",
    "description": (
        "Return the onsets that are NOT real musical hits; separation "
        "bleed, detector double-triggers, or noise; each paired with a "
        "short reason code so the operator can see WHY it was flagged. "
        "For `double_trigger` you MUST also give `double_of`; the index "
        "of the real strike this onset duplicates. "
        "Keep the list empty if every onset is a genuine hit. Never "
        "include an index that wasn't shown to you."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "rejected_onsets": {
                "type": "array",
                "description": (
                    "Onsets to drop. Should be the minority; only clear "
                    "artifacts."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "The `#N` index of the onset to drop.",
                        },
                        "reason": {
                            "type": "string",
                            "enum": sorted(REASON_CODES),
                            "description": (
                                "Short reason code. Use `custom` only "
                                "when none of the standard codes fit; in "
                                "that case `reason_text` is required."
                            ),
                        },
                        "reason_text": {
                            "type": "string",
                            "maxLength": 200,
                            "description": (
                                "Required when `reason` is `custom`. "
                                "Optional brief extra detail for the "
                                "standard reasons (e.g. which pitch you "
                                "think the bleed is from). Keep short."
                            ),
                        },
                        "double_of": {
                            "type": "integer",
                            "minimum": 0,
                            "description": (
                                "Required when `reason` is `double_trigger`: "
                                "the `#N` index of the REAL strike this onset "
                                "is a duplicate of (the one actually played). "
                                "Omit for other reasons."
                            ),
                        },
                    },
                    "required": ["index", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["rejected_onsets"],
        "additionalProperties": False,
    },
}


def filter_onsets_for_instrument(
    pitch: str,
    instrument_name: str,
    candidates_for_pitch: list[OnsetCandidate],
    structure: BeatStructure,
    others_by_pitch: dict[str, list[OnsetCandidate]],
    debug_purpose: str | None = None,
    llm_model: str | None = None,
) -> tuple[list[OnsetCandidate], dict[int, dict[str, Any]]]:
    """One LLM call. Returns `(kept, rejection_info)`:

    - `kept`, the non-rejected, in-range onsets for `pitch`, original
      times preserved.
    - `rejection_info`; `{id(c): {"reason": str, "reason_text": str | None}}`
      for every onset the LLM rejected, keyed by Python object identity
      so the caller can match it against `id(c)` membership in the
      pre-filter candidate list. Empty when the LLM kept everything.

    Anthropic-call failures propagate so the runner can map them to
    HTTP 502 (per CLEANROOM_SPEC §11.14: no silent fallback to "keep
    everything", which would silently deliver an artifact-heavy MIDI).
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot call the LLM. "
            "Configure it in transcriber/.env."
        )

    indexed = _index_in_range(candidates_for_pitch)
    if not indexed:
        return [], {}
    ordered = [c for _, c in indexed]

    bar_blocks = _format_indexed_bars(indexed, pitch, structure, others_by_pitch)
    initial_sig = structure.initial_time_signature
    prompt = (
        _load_prompt_template()
        .replace("{PITCH}", pitch)
        .replace("{INSTRUMENT_NAME}", instrument_name)
        .replace("{INITIAL_TEMPO}", f"{structure.initial_tempo:.2f}")
        .replace("{INITIAL_TIME_SIG}", f"{initial_sig[0]}/{initial_sig[1]}")
        .replace("{BAR_COUNT}", str(len(structure.bars)))
        .replace("{ONSET_COUNT}", str(len(ordered)))
        .replace("{BARS}", bar_blocks)
    )

    purpose = debug_purpose or f"filter_{pitch}"
    model = llm_model or settings.llm_model
    log.info(
        "Calling filter LLM (instrument=%s) model=%s prompt_chars=%d onsets=%d",
        pitch, model, len(prompt), len(ordered),
    )
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = call_messages_with_refusal_retry(
        client,
        {
            "model": model,
            "max_tokens": settings.llm_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [_FILTER_TOOL],
            "tool_choice": {"type": "tool", "name": _FILTER_TOOL["name"]},
        },
        base_prompt=prompt,
        purpose=purpose,
    )

    rejected_raw = _extract_rejected(response, len(ordered))
    # Deterministic backstop: the LLM can over-eagerly call two well-spaced
    # hits a "double_trigger" (a snare roll, a fast kick double). Overturn
    # any such rejection whose gap to the strike it claims to duplicate
    # exceeds this lane's physical refractory window, so real fast playing
    # always survives regardless of the model's judgement.
    rejected, overturned = _apply_refractory_guardrail(ordered, rejected_raw, pitch)
    kept = [c for i, c in enumerate(ordered) if i not in rejected]
    rejection_info: dict[int, dict[str, Any]] = {
        id(ordered[i]): info for i, info in rejected.items()
    }
    log.info(
        "filter %s: rejected %d / %d onsets, kept %d",
        pitch, len(rejected), len(ordered), len(kept),
    )
    if overturned:
        log.info(
            "filter %s: overturned %d double_trigger rejection(s) past the "
            "%.0f ms refractory window",
            pitch, len(overturned), _refractory_window_s(pitch) * 1000.0,
        )

    sink = current_debug_sink()
    if sink is not None:
        sink.write_json(
            f"filter/{pitch}.json",
            {
                "pitch": pitch,
                "n_input": len(ordered),
                "rejected": [
                    {"index": i, **info} for i, info in sorted(rejected.items())
                ],
                # double_trigger rejections the refractory guardrail
                # restored as real hits (too far apart to be one strike).
                "overturned": [
                    {"index": i, **info}
                    for i, info in sorted(overturned.items())
                ],
                "n_kept": len(kept),
            },
        )
    return kept, rejection_info


def filter_onsets_all_instruments(
    candidates_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    max_workers: int | None = None,
    on_complete: Callable[[str, int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
    skip_pitches: set[str] | None = None,
    llm_model: str | None = None,
) -> tuple[dict[str, list[OnsetCandidate]], dict[str, dict[int, dict[str, Any]]]]:
    """Filter every instrument that has in-range onsets, in parallel.

    Returns `(kept_by_pitch, reasons_by_pitch)` where `reasons_by_pitch`
    is `{pitch: {id(c): {"reason": ..., "reason_text": ...}}}` for every
    rejected onset across every instrument. Skipped pitches contribute
    nothing to either map.

    Mirrors `llm.transcribe_all_instruments`' threading + debug-sink
    ContextVar propagation. Instruments with no in-range candidates are
    omitted (they weren't played).

    `on_complete`, when provided, is invoked from the futures-completion
    loop with `(pitch, done, total)` each time an instrument's filter
    call returns. Used by the HTTP layer to surface live "N/M filtered"
    progress to the client. Errors in the callback are caught so a
    misbehaving subscriber can't abort the filter pass.

    `cancel_event`, when set, causes the pool to stop submitting/awaiting
    new instruments — already-in-flight LLM calls run to completion
    (the anthropic SDK exposes no cross-thread cancel) but pending ones
    are cancelled and the function returns whatever it has so far.

    `skip_pitches`, when provided, names pitches whose lanes have
    already been LLM-vetted upstream (e.g. `h` / `H` after
    `hihat_split`'s unified ternary classifier). Those pitches are not
    submitted to the pool — re-filtering would duplicate work and risk
    double-rejecting soft real hits. They are NOT re-attached to the
    result here; the caller is responsible for merging them in.
    """
    skip = skip_pitches or set()
    pitches = sorted(
        p for p, cands in candidates_by_pitch.items()
        if p not in skip and any(c.bar >= 0 for c in cands)
    )
    if not pitches:
        log.warning("filter: no instrument had any in-range onsets")
        return {}, {}

    workers = max(1, max_workers or settings.instrument_concurrency)
    sink = current_debug_sink()

    def work(
        pitch: str,
    ) -> tuple[str, list[OnsetCandidate], dict[int, dict[str, Any]]]:
        token = set_current_debug_sink(sink)
        try:
            name = PITCH_DISPLAY_NAMES.get(pitch, pitch)
            others = {
                p: c for p, c in candidates_by_pitch.items() if p != pitch
            }
            kept, reasons = filter_onsets_for_instrument(
                pitch,
                name,
                candidates_by_pitch.get(pitch, []),
                structure,
                others,
                debug_purpose=f"filter_{pitch}",
                llm_model=llm_model,
            )
            return pitch, kept, reasons
        finally:
            reset_current_debug_sink(token)

    kept_by_pitch: dict[str, list[OnsetCandidate]] = {}
    reasons_by_pitch: dict[str, dict[int, dict[str, Any]]] = {}
    total = len(pitches)
    done = 0
    cancelled = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, p): p for p in pitches}
        try:
            for fut in concurrent.futures.as_completed(futures):
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                p, kept, reasons = fut.result()
                if kept:
                    kept_by_pitch[p] = kept
                if reasons:
                    reasons_by_pitch[p] = reasons
                done += 1
                if on_complete is not None:
                    try:
                        on_complete(p, done, total)
                    except Exception:  # pragma: no cover - best-effort
                        log.exception("on_complete callback raised; ignoring")
        finally:
            if cancelled:
                # Cancel everything not yet started; in-flight LLM calls
                # cannot be interrupted but the pool's __exit__ will at
                # least skip their result collection.
                for f in futures:
                    if not f.done():
                        f.cancel()
                log.info(
                    "filter: cancelled after %d/%d instruments (client disconnected)",
                    done, total,
                )

    return kept_by_pitch, reasons_by_pitch


def _index_in_range(
    candidates: list[OnsetCandidate],
) -> list[tuple[int, OnsetCandidate]]:
    """Assign each in-range onset a stable index in prompt order.

    Order = bar ascending, then `beat_in_bar` ascending — identical to
    `llm._format_bars` so a human can cross-read the two debug dumps.
    Out-of-range onsets (`bar < 0`) are excluded entirely.
    """
    in_range = [c for c in candidates if c.bar >= 0]
    in_range.sort(key=lambda c: (c.bar, c.beat_in_bar))
    return list(enumerate(in_range))


def _format_indexed_bars(
    indexed: list[tuple[int, OnsetCandidate]],
    pitch: str,
    structure: BeatStructure,
    others_by_pitch: dict[str, list[OnsetCandidate]],
) -> str:
    """Render per-bar blocks: the target pitch's indexed onsets plus a
    compact one-line summary of what every *other* instrument did in
    that bar (pitch + beat position only — the bleed-discriminating
    signal, without the token cost of full listings)."""
    if not structure.bars:
        return "(no bars detected)"

    by_bar: dict[int, list[tuple[int, OnsetCandidate]]] = {}
    for idx, c in indexed:
        by_bar.setdefault(c.bar, []).append((idx, c))

    others_by_bar: dict[int, list[tuple[str, float]]] = {}
    for op, cands in others_by_pitch.items():
        for c in cands:
            if c.bar < 0:
                continue
            others_by_bar.setdefault(c.bar, []).append((op, c.beat_in_bar))

    blocks: list[str] = []
    for bar in structure.bars:
        header = (
            f"Bar {bar.index} "
            f"[{bar.time_signature[0]}/{bar.time_signature[1]}, "
            f"{bar.tempo_bpm:.1f} BPM, feel={bar.feel}]:"
        )
        rows = [header]

        entries = by_bar.get(bar.index, [])
        if entries:
            rendered = " ".join(
                f"#{idx}({c.beat_in_bar:.3f},{c.strength:.2f})"
                for idx, c in entries
            )
            rows.append(f"  {pitch}: {rendered}")
        else:
            rows.append(f"  {pitch}: (none)")

        others = sorted(others_by_bar.get(bar.index, []), key=lambda x: x[1])
        if others:
            summary = " ".join(f"{op}{pos:.2f}" for op, pos in others)
            rows.append(f"  others: {summary}")

        blocks.append("\n".join(rows))

    return "\n\n".join(blocks)


def _extract_rejected(
    response: anthropic.types.Message, n: int
) -> dict[int, dict[str, Any]]:
    """Pull `rejected_onsets` from the forced tool call.

    Returns `{index: {"reason": str, "reason_text": str | None}}` for
    every rejected onset. Duplicate indices keep the LAST entry the
    model emitted (it shouldn't emit duplicates, but if it does we
    accept the most recent reason).

    Raises on malformed responses (non-list, non-dict items, missing /
    invalid `index` or `reason`, out-of-range indices, `custom` reason
    without `reason_text`) so the runner can surface the model bug as
    HTTP 502 rather than silently delivering a degraded filter pass.
    """
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != _FILTER_TOOL["name"]:
            continue
        raw = block.input.get("rejected_onsets", [])
        if not isinstance(raw, list):
            raise RuntimeError(
                f"filter: tool returned non-list rejected_onsets "
                f"({type(raw).__name__}); model violated schema"
            )
        out: dict[int, dict[str, Any]] = {}
        for v in raw:
            if not isinstance(v, dict):
                raise RuntimeError(
                    f"filter: rejected_onsets item not an object ({v!r})"
                )
            try:
                i = int(v["index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"filter: rejected_onsets item missing/invalid `index` ({v!r})"
                ) from exc
            if not 0 <= i < n:
                raise RuntimeError(
                    f"filter: rejected_onsets contains out-of-range index "
                    f"{i} (valid range: [0, {n}))"
                )
            reason = v.get("reason")
            if not isinstance(reason, str) or reason not in REASON_CODES:
                raise RuntimeError(
                    f"filter: rejected_onsets item has missing/invalid "
                    f"`reason` ({reason!r}); expected one of "
                    f"{sorted(REASON_CODES)}"
                )
            reason_text_raw = v.get("reason_text")
            if reason_text_raw is not None and not isinstance(reason_text_raw, str):
                raise RuntimeError(
                    f"filter: rejected_onsets item `reason_text` must be a "
                    f"string when present (got {type(reason_text_raw).__name__})"
                )
            reason_text: str | None = (
                reason_text_raw.strip() if isinstance(reason_text_raw, str) else None
            ) or None
            if reason == "custom" and not reason_text:
                raise RuntimeError(
                    f"filter: rejected_onsets item with reason='custom' "
                    f"must include a non-empty `reason_text` (index {i})"
                )
            info: dict[str, Any] = {"reason": reason, "reason_text": reason_text}
            # `double_of` only carries meaning for double_trigger: it's the
            # index of the real strike this onset duplicates, used by the
            # refractory guardrail to verify the two are actually close
            # enough to be one physical hit. A missing/garbage value is NOT
            # a hard error (it just leaves the rejection unverifiable, which
            # the guardrail resolves by keeping the onset); but a non-integer
            # `double_of` is a schema violation, surfaced like the others.
            if reason == "double_trigger":
                double_of_raw = v.get("double_of")
                if double_of_raw is not None:
                    try:
                        info["double_of"] = int(double_of_raw)
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"filter: rejected_onsets item `double_of` must be "
                            f"an integer when present (got {double_of_raw!r}, "
                            f"index {i})"
                        ) from exc
            out[i] = info
        return out
    raise RuntimeError(
        "filter: no tool_use block in response; tool call was not made"
    )


def _refractory_window_s(pitch: str) -> float:
    """Minimum gap below which a `double_trigger` rejection may stand for
    this lane.

    Kick gets a wider window (a beater can't re-strike as fast as a stick
    bounce); every other filter-LLM lane uses the default, which sits just
    above the detector's 20 ms min-distance so rolls / drags / flams /
    fast doubles survive. Crash/ride never reach here (cymbal_split vets
    them upstream)."""
    if pitch == "k":
        return settings.double_trigger_refractory_kick_s
    return settings.double_trigger_refractory_default_s


def _apply_refractory_guardrail(
    ordered: list[OnsetCandidate],
    rejected: dict[int, dict[str, Any]],
    pitch: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Overturn physically-implausible `double_trigger` rejections.

    For each `double_trigger` rejection, look up the real strike the LLM
    said it duplicates (`double_of`) and measure the time gap. The
    rejection only STANDS when that strike exists and the gap is below this
    lane's refractory window; a gap at/above the window (two real hits) or
    a missing / out-of-range / self-referential `double_of` (unverifiable)
    overturns it, keeping the onset.

    Returns `(rejected_after, overturned)`, both `{index: info}`. Only
    `double_trigger` entries are ever touched; bleed / noise / custom pass
    through unchanged. The guardrail never adds rejections, only restores
    hits, so it is strictly recall-positive."""
    window = _refractory_window_s(pitch)
    n = len(ordered)
    rejected_after: dict[int, dict[str, Any]] = {}
    overturned: dict[int, dict[str, Any]] = {}
    for i, info in rejected.items():
        if info["reason"] != "double_trigger":
            rejected_after[i] = info
            continue
        j = info.get("double_of")
        if isinstance(j, int) and 0 <= j < n and j != i:
            gap = abs(float(ordered[i].time) - float(ordered[j].time))
            if gap < window:
                rejected_after[i] = info
                continue
        # Gap too wide to be one strike, or no usable reference: keep it.
        overturned[i] = info
    return rejected_after, overturned


def _load_prompt_template() -> str:
    return (PROMPT_DIR / "filter_onsets.md").read_text(encoding="utf-8")
