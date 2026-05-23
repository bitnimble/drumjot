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

_FILTER_TOOL: dict[str, Any] = {
    "name": "report_artifact_onsets",
    "description": (
        "Return the indices of the detected onsets that are NOT real "
        "musical hits — separation bleed, detector double-triggers, or "
        "noise. Keep the list empty if every onset is a genuine hit. "
        "Never include an index that wasn't shown to you."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "rejected_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "description": (
                    "The `#N` indices of onsets to drop. These should be "
                    "the minority — only clear artifacts."
                ),
            },
        },
        "required": ["rejected_indices"],
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
) -> list[OnsetCandidate]:
    """One LLM call. Returns the kept (non-rejected, in-range) onsets for
    `pitch`, original times preserved.

    On any failure the input onsets are returned unfiltered — a missed
    filter pass degrades to "no filtering", never to dropped audio.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot call the LLM. "
            "Configure it in transcriber/.env."
        )

    indexed = _index_in_range(candidates_for_pitch)
    if not indexed:
        return []
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
    log.info(
        "Calling filter LLM (instrument=%s) model=%s prompt_chars=%d onsets=%d",
        pitch, settings.llm_model, len(prompt), len(ordered),
    )
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        response = call_messages_with_refusal_retry(
            client,
            {
                "model": settings.llm_model,
                "max_tokens": settings.llm_max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [_FILTER_TOOL],
                "tool_choice": {"type": "tool", "name": _FILTER_TOOL["name"]},
            },
            base_prompt=prompt,
            purpose=purpose,
        )
    except Exception as exc:
        log.warning(
            "filter %s: LLM call failed (%s); keeping all %d onsets",
            pitch, exc, len(ordered),
        )
        return ordered

    rejected = _extract_rejected(response, len(ordered))
    kept = [c for i, c in enumerate(ordered) if i not in rejected]
    log.info(
        "filter %s: rejected %d / %d onsets, kept %d",
        pitch, len(rejected), len(ordered), len(kept),
    )

    sink = current_debug_sink()
    if sink is not None:
        sink.write_json(
            f"filter/{pitch}.json",
            {
                "pitch": pitch,
                "n_input": len(ordered),
                "rejected_indices": sorted(rejected),
                "n_kept": len(kept),
            },
        )
    return kept


def filter_onsets_all_instruments(
    candidates_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    max_workers: int | None = None,
    on_complete: Callable[[str, int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, list[OnsetCandidate]]:
    """Filter every instrument that has in-range onsets, in parallel.

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
    """
    pitches = sorted(
        p for p, cands in candidates_by_pitch.items()
        if any(c.bar >= 0 for c in cands)
    )
    if not pitches:
        log.warning("filter: no instrument had any in-range onsets")
        return {}

    workers = max(1, max_workers or settings.instrument_concurrency)
    sink = current_debug_sink()

    def work(pitch: str) -> tuple[str, list[OnsetCandidate]]:
        token = set_current_debug_sink(sink)
        try:
            name = PITCH_DISPLAY_NAMES.get(pitch, pitch)
            others = {
                p: c for p, c in candidates_by_pitch.items() if p != pitch
            }
            kept = filter_onsets_for_instrument(
                pitch,
                name,
                candidates_by_pitch.get(pitch, []),
                structure,
                others,
                debug_purpose=f"filter_{pitch}",
            )
            return pitch, kept
        finally:
            reset_current_debug_sink(token)

    kept_by_pitch: dict[str, list[OnsetCandidate]] = {}
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
                pitch = futures[fut]
                try:
                    p, kept = fut.result()
                except Exception as exc:
                    log.warning(
                        "filter: instrument %s failed entirely (%s); "
                        "keeping its raw onsets", pitch, exc,
                    )
                    kept_by_pitch[pitch] = [
                        c for c in candidates_by_pitch.get(pitch, []) if c.bar >= 0
                    ]
                    p = pitch
                else:
                    if kept:
                        kept_by_pitch[p] = kept
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

    return kept_by_pitch


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
) -> set[int]:
    """Pull `rejected_indices` from the forced tool call, clamped to the
    valid `[0, n)` range (hallucinated / out-of-range indices ignored)."""
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != _FILTER_TOOL["name"]:
            continue
        raw = block.input.get("rejected_indices", [])
        if not isinstance(raw, list):
            log.warning(
                "filter: tool returned non-list rejected_indices (%s); "
                "treating as no rejections", type(raw).__name__,
            )
            return set()
        out: set[int] = set()
        for v in raw:
            try:
                i = int(v)
            except (TypeError, ValueError):
                continue
            if 0 <= i < n:
                out.add(i)
        return out
    log.warning("filter: no tool_use block in response; no rejections")
    return set()


def _load_prompt_template() -> str:
    return (PROMPT_DIR / "filter_onsets.md").read_text(encoding="utf-8")
