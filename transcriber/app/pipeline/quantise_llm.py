"""LLM residual pass: Haiku shifts on-grid onsets in cross-instrument context.

Sends every on-grid kept onset with its current slot to Haiku across
parallel windows of consecutive bars; the model returns a clamped integer
slot shift per onset. This is where cross-instrument musical reasoning
earns its keep. Split out of `quantise.py` as one cohesive pass (indexing,
windowing, prompt formatting, the forced-tool call, shift extraction).
"""
from __future__ import annotations

import contextvars
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from app.config import settings
from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.llm_util import call_messages_with_refusal_retry
from app.pipeline.quantise_config import (
    _CONTEXT_BARS,
    _LLM_MODEL,
    _MAX_BARS_PER_WINDOW,
    _MAX_LLM_SHIFT,
    _MAX_PARALLEL_CHUNKS,
    _QUANTISE_TOOL,
    _TARGET_ONSETS_PER_WINDOW,
    PROMPT_DIR,
    SLOTS_PER_BEAT,
    _max_tokens_for,
)

log = logging.getLogger(__name__)


def _llm_residual_pass(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
    cancel_event: threading.Event | None = None,
) -> tuple[dict[tuple[str, int], int], str]:
    """Haiku residual pass over parallel windows of consecutive bars.

    Returns `(shifts, status)`, where `shifts` is the merged
    (pitch, idx_in_pitch_list) -> shift_slots map the model proposed
    (clamped to ±_MAX_LLM_SHIFT, unknown ids dropped) and `status` is a
    short summary suitable for `quantise/shifts.json` ("ok",
    "partial: F/N windows failed", "error: ...", or "cancelled").

    The on-grid onsets are split into windows (`_build_windows`); each
    window is one forced-tool call carrying its core bars (shiftable,
    `#id`-tagged) plus ±_CONTEXT_BARS read-only neighbour bars. Windows
    run concurrently. A window that raises degrades only its own bars to
    geometric placement, the rest keep their LLM shifts. Off-grid onsets
    are never offered as shift targets; their position is geometric truth.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot call the quantise LLM."
        )

    indexed = _index_for_llm(kept_by_pitch, structure, slots_per_beat=slots_per_beat)
    if not indexed:
        return {}, "ok"

    windows = _build_windows(
        indexed, structure,
        target_onsets=_TARGET_ONSETS_PER_WINDOW,
        max_bars=_MAX_BARS_PER_WINDOW,
        context_bars=_CONTEXT_BARS,
    )
    template = _load_prompt_template()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    log.info(
        "quantise LLM: %d onsets across %d bars -> %d window(s), "
        "<=%d concurrent",
        len(indexed), len(structure.bars), len(windows), _MAX_PARALLEL_CHUNKS,
    )

    shifts: dict[tuple[str, int], int] = {}
    failed = 0
    cancelled = False
    max_workers = max(1, min(_MAX_PARALLEL_CHUNKS, len(windows)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for wi, window in enumerate(windows):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            # Submit through a copy of the submitting thread's context so
            # contextvars propagate into the pool workers;
            # ThreadPoolExecutor does NOT copy them. This carries the
            # request id (for log correlation) plus the debug-sink /
            # run-log contextvars; copy_context() runs here, on the
            # submitting thread, which is already under the pipeline's
            # asyncio.to_thread context and so holds those values.
            futures[pool.submit(
                contextvars.copy_context().run,
                _call_window, client, template, structure, window, wi,
                slots_per_beat, cancel_event,
            )] = wi
        for fut in as_completed(futures):
            wi = futures[fut]
            try:
                window_shifts = fut.result()
            except Exception as exc:
                failed += 1
                log.warning(
                    "quantise LLM: window %d failed (%s); keeping geometric "
                    "placement for its bars", wi, exc,
                )
                continue
            if window_shifts is None:  # window short-circuited on cancel
                cancelled = True
                continue
            shifts.update(window_shifts)

    n = len(windows)
    if cancelled:
        status = "cancelled"
    elif failed == 0:
        status = "ok"
    elif failed >= n:
        status = f"error: all {n} windows failed"
    else:
        status = f"partial: {failed}/{n} windows failed"
    log.info(
        "quantise LLM: %d onsets received a non-zero shift (status=%s)",
        len(shifts), status,
    )
    return shifts, status


def _build_windows(
    indexed: list[_LlmEntry],
    structure: BeatStructure,
    *,
    target_onsets: int,
    max_bars: int,
    context_bars: int,
) -> list[_Window]:
    """Partition the globally-sorted `indexed` entries into bar windows.

    Walks the onset-bearing bars in ascending order, accumulating them
    into a window until adding the next bar would exceed `target_onsets`
    (a soft cap, a lone dense bar may exceed it) or push the window's
    bar-index span past `max_bars`. Each window also picks up the
    ±`context_bars` neighbour bars (clamped to the score) as read-only
    context. Local `#id`s are assigned to core-bar onsets in global sort
    order so a window's response maps cleanly back to (pitch, idx).
    """
    by_bar: dict[int, list[tuple[int, _LlmEntry]]] = {}
    for gid, e in enumerate(indexed):
        by_bar.setdefault(e.bar, []).append((gid, e))
    core_bars_sorted = sorted(by_bar.keys())

    groups: list[list[int]] = []
    cur: list[int] = []
    cur_onsets = 0
    for bar_idx in core_bars_sorted:
        n = len(by_bar[bar_idx])
        if cur and (
            cur_onsets + n > target_onsets or bar_idx - cur[0] >= max_bars
        ):
            groups.append(cur)
            cur, cur_onsets = [], 0
        cur.append(bar_idx)
        cur_onsets += n
    if cur:
        groups.append(cur)

    last_bar = len(structure.bars) - 1
    windows: list[_Window] = []
    for core in groups:
        core_set = set(core)
        lo = max(0, core[0] - context_bars)
        hi = min(last_bar, core[-1] + context_bars)
        # Local id assignment in global sort order (core bars only).
        local_to_global: list[tuple[str, int]] = []
        gid_to_lid: dict[int, int] = {}
        for bar_idx in core:
            for gid, e in by_bar[bar_idx]:
                gid_to_lid[gid] = len(local_to_global)
                local_to_global.append((e.pitch, e.idx))
        windows.append(_Window(
            core_set=core_set,
            render_bars=list(range(lo, hi + 1)),
            by_bar=by_bar,
            gid_to_lid=gid_to_lid,
            local_to_global=local_to_global,
        ))
    return windows


def _call_window(
    client: anthropic.Anthropic,
    template: str,
    structure: BeatStructure,
    window: _Window,
    window_index: int,
    slots_per_beat: int,
    cancel_event: threading.Event | None,
) -> dict[tuple[str, int], int] | None:
    """Run one window's forced-tool call; return its (pitch, idx) -> shift.

    Returns None if `cancel_event` is set before the call is made (the
    caller treats this as cancelled, not failed). Raises on API error so
    the caller can degrade just this window's bars to geometric.
    """
    if cancel_event is not None and cancel_event.is_set():
        return None

    n_local = len(window.local_to_global)
    bar_blocks = _format_window(structure, window, slots_per_beat=slots_per_beat)
    initial_sig = structure.initial_time_signature
    prompt = (
        template
        .replace("{INITIAL_TEMPO}", f"{structure.initial_tempo:.2f}")
        .replace("{INITIAL_TIME_SIG}", f"{initial_sig[0]}/{initial_sig[1]}")
        .replace("{BAR_COUNT}", str(len(window.render_bars)))
        .replace("{ONSET_COUNT}", str(n_local))
        .replace("{SLOTS_PER_BEAT}", str(slots_per_beat))
        .replace("{MAX_SHIFT}", str(_MAX_LLM_SHIFT))
        .replace("{BARS}", bar_blocks)
    )

    max_tokens = _max_tokens_for(n_local)
    log.info(
        "quantise LLM window %d: prompt_chars=%d shiftable_onsets=%d "
        "max_tokens=%d", window_index, len(prompt), n_local, max_tokens,
    )
    response = call_messages_with_refusal_retry(
        client,
        {
            "model": _LLM_MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [_QUANTISE_TOOL],
            "tool_choice": {"type": "tool", "name": _QUANTISE_TOOL["name"]},
        },
        base_prompt=prompt,
        purpose=f"quantise_w{window_index:02d}",
    )

    if getattr(response, "stop_reason", None) == "max_tokens":
        # A forced tool call that hits max_tokens emits unparseable JSON;
        # `block.input` will be empty / partial and we'd silently report
        # "0 shifts". Flag it so it's obvious from the log next time.
        log.warning(
            "quantise LLM window %d hit max_tokens (%d) for %d onsets; "
            "response will be truncated and most shifts lost. Bump "
            "_LLM_MAX_TOKENS_PER_ONSET in quantise.py.",
            window_index, max_tokens, n_local,
        )
    raw_shifts = _extract_shifts(response, n_local)
    out: dict[tuple[str, int], int] = {}
    for local_id, shift in raw_shifts.items():
        if local_id < 0 or local_id >= n_local:
            continue
        clamped = max(-_MAX_LLM_SHIFT, min(_MAX_LLM_SHIFT, shift))
        if clamped == 0:
            continue
        out[window.local_to_global[local_id]] = clamped
    return out


def _index_for_llm(
    kept_by_pitch: dict[str, list[OnsetCandidate]],
    structure: BeatStructure,
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
) -> list[_LlmEntry]:
    """Build a stable LLM-facing list of `(bar, slot, pitch)` entries.

    Ordering: by `(bar, current_slot, pitch)` so a human can cross-read
    the prompt and the response. Out-of-range and off-grid onsets are
    excluded (off-grid onsets are geometric findings, not jitter to
    correct, so they're never offered to the model as shift targets).
    """
    entries: list[_LlmEntry] = []
    for pitch in sorted(kept_by_pitch.keys()):
        for idx, c in enumerate(kept_by_pitch[pitch]):
            if c.off_grid:
                continue
            bar_idx = int(c.bar)
            if bar_idx < 0 or bar_idx >= len(structure.bars):
                continue
            bar = structure.bars[bar_idx]
            current_time = c.quantised_time if c.quantised_time is not None else c.time
            num_beats = max(int(bar.time_signature[0]), 1)
            slot_span = (bar.end_time - bar.start_time) / (num_beats * slots_per_beat)
            if slot_span <= 0:
                continue
            current_slot = round((current_time - bar.start_time) / slot_span)
            current_slot = max(0, min(num_beats * slots_per_beat - 1, current_slot))
            entries.append(_LlmEntry(
                pitch=pitch, idx=idx, bar=bar_idx, slot=current_slot,
                residual=c.quantised_residual_slots,
            ))
    entries.sort(key=lambda e: (e.bar, e.slot, e.pitch))
    return entries


class _LlmEntry:
    __slots__ = ("pitch", "idx", "bar", "slot", "residual")

    def __init__(
        self, pitch: str, idx: int, bar: int, slot: int,
        residual: float | None = None,
    ) -> None:
        self.pitch = pitch
        self.idx = idx
        self.bar = bar
        self.slot = slot
        # Geometric snap's signed sub-slot residual (range (-0.5, +0.5]),
        # surfaced to the model as a near-miss hint. None when unknown.
        self.residual = residual


class _Window:
    """One LLM call's worth of bars.

    `core_set` holds the bar indices whose onsets are shiftable (rendered
    with `#id`s); `render_bars` is the full ordered span to print (core +
    read-only context bars). `gid_to_lid` maps a global onset id (its
    position in the shared `indexed` list) to this window's local id;
    `local_to_global` is the inverse for shiftable onsets only, mapping a
    local id back to its `(pitch, idx)`. `by_bar` is the shared
    global-id-keyed grouping (read-only here).
    """
    __slots__ = ("core_set", "render_bars", "by_bar", "gid_to_lid", "local_to_global")

    def __init__(
        self,
        core_set: set[int],
        render_bars: list[int],
        by_bar: dict[int, list[tuple[int, _LlmEntry]]],
        gid_to_lid: dict[int, int],
        local_to_global: list[tuple[str, int]],
    ) -> None:
        self.core_set = core_set
        self.render_bars = render_bars
        self.by_bar = by_bar
        self.gid_to_lid = gid_to_lid
        self.local_to_global = local_to_global


def _format_window(
    structure: BeatStructure,
    window: _Window,
    *,
    slots_per_beat: int = SLOTS_PER_BEAT,
) -> str:
    """Render a window's bars: one block per bar, onset rows grouped by slot.

    Core bars carry shiftable onsets tagged with their window-local `#id`.
    Context bars are tagged `[context - read-only]` and their onsets are
    rendered WITHOUT an id, so the model can see the surrounding groove
    but cannot (and is told not to) shift them.
    """
    blocks: list[str] = []
    for bar_idx in window.render_bars:
        bar = structure.bars[bar_idx]
        is_core = bar_idx in window.core_set
        tag = "" if is_core else " [context - read-only]"
        header = (
            f"Bar {bar.index} "
            f"[{bar.time_signature[0]}/{bar.time_signature[1]}, "
            f"{bar.tempo_bpm:.1f} BPM, feel={bar.feel}]{tag}:"
        )
        rows = [header]
        bar_entries = window.by_bar.get(bar_idx, [])
        if not bar_entries:
            rows.append("  (no onsets)")
            blocks.append("\n".join(rows))
            continue
        # Group by slot for compactness.
        by_slot: dict[int, list[tuple[int, _LlmEntry]]] = {}
        for gid, e in bar_entries:
            by_slot.setdefault(e.slot, []).append((gid, e))
        for slot in sorted(by_slot.keys()):
            slot_entries = by_slot[slot]
            beat_label = _slot_label(slot, slots_per_beat)
            if is_core:
                rendered = " ".join(
                    f"#{window.gid_to_lid[gid]}({e.pitch}{_residual_tag(e.residual)})"
                    for gid, e in slot_entries
                )
            else:
                rendered = " ".join(f"({e.pitch})" for _gid, e in slot_entries)
            # Display slot is 1-indexed (slot 1 = downbeat, matching the
            # 1-indexed beats); the internal `slot` int stays 0-indexed
            # everywhere else (geometric DP, envelope re-snap, grid snap)
            # so this is purely a render-time convention for the prompt.
            rows.append(f"  slot {slot + 1:>2} {beat_label}: {rendered}")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def _residual_tag(residual: float | None) -> str:
    """Compact near-miss hint for a core onset, e.g. " r+0.45".

    Only emitted for a notable sub-slot residual (|r| >= 0.25); a small or
    absent residual is left unannotated. Surfaces how far the raw audio sat
    from the slot so the model knows which rounds were coin-flips. NOT a
    correctness signal on its own (a systematic full-slot offset rounds
    cleanly, residual ~ 0), the prompt says to weigh musical context, not r.
    """
    if residual is None or abs(residual) < 0.25:
        return ""
    return f" r{residual:+.2f}"


def _slot_label(slot: int, slots_per_beat: int) -> str:
    """Human-readable label for a slot, e.g. "(beat 2)" or "(& of 1)".

    Positions are fractions of `slots_per_beat`, so the labels track the
    active grid density rather than a hardcoded 12. Named positions are
    emitted only when they land on an integer slot for this density
    (always so at 12 = 1/48); otherwise a generic fallback names the
    whole-note subdivision.
    """
    beat = slot // slots_per_beat + 1  # 1-indexed beat
    within = slot % slots_per_beat
    if within == 0:
        return f"(beat {beat})"
    if within == slots_per_beat // 2:
        return f"(& of {beat})"
    if within == slots_per_beat // 4:
        return f"(e of {beat})"
    if within == 3 * slots_per_beat // 4:
        return f"(a of {beat})"
    if slots_per_beat % 3 == 0 and within == slots_per_beat // 3:
        return f"(trip-2 of {beat})"
    if slots_per_beat % 3 == 0 and within == 2 * slots_per_beat // 3:
        return f"(trip-3 of {beat})"
    return f"(1/{4 * slots_per_beat} +{within} of {beat})"


def _extract_shifts(
    response: anthropic.types.Message, n: int
) -> dict[int, int]:
    """Pull `shifts` from the forced tool call. Returns `{id: shift}`,
    ignoring entries that aren't well-formed; clamping happens at the
    call site (so the dict here can carry the raw values for logging)."""
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != _QUANTISE_TOOL["name"]:
            continue
        raw = block.input.get("shifts", [])
        if not isinstance(raw, list):
            log.warning(
                "quantise: tool returned non-list shifts (%s); "
                "treating as no shifts", type(raw).__name__,
            )
            return {}
        out: dict[int, int] = {}
        for row in raw:
            if not isinstance(row, dict):
                continue
            try:
                rid = int(row.get("id"))
                shift = int(row.get("shift"))
            except (TypeError, ValueError):
                continue
            if 0 <= rid < n:
                out[rid] = shift
        return out
    log.warning("quantise: no tool_use block in response; no shifts applied")
    return {}


def _load_prompt_template() -> str:
    return (PROMPT_DIR / "quantise_onsets.md").read_text(encoding="utf-8")
