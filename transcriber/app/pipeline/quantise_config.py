"""Shared constants + the LLM tool schema for the quantise passes.

Split out of `quantise.py` so the config the passes share sits in one
place with no import cycle (every quantise submodule imports from here,
none imports back).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# 1/48 of a whole note = 12 slots per quarter-note beat. Matches
# `src/midi/from_midi.ts::gridDivision`'s default; the frontend snap
# becomes a no-op on MIDI produced by this stage. Threaded as a parameter
# (default here) so the grid density is not hard-assumed downstream.
SLOTS_PER_BEAT = 12

# Match band for the geometric snap: the farthest (in slots) an onset may
# be pulled to reach a free slot. Beyond it, the onset is left off-grid.
# ±2 slots ≈ 83 ms at 120 BPM, so rejection is rare; it's the only control
# on off-grid promotion now that the cross-instrument cluster pull is gone.
_MATCH_BAND = 2

# Off-grid penalty for the DP: strictly worse than any in-band placement
# (cost <= band^2), so an onset goes off-grid only when no in-band slot is
# free. `(band + 1) ** 2` per the design spec.
_OFF_GRID_PENALTY = float((_MATCH_BAND + 1) ** 2)

# Maximum |shift| accepted from the LLM. Clamped server-side regardless
# of what the model returns; the model isn't a re-quantiser, it's a
# jitter corrector.
_MAX_LLM_SHIFT = 2

# Haiku model id for this stage. Quantisation correction is constrained
# pattern-matching; same tier as the filter stage.
_LLM_MODEL = "claude-haiku-4-5-20251001"

# Per-call token budget. The prompt asks the model to return entries
# ONLY for non-zero shifts (mirror of `filter_llm`'s "rejected_onsets"
# pattern), so the natural response size is proportional to the number
# of jitter corrections needed; not the total onset count. We still
# size the cap from `n_onsets` as defence-in-depth: a model that
# ignores the prompt and emits one entry per onset measured at
# ~13 tokens/entry in the wild (Haiku 4.5: `{"id":1234,"shift":-2},`
# tokenises to ~10–13 tokens depending on id width and field ordering),
# so the per-onset multiplier is set to 16 for headroom. Haiku 4.5
# supports 64K output tokens so the cap stays generous.
_LLM_MAX_TOKENS_PER_ONSET = 16
_LLM_MAX_TOKENS_FLOOR = 8192
_LLM_MAX_TOKENS_OVERHEAD = 1024

# The LLM residual pass is split into windows of consecutive bars that run
# concurrently. Splitting (a) shrinks per-call latency and lets windows
# overlap on the wire, and (b) keeps each prompt small enough that Haiku
# reasons sharply over it instead of skimming a 1000+-onset wall. Onsets
# never cross a bar boundary, so windows always break on bar boundaries.
#
# A window accumulates consecutive (onset-bearing) bars until either the
# onset target or the bar-span cap is reached, whichever comes first. The
# onset target is a SOFT cap: a single dense bar that exceeds it alone
# still becomes its own window (we never split a bar). Each window also
# renders +/-_CONTEXT_BARS neighbour bars read-only so groove continuity
# across the window seam is visible without those bars being shiftable.
_TARGET_ONSETS_PER_WINDOW = 150
_MAX_BARS_PER_WINDOW = 8       # max bar-index span of a window's core bars
_CONTEXT_BARS = 1              # read-only neighbour bars rendered per side
_MAX_PARALLEL_CHUNKS = 8       # cap on concurrent Anthropic requests

# Deterministic musical-grid pass (runs between the geometric snap and the
# LLM residual pass). It infers, per (lane, bar), which subdivision grid
# the surrounding rhythm is using, then snaps onsets onto that grid. Unlike
# the geometric snap (which reasons purely from audio timing), this pass
# uses the *population* of onsets to recover the slot a hit musically
# belongs on, including the case where a performer played a consistent full
# slot off the beat and rounded cleanly onto the wrong slot. Tuplet/swing
# safety is structural: a note is only judged off-grid relative to a grid
# its own lane voted for, so genuine triplets/shuffle/poly-rhythm survive.
_GRID_MIN_ONSETS = 4            # min onsets for a (lane, bar) to vote a grid
_GRID_COMPLEXITY_PENALTY = 0.5  # Occam cost per grid slot (favours simpler)
_GRID_DECISIVE_MARGIN = 0.3     # winner must beat runner-up by this (else skip)
_GRID_SNAP_TOLERANCE = 1        # max |slot shift| this pass will apply

# Per-note envelope re-snap (runs right after the geometric snap). Every
# other quantise pass trusts the onset *time* and never looks at the audio,
# so a hit whose detection locked onto the wrong (early) envelope max stays
# misplaced, on a real transient, just the wrong one. This pass samples the
# lane's onset-strength envelope in each candidate slot's time-bin and moves
# the note to the bin holding the strongest transient, but only when that
# bin's energy clearly dominates the current slot's (so it never nudges a
# note that's already on its hit, and stays put when the audio is ambiguous).
# Bounded to ±tolerance slots and applied through the monotonic-injective
# guard, so it can't collide or reorder onsets.
_ENV_RESNAP_TOLERANCE = 2       # max |slot shift| this pass will apply
_ENV_RESNAP_DOMINANCE = 2.0     # target bin must be >this × the current bin
_ENV_RESNAP_FLOOR_FRAC = 0.15   # target bin must clear this × envelope ref


def _max_tokens_for(n_onsets: int) -> int:
    """Headroom-aware token budget for the forced-tool quantise response."""
    return max(
        _LLM_MAX_TOKENS_FLOOR,
        n_onsets * _LLM_MAX_TOKENS_PER_ONSET + _LLM_MAX_TOKENS_OVERHEAD,
    )


_QUANTISE_TOOL: dict[str, Any] = {
    "name": "shift_onsets",
    "description": (
        "Return ONLY the onsets that need to move; omit any onset that "
        "should stay where it is. Each entry is a signed integer "
        "1/48-slot shift (negative = earlier; positive = later). Use "
        "surrounding musical context across instruments to decide which "
        "onsets to shift. Bounded |shift| <= 2; anything larger will be "
        "clamped. An empty `shifts` array is the correct answer when "
        "every onset is already correctly placed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "shifts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "minimum": 0},
                        "shift": {
                            "type": "integer",
                            "minimum": -_MAX_LLM_SHIFT,
                            "maximum": _MAX_LLM_SHIFT,
                        },
                    },
                    "required": ["id", "shift"],
                    "additionalProperties": False,
                },
                "description": (
                    "Onsets to shift; identified by the `#N` id shown in "
                    "the prompt. Include ONLY onsets that need to move, "
                    "omit any onset that should stay where it is. Empty "
                    "array means nothing needs shifting. Unknown ids are "
                    "ignored; the server clamps shift to the allowed range."
                ),
            },
        },
        "required": ["shifts"],
        "additionalProperties": False,
    },
}
