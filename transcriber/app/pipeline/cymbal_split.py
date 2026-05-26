"""Split the merged cymbals stem's onsets into ride (`d`), crash (`c`),
and discard (artifact rejected upstream of the filter LLM).

The active Stage-2 separator (jarredou 5-stem MDX23C DrumSep) does not
separate ride from crash; it emits ONE `cymbals` stem, mapped to pitch
`c` by `separate.STEM_NAME_TO_PITCH`. Ride vs crash is a poor fit for a
fixed timbre threshold (a washy or "crash-ridden" ride decays long and
noisy like a crash; the reliably discriminating signal is the musical
*role*; a sustained timekeeping stream vs a sparse accent; not timbre),
so we follow the same line this pipeline draws everywhere: deterministic
measurement, LLM judgement.

  - Deterministic: for each cymbals onset we measure, off the cymbals
    stem audio, a post-onset decay time, spectral flatness, spectral
    centroid, and the gap to the nearest neighbouring cymbal onset; the
    acoustic / density cues the LLM cannot otherwise perceive.
  - LLM: those features plus the musical context (beat positions + a
    one-line summary of what every other instrument did per bar, so it
    can see kick-coincident accents) go to one forced-tool call that
    returns the ride / crash / discard classification. Ride is the
    default (anything the model does not flag is ride).

The **discard** category mirrors `hihat_split.py`'s ternary classifier:
the LLM rejects detector artifacts (bleed from hi-hat / snare /
percussion that lines up with `others:`, double-triggers, and sizzle
re-triggers inside a long crash tail) in the same call that classifies
ride vs crash. This collapses two passes (split, then per-lane
`filter_llm`) into one; the model uses the same picture (full features
+ cross-instrument context) to judge "is this an artifact?" that it
uses to judge "is this a ride?". The `filter_llm` pass is skipped
entirely for `c` / `d` afterwards; see
`pipeline/runner.py::_do_transcribe`.

On any failure (no API key, call error, no tool block) we fall back to
a coarse deterministic ride/crash rule over the same features (no
discards) so the cymbal lane is never dropped; mirroring `filter_llm`'s
degrade-to-safe contract.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anthropic
import librosa
import numpy as np

from app.config import settings
from app.debug import current_debug_sink
from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.llm_util import call_messages_with_refusal_retry
from app.pipeline.separate import PITCH_DISPLAY_NAMES

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# Pitch letters (must match separate.STEM_NAME_TO_PITCH).
_CYMBALS_PITCH = "c"  # the merged cymbals stem lands here pre-split
_RIDE_PITCH = "d"
_CRASH_PITCH = "c"

# Output-pitch -> input-stem-pitch aliases. After the split runs there
# are two output lanes (`c` crash and `d` ride) sharing the single
# combined cymbals stem (`stem_c.mp3`). The debug-bundle builder
# (`debug_bundle.py`) reads this so the manifest's `mapping` declares
# the shared stem under BOTH keys; `c → stem_c.mp3` AND
# `d → stem_c.mp3`; letting the frontend cluster ride alongside crash
# under the cymbals audio row instead of recomputing the relationship.
# The crash → crash entry is implicit (it's just the stem's own pitch);
# only the non-identity alias needs to be listed here.
STEM_PITCH_ALIASES: dict[str, str] = {
    _RIDE_PITCH: _CYMBALS_PITCH,
}

# Feature-extraction windows (seconds). The decay search is capped so a
# long crash tail can't run into the next phrase; the timbre window is
# short so it characterises the attack/early-decay, not room ambience.
_DECAY_MAX_S = 3.0
_DECAY_DROP_DB = -20.0  # decay = time for RMS to fall this far below peak
_TIMBRE_WIN_S = 0.18

# Coarse deterministic fallback: a crash both rings long AND is not part
# of a tight stream (an isolated, sustained hit). A ride ping in a
# timekeeping pattern is short and densely spaced.
_FALLBACK_DECAY_S = 0.70
_FALLBACK_ISOLATION_S = 0.25

_SPLIT_TOOL: dict[str, Any] = {
    "name": "report_cymbal_classification",
    "description": (
        "Classify each detected cymbal onset into one of three buckets: "
        "CRASH (sparse accent, long sustained decay, often coincident "
        "with a kick on a strong beat), RIDE (steady timekeeping stream, "
        "short articulate decay), or DISCARD (not a real hit; bleed "
        "from another cymbal-like instrument lining up with `others:`, "
        "double-triggers, or sizzle re-triggers inside a long crash "
        "tail). Return TWO arrays of `#N` indices: the crashes and the "
        "discards. Every onset NOT in either array is treated as RIDE. "
        "The two arrays must be disjoint. Return empty arrays when "
        "appropriate (pure ride; nothing to discard). Never include an "
        "index that wasn't shown to you."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "crash_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "description": (
                    "The `#N` indices of onsets that are CRASH hits."
                ),
            },
            "discard_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "description": (
                    "The `#N` indices of onsets that are NOT real hits "
                    "(bleed, double-triggers, sizzle re-triggers in a "
                    "crash tail). These should be the minority; only "
                    "clear artifacts."
                ),
            },
        },
        "required": ["crash_indices", "discard_indices"],
        "additionalProperties": False,
    },
}


class _Feat:
    """Per-onset measured features (kept off `OnsetCandidate` so the split
    stays local and doesn't widen the pipeline-wide onset schema)."""

    __slots__ = ("decay_s", "flatness", "centroid_hz", "gap_s")

    def __init__(
        self, decay_s: float, flatness: float, centroid_hz: float, gap_s: float
    ) -> None:
        self.decay_s = decay_s
        self.flatness = flatness
        self.centroid_hz = centroid_hz
        self.gap_s = gap_s


def split_cymbal_onsets(
    onsets_by_pitch: dict[str, list[OnsetCandidate]],
    per_instrument_stems: dict[str, Path],
    structure: BeatStructure,
    *,
    llm_model: str | None = None,
) -> tuple[dict[str, list[OnsetCandidate]], list[OnsetCandidate]]:
    """Return `(onsets_by_pitch_with_split, discarded_onsets)`.

    The first element is `onsets_by_pitch` with the cymbals lane split
    into ride (`d`) and crash (`c`). The second is the list of in-range
    cymbal onsets the classifier rejected as artifacts; kept around so
    the UI's "Show filtered" ghost overlay can surface them via
    `note_provenance`. The discards are *not* present in either lane;
    the runner merges them back into `all_onsets_by_pitch[c]` at the
    provenance boundary only.

    No-op (returns the input mapping unchanged + an empty discarded list)
    when there is no cymbals stem, no cymbals onsets, or no in-range
    cymbals onsets to classify.
    """
    cym = onsets_by_pitch.get(_CYMBALS_PITCH)
    stem_path = per_instrument_stems.get(_CYMBALS_PITCH)
    if not cym or stem_path is None or not stem_path.exists():
        return onsets_by_pitch, []

    in_range = sorted(
        (c for c in cym if c.bar >= 0), key=lambda c: (c.bar, c.beat_in_bar)
    )
    out_of_range = [c for c in cym if c.bar < 0]
    if not in_range:
        return onsets_by_pitch, []

    feats = _measure(stem_path, in_range)

    llm_result = _classify_llm(in_range, feats, structure, onsets_by_pitch, llm_model=llm_model)
    if llm_result is None:
        crash_idx, discard_idx = _classify_fallback(in_range, feats)
        source = "fallback"
    else:
        crash_idx, discard_idx = llm_result
        source = "llm"

    ride = [
        c for i, c in enumerate(in_range)
        if i not in crash_idx and i not in discard_idx
    ]
    crash = [c for i, c in enumerate(in_range) if i in crash_idx]
    discarded = [c for i, c in enumerate(in_range) if i in discard_idx]
    # Out-of-range cymbal onsets are never consumed downstream (bar < 0);
    # park them on the crash lane so nothing is silently discarded.
    crash.extend(out_of_range)

    log.info(
        "cymbal split (%s): %d onsets -> %d ride, %d crash, %d discard",
        source,
        len(in_range),
        len(ride),
        len(crash) - len(out_of_range),
        len(discarded),
    )

    sink = current_debug_sink()
    if sink is not None:
        sink.write_json(
            "cymbal_split/decision.json",
            {
                "source": source,
                "n_input": len(in_range),
                "n_ride": len(ride),
                "n_crash": len(crash) - len(out_of_range),
                "n_discard": len(discarded),
                "onsets": [
                    {
                        "index": i,
                        "bar": c.bar,
                        "beat_in_bar": round(c.beat_in_bar, 3),
                        "strength": round(c.strength, 3),
                        "decay_s": round(feats[i].decay_s, 3),
                        "flatness": round(feats[i].flatness, 4),
                        "centroid_hz": round(feats[i].centroid_hz, 1),
                        "gap_s": round(feats[i].gap_s, 3),
                        "label": (
                            "discard" if i in discard_idx
                            else "crash" if i in crash_idx
                            else "ride"
                        ),
                    }
                    for i, c in enumerate(in_range)
                ],
            },
        )

    out = dict(onsets_by_pitch)
    # Merge rather than overwrite in case a `d` lane somehow already
    # exists (it won't with the 5-stem model, but stay defensive).
    if ride:
        out[_RIDE_PITCH] = sorted(
            out.get(_RIDE_PITCH, []) + ride,
            key=lambda c: (c.bar, c.beat_in_bar),
        )
    if crash:
        out[_CRASH_PITCH] = sorted(
            crash, key=lambda c: (c.bar, c.beat_in_bar)
        )
    else:
        out.pop(_CRASH_PITCH, None)
    return out, discarded


def _measure(
    stem_path: Path, onsets: list[OnsetCandidate]
) -> list[_Feat]:
    """Measure decay / flatness / centroid / neighbour-gap per onset.

    The cymbals stem is loaded once. Decay is the time for post-onset RMS
    to fall `_DECAY_DROP_DB` below its local peak, searched only up to the
    next cymbal onset (capped at `_DECAY_MAX_S`) — so a ride ping in a
    dense stream measures short by construction, an isolated crash long.
    """
    sr = 44100
    audio, sr = librosa.load(str(stem_path), sr=sr, mono=True)
    hop = 512
    rms = librosa.feature.rms(y=audio, hop_length=hop)[0]
    rms_t = librosa.times_like(rms, sr=sr, hop_length=hop)
    n = len(onsets)
    out: list[_Feat] = []
    for i, c in enumerate(onsets):
        t = c.time
        nxt = onsets[i + 1].time if i + 1 < n else t + _DECAY_MAX_S
        prev = onsets[i - 1].time if i > 0 else None
        gap = nxt - t
        if prev is not None:
            gap = min(gap, t - prev)

        win_end = min(nxt, t + _DECAY_MAX_S)
        seg = (rms_t >= t) & (rms_t <= win_end)
        if not np.any(seg):
            out.append(_Feat(0.0, 0.0, 0.0, float(gap)))
            continue
        seg_rms = rms[seg]
        seg_t = rms_t[seg]
        peak_i = int(np.argmax(seg_rms))
        peak = float(seg_rms[peak_i])
        if peak <= 0.0:
            decay_s = 0.0
        else:
            thresh = peak * (10.0 ** (_DECAY_DROP_DB / 20.0))
            below = np.where(seg_rms[peak_i:] < thresh)[0]
            decay_s = (
                float(seg_t[peak_i + below[0]] - seg_t[peak_i])
                if below.size
                else float(seg_t[-1] - seg_t[peak_i])
            )

        a0 = int(t * sr)
        a1 = min(len(audio), int((t + _TIMBRE_WIN_S) * sr))
        clip = audio[a0:a1]
        if clip.size >= hop:
            flat = float(np.mean(librosa.feature.spectral_flatness(y=clip)))
            cen = float(
                np.mean(librosa.feature.spectral_centroid(y=clip, sr=sr))
            )
        else:
            flat, cen = 0.0, 0.0
        out.append(_Feat(decay_s, flat, cen, float(gap)))
    return out


def _classify_llm(
    onsets: list[OnsetCandidate],
    feats: list[_Feat],
    structure: BeatStructure,
    onsets_by_pitch: dict[str, list[OnsetCandidate]],
    *,
    llm_model: str | None = None,
) -> tuple[set[int], set[int]] | None:
    """Ask the LLM to classify each onset crash / ride / discard.

    Returns `(crash_indices, discard_indices)`; everything not in either
    set is implicitly ride. The two sets are guaranteed disjoint; overlapping entries resolve to **discard** (the safer error: a real
    hit lost as discard is one missed note; a sizzle re-trigger
    mislabelled as a crash creates a phantom accent that the rest of the
    pipeline will treat as a section boundary).

    Returns `None` to signal the caller to use the deterministic
    fallback (no API key, call error, or malformed tool output).
    """
    if not settings.anthropic_api_key:
        log.info("cymbal split: no ANTHROPIC_API_KEY; using fallback")
        return None

    bar_blocks = _format_bars(onsets, feats, structure, onsets_by_pitch)
    initial_sig = structure.initial_time_signature
    prompt = (
        _load_prompt_template()
        .replace("{INITIAL_TEMPO}", f"{structure.initial_tempo:.2f}")
        .replace("{INITIAL_TIME_SIG}", f"{initial_sig[0]}/{initial_sig[1]}")
        .replace("{BAR_COUNT}", str(len(structure.bars)))
        .replace("{ONSET_COUNT}", str(len(onsets)))
        .replace("{BARS}", bar_blocks)
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    model = llm_model or settings.llm_model
    try:
        response = call_messages_with_refusal_retry(
            client,
            {
                "model": model,
                "max_tokens": settings.llm_max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [_SPLIT_TOOL],
                "tool_choice": {"type": "tool", "name": _SPLIT_TOOL["name"]},
            },
            base_prompt=prompt,
            purpose="cymbal_split",
        )
    except Exception as exc:
        log.warning(
            "cymbal split: LLM call failed (%s); using fallback", exc
        )
        return None

    n = len(onsets)
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != _SPLIT_TOOL["name"]:
            continue
        crash_raw = block.input.get("crash_indices", [])
        discard_raw = block.input.get("discard_indices", [])
        if not isinstance(crash_raw, list) or not isinstance(discard_raw, list):
            log.warning(
                "cymbal split: non-list crash/discard indices; using fallback"
            )
            return None
        discard_set = _coerce_index_set(discard_raw, n)
        # Disjointness: discard wins on overlap (see docstring).
        crash_set = _coerce_index_set(crash_raw, n) - discard_set
        return crash_set, discard_set
    log.warning("cymbal split: no tool_use block; using fallback")
    return None


def _coerce_index_set(raw: list[Any], n: int) -> set[int]:
    """Clamp a list of tool-returned indices into `[0, n)`, ignoring
    non-int entries and out-of-range values."""
    out: set[int] = set()
    for v in raw:
        try:
            idx = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n:
            out.add(idx)
    return out


def _classify_fallback(
    onsets: list[OnsetCandidate], feats: list[_Feat]
) -> tuple[set[int], set[int]]:
    """Coarse deterministic ride/crash split over the measured features.

    Crash = rings long AND is isolated (not part of a tight stream).
    Never discards; "do nothing about artifacts" is acceptable degraded
    behaviour when the LLM is unavailable; the goal is "never drop the
    lane", not accuracy parity with the model. Runs only when the LLM
    is unavailable.
    """
    crash: set[int] = set()
    for i, f in enumerate(feats):
        if f.decay_s >= _FALLBACK_DECAY_S and f.gap_s >= _FALLBACK_ISOLATION_S:
            crash.add(i)
    return crash, set()


def _format_bars(
    onsets: list[OnsetCandidate],
    feats: list[_Feat],
    structure: BeatStructure,
    onsets_by_pitch: dict[str, list[OnsetCandidate]],
) -> str:
    """Render per-bar blocks: indexed cymbal onsets with their measured
    features, plus a compact one-line summary of every other instrument's
    hits in that bar (so the model can spot kick-coincident accents)."""
    if not structure.bars:
        return "(no bars detected)"

    by_bar: dict[int, list[tuple[int, OnsetCandidate, _Feat]]] = {}
    for i, c in enumerate(onsets):
        by_bar.setdefault(c.bar, []).append((i, c, feats[i]))

    others_by_bar: dict[int, list[tuple[str, float]]] = {}
    for op, cands in onsets_by_pitch.items():
        if op == _CYMBALS_PITCH:
            continue
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
                f"#{i}(b{c.beat_in_bar:.2f},str{c.strength:.2f},"
                f"dec{ft.decay_s:.2f}s,flat{ft.flatness:.3f},"
                f"cen{ft.centroid_hz/1000.0:.1f}k,gap{ft.gap_s:.2f}s)"
                for i, c, ft in entries
            )
            rows.append(f"  cymbals: {rendered}")
        else:
            rows.append("  cymbals: (none)")
        others = sorted(others_by_bar.get(bar.index, []), key=lambda x: x[1])
        if others:
            summary = " ".join(
                f"{PITCH_DISPLAY_NAMES.get(op, op)}{pos:.2f}"
                for op, pos in others
            )
            rows.append(f"  others: {summary}")
        blocks.append("\n".join(rows))

    return "\n\n".join(blocks)


def _load_prompt_template() -> str:
    return (PROMPT_DIR / "split_cymbals.md").read_text(encoding="utf-8")
