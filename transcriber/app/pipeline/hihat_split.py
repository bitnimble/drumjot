"""Split the merged hi-hat stem's onsets into closed (`h`) vs open (`H`).

The Stage-2 separator emits a single `(hh)` stem mapped to pitch `h`, but
a real hi-hat performance interleaves closed strikes and open (sustained)
hits — and ADTOF (even in-distribution on the drum stem) reads the
open-hat sizzle/ring as a stream of confident frame-level activations.
Pushing both flavours through one per-instrument LLM call buries the
closed pattern under that sizzle.

The split classifies each detected hi-hat onset open vs closed off the
isolated stem audio (deterministic features the LLM can't hear) and
routes them into two lanes that the per-instrument transcribe pass handles
as separate instruments. Identity comes from the audio; the transcription
LLM gets to reason about a (mostly) clean closed lane and an (open) lane
in isolation.

Architectural notes:

* **Synthetic pitch `H` for open hi-hat.** Drumjot's DSL has only one
  notational hi-hat pitch (`h`) with `:o` / `:c` modifiers, so `H` is an
  *internal* routing key for the transcribe pass — it must not leak as a
  permanent extra voice in finished transcriptions. First cut keeps it
  visible: `recompose.PITCH_DISPLAY_NAMES["H"] = "Open Hi-Hat"`, so the
  Jot ends up with an explicit Open Hi-Hat voice while we validate the
  classifier. The notation-correct follow-up is to fold the `H` fragment
  into `h` with `:o` applied per note before recompose runs (requires
  parser-based per-bar merging, easiest as a new bun bridge).
* Same shape as `cymbal_split.py` on purpose — the `_classify_llm` /
  `_classify_fallback` seam is re-usable from a future "option B"
  (single hi-hat lane, per-onset modifier hint to the existing
  transcribe LLM call) without rewiring the audio analysis.
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
from app.pipeline.recompose import PITCH_DISPLAY_NAMES

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# Pitch letters. The hi-hat stem lands on `h`; the synthetic `H` is the
# open-hi-hat routing key consumed by transcribe (see module docstring).
_HIHAT_PITCH = "h"
_CLOSED_PITCH = "h"
_OPEN_PITCH = "H"

# Feature-extraction windows (seconds). All measurements are robust to
# re-triggering: when ADTOF emits a sizzle-train of phantom onsets inside
# an open-hat ring, a threshold-crossing decay measurement gets capped at
# the next phantom onset's time and reports identical short decays for
# everything — open hats end up indistinguishable from closed. Instead we
# measure mean RMS over fixed windows (late = after the strike, pre =
# before it), which average over re-trigger noise and directly capture
# the two open-hat signatures the classifier needs.
_PEAK_WIN_S = 0.08      # local peak is searched in [t, t+PEAK_WIN_S]
_LATE_START_S = 0.20    # "still ringing" window
_LATE_END_S = 0.50
_PRE_START_S = 0.30     # "riding on existing ring" lookback start (back in time)
_PRE_END_S = 0.05       # ...ends just before the attack so we don't sample it
_TIMBRE_WIN_S = 0.15
_ATTACK_WIN_S = 0.06    # short window for 10-90% rise of post-onset envelope

# Coarse deterministic fallback: open if STILL RINGING 200-500ms after
# the strike OR onset is sitting inside existing ring energy. Either
# signature alone is sufficient. Conservative-ish (open over-call is the
# safer error here — pushes ambiguous hits into the sustain-friendly lane).
_FALLBACK_LATE_RMS = 0.35
_FALLBACK_PRE_RMS = 0.35

# Open-tail post-filter: a closed hi-hat strike inside an open hi-hat's
# audible tail is physically impossible (the drummer cannot pedal-close
# the cymbal mid-ring instantaneously and then strike it closed). Any
# closed-labelled onset that falls inside a measured open tail is
# therefore a spurious detection — a sizzle bump, an ADTOF phantom, or a
# classifier error — and we drop it deterministically after the LLM/
# fallback classification runs.
#
# `_TAIL_END_FRAC`: tail considered "over" when smoothed RMS drops below
# this fraction of the strike's local peak.
# `_TAIL_SMOOTH_S`: moving-average window applied to RMS before the
# threshold search, so transient sizzle bumps don't end the tail early.
# `_TAIL_MAX_S`: hard cap regardless of measurement (avoids the
# pathological "open hat at the very end of the take, RMS never decays").
# `_TAIL_MIN_S`: physical floor. Even an open hat whose RMS technically
# crossed the threshold quickly enforces this minimum no-closed window —
# nobody pedal-closes a struck hat faster than ~80ms.
_TAIL_END_FRAC = 0.30
_TAIL_SMOOTH_S = 0.04
_TAIL_MAX_S = 2.0
_TAIL_MIN_S = 0.08
# Open-within-open: an open hit inside a previous open's tail is dropped
# IF it lacks a fresh-attack signature. The discriminator is `pre_rms`
# (mean stem RMS just before the strike, normalized to the strike's
# peak): a genuine repeated open strike spikes well above the ring
# (pre_rms low, ~0.2-0.5); a sizzle bump's "peak" is barely above the
# ring (pre_rms very high, often > 0.8). Threshold above this = sizzle
# (drop); at or below = real strike (keep, extend the tail window).
# Raise toward 0.75-0.85 if real fast open patterns get cut; lower
# toward 0.55 if sizzle-train bumps keep surviving.
_OPEN_IN_TAIL_MAX_PRE_RMS = 0.65

_SPLIT_TOOL: dict[str, Any] = {
    "name": "report_open_hihat_onsets",
    "description": (
        "Return the indices of the hi-hat onsets that are OPEN hi-hat "
        "hits — the cymbals struck while the foot-pedal is up so the two "
        "cymbals ring/sizzle together, producing a long sustained decay. "
        "Every onset NOT returned is treated as a CLOSED hi-hat hit "
        "(short, articulate, ticky). Return an empty list if the part is "
        "all closed; return all indices if it is all open. Never include "
        "an index that wasn't shown to you."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "open_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "description": "The `#N` indices of onsets that are open.",
            },
        },
        "required": ["open_indices"],
        "additionalProperties": False,
    },
}


class _Feat:
    """Per-onset measured features (kept off `OnsetCandidate` so the
    split stays local and doesn't widen the pipeline-wide onset schema).

    `late_rms` = mean RMS in [t+0.2, t+0.5] / local peak. High = still
    ringing 200-500ms after the strike (open). `pre_rms` = mean RMS in
    [t-0.3, t-0.05] / local peak. High = riding on existing ring energy
    (also open — this is the in-passage sizzle-train signature).
    """

    __slots__ = (
        "late_rms",
        "pre_rms",
        "attack_s",
        "flatness",
        "centroid_hz",
        "gap_s",
        "tail_end_t",
    )

    def __init__(
        self,
        late_rms: float,
        pre_rms: float,
        attack_s: float,
        flatness: float,
        centroid_hz: float,
        gap_s: float,
        tail_end_t: float,
    ) -> None:
        self.late_rms = late_rms
        self.pre_rms = pre_rms
        self.attack_s = attack_s
        self.flatness = flatness
        self.centroid_hz = centroid_hz
        self.gap_s = gap_s
        # Absolute time at which this onset's ring is considered over (per
        # `_TAIL_END_FRAC` / `_TAIL_MIN_S`). Only consulted by the
        # open-tail post-filter for onsets the classifier labelled OPEN.
        self.tail_end_t = tail_end_t


def split_hihat_onsets(
    onsets_by_pitch: dict[str, list[OnsetCandidate]],
    per_instrument_stems: dict[str, Path],
    structure: BeatStructure,
) -> dict[str, list[OnsetCandidate]]:
    """Return `onsets_by_pitch` with the hi-hat lane split into closed/open.

    No-op (returns the input mapping unchanged) when there is no hi-hat
    stem, no hi-hat onsets, or no in-range hi-hat onsets to classify.
    """
    hh = onsets_by_pitch.get(_HIHAT_PITCH)
    stem_path = per_instrument_stems.get(_HIHAT_PITCH)
    if not hh or stem_path is None or not stem_path.exists():
        return onsets_by_pitch

    in_range = sorted(
        (c for c in hh if c.bar >= 0), key=lambda c: (c.bar, c.beat_in_bar)
    )
    out_of_range = [c for c in hh if c.bar < 0]
    if not in_range:
        return onsets_by_pitch

    feats = _measure(stem_path, in_range)

    open_idx = _classify_llm(in_range, feats, structure, onsets_by_pitch)
    source = "llm"
    if open_idx is None:
        open_idx = _classify_fallback(in_range, feats)
        source = "fallback"

    # Deterministic post-filter: drop spurious onsets that fall inside
    # an open's measured ring tail. Closed-inside-tail is dropped
    # unconditionally (physically impossible); open-inside-tail is
    # dropped only when it lacks a fresh-attack signature (a sizzle bump
    # the model labelled open, not a real repeated strike).
    closed_dropped, open_dropped = _open_tail_filter(
        in_range, feats, open_idx
    )

    closed = [
        c for i, c in enumerate(in_range)
        if i not in open_idx and i not in closed_dropped
    ]
    opened = [
        c for i, c in enumerate(in_range)
        if i in open_idx and i not in open_dropped
    ]
    # Out-of-range hi-hat onsets are never consumed downstream (bar < 0);
    # park them on the closed lane so nothing is silently discarded.
    closed.extend(out_of_range)

    log.info(
        "hihat split (%s): %d onsets -> %d closed, %d open "
        "(tail-filter dropped %d closed + %d open inside open tails)",
        source,
        len(in_range),
        len(closed) - len(out_of_range),
        len(opened),
        len(closed_dropped),
        len(open_dropped),
    )

    sink = current_debug_sink()
    if sink is not None:
        sink.write_json(
            "hihat_split/decision.json",
            {
                "source": source,
                "n_input": len(in_range),
                "n_closed": len(closed) - len(out_of_range),
                "n_open": len(opened),
                "n_closed_in_tail": len(closed_dropped),
                "n_open_in_tail": len(open_dropped),
                "onsets": [
                    {
                        "index": i,
                        "bar": c.bar,
                        "beat_in_bar": round(c.beat_in_bar, 3),
                        "strength": round(c.strength, 3),
                        "late_rms": round(feats[i].late_rms, 3),
                        "pre_rms": round(feats[i].pre_rms, 3),
                        "attack_s": round(feats[i].attack_s, 4),
                        "flatness": round(feats[i].flatness, 4),
                        "centroid_hz": round(feats[i].centroid_hz, 1),
                        "gap_s": round(feats[i].gap_s, 3),
                        "tail_end_s": round(
                            feats[i].tail_end_t - c.time, 3
                        ),
                        "label": (
                            "open_in_tail" if i in open_dropped
                            else "open" if i in open_idx
                            else "closed_in_tail" if i in closed_dropped
                            else "closed"
                        ),
                    }
                    for i, c in enumerate(in_range)
                ],
            },
        )

    out = dict(onsets_by_pitch)
    if closed:
        out[_CLOSED_PITCH] = sorted(
            closed, key=lambda c: (c.bar, c.beat_in_bar)
        )
    else:
        out.pop(_CLOSED_PITCH, None)
    if opened:
        # Merge into any pre-existing `H` lane just to stay defensive;
        # nothing else in the pipeline produces this pitch today.
        out[_OPEN_PITCH] = sorted(
            out.get(_OPEN_PITCH, []) + opened,
            key=lambda c: (c.bar, c.beat_in_bar),
        )
    return out


def _open_tail_filter(
    onsets: list[OnsetCandidate],
    feats: list[_Feat],
    open_idx: set[int],
) -> tuple[set[int], set[int]]:
    """Drop spurious onsets inside an open hi-hat's measured ring tail.

    Returns `(closed_dropped, open_dropped)`. Single time-ordered pass
    with a rolling "current open tail end" tracker:

      * CLOSED inside an open tail -> dropped unconditionally. A
        struck-closed hi-hat needs the pedal down on a ringing cymbal,
        which is physically impossible in zero time.
      * OPEN inside an open tail -> dropped IF its `pre_rms` exceeds
        `_OPEN_IN_TAIL_MAX_PRE_RMS` (no fresh attack on top of the
        ring; the "peak" is just sizzle the model picked up). Kept
        otherwise, treated as a genuine repeated strike that
        re-energizes the ring, and the tracked tail end extends to the
        max of its own tail and the prior remainder.
      * Either kind OUTSIDE any tail -> always kept; an open here
        starts a new tracked tail.

    Single-pass and order-preserving by construction; no interval-merge
    bookkeeping needed because the tracker IS the merged interval.
    """
    closed_dropped: set[int] = set()
    open_dropped: set[int] = set()
    if not open_idx:
        return closed_dropped, open_dropped
    order = sorted(range(len(onsets)), key=lambda i: onsets[i].time)
    current_tail_end = -1.0
    for i in order:
        t = onsets[i].time
        in_tail = t <= current_tail_end
        if i in open_idx:
            if in_tail and feats[i].pre_rms > _OPEN_IN_TAIL_MAX_PRE_RMS:
                # Sizzle bump within previous open's ring (no fresh
                # attack). Drop and DO NOT extend the tail.
                open_dropped.add(i)
            else:
                # Outside any tail, OR genuine repeated strike with a
                # real fresh attack. Keep and extend.
                current_tail_end = max(current_tail_end, feats[i].tail_end_t)
        else:
            if in_tail:
                closed_dropped.add(i)
    return closed_dropped, open_dropped


def _measure(
    stem_path: Path, onsets: list[OnsetCandidate]
) -> list[_Feat]:
    """Measure late-RMS / pre-RMS / attack / flatness / centroid / gap per onset.

    The hi-hat stem is loaded once. `late_rms` and `pre_rms` are the two
    discriminators robust to ADTOF re-triggering inside the open-hat
    ring — both are mean RMS over fixed windows, normalized to the
    onset's local peak, so neither depends on neighbouring onset times
    (the bug that made decay-to-threshold useless: with sizzle-train
    onsets ~80ms apart the decay window collapsed to ~80ms regardless of
    how long the ring really lasted). `attack_s` is the 10-90% rise time
    of the early post-onset envelope (closed = sharp; open = slower
    swell as the cymbals sizzle).
    """
    sr = 44100
    audio, sr = librosa.load(str(stem_path), sr=sr, mono=True)
    hop = 512
    rms = librosa.feature.rms(y=audio, hop_length=hop)[0]
    rms_t = librosa.times_like(rms, sr=sr, hop_length=hop)
    # Smoothed RMS for tail-end detection: averaging over ~40ms prevents
    # a momentary dip between sizzle bumps from being misread as "ring is
    # over" (the bug we just fixed in the decay measurement, in the
    # other direction).
    smooth_n = max(1, int(round(_TAIL_SMOOTH_S * sr / hop)))
    if smooth_n > 1 and rms.size >= smooth_n:
        kernel = np.ones(smooth_n) / smooth_n
        rms_smooth = np.convolve(rms, kernel, mode="same")
    else:
        rms_smooth = rms
    n = len(onsets)
    out: list[_Feat] = []
    for i, c in enumerate(onsets):
        t = c.time
        nxt = onsets[i + 1].time if i + 1 < n else t + _LATE_END_S
        prev = onsets[i - 1].time if i > 0 else None
        gap = nxt - t
        if prev is not None:
            gap = min(gap, t - prev)

        # --- local peak (search a short post-onset window) ---------
        peak_mask = (rms_t >= t) & (rms_t <= t + _PEAK_WIN_S)
        if not np.any(peak_mask):
            out.append(
                _Feat(0.0, 0.0, 0.0, 0.0, 0.0, float(gap), t + _TAIL_MIN_S)
            )
            continue
        peak = float(rms[peak_mask].max())
        if peak <= 0.0:
            late_rms = 0.0
            pre_rms = 0.0
            tail_end_t = t + _TAIL_MIN_S
        else:
            # --- late RMS [t+0.2, t+0.5] / peak --------------------
            late_mask = (rms_t >= t + _LATE_START_S) & (rms_t <= t + _LATE_END_S)
            late_rms = (
                float(rms[late_mask].mean()) / peak if np.any(late_mask) else 0.0
            )
            # --- pre RMS [t-0.3, t-0.05] / peak --------------------
            pre_mask = (rms_t >= t - _PRE_START_S) & (rms_t <= t - _PRE_END_S)
            pre_rms = (
                float(rms[pre_mask].mean()) / peak if np.any(pre_mask) else 0.0
            )
            # --- tail end: first time after the peak window when the
            # smoothed RMS drops below _TAIL_END_FRAC * peak. Capped at
            # _TAIL_MAX_S; floored at _TAIL_MIN_S (physical pedal-close
            # minimum). Consulted only by the open-tail post-filter.
            tail_threshold = _TAIL_END_FRAC * peak
            search_mask = (
                (rms_t > t + _PEAK_WIN_S) & (rms_t <= t + _TAIL_MAX_S)
            )
            if np.any(search_mask):
                seg = rms_smooth[search_mask]
                seg_t = rms_t[search_mask]
                below = np.where(seg < tail_threshold)[0]
                tail_end_t = (
                    float(seg_t[below[0]]) if below.size
                    else float(seg_t[-1])
                )
            else:
                tail_end_t = t + _TAIL_MIN_S
            tail_end_t = max(tail_end_t, t + _TAIL_MIN_S)
            tail_end_t = min(tail_end_t, t + _TAIL_MAX_S)

        # --- attack-sharpness (10-90% rise time of post-onset envelope)
        a0 = int(t * sr)
        a1 = min(len(audio), int((t + _ATTACK_WIN_S) * sr))
        attack_s = 0.0
        if a1 - a0 >= hop:
            env = np.abs(audio[a0:a1])
            if env.size:
                env_peak = float(env.max())
                if env_peak > 0.0:
                    lo = 0.1 * env_peak
                    hi = 0.9 * env_peak
                    above_lo = np.where(env >= lo)[0]
                    above_hi = np.where(env >= hi)[0]
                    if above_lo.size and above_hi.size:
                        attack_s = max(
                            0.0,
                            float(above_hi[0] - above_lo[0]) / float(sr),
                        )

        # --- timbre (flatness + centroid) ---------------------------
        t1 = min(len(audio), int((t + _TIMBRE_WIN_S) * sr))
        clip = audio[a0:t1]
        if clip.size >= hop:
            flat = float(np.mean(librosa.feature.spectral_flatness(y=clip)))
            cen = float(
                np.mean(librosa.feature.spectral_centroid(y=clip, sr=sr))
            )
        else:
            flat, cen = 0.0, 0.0
        out.append(
            _Feat(late_rms, pre_rms, attack_s, flat, cen, float(gap), tail_end_t)
        )
    return out


def _classify_llm(
    onsets: list[OnsetCandidate],
    feats: list[_Feat],
    structure: BeatStructure,
    onsets_by_pitch: dict[str, list[OnsetCandidate]],
) -> set[int] | None:
    """Ask the LLM which onsets are open. Returns the open index set, or
    `None` to signal the caller to use the deterministic fallback."""
    if not settings.anthropic_api_key:
        log.info("hihat split: no ANTHROPIC_API_KEY; using fallback")
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
    try:
        response = call_messages_with_refusal_retry(
            client,
            {
                "model": settings.llm_model,
                "max_tokens": settings.llm_max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [_SPLIT_TOOL],
                "tool_choice": {"type": "tool", "name": _SPLIT_TOOL["name"]},
            },
            base_prompt=prompt,
            purpose="hihat_split",
        )
    except Exception as exc:
        log.warning(
            "hihat split: LLM call failed (%s); using fallback", exc
        )
        return None

    n = len(onsets)
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != _SPLIT_TOOL["name"]:
            continue
        raw = block.input.get("open_indices", [])
        if not isinstance(raw, list):
            log.warning(
                "hihat split: non-list open_indices; using fallback"
            )
            return None
        out: set[int] = set()
        for v in raw:
            try:
                idx = int(v)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n:
                out.add(idx)
        return out
    log.warning("hihat split: no tool_use block; using fallback")
    return None


def _classify_fallback(
    onsets: list[OnsetCandidate], feats: list[_Feat]
) -> set[int]:
    """Coarse deterministic open/closed split over the measured features.

    Open if STILL RINGING after the strike (high `late_rms`) OR riding
    on existing ring energy (high `pre_rms`). Either signature alone is
    sufficient. Runs only when the LLM is unavailable.
    """
    opened: set[int] = set()
    for i, f in enumerate(feats):
        if f.late_rms >= _FALLBACK_LATE_RMS or f.pre_rms >= _FALLBACK_PRE_RMS:
            opened.add(i)
    return opened


def _format_bars(
    onsets: list[OnsetCandidate],
    feats: list[_Feat],
    structure: BeatStructure,
    onsets_by_pitch: dict[str, list[OnsetCandidate]],
) -> str:
    """Per-bar render: indexed hi-hat onsets with measured features +
    a compact one-line summary of every other instrument's hits in that
    bar (so the model can spot kick-coincident accents / open-hat fills).
    """
    if not structure.bars:
        return "(no bars detected)"

    by_bar: dict[int, list[tuple[int, OnsetCandidate, _Feat]]] = {}
    for i, c in enumerate(onsets):
        by_bar.setdefault(c.bar, []).append((i, c, feats[i]))

    others_by_bar: dict[int, list[tuple[str, float]]] = {}
    for op, cands in onsets_by_pitch.items():
        if op == _HIHAT_PITCH:
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
                f"late{ft.late_rms:.2f},pre{ft.pre_rms:.2f},"
                f"atk{ft.attack_s * 1000.0:.0f}ms,"
                f"flat{ft.flatness:.3f},cen{ft.centroid_hz / 1000.0:.1f}k,"
                f"gap{ft.gap_s:.2f}s)"
                for i, c, ft in entries
            )
            rows.append(f"  hihat: {rendered}")
        else:
            rows.append("  hihat: (none)")
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
    return (PROMPT_DIR / "split_hihat.md").read_text(encoding="utf-8")
