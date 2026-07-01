"""Split the merged hi-hat stem's onsets into closed (`h`), open (`H`),
and discard (artifact rejected upstream of the filter LLM).

The Stage-2 separator emits a single `(hh)` stem mapped to pitch `h`, but
a real hi-hat performance interleaves closed strikes and open (sustained)
hits — and ADTOF (even in-distribution on the drum stem) reads the
open-hat sizzle/ring as a stream of confident frame-level activations.
Pushing both flavours through one per-instrument LLM call buries the
closed pattern under that sizzle.

This module classifies each detected hi-hat onset into one of three
buckets off the isolated stem audio (deterministic features the LLM
can't hear) plus the LLM's judgement:

* **OPEN** — foot up, the cymbals ring/sizzle together.
* **CLOSED** — foot down, ticky/articulate.
* **DISCARD** — not a real hit. Sizzle re-trigger inside an open tail,
  bleed from another instrument that lines up with `others:`, or a
  double-trigger ~<30 ms after a real hit.

The unified ternary call replaces an earlier two-pass design (binary
open/closed here, then `filter_llm` per lane). The seam between the two
passes was the structural failure mode: when an open tail produced a
sizzle train, this stage's binary call mis-labelled some bumps as
*closed*; `filter_llm` then saw `h` in isolation (no features, no view of
`H`) and had no signal to reject them. By collapsing the decisions, the
LLM uses the same picture (full features + cross-instrument context) to
judge "is this a sizzle bump posing as closed?" that it uses to judge
"is this open?". The `filter_llm` pass is skipped entirely for `h` / `H`
afterwards — see `pipeline/runner.py::_do_transcribe`.

Architectural notes:

* **Synthetic pitch `H` for open hi-hat.** Drumjot's DSL has only one
  notational hi-hat pitch (`h`) with `:o` / `:c` modifiers, so `H` is an
  *internal* routing key for the transcribe pass; it must not leak as a
  permanent extra layer in finished transcriptions. Today the backend
  emits `H` as a distinct MIDI note (46 = GM open hi-hat) and the
  frontend folds it back via `canonicalProvenanceLane` in
  `src/editing/provenance/provenance_store.ts` plus the GM table in
  `src/midi/from_midi.ts`.
  The notation-correct follow-up is to fold `H` → `h:o` backend-side
  before MIDI emission so `note_provenance.json` carries the canonical
  pitch directly (eliminates the asymmetric coupling). See
  CLEANROOM_SPEC §11.17.
* The deterministic `_open_tail_filter` stays as a backstop. The ternary
  LLM call should now catch sizzle bumps directly, but the rule
  ("closed-inside-confirmed-open-tail is physically impossible") is a
  free safety net the LLM cannot violate.
* The discarded list is returned alongside the split lanes so the
  runner can hand it to `note_provenance` for the UI's "Show filtered"
  ghost overlay.
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

# Pitch letters. The hi-hat stem lands on `h`; the synthetic `H` is the
# open-hi-hat routing key consumed by transcribe (see module docstring).
_HIHAT_PITCH = "h"
_CLOSED_PITCH = "h"
_OPEN_PITCH = "H"

# Feature-extraction windows (seconds). All measurements are robust to
# re-triggering: when ADTOF emits a sizzle-train of phantom onsets inside
# an open-hat ring, a threshold-crossing decay measurement gets capped at
# the next phantom onset's time and reports identical short decays for
# everything, open hats end up indistinguishable from closed. Instead we
# measure mean RMS over fixed windows (late = after the strike, pre =
# before it), which average over re-trigger noise and directly capture
# the two open-hat signatures the classifier needs.
_PEAK_WIN_S = 0.08      # local peak is searched in [t-_PEAK_BACK_S, t+_PEAK_WIN_S]
# Extend the peak search backward by ~one RMS hop so a slightly-late
# onset time (ADTOF's peak-picking can lag the transient by a frame or
# two) still catches the real strike amplitude rather than its decay.
# A peak that lands in the decay portion is artificially small, which
# makes the late/pre RATIOS explode (a 4.4 late_rms on a closed hit is
# the diagnostic signature of this failure mode).
_PEAK_BACK_S = 0.02
_LATE_START_S = 0.20    # "still ringing" window (then clipped to next onset)
_LATE_END_S = 0.50
_PRE_START_S = 0.30     # "riding on existing ring" lookback (then clipped to prev onset)
_PRE_END_S = 0.05       # ...ends just before the attack so we don't sample it
# Guard band between this strike's late/pre window and a neighbouring
# strike. The raw windows ([+200,+500] late, [-300,-50] pre) are wider
# than a typical hi-hat gap (16ths @120 BPM = 125 ms, 8ths = 250 ms),
# so without clipping the late window samples the NEXT strike's
# transient and reports "still ringing" energy that's really just the
# next hit. Clip both windows to leave this much breathing room from
# the adjacent onset.
_NEIGHBOR_GUARD_S = 0.02
# After clipping, a window narrower than this can't average reliably
# over re-trigger noise; report 0 (silence) instead of a polluted
# measurement.
_MIN_WINDOW_S = 0.08
_TIMBRE_WIN_S = 0.15
_ATTACK_WIN_S = 0.06    # short window for 10-90% rise of post-onset envelope
# Bleed discriminator bands. A real hi-hat is high-band noise with little
# low-mid body; snare/kick bleed leaking into the hat stem dumps energy
# into ~200-1500 Hz (drum bodies / fundamentals). `lowband_ratio` = the
# fraction of OCCUPIED-band energy that sits in the low band. The occupied
# band tops out at ~14 kHz because the MP3 source lowpasses there and the
# separator doesn't restore it; measuring to Nyquist would divide by dead
# air (the trap the cymbal-classifier work hit with spectral flatness).
_LOWBAND_HZ = (200.0, 1500.0)
_OCCUPIED_HZ = (200.0, 14000.0)

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
# IF it lacks a fresh-attack signature. The discriminator is `attack_flux`
# (peak onset-strength flux at the strike, normalized to the stem's median
# flux): a real strike, even a soft one riding on a loud ring, produces a
# fresh half-wave-rectified spectral-flux SPIKE; a sizzle re-trigger is just
# ring wobble with no fresh transient, so its flux stays near the floor.
# Below this = no fresh attack = sizzle (drop); at or above = real strike
# (keep, extend the tail window).
#
# This REPLACES the earlier `pre_rms > 0.65` rule, which conflated
# "consecutive open strikes" with "sizzle bumps": in a sustained open-hat
# groove every strike rides the previous ring, so `pre_rms` is high (0.5-1.0)
# for genuine strikes too, and the rule decimated real open passages
# (validated on debug run 20260531_3acbcfa8 bars 73-80: 12 of 36 genuine
# open strikes were dropped). `attack_flux` is level-relative, so it keeps a
# real strike on top of a loud ring while still rejecting a flux-less sizzle.
# Raise if sizzle-train bumps survive; lower if fast real open patterns get cut.
_OPEN_IN_TAIL_MIN_FLUX = 3.0

# --- Deterministic envelope guardrail (open vs closed) ---------------------
# The RMS ring shape is a far more reliable open/closed signal than the LLM:
# on the labeled debug run the LLM mislabeled an all-closed passage as
# 13-open/10-discard and kept only 5 of 36 hits in an all-open passage,
# despite per-onset features that separate the two classes at ~96% with a
# single threshold. So after the LLM (or fallback) classifies, we OVERRIDE
# its open/closed call whenever the envelope is DECISIVE; ambiguous onsets
# (and every `discard`) are left to the LLM. The guardrail only re-routes
# hits the classifier already accepted as real, it never creates or removes
# onsets, so it cannot re-admit artifacts.
#
# OPEN if ANY of: ring lasts >= _VERDICT_OPEN_TAIL_S, still ringing
# (`late_rms` >= ...), or riding on an existing ring (`pre_rms` >= ...).
# CLOSED only if ALL of: short ring AND low late AND low pre. Anything else
# is ambiguous -> trust the LLM.
_VERDICT_OPEN_TAIL_S = 0.30
_VERDICT_OPEN_LATE_RMS = 0.20
_VERDICT_OPEN_PRE_RMS = 0.45
# The `pre_rms` ("riding on an existing ring") open-signature must be
# CORROBORATED by real sustain (`late_rms >= this`). `pre_rms` is a ratio to
# the strike's local peak, so a near-zero peak (a phantom onset on the noise
# floor / a previous hit's decay, no fresh transient) makes it explode to
# 2-9 and falsely vote "open" even with zero tail and zero late. A genuine
# open hit always rings (high tail and/or late), so requiring a little
# measured ring after the strike rejects the degenerate case while costing
# real opens nothing (they pass on the tail signature anyway). Validated on
# Cold-Hard-Bitch: kills 88 phantom 8th-note opens, keeps 252 real opens.
_VERDICT_OPEN_PRE_CORROB_LATE = 0.10
_VERDICT_CLOSED_TAIL_S = 0.18
_VERDICT_CLOSED_LATE_RMS = 0.12
_VERDICT_CLOSED_PRE_RMS = 0.30

# --- Discard-rescue (recall-positive overturn of LLM discards) -------------
# The LLM sometimes discards real hits (on the labeled run it discarded ~24
# genuine hats across two passages). We overturn a discard back to its
# envelope verdict ONLY when every guard agrees it's a real hat:
#   * the envelope is decisive (open or closed, not ambiguous), AND
#   * it looks like a hat not bleed: `lowband_ratio <= _BLEED_LOWBAND_RATIO_MAX`
#     (snare/kick bleed dumps energy into 200-1500 Hz; a hat doesn't; #     validated hat ratios ~0.01 vs snare/kick ~0.6), AND
#   * a fresh transient (not sizzle): `attack_flux >= _RESCUE_MIN_FLUX`, AND
#   * not a double-trigger: `gap_s >= _RESCUE_MIN_GAP_S`, AND
#   * the LLM was UNSURE (`low_confidence_discards`) OR the envelope is
#     OVERWHELMING (`_RESCUE_STRONG_*`), so a confident LLM discard is
#     respected unless the signal is beyond doubt.
# Recall-positive only: it moves discards into open/closed, never the reverse.
_BLEED_LOWBAND_RATIO_MAX = 0.15
_RESCUE_MIN_FLUX = 3.0
_RESCUE_MIN_GAP_S = 0.03
_RESCUE_STRONG_OPEN_TAIL_S = 0.50
_RESCUE_STRONG_CLOSED_FLUX = 20.0

_SPLIT_TOOL: dict[str, Any] = {
    "name": "report_hihat_classification",
    "description": (
        "Classify each detected hi-hat onset into one of three buckets: "
        "OPEN (foot up, the cymbals ring/sizzle together), CLOSED (foot "
        "down, short/articulate/ticky), or DISCARD (not a real hit — a "
        "sizzle re-trigger inside an open tail, bleed from another "
        "instrument that lines up with the `others:` summary, or a "
        "double-trigger immediately after a real hit). Return TWO arrays "
        "of `#N` indices: the open ones and the discard ones. Every "
        "onset NOT in either array is treated as CLOSED. The two arrays "
        "must be disjoint. Return empty arrays when appropriate (all "
        "closed; nothing to discard). Never include an index that wasn't "
        "shown to you."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "open_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "description": (
                    "The `#N` indices of onsets that are OPEN hi-hat hits."
                ),
            },
            "discard_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "description": (
                    "The `#N` indices of onsets that are NOT real hits "
                    "(sizzle bumps inside an open tail, bleed, double-"
                    "triggers). These should be the minority, only "
                    "clear artifacts."
                ),
            },
            "low_confidence_discards": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "description": (
                    "The subset of `discard_indices` you are NOT confident "
                    "about, borderline calls where the onset might be a "
                    "real (soft / fast) hit rather than an artifact. "
                    "Downstream acoustic checks may overturn these back to "
                    "a real hit; leave a discard OUT of this list only when "
                    "you are confident it is an artifact. Must be a subset "
                    "of `discard_indices`."
                ),
            },
        },
        "required": ["open_indices", "discard_indices"],
        "additionalProperties": False,
    },
}


class _Feat:
    """Per-onset measured features (kept off `OnsetCandidate` so the
    split stays local and doesn't widen the pipeline-wide onset schema).

    `late_rms` = mean RMS in [t+0.2, min(t+0.5, nxt-guard)] / local peak.
    High = still ringing 200-500ms after the strike (open). The window
    is clipped against the next onset so a dense hi-hat pattern's late
    window can't sample the next strike's transient and report it as
    "still ringing"; an over-narrow clipped window reports 0.
    `pre_rms` = mean RMS in [max(t-0.3, prev+guard), t-0.05] / local
    peak. High = riding on existing ring energy (also open, this is
    the in-passage sizzle-train signature). Same clipping logic against
    the previous onset.
    """

    __slots__ = (
        "late_rms",
        "pre_rms",
        "attack_s",
        "attack_flux",
        "flatness",
        "centroid_hz",
        "lowband_ratio",
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
        attack_flux: float = 0.0,
        lowband_ratio: float = 0.0,
    ) -> None:
        self.late_rms = late_rms
        self.pre_rms = pre_rms
        self.attack_s = attack_s
        # Peak onset-strength flux at the strike / stem-median flux. A fresh
        # transient (real strike, even soft on a loud ring) spikes; a sizzle
        # re-trigger inside a ring does not. Drives the open-within-open drop.
        self.attack_flux = attack_flux
        self.flatness = flatness
        self.centroid_hz = centroid_hz
        # Fraction of occupied-band energy in the low band (~200-1500 Hz).
        # Hi-hat = low (high-band noise); snare/kick bleed = high (low-mid
        # body). The bleed guard on the discard-rescue reads this.
        self.lowband_ratio = lowband_ratio
        self.gap_s = gap_s
        # Absolute time at which this onset's ring is considered over (per
        # `_TAIL_END_FRAC` / `_TAIL_MIN_S`). Only consulted by the
        # open-tail post-filter for onsets the classifier labelled OPEN.
        self.tail_end_t = tail_end_t


def split_hihat_onsets(
    onsets_by_pitch: dict[str, list[OnsetCandidate]],
    per_instrument_stems: dict[str, Path],
    structure: BeatStructure,
    *,
    llm_model: str | None = None,
) -> tuple[dict[str, list[OnsetCandidate]], list[OnsetCandidate]]:
    """Return `(onsets_by_pitch_with_split, discarded_onsets)`.

    The first element is `onsets_by_pitch` with the hi-hat lane split
    into closed (`h`) and synthetic open (`H`). The second is the list
    of in-range hi-hat onsets the classifier (or the deterministic
    open-tail backstop) rejected as artifacts — kept around so the UI's
    "Show filtered" ghost overlay can surface them via
    `note_provenance`. The discards are *not* present in either lane;
    the runner merges them back into `all_onsets_by_pitch[h]` at the
    provenance boundary only.

    No-op (returns the input mapping unchanged + an empty discarded list)
    when there is no hi-hat stem, no hi-hat onsets, or no in-range hi-hat
    onsets to classify.
    """
    hh = onsets_by_pitch.get(_HIHAT_PITCH)
    stem_path = per_instrument_stems.get(_HIHAT_PITCH)
    if not hh or stem_path is None or not stem_path.exists():
        return onsets_by_pitch, []

    in_range = sorted(
        (c for c in hh if c.bar >= 0), key=lambda c: (c.bar, c.beat_in_bar)
    )
    out_of_range = [c for c in hh if c.bar < 0]
    if not in_range:
        return onsets_by_pitch, []

    feats = _measure(stem_path, in_range)
    # Attach the per-onset measurements to the candidates so the UI's
    # per-note "Acoustic properties" subsection can show the same
    # numbers the classifier saw. Pure mutation; the split's downstream
    # logic still reads from the local `feats` list for indexing.
    for c, f in zip(in_range, feats, strict=True):
        c.decay_s = None  # hi-hat split tracks rise/ring instead of a 20dB decay
        c.flatness = f.flatness
        c.centroid_hz = f.centroid_hz
        c.gap_s = f.gap_s
        c.attack_s = f.attack_s
        c.attack_flux = f.attack_flux
        c.lowband_ratio = f.lowband_ratio
        c.late_rms = f.late_rms
        c.pre_rms = f.pre_rms
        c.tail_end_s = f.tail_end_t - c.time

    llm_result = _classify_llm(in_range, feats, structure, onsets_by_pitch, llm_model=llm_model)
    if llm_result is None:
        open_idx, discard_idx, low_conf_discards = _classify_fallback(in_range, feats)
        source = "fallback"
    else:
        open_idx, discard_idx, low_conf_discards = llm_result
        source = "llm"

    # Deterministic envelope guardrail: the RMS ring shape separates open
    # from closed far more reliably than the LLM (which mislabeled whole
    # passages on the labeled debug run). Override the classifier's
    # open/closed call wherever the envelope is DECISIVE; leave ambiguous
    # onsets and every `discard` to the LLM. Re-routes accepted hits only;
    # never adds or removes onsets, so it can't re-admit artifacts.
    forced_open: set[int] = set()
    forced_closed: set[int] = set()
    for i, (c, f) in enumerate(zip(in_range, feats, strict=True)):
        if i in discard_idx:
            continue
        v = _envelope_open_verdict(f, c.time)
        if v == "open" and i not in open_idx:
            open_idx.add(i)
            forced_open.add(i)
        elif v == "closed" and i in open_idx:
            open_idx.discard(i)
            forced_closed.add(i)

    # Discard-rescue (recall-positive): overturn an LLM discard back to a
    # real hit when every guard agrees it isn't an artifact. Mutates
    # open_idx / discard_idx in place; only ever moves discards into
    # open/closed (see `_rescue_discards`).
    rescued_open, rescued_closed = _rescue_discards(
        in_range, feats, open_idx, discard_idx, low_conf_discards,
    )

    # Deterministic backstop: a closed-labelled onset inside a confirmed
    # open tail is physically impossible; an open-labelled onset inside
    # an open tail without a fresh attack is a sizzle bump. The ternary
    # LLM call should catch these directly — the rule is left in place
    # as a free invariant the LLM cannot violate. Discards are excluded
    # from the sweep (already dropped; do not extend the tail off them).
    closed_in_tail, open_in_tail = _open_tail_filter(
        in_range, feats, open_idx, discard_idx,
    )

    closed = [
        c for i, c in enumerate(in_range)
        if i not in open_idx
        and i not in discard_idx
        and i not in closed_in_tail
    ]
    opened = [
        c for i, c in enumerate(in_range)
        if i in open_idx and i not in open_in_tail
    ]
    discarded = [
        c for i, c in enumerate(in_range)
        if i in discard_idx or i in closed_in_tail or i in open_in_tail
    ]
    # Out-of-range hi-hat onsets are never consumed downstream (bar < 0);
    # park them on the closed lane so nothing is silently discarded.
    closed.extend(out_of_range)

    log.info(
        "hihat split (%s): %d onsets -> %d closed, %d open, %d discard "
        "(post-rescue discard %d + tail-filter %d closed + %d open inside "
        "open tails; guardrail forced %d->open %d->closed; rescued "
        "%d->open %d->closed)",
        source,
        len(in_range),
        len(closed) - len(out_of_range),
        len(opened),
        len(discarded),
        len(discard_idx),
        len(closed_in_tail),
        len(open_in_tail),
        len(forced_open),
        len(forced_closed),
        len(rescued_open),
        len(rescued_closed),
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
                "n_discard": len(discarded),
                "n_discard_post_rescue": len(discard_idx),
                "n_closed_in_tail": len(closed_in_tail),
                "n_open_in_tail": len(open_in_tail),
                "n_forced_open": len(forced_open),
                "n_forced_closed": len(forced_closed),
                "n_rescued_open": len(rescued_open),
                "n_rescued_closed": len(rescued_closed),
                "onsets": [
                    {
                        "index": i,
                        "bar": c.bar,
                        "beat_in_bar": round(c.beat_in_bar, 3),
                        "strength": round(c.strength, 3),
                        "late_rms": round(feats[i].late_rms, 3),
                        "pre_rms": round(feats[i].pre_rms, 3),
                        "attack_s": round(feats[i].attack_s, 4),
                        "attack_flux": round(feats[i].attack_flux, 2),
                        "lowband_ratio": round(feats[i].lowband_ratio, 3),
                        "flatness": round(feats[i].flatness, 4),
                        "centroid_hz": round(feats[i].centroid_hz, 1),
                        "gap_s": round(feats[i].gap_s, 3),
                        "tail_end_s": round(
                            feats[i].tail_end_t - c.time, 3
                        ),
                        "forced": (
                            "open" if i in forced_open
                            else "closed" if i in forced_closed
                            else None
                        ),
                        "rescued": (
                            "open" if i in rescued_open
                            else "closed" if i in rescued_closed
                            else None
                        ),
                        "label": _label_for(
                            i, open_idx, discard_idx,
                            closed_in_tail, open_in_tail,
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
    return out, discarded


def _label_for(
    i: int,
    open_idx: set[int],
    discard_idx: set[int],
    closed_in_tail: set[int],
    open_in_tail: set[int],
) -> str:
    """Per-onset label for the debug dump. Tail-filter drops are
    surfaced as their own labels (distinct from LLM `discard`) so an
    eyeball pass on `decision.json` can tell whether the backstop is
    catching things the LLM missed.
    """
    if i in open_in_tail:
        return "open_in_tail"
    if i in closed_in_tail:
        return "closed_in_tail"
    if i in discard_idx:
        return "discard"
    if i in open_idx:
        return "open"
    return "closed"


def _open_tail_filter(
    onsets: list[OnsetCandidate],
    feats: list[_Feat],
    open_idx: set[int],
    discard_idx: set[int],
) -> tuple[set[int], set[int]]:
    """Drop spurious onsets inside an open hi-hat's measured ring tail.

    Returns `(closed_dropped, open_dropped)` — the extra drops the
    deterministic backstop catches on top of the LLM's `discard_idx`.
    Onsets already in `discard_idx` are skipped (already dropped; do
    not extend the tail off them, since the LLM judged them artifacts).

    Single time-ordered pass with a rolling "current open tail end"
    tracker:

      * CLOSED inside an open tail -> dropped unconditionally. A
        struck-closed hi-hat needs the pedal down on a ringing cymbal,
        which is physically impossible in zero time.
      * OPEN inside an open tail -> dropped IF its `attack_flux` is
        below `_OPEN_IN_TAIL_MIN_FLUX` (no fresh transient, the "peak"
        is just sizzle the model picked up on the existing ring). Kept
        otherwise, treated as a genuine repeated strike (which produces
        a fresh flux spike even riding on a loud ring), and the tracked
        tail end extends to the max of its own tail and the prior
        remainder.
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
        if i in discard_idx:
            continue
        t = onsets[i].time
        in_tail = t <= current_tail_end
        if i in open_idx:
            if in_tail and feats[i].attack_flux < _OPEN_IN_TAIL_MIN_FLUX:
                # Sizzle bump within previous open's ring (no fresh
                # transient). Drop and DO NOT extend the tail.
                open_dropped.add(i)
            else:
                # Outside any tail, OR genuine repeated strike with a
                # real fresh attack. Keep and extend.
                current_tail_end = max(current_tail_end, feats[i].tail_end_t)
        else:
            if in_tail:
                closed_dropped.add(i)
    return closed_dropped, open_dropped


def _envelope_open_verdict(f: _Feat, onset_time: float) -> str | None:
    """Deterministic open/closed verdict from the RMS ring envelope, or
    `None` when the evidence is ambiguous (defer to the LLM).

    OPEN when ANY signature is decisive: a long ring
    (`tail_end_t - onset_time >= _VERDICT_OPEN_TAIL_S`), still ringing
    200-500 ms later (`late_rms`), or riding on an existing ring
    (`pre_rms`), the last only when CORROBORATED by measured sustain
    (`late_rms >= _VERDICT_OPEN_PRE_CORROB_LATE`), since `pre_rms` explodes
    on a degenerate (near-zero-peak) phantom onset and would otherwise vote
    "open" with no tail and no late. CLOSED only when ALL three say "short
    and dry". The asymmetry is deliberate: a hit needs just one strong open
    signature to be open, but must look closed on every axis to be
    force-closed, so a genuine open is never force-closed on a single soft
    measurement.

    `tail_end_t` is an absolute time on `_Feat`; the tail DURATION is
    `tail_end_t - onset_time` (the same value the split stores as
    `tail_end_s` on the candidate).
    """
    tail_s = f.tail_end_t - onset_time
    if (
        tail_s >= _VERDICT_OPEN_TAIL_S
        or f.late_rms >= _VERDICT_OPEN_LATE_RMS
        or (
            f.pre_rms >= _VERDICT_OPEN_PRE_RMS
            and f.late_rms >= _VERDICT_OPEN_PRE_CORROB_LATE
        )
    ):
        return "open"
    if (
        tail_s <= _VERDICT_CLOSED_TAIL_S
        and f.late_rms <= _VERDICT_CLOSED_LATE_RMS
        and f.pre_rms <= _VERDICT_CLOSED_PRE_RMS
    ):
        return "closed"
    return None


def _rescue_discards(
    onsets: list[OnsetCandidate],
    feats: list[_Feat],
    open_idx: set[int],
    discard_idx: set[int],
    low_conf_discards: set[int],
) -> tuple[set[int], set[int]]:
    """Overturn LLM discards that every guard says are real hits.

    Mutates `open_idx` / `discard_idx` in place and returns
    `(rescued_open, rescued_closed)`. Recall-positive only: a discard is
    overturned to its envelope verdict (open/closed) iff ALL hold:

      * the envelope verdict is decisive (not ambiguous), AND
      * it isn't bleed: `lowband_ratio <= _BLEED_LOWBAND_RATIO_MAX`
        (snare/kick bleed dumps energy into the low band; a hat doesn't),
        AND
      * a fresh transient: `attack_flux >= _RESCUE_MIN_FLUX` (not sizzle),
        AND
      * not a double-trigger: `gap_s >= _RESCUE_MIN_GAP_S`, AND
      * the LLM was UNSURE (`low_conf_discards`) OR the envelope is
        OVERWHELMING (`_RESCUE_STRONG_*`).

    Never moves anything INTO discard, so it can't add artifacts.
    """
    rescued_open: set[int] = set()
    rescued_closed: set[int] = set()
    for i, (c, f) in enumerate(zip(onsets, feats, strict=True)):
        if i not in discard_idx:
            continue
        v = _envelope_open_verdict(f, c.time)
        if v is None:
            continue  # ambiguous envelope: respect the LLM's discard
        if (
            f.lowband_ratio > _BLEED_LOWBAND_RATIO_MAX  # looks like bleed
            or f.attack_flux < _RESCUE_MIN_FLUX          # no fresh strike
            or f.gap_s < _RESCUE_MIN_GAP_S               # double-trigger
        ):
            continue
        overwhelming = (
            (v == "open"
             and (f.tail_end_t - c.time) >= _RESCUE_STRONG_OPEN_TAIL_S)
            or (v == "closed" and f.attack_flux >= _RESCUE_STRONG_CLOSED_FLUX)
        )
        if i not in low_conf_discards and not overwhelming:
            continue
        discard_idx.discard(i)
        if v == "open":
            open_idx.add(i)
            rescued_open.add(i)
        else:
            rescued_closed.add(i)
    return rescued_open, rescued_closed


def _measure(
    stem_path: Path, onsets: list[OnsetCandidate]
) -> list[_Feat]:
    """Measure late-RMS / pre-RMS / attack / attack-flux / flatness / centroid / gap per onset.

    The hi-hat stem is loaded once. `late_rms` and `pre_rms` are the
    two discriminators for the "still ringing" / "riding on ring" open
    signatures, both are mean RMS over fixed windows, normalized to
    the onset's local peak. Two corrections vs a naïve implementation:

    1. The peak window extends BEFORE the onset by `_PEAK_BACK_S` so a
       slightly-late onset time still catches the real transient peak.
       Without this, an onset that lands a frame past the transient
       takes `peak = decay`, which inflates the late/pre ratios by an
       order of magnitude.
    2. The late / pre windows are clipped to leave `_NEIGHBOR_GUARD_S`
       away from adjacent onsets, so neighbour transients can't
       pollute this strike's tail / pre-ring measurements. The raw
       windows ([+200,+500]ms late, [-300,-50]ms pre) are wider than
       typical hi-hat gaps; without clipping, `late_rms` reports the
       NEXT strike's energy at any density above quarter notes.

    `attack_s` is the 10-90% rise time of the early post-onset envelope
    (closed = sharp; open = slower swell as the cymbals sizzle).
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
    # Onset-strength (half-wave-rectified spectral flux) envelope: responds
    # to ENERGY INCREASES (a fresh strike) and ignores steady ring level, so
    # a soft strike on top of a loud open-hat ring still shows a flux spike
    # while a sizzle re-trigger does not. `attack_flux` below normalizes each
    # onset's local flux peak to the stem's median flux, the fresh-attack
    # signal the open-within-open drop and the LLM prompt both consume.
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
    onset_env_t = librosa.times_like(onset_env, sr=sr, hop_length=hop)
    pos = onset_env[onset_env > 0.0]
    flux_med = float(np.median(pos)) if pos.size else float("inf")
    n = len(onsets)
    out: list[_Feat] = []
    for i, c in enumerate(onsets):
        t = c.time
        nxt = onsets[i + 1].time if i + 1 < n else t + _LATE_END_S
        prev = onsets[i - 1].time if i > 0 else None
        gap = nxt - t
        if prev is not None:
            gap = min(gap, t - prev)

        # --- fresh-attack flux: peak onset-strength in [t-0.02, t+0.04],
        # normalized to the stem's median flux. A real strike spikes here
        # regardless of how loud the surrounding ring is; a sizzle bump
        # (ring wobble, no fresh transient) stays near the floor.
        flux_mask = (
            (onset_env_t >= t - 0.02) & (onset_env_t <= t + 0.04)
        )
        attack_flux = (
            float(onset_env[flux_mask].max()) / flux_med
            if np.any(flux_mask) and np.isfinite(flux_med) and flux_med > 0.0
            else 0.0
        )

        # --- local peak (search a short window around the onset) ---
        # See `_PEAK_BACK_S`: the search starts BEFORE `t` to absorb
        # onset-time jitter; otherwise an onset that lands one frame
        # past the real transient gets `peak = decay` and blows up the
        # late/pre ratios downstream.
        peak_mask = (rms_t >= t - _PEAK_BACK_S) & (rms_t <= t + _PEAK_WIN_S)
        if not np.any(peak_mask):
            out.append(
                _Feat(0.0, 0.0, 0.0, 0.0, 0.0, float(gap), t + _TAIL_MIN_S,
                      attack_flux=attack_flux)
            )
            continue
        peak = float(rms[peak_mask].max())
        if peak <= 0.0:
            late_rms = 0.0
            pre_rms = 0.0
            tail_end_t = t + _TAIL_MIN_S
        else:
            # --- late RMS [t+_LATE_START_S, min(t+_LATE_END_S, nxt-guard)] / peak
            # Clipped to the next onset so this strike's "still
            # ringing" measurement doesn't pick up the next strike's
            # transient. Without the clip, [+200,+500]ms catches the
            # next hit at any density above quarter notes, and the
            # ratio reports the neighbour's energy rather than this
            # strike's tail.
            late_start_t = t + _LATE_START_S
            late_end_t = min(t + _LATE_END_S, nxt - _NEIGHBOR_GUARD_S)
            if late_end_t - late_start_t >= _MIN_WINDOW_S:
                late_mask = (rms_t >= late_start_t) & (rms_t <= late_end_t)
                late_rms = (
                    float(rms[late_mask].mean()) / peak
                    if np.any(late_mask) else 0.0
                )
            else:
                late_rms = 0.0
            # --- pre RMS [max(t-_PRE_START_S, prev+guard), t-_PRE_END_S] / peak
            # Symmetric clip against the previous onset: the previous
            # strike's transient must not pollute the "background ring"
            # measurement, otherwise a tight closed pattern reports
            # high `pre` (= previous transient) and looks like an open
            # passage.
            pre_end_t = t - _PRE_END_S
            pre_start_t = t - _PRE_START_S
            if prev is not None:
                pre_start_t = max(pre_start_t, prev + _NEIGHBOR_GUARD_S)
            if pre_end_t - pre_start_t >= _MIN_WINDOW_S:
                pre_mask = (rms_t >= pre_start_t) & (rms_t <= pre_end_t)
                pre_rms = (
                    float(rms[pre_mask].mean()) / peak
                    if np.any(pre_mask) else 0.0
                )
            else:
                pre_rms = 0.0
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

        # --- timbre (flatness + centroid) + low-band bleed ratio -----
        t1 = min(len(audio), int((t + _TIMBRE_WIN_S) * sr))
        clip = audio[a0:t1]
        if clip.size >= hop:
            flat = float(np.mean(librosa.feature.spectral_flatness(y=clip)))
            cen = float(
                np.mean(librosa.feature.spectral_centroid(y=clip, sr=sr))
            )
            lowband_ratio = _lowband_ratio(clip, sr)
        else:
            flat, cen, lowband_ratio = 0.0, 0.0, 0.0
        out.append(
            _Feat(late_rms, pre_rms, attack_s, flat, cen, float(gap),
                  tail_end_t, attack_flux=attack_flux,
                  lowband_ratio=lowband_ratio)
        )
    return out


def _lowband_ratio(clip: np.ndarray, sr: int) -> float:
    """Fraction of OCCUPIED-band energy (`_OCCUPIED_HZ`) that falls in the
    low band (`_LOWBAND_HZ`). High for snare/kick bleed (low-mid body),
    low for a hi-hat (high-band noise). Occupied band caps at ~14 kHz to
    avoid dividing by the dead air above the MP3 lowpass."""
    win = clip.astype(np.float64) * np.hanning(clip.size)
    spec = np.abs(np.fft.rfft(win)) ** 2
    freqs = np.fft.rfftfreq(clip.size, 1.0 / sr)
    occ = (freqs >= _OCCUPIED_HZ[0]) & (freqs <= _OCCUPIED_HZ[1])
    low = (freqs >= _LOWBAND_HZ[0]) & (freqs <= _LOWBAND_HZ[1])
    occ_e = float(spec[occ].sum())
    return float(spec[low].sum() / occ_e) if occ_e > 0.0 else 0.0


def _classify_llm(
    onsets: list[OnsetCandidate],
    feats: list[_Feat],
    structure: BeatStructure,
    onsets_by_pitch: dict[str, list[OnsetCandidate]],
    *,
    llm_model: str | None = None,
) -> tuple[set[int], set[int], set[int]] | None:
    """Ask the LLM to classify each onset open / closed / discard.

    Returns `(open_indices, discard_indices, low_confidence_discards)`;
    everything not in open or discard is implicitly closed. open/discard
    are guaranteed disjoint; overlapping entries resolve to **discard**
    (the safer error: a real hit lost as discard is one missed note; a
    sizzle bump mislabelled as open creates a phantom note AND extends the
    open-tail backstop's window, masking nearby closed hits too).
    `low_confidence_discards` is the subset of discards the model was
    unsure about; the discard-rescue overturns those readily.

    Returns `None` to signal the caller to use the deterministic
    fallback (no API key, call error, or malformed tool output).
    """
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
    model = llm_model or settings.llm_model
    try:
        response = call_messages_with_refusal_retry(
            client,
            {
                "model": model,
                "max_tokens": settings.llm_max_tokens,
                "temperature": 0,  # deterministic open/closed split (A/B + debug-bundle replay reproducibility)
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
        open_raw = block.input.get("open_indices", [])
        discard_raw = block.input.get("discard_indices", [])
        low_conf_raw = block.input.get("low_confidence_discards", [])
        if not isinstance(open_raw, list) or not isinstance(discard_raw, list):
            log.warning(
                "hihat split: non-list open/discard indices; using fallback"
            )
            return None
        discard_set = _coerce_index_set(discard_raw, n)
        # Disjointness: discard wins on overlap (see docstring).
        open_set = _coerce_index_set(open_raw, n) - discard_set
        # Low-confidence discards: keep only those actually in the discard
        # set (the model is told it's a subset; enforce it defensively).
        low_conf_set = _coerce_index_set(
            low_conf_raw if isinstance(low_conf_raw, list) else [], n
        ) & discard_set
        return open_set, discard_set, low_conf_set
    log.warning("hihat split: no tool_use block; using fallback")
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
) -> tuple[set[int], set[int], set[int]]:
    """Coarse deterministic open/closed split over the measured features.

    Open if STILL RINGING after the strike (high `late_rms`) OR riding
    on existing ring energy (high `pre_rms`). Either signature alone is
    sufficient. Never discards, "do nothing about sizzle" is acceptable
    degraded behaviour when the LLM is unavailable; the open-tail
    backstop still catches the most egregious cases. Runs only when the
    LLM is unavailable. Returns an empty discard + low-confidence set to
    match the `_classify_llm` contract.
    """
    opened: set[int] = set()
    for i, f in enumerate(feats):
        if f.late_rms >= _FALLBACK_LATE_RMS or f.pre_rms >= _FALLBACK_PRE_RMS:
            opened.add(i)
    return opened, set(), set()


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
                f"tail{ft.tail_end_t - c.time:.2f}s,"
                f"atk{ft.attack_s * 1000.0:.0f}ms,flux{ft.attack_flux:.1f},"
                f"lb{ft.lowband_ratio:.2f},gap{ft.gap_s:.2f}s)"
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
