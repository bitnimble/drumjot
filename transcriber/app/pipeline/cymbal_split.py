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

# STFT size for the timbre measurements (flatness / band slicing).
_TIMBRE_NFFT = 2048

# Flatness is measured only across the band cymbals actually occupy.
# Computed over the full 0-Nyquist range the metric collapses toward zero:
# the source/stem is dead above ~14 kHz (lossy-source lowpass + separator
# bandwidth), so the silent high bins crush the geometric mean (a single
# near-zero bin drags it down). The HI cut removes that dead region. The
# LO cut sits low (~250 Hz, not up in the kHz) on purpose: the ride/crash
# tonal cue is the pitched "ping" partial around 300-700 Hz, so the
# flatness band must reach down far enough to feel it; cutting higher hides
# exactly what distinguishes a tonal ride from a noisy crash. Below ~200 Hz
# is sub-band rumble / bleed with no cymbal content, so it stays excluded.
_FLATNESS_LO_HZ = 250.0
_FLATNESS_HI_HZ = 14000.0

# The low-band spectral-crest ("tonal") feature: peak-to-mean power over
# this band, in dB. It is the perception-matching ride/crash discriminator
# the flatness band can't capture. A ride has a tall narrow partial here
# (its pitched ping -> high crest); a crash is flat broadband noise here
# (-> crest near 0 dB). Measured low because that is where the pitched
# partial lives; everything above ~2 kHz is broadband mush for both
# classes. See `_band_crest_db`.
_TONAL_LO_HZ = 200.0
_TONAL_HI_HZ = 1500.0

# Post-onset energy envelope: RMS at these offsets past the hit's own peak,
# expressed in dB below that peak. Gives the model the decay *shape*
# (a crash sustains, a ride ping drops fast) instead of the single
# `decay_s` scalar, which collapses to a near-constant value whenever the
# next onset truncates the decay window. Offsets past the next onset are
# dropped, so a crowded hit returns a short list.
_ENV_OFFSETS_S = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
_ENV_FLOOR_DB = -60.0

# Lane letters used as deterministic voice labels.
_CRASH_LABEL = "crash"
_RIDE_LABEL = "ride"

# --- Voice-based classification ---------------------------------------
#
# Strategy: a song has a small number of distinct cymbal *voices* (a ride
# and/or a crash, occasionally two crashes). Each voice has a consistent
# timbre fingerprint. We cluster onsets into voices by timbre, then label
# each voice by whether it is the *timekeeping* cymbal -- the ride -- using
# the voice's GLOBAL stream fraction (how much of the whole song that voice
# spends in dense, evenly-spaced runs), defaulting to crash.
#
# Why voice-level density and not per-onset density: a crash played as a
# fast stream in one chorus (Cold Hard Bitch) still belongs to a voice
# that is sparse *across the whole song*, so its streamed hits inherit the
# crash label. A ride is the timekeeper, so its voice is dense everywhere.
# Per-onset density was the old bug (it called those streamed crashes ride);
# the global voice statistic is what fixes it. Timbre alone can't decide
# (a crash's tonal/decay overlaps a ride's, and neither is portable across
# songs), so the ride/crash call is relative-within-the-song.
#
# Intrinsic decay was tried and rejected: with room, a ride and a crash
# decay about the same (~0.9 s in both test songs), so `decay_s` / sustain
# / decay-rate don't separate them. Those features are still measured for
# the LLM discard pass and debug visibility, just not for the label.

# The LLM only flags artifacts to discard; it no longer makes the
# ride/crash call.

# Decay/sustain measurement window: look at the post-peak fall only this
# far past the peak (or to the next onset, whichever is sooner). Feeds the
# `sustain_db` / `decay_rate_db_s` debug features (not the label).
_SUSTAIN_HORIZON_S = 0.5

# An onset is "in a stream" when its nearest-neighbour gap is at or below
# this. A voice is labelled RIDE when at least `_RIDE_STREAM_FRACTION` of
# its onsets are in-stream (it is predominantly timekeeping); otherwise
# CRASH (the default). Measured CHB crash voices sit at ~0.16-0.19, ride
# voices at ~0.69-0.87, so the 0.45 bar separates them with wide margin.
_RIDE_STREAM_GAP_S = 0.35
_RIDE_STREAM_FRACTION = 0.45

# Accent recovery (a pure TIMBRE gate): a hit inside a ride voice is
# relabelled crash when its low/mid energy ratio (`low_mid_db`) sits at
# least this far below the median for the voice's in-stream notes. A ride
# has a pitched fundamental that fills the LOW band (~250-800 Hz), so its
# low/mid reads high; a crash is a mid "wash" with little low-band energy,
# so it reads low. Used RELATIVE to the voice (the absolute level isn't
# portable across kits: one kit's crash can have more low end than another
# kit's ride), so a genuine sparse RIDE note (which still has the
# fundamental) is spared while a foreign crash that fell inside the
# ride's cluster is caught -- regardless of rhythm/isolation.
#
# Ear-confirmed on itte (`low_mid_db` measured on the post-attack window):
# the 6 labelled crashes sit at -28..-36 dB, the ride stream median is
# ~-18 with no note below ~-27, so a 9 dB margin (threshold ~-27)
# recovers all 6 crashes -- including one embedded at ride spacing -- with
# zero in-stream false positives. `tonal` (pitched-ping crest) was tried
# and rejected: a crash's crest overlaps a ride's, so it didn't separate.
_RIDE_ACCENT_LOWMID_MARGIN_DB = 9.0

# Bands for `low_mid_db`: the ride's fundamental "tone" band over the
# crash's mid "wash" band.
_FUND_LO_HZ = 250.0
_FUND_HI_HZ = 800.0
_WASH_LO_HZ = 1500.0
_WASH_HI_HZ = 5000.0

# `low_mid_db` is measured on this post-attack window (seconds from the
# onset), NOT the short attack window the other timbre features use. The
# attack is a broadband transient for BOTH classes and blurs the ratio;
# after it, a ride's pitched fundamental rings on while a crash's wash
# fades, which is where the two separate cleanly. Measured on itte: this
# window gives zero overlap between the crash and ride distributions,
# versus 3/67 overlap on the attack window.
_LOWMID_WIN_START_S = 0.08
_LOWMID_WIN_END_S = 0.35

# Voice clustering (deterministic k-means on the standardized timbre
# fingerprint [centroid, tonal, flatness]). We try k = 2.._VOICE_MAX_K and
# keep the largest split where every cluster is big enough and the clusters
# are clearly separated; else fewer voices (down to one). The separation
# gate is what stops a single cymbal's articulations from over-splitting.
_KMEANS_ITERS = 20
_VOICE_MAX_K = 3              # ride + crash, occasionally a third cymbal
_VOICE_MIN_SIZE = 4           # min onsets for a voice to be real
_VOICE_SEPARATION_MIN = 1.5   # min pairwise centroid dist / within spread

_SPLIT_TOOL: dict[str, Any] = {
    "name": "report_cymbal_artifacts",
    "description": (
        "Report which detected cymbal onsets are NOT real hits and should "
        "be DISCARDED. The ride-vs-crash split is already decided; your "
        "only job is to flag artifacts: bleed (a weak onset that exists "
        "only because a louder instrument in `others:` hit at the same "
        "moment, with an off-voice fingerprint of its own), double- "
        "triggers (two onsets implausibly close where the drummer struck "
        "once), and sizzle re-triggers (weak bumps riding inside a real "
        "crash's decay tail). Return a single array of `#N` indices to "
        "discard; it should be the minority -- only clear artifacts. "
        "Return an empty array when nothing is a clear artifact. Never "
        "include an index that wasn't shown to you."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "discard_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "description": (
                    "The `#N` indices of onsets that are NOT real hits "
                    "(bleed, double-triggers, sizzle re-triggers in a "
                    "crash tail). The minority; only clear artifacts."
                ),
            },
        },
        "required": ["discard_indices"],
        "additionalProperties": False,
    },
}


class _Feat:
    """Per-onset measured features (kept off `OnsetCandidate` so the split
    stays local and doesn't widen the pipeline-wide onset schema).

    `env_db` is the post-onset decay envelope (see `_envelope_db`): RMS at
    fixed offsets past the peak, in dB below it, truncated at the next
    onset (so it can be shorter than `_ENV_OFFSETS_S`, or empty).

    `tonal_db` is the low-band spectral crest (see `_band_crest_db`): high
    when a pitched partial dominates (ride), near 0 when the band is flat
    noise (crash).

    `sustain_db` / `decay_rate_db_s` are the intrinsic-decay measures (see
    `_decay_metrics`); kept for debug / LLM context (they don't separate
    ride from crash on their own).

    `low_mid_db` is the low-band / mid-band energy ratio (see
    `_low_mid_db`): high when a pitched fundamental fills the low band (a
    ride), low for a mid-only "wash" (a crash). It drives accent recovery,
    relative to the voice's stream median."""

    __slots__ = (
        "decay_s", "flatness", "centroid_hz", "gap_s", "env_db", "tonal_db",
        "sustain_db", "decay_rate_db_s", "low_mid_db",
    )

    def __init__(
        self,
        decay_s: float,
        flatness: float,
        centroid_hz: float,
        gap_s: float,
        env_db: list[float] | None = None,
        tonal_db: float = 0.0,
        sustain_db: float = 0.0,
        decay_rate_db_s: float = 0.0,
        low_mid_db: float = 0.0,
    ) -> None:
        self.decay_s = decay_s
        self.flatness = flatness
        self.centroid_hz = centroid_hz
        self.gap_s = gap_s
        self.env_db = env_db if env_db is not None else []
        self.tonal_db = tonal_db
        self.sustain_db = sustain_db
        self.decay_rate_db_s = decay_rate_db_s
        self.low_mid_db = low_mid_db


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
    # Attach the per-onset measurements to the candidates so the UI's
    # per-note "Acoustic properties" subsection can show the same
    # numbers the classifier saw. Pure mutation; the split's downstream
    # logic still reads from the local `feats` list for indexing.
    for c, f in zip(in_range, feats, strict=True):
        c.decay_s = f.decay_s
        c.flatness = f.flatness
        c.centroid_hz = f.centroid_hz
        c.gap_s = f.gap_s

    # Deterministic ride/crash split: cluster onsets into cymbal voices by
    # timbre, then label each voice by its intrinsic decay rate (crash by
    # default). Rhythm/density does NOT decide ride vs crash.
    voice_ids = _cluster_voices(feats)
    voice_labels = _label_voices(feats, voice_ids)
    prov_labels = [voice_labels[voice_ids[i]] for i in range(len(in_range))]
    # Isolated, crash-timbred hits inside a ride voice are accent crashes.
    prov_labels = _demote_ride_accents(prov_labels, feats, voice_ids)

    # The LLM only prunes artifacts (bleed / double-trigger / sizzle).
    discard_idx = _discard_llm(
        in_range, feats, voice_ids, prov_labels, structure, onsets_by_pitch,
        llm_model=llm_model,
    )
    if discard_idx is None:
        discard_idx = _discard_fallback()
        source = "deterministic"
    else:
        source = "deterministic+llm_discard"

    ride = [
        c for i, c in enumerate(in_range)
        if prov_labels[i] == _RIDE_LABEL and i not in discard_idx
    ]
    crash = [
        c for i, c in enumerate(in_range)
        if prov_labels[i] == _CRASH_LABEL and i not in discard_idx
    ]
    discarded = [c for i, c in enumerate(in_range) if i in discard_idx]
    # Out-of-range cymbal onsets are never consumed downstream (bar < 0);
    # park them on the crash lane so nothing is silently discarded.
    crash.extend(out_of_range)

    n_voices = len(set(voice_ids))
    log.info(
        "cymbal split (%s): %d onsets, %d voice(s) %s -> %d ride, %d crash, "
        "%d discard",
        source,
        len(in_range),
        n_voices,
        {v: voice_labels[v] for v in sorted(set(voice_ids))},
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
                "n_voices": n_voices,
                "voice_labels": {
                    str(v): voice_labels[v] for v in sorted(set(voice_ids))
                },
                "n_ride": len(ride),
                "n_crash": len(crash) - len(out_of_range),
                "n_discard": len(discarded),
                "onsets": [
                    {
                        "index": i,
                        "bar": c.bar,
                        "beat_in_bar": round(c.beat_in_bar, 3),
                        "strength": round(c.strength, 3),
                        "voice": voice_ids[i],
                        "decay_s": round(feats[i].decay_s, 3),
                        "sustain_db": round(feats[i].sustain_db, 2),
                        "decay_rate_db_s": round(feats[i].decay_rate_db_s, 1),
                        "low_mid_db": round(feats[i].low_mid_db, 2),
                        "tonal_db": round(feats[i].tonal_db, 2),
                        "flatness": round(feats[i].flatness, 4),
                        "centroid_hz": round(feats[i].centroid_hz, 1),
                        "gap_s": round(feats[i].gap_s, 3),
                        "envelope_db": list(feats[i].env_db),
                        "label": (
                            "discard" if i in discard_idx else prov_labels[i]
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


def _band_flatness(
    power_spec: np.ndarray, freqs: np.ndarray, lo_hz: float, hi_hz: float
) -> float:
    """Spectral flatness (geometric mean / arithmetic mean of power)
    restricted to the `[lo_hz, hi_hz]` band, averaged over frames.

    Over the full 0-Nyquist range this metric collapses toward zero for
    cymbals: the occupied band is only ~1.5-14 kHz and the many near-empty
    bins outside it crush the geometric mean. Restricting to the occupied
    band keeps the number meaningful. `power_spec` is `(n_freq, n_frames)`;
    returns `0.0` when no FFT bin falls inside the band.
    """
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(mask):
        return 0.0
    band = power_spec[mask, :] + 1e-10
    gmean = np.exp(np.mean(np.log(band), axis=0))
    amean = np.mean(band, axis=0)
    return float(np.mean(gmean / amean))


def _band_crest_db(
    power_spec: np.ndarray, freqs: np.ndarray, lo_hz: float, hi_hz: float
) -> float:
    """Spectral crest (peak-to-mean power, in dB) over `[lo_hz, hi_hz]`,
    computed on the frame-averaged spectrum.

    High when a narrow tonal partial dominates the band -- a ride's pitched
    "ping" -- and near 0 dB for flat broadband noise -- a crash. Measured
    low (~200-1500 Hz) because that is where the ride/crash distinction
    lives; above ~2 kHz both classes are broadband and indistinguishable,
    and the flatness band sits entirely above the pitched partial. Returns
    `0.0` when no FFT bin falls inside the band or the band has no energy.
    """
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(mask):
        return 0.0
    band = power_spec[mask, :].mean(axis=1)
    mean = float(np.mean(band))
    if mean <= 0.0:
        return 0.0
    return float(10.0 * np.log10(float(np.max(band)) / mean))


def _low_mid_db(
    power_spec: np.ndarray,
    freqs: np.ndarray,
    fund: tuple[float, float],
    wash: tuple[float, float],
) -> float:
    """Low-band ("fundamental") over mid-band ("wash") energy, in dB.

    A ride's pitched tone fills the low band, so its ratio reads high; a
    crash is a mid-band wash with little low-band energy, so it reads low.
    The absolute level is NOT portable across kits, so it is consumed only
    relative to a voice's own stream median (see `_demote_ride_accents`).
    Returns `0.0` if the mid band has no energy.
    """
    lo = float(power_spec[(freqs >= fund[0]) & (freqs <= fund[1]), :].sum())
    mid = float(power_spec[(freqs >= wash[0]) & (freqs <= wash[1]), :].sum())
    if mid <= 0.0:
        return 0.0
    return float(10.0 * np.log10(max(lo, 1e-20) / mid))


def _decay_metrics(
    seg_rms: np.ndarray,
    seg_t: np.ndarray,
    peak_i: int,
    peak: float,
    horizon_end: float,
) -> tuple[float, float]:
    """Intrinsic decay over `[peak, horizon_end]`: returns
    `(sustain_db, decay_rate_db_s)`.

    `sustain_db` is the deepest the RMS fell after the peak, in dB below
    it (<= 0). `decay_rate_db_s` is that fall divided by the time it took,
    in dB/s -- a RATE, so a fast-articulate ride reads high and a washy
    crash reads low *regardless of how soon the next onset truncates the
    window*. That truncation-robustness is the whole point: `decay_s`
    collapses to a near-constant short value in a dense stream, but the
    rate still separates a crash wall (shallow) from a ride stream
    (steep). Returns `(0.0, 0.0)` for a zero peak or a window too short to
    measure.
    """
    if peak <= 0.0 or seg_rms.size < 2:
        return 0.0, 0.0
    pt = float(seg_t[peak_i])
    sel = (seg_t >= pt) & (seg_t <= horizon_end)
    rt = seg_t[sel]
    rr = seg_rms[sel]
    if rr.size < 2:
        return 0.0, 0.0
    vi = int(np.argmin(rr))
    valley = float(rr[vi])
    sustain_db = 20.0 * float(np.log10(max(valley, 1e-10) / peak))
    elapsed = float(rt[vi] - rt[0])
    rate = (-sustain_db / elapsed) if elapsed > 0.0 else 0.0
    return sustain_db, rate


def _envelope_db(
    rms: np.ndarray,
    rms_t: np.ndarray,
    peak_time: float,
    peak: float,
    win_end: float,
) -> list[float]:
    """Post-onset decay envelope: RMS sampled at `_ENV_OFFSETS_S` past the
    hit's peak, each expressed in dB below that peak and floored at
    `_ENV_FLOOR_DB`.

    Offsets landing past `win_end` (the next onset) are dropped, so a hit
    crowded by the next onset returns a short list, the model reads "few
    samples = tail cut short", same truncation `decay_s` suffers but with
    the *shape* up to the cut preserved. Returns `[]` for a zero/empty
    peak.
    """
    if peak <= 0.0 or rms_t.size == 0:
        return []
    out: list[float] = []
    for off in _ENV_OFFSETS_S:
        sample_t = peak_time + off
        if sample_t > win_end:
            break
        idx = int(np.searchsorted(rms_t, sample_t))
        if idx >= rms_t.size:
            break
        val = float(rms[idx])
        db = 20.0 * float(np.log10(max(val, 1e-10) / peak))
        out.append(round(max(db, _ENV_FLOOR_DB), 1))
    return out


def _measure(
    stem_path: Path, onsets: list[OnsetCandidate]
) -> list[_Feat]:
    """Measure decay / flatness / centroid / neighbour-gap / envelope per
    onset.

    The cymbals stem is loaded once. Decay is the time for post-onset RMS
    to fall `_DECAY_DROP_DB` below its local peak, searched only up to the
    next cymbal onset (capped at `_DECAY_MAX_S`); so a ride ping in a
    dense stream measures short by construction, an isolated crash long.
    Flatness is band-restricted (see `_band_flatness`); the envelope
    captures the decay shape that the single `decay_s` scalar loses when
    the window is truncated (see `_envelope_db`).
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
            power = (
                np.abs(
                    librosa.stft(clip, n_fft=_TIMBRE_NFFT, hop_length=hop)
                )
                ** 2
            )
            band_freqs = librosa.fft_frequencies(sr=sr, n_fft=_TIMBRE_NFFT)
            flat = _band_flatness(
                power, band_freqs, _FLATNESS_LO_HZ, _FLATNESS_HI_HZ
            )
            tonal = _band_crest_db(
                power, band_freqs, _TONAL_LO_HZ, _TONAL_HI_HZ
            )
            cen = float(
                np.mean(librosa.feature.spectral_centroid(y=clip, sr=sr))
            )
        else:
            flat, cen, tonal = 0.0, 0.0, 0.0
        # low/mid on the post-attack "tone" window (see _LOWMID_WIN_*).
        lm0 = int((t + _LOWMID_WIN_START_S) * sr)
        lm1 = min(len(audio), int((t + _LOWMID_WIN_END_S) * sr))
        lm_clip = audio[lm0:lm1]
        if lm_clip.size >= hop:
            lm_power = (
                np.abs(librosa.stft(lm_clip, n_fft=_TIMBRE_NFFT, hop_length=hop))
                ** 2
            )
            lm_freqs = librosa.fft_frequencies(sr=sr, n_fft=_TIMBRE_NFFT)
            low_mid = _low_mid_db(
                lm_power, lm_freqs,
                (_FUND_LO_HZ, _FUND_HI_HZ), (_WASH_LO_HZ, _WASH_HI_HZ),
            )
        else:
            low_mid = 0.0
        env = _envelope_db(rms, rms_t, float(seg_t[peak_i]), peak, win_end)
        horizon_end = min(win_end, float(seg_t[peak_i]) + _SUSTAIN_HORIZON_S)
        sustain_db, decay_rate = _decay_metrics(
            seg_rms, seg_t, peak_i, peak, horizon_end
        )
        out.append(
            _Feat(
                decay_s, flat, cen, float(gap), env, tonal,
                sustain_db, decay_rate, low_mid,
            )
        )
    return out


def _kmeans(z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic Lloyd k-means on standardized rows `z`. Greedy
    farthest-point seeding (first the point farthest from the global mean,
    then each next point farthest from the chosen set), so no RNG is
    needed. Returns `(labels, centroids)`.
    """
    centroids = [z[int(np.argmax(((z - z.mean(axis=0)) ** 2).sum(axis=1)))]]
    for _ in range(k - 1):
        d = np.min(
            [((z - c) ** 2).sum(axis=1) for c in centroids], axis=0
        )
        centroids.append(z[int(np.argmax(d))])
    c = np.array(centroids)
    labels = np.zeros(len(z), dtype=int)
    for it in range(_KMEANS_ITERS):
        d2 = np.stack([((z - ci) ** 2).sum(axis=1) for ci in c])
        new = d2.argmin(axis=0)
        if it > 0 and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                c[j] = z[labels == j].mean(axis=0)
    return labels, c


def _cluster_separation(
    z: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> float:
    """Min pairwise centroid distance divided by mean within-cluster
    spread. Higher = more clearly separated clusters; used to reject
    splits that merely carve up a single cymbal's natural variation.
    """
    k = len(centroids)
    within = [
        float(np.sqrt(((z[labels == j] - centroids[j]) ** 2).sum(axis=1)).mean())
        for j in range(k)
        if (labels == j).any()
    ]
    spread = float(np.mean(within)) if within else 0.0
    if spread <= 0.0:
        return float("inf")
    dmin = min(
        float(np.sqrt(((centroids[a] - centroids[b]) ** 2).sum()))
        for a in range(k)
        for b in range(a + 1, k)
    )
    return dmin / spread


def _cluster_voices(feats: list[_Feat]) -> list[int]:
    """Group onsets into cymbal voices by timbre fingerprint (standardized
    `[centroid, tonal, flatness]`). Returns a 0-based voice id per onset.

    Tries `k = 2.._VOICE_MAX_K` and keeps the LARGEST split where every
    cluster has at least `_VOICE_MIN_SIZE` onsets and the clusters are
    separated by at least `_VOICE_SEPARATION_MIN`; otherwise falls back to
    fewer voices, down to a single voice 0. The separation gate is what
    keeps one cymbal's articulations from over-splitting into spurious
    voices that could then be mislabelled.
    """
    n = len(feats)
    if _VOICE_MAX_K < 2 or n < _VOICE_MIN_SIZE * 2:
        return [0] * n
    x = np.array(
        [[f.centroid_hz, f.tonal_db, f.flatness] for f in feats], dtype=float
    )
    sd = x.std(axis=0)
    sd[sd == 0.0] = 1.0
    z = (x - x.mean(axis=0)) / sd
    best = [0] * n
    for k in range(2, _VOICE_MAX_K + 1):
        if n < _VOICE_MIN_SIZE * k:
            break
        labels, centroids = _kmeans(z, k)
        sizes = [int((labels == j).sum()) for j in range(k)]
        if min(sizes) < _VOICE_MIN_SIZE:
            continue
        if _cluster_separation(z, labels, centroids) >= _VOICE_SEPARATION_MIN:
            best = [int(v) for v in labels]
    return best


def _label_voices(
    feats: list[_Feat], voice_ids: list[int]
) -> dict[int, str]:
    """Label each voice ride or crash by its GLOBAL stream fraction. Crash
    is the default: a voice is `_RIDE_LABEL` only if it is big enough
    (`_VOICE_MIN_SIZE`) and at least `_RIDE_STREAM_FRACTION` of its onsets
    are in-stream (gap <= `_RIDE_STREAM_GAP_S`) -- i.e. the voice is
    predominantly timekeeping across the whole song.

    The statistic is per-voice and global, NOT per-onset: a crash played
    as a fast stream in one section still belongs to a voice that is sparse
    over the whole song, so its streamed hits stay crash. Only a genuine
    timekeeping cymbal is dense everywhere.
    """
    labels: dict[int, str] = {}
    for v in sorted(set(voice_ids)):
        gaps = [
            feats[i].gap_s for i, vi in enumerate(voice_ids) if vi == v
        ]
        in_stream = sum(1 for g in gaps if g <= _RIDE_STREAM_GAP_S)
        frac = in_stream / len(gaps) if gaps else 0.0
        if len(gaps) >= _VOICE_MIN_SIZE and frac >= _RIDE_STREAM_FRACTION:
            labels[v] = _RIDE_LABEL
        else:
            labels[v] = _CRASH_LABEL
    return labels


def _demote_ride_accents(
    prov_labels: list[str], feats: list[_Feat], voice_ids: list[int]
) -> list[str]:
    """Reclassify isolated, crash-timbred hits inside a ride voice as crash
    accents.

    A ride voice is the timekeeping cymbal, but clustering can sweep up
    accent crashes that fall inside its timbre. A hit is demoted when its
    `low_mid_db` sits at least `_RIDE_ACCENT_LOWMID_MARGIN_DB` below the
    median for its voice's in-stream notes -- it lacks the ride's pitched
    low fundamental, so it is timbrally a crash, not a sparse ride note.
    Purely timbre-based (no rhythm/isolation test): a sparse ride note
    keeps the fundamental and is spared, while a crash at any spacing
    (even embedded at ride spacing) is caught. Crash voices have no ride
    labels to demote. The reference is the median over IN-STREAM notes
    (gap <= `_RIDE_STREAM_GAP_S`), which are the genuine timekeeping hits,
    so a few swept-up crashes can't skew it.
    """
    ref_low_mid: dict[int, float] = {}
    for v in set(voice_ids):
        in_stream = [
            feats[i].low_mid_db
            for i in range(len(feats))
            if voice_ids[i] == v
            and prov_labels[i] == _RIDE_LABEL
            and feats[i].gap_s <= _RIDE_STREAM_GAP_S
        ]
        if in_stream:
            ref_low_mid[v] = float(np.median(in_stream))

    out = list(prov_labels)
    for i, lab in enumerate(prov_labels):
        if lab != _RIDE_LABEL:
            continue
        ref = ref_low_mid.get(voice_ids[i])
        if (
            ref is not None
            and feats[i].low_mid_db < ref - _RIDE_ACCENT_LOWMID_MARGIN_DB
        ):
            out[i] = _CRASH_LABEL
    return out


def _discard_llm(
    onsets: list[OnsetCandidate],
    feats: list[_Feat],
    voice_ids: list[int],
    prov_labels: list[str],
    structure: BeatStructure,
    onsets_by_pitch: dict[str, list[OnsetCandidate]],
    *,
    llm_model: str | None = None,
) -> set[int] | None:
    """Ask the LLM which onsets are artifacts to DISCARD.

    The ride/crash split is already decided deterministically (voice
    clustering + intrinsic decay); `voice_ids` / `prov_labels` are passed
    only as context so the model can spot off-voice bleed. Returns the set
    of indices to discard, or `None` to signal the caller to skip discards
    (no API key, call error, or malformed tool output).
    """
    if not settings.anthropic_api_key:
        log.info("cymbal split: no ANTHROPIC_API_KEY; skipping discard pass")
        return None

    bar_blocks = _format_bars(
        onsets, feats, voice_ids, prov_labels, structure, onsets_by_pitch
    )
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
                "temperature": 0,  # deterministic ride/crash split (A/B + debug-bundle replay reproducibility)
                "messages": [{"role": "user", "content": prompt}],
                "tools": [_SPLIT_TOOL],
                "tool_choice": {"type": "tool", "name": _SPLIT_TOOL["name"]},
            },
            base_prompt=prompt,
            purpose="cymbal_split",
        )
    except Exception as exc:
        log.warning(
            "cymbal split: LLM call failed (%s); skipping discard pass", exc
        )
        return None

    n = len(onsets)
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != _SPLIT_TOOL["name"]:
            continue
        discard_raw = block.input.get("discard_indices", [])
        if not isinstance(discard_raw, list):
            log.warning(
                "cymbal split: non-list discard indices; skipping discards"
            )
            return None
        return _coerce_index_set(discard_raw, n)
    log.warning("cymbal split: no tool_use block; skipping discards")
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


def _discard_fallback() -> set[int]:
    """No-LLM discard pass: discard nothing.

    The ride/crash split is fully deterministic (voice clustering +
    intrinsic decay), so when the LLM is unavailable we still produce
    correct lanes; we just skip artifact removal. "Keep everything" is the
    safe degraded behaviour -- the goal is "never drop a real hit", and a
    few un-pruned bleed/sizzle artifacts are acceptable without the model.
    """
    return set()


def _format_bars(
    onsets: list[OnsetCandidate],
    feats: list[_Feat],
    voice_ids: list[int],
    prov_labels: list[str],
    structure: BeatStructure,
    onsets_by_pitch: dict[str, list[OnsetCandidate]],
) -> str:
    """Render per-bar blocks: indexed cymbal onsets with their measured
    features and the deterministic voice id + provisional ride/crash label,
    plus a compact one-line summary of every other instrument's hits in
    that bar (so the model can spot kick-coincident bleed). The label is
    already decided; these rows let the model flag artifacts to discard."""
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
                f"#{i}({prov_labels[i][0]}{voice_ids[i]},"
                f"b{c.beat_in_bar:.2f},str{c.strength:.2f},"
                f"dec{ft.decay_s:.2f}s,sus{ft.sustain_db:.0f}dB,"
                f"rate{ft.decay_rate_db_s:.0f},tonal{ft.tonal_db:.0f}dB,"
                f"flat{ft.flatness:.3f},cen{ft.centroid_hz/1000.0:.1f}k,"
                f"gap{ft.gap_s:.2f}s,"
                f"env[{','.join(f'{v:.0f}' for v in ft.env_db)}])"
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
