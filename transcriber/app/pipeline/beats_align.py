"""Chart<->audio grid alignment (coarse phase + fine lag removal).

Two passes shift the whole beat grid onto the drum transients without changing
its metric regularity:

- **Coarse** (`align_beats_to_envelope` + the pure `_coarse_offset_from_envelope`):
  a single global shift, up to ±2 quarter-notes, that best seats the grid on
  the onset-strength envelope, killing a multi-slot phase error the fine
  pass's ±50 ms window can't see.
- **Fine** (`align_beats_to_onsets`): one median offset removing the tracker's
  systematic ~30-50 ms activation lag, gated by onset coverage.

`detect_envelope_onsets_for_alignment` produces the audio-only onset list the
fine pass consumes. Depends on `beats_types` + `beats_structure` (+ librosa).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.pipeline.beats_structure import _rebuild_bar_fields
from app.pipeline.beats_types import BeatStructure

log = logging.getLogger(__name__)

# Minimum fraction of beats that must have a strong drum onset within
# the alignment window before we trust the median offset enough to
# shift the whole grid. Drum stems normally hit this easily (most beats
# coincide with a kick/snare/hat); sparse or heavily-syncopated material
# that falls below it keeps the raw tracker grid rather than risk a
# bogus global shift.
MIN_ALIGN_COVERAGE = 0.30

# ---- Coarse envelope phase alignment (runs before the fine onset snap) ----
#
# The fine `align_beats_to_onsets` only searches ±50 ms (~1 slot). When the
# beat tracker locks onto a phase that's a few slots off, the true hits sit
# outside that window, so the fine pass no-ops or misfires and a constant
# multi-slot offset survives into the score. This coarse pass first finds a
# single global shift (up to ±`COARSE_MAX_SHIFT_BEATS` quarter notes) that
# best seats the beat grid on the drum-stem onset-strength envelope, then
# hands a well-phased grid to the fine pass for sub-frame lag removal.
COARSE_MAX_SHIFT_BEATS = 2.0   # cap: ± two quarter notes (half a 4/4 bar)
COARSE_SEARCH_STEP_SEC = 0.002  # offset search resolution (~0.05 slot @120)
COARSE_ENV_HOP = 256            # onset-strength hop (~5.8 ms @ 44.1 kHz)
# Multiplicative taper favouring small shifts: a shift at the ±cap must beat
# the zero-shift score by >this fraction to win, so we never lock onto a
# louder backbeat a full beat away when the grid is already close.
COARSE_CENTER_PENALTY = 0.15
# The winning comb score must exceed the mean comb score over the whole
# search range by this factor, else there's no clear pulse and we shift
# nothing (envelope too flat / not enough drum energy).
COARSE_PROMINENCE = 1.10


def detect_envelope_onsets_for_alignment(audio_path: Path) -> list[tuple[float, float]]:
    """Audio-only `(time, strength)` onset list for grid alignment, no model.

    Peaks of the librosa onset-strength envelope. Beat This! reports beats
    essentially on the transient (measured median lag ~2.5 ms on real drum
    stems, 100 % envelope coverage), so the heavy ADTOF onset pass the old
    neural trackers needed for their ~30-50 ms activation-peak lag is
    unnecessary; these envelope peaks feed the same `align_beats_to_onsets`
    median snap on CPU. Returns `[]` on empty/failed audio so the caller
    degrades to "no fine alignment".
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if y.size == 0:
        return []
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    times = librosa.times_like(env, sr=sr, hop_length=256)
    peaks = librosa.util.peak_pick(
        env, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.0, wait=2
    )
    return [(float(times[i]), float(env[i])) for i in peaks]


def align_beats_to_envelope(
    structure: BeatStructure,
    audio_path: Path,
) -> None:
    """Coarse global phase align: seat the beat grid on the drum envelope.

    Computes the onset-strength envelope of `audio_path` (the drum stem
    when available) and finds the single global time shift `δ`, within
    ±`COARSE_MAX_SHIFT_BEATS` quarter notes, that maximises the envelope
    energy summed over all beat positions shifted by `δ`. A multiplicative
    centre taper (`COARSE_CENTER_PENALTY`) biases the search toward small
    shifts so a grid that's already close isn't dragged a full beat onto a
    louder backbeat, and a prominence gate (`COARSE_PROMINENCE`) leaves the
    grid untouched when there's no clear pulse to lock onto.

    The shift is applied uniformly to every beat (inter-beat gaps and hence
    per-bar tempo are preserved) and accumulated into `align_offset_sec`.
    This runs *before* `align_beats_to_onsets`: it kills a multi-slot phase
    error the fine pass's ±50 ms window can't see, leaving the fine pass a
    well-phased grid on which to do precise lag removal.
    """
    if len(structure.beats) < 2:
        return
    import librosa

    beat_times = np.asarray([b.time for b in structure.beats], dtype=np.float64)
    beat_period = float(np.median(np.diff(np.sort(beat_times))))
    if not np.isfinite(beat_period) or beat_period <= 0:
        return
    max_shift = COARSE_MAX_SHIFT_BEATS * beat_period

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as exc:
        log.warning("coarse align: could not load %s (%s); skipping", audio_path, exc)
        return
    if y.size == 0:
        return
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=COARSE_ENV_HOP)
    if env.size == 0 or not np.any(env):
        return
    frame_times = librosa.frames_to_time(
        np.arange(env.size), sr=sr, hop_length=COARSE_ENV_HOP
    )

    offset = _coarse_offset_from_envelope(
        beat_times, env.astype(np.float64), frame_times,
        max_shift=max_shift, step=COARSE_SEARCH_STEP_SEC,
        center_penalty=COARSE_CENTER_PENALTY, prominence=COARSE_PROMINENCE,
    )
    if offset == 0.0:
        log.info(
            "coarse align: no confident global phase shift found; grid unchanged"
        )
        return
    for beat in structure.beats:
        beat.time += offset
    structure.align_offset_sec += offset
    structure.align_coarse_offset_sec += offset
    log.info(
        "coarse align: shifted all %d beats by %+.1f ms (%.2f slot @ %.1f BPM, "
        "search ±%.0f ms)",
        len(structure.beats), offset * 1000.0,
        offset / (beat_period / 12.0), 60.0 / beat_period, max_shift * 1000.0,
    )
    _rebuild_bar_fields(structure)


def _coarse_offset_from_envelope(
    beat_times: np.ndarray,
    env: np.ndarray,
    frame_times: np.ndarray,
    *,
    max_shift: float,
    step: float,
    center_penalty: float,
    prominence: float,
) -> float:
    """Global shift `δ` maximising Σ env(beat + δ), centre-biased and gated.

    Pure (no audio I/O) so it's unit-testable: sweeps `δ` over
    [-max_shift, +max_shift] in `step` increments, sampling the envelope at
    each shifted beat position by linear interpolation. The raw comb score
    is tapered by `(1 - center_penalty·|δ|/max_shift)` to favour small
    shifts, and the winner is accepted only if its raw score clears
    `prominence × mean(raw scores)`. Returns 0.0 when nothing is confident.
    """
    deltas = np.arange(-max_shift, max_shift + step, step)
    if deltas.size == 0:
        return 0.0
    raw = np.array(
        [float(np.interp(beat_times + d, frame_times, env).sum()) for d in deltas]
    )
    if not np.any(raw > 0):
        return 0.0
    taper = 1.0 - center_penalty * (np.abs(deltas) / max_shift)
    best = int(np.argmax(raw * taper))
    if raw[best] < prominence * float(raw.mean()):
        return 0.0
    return float(deltas[best])


def align_beats_to_onsets(
    structure: BeatStructure,
    onsets: list[tuple[float, float]],
    max_distance: float = 0.05,
) -> None:
    """Shift the whole beat grid by the tracker's *systematic* lag.

    Neural beat trackers report each beat
    ~30-50 ms after the transient, because the activation peak lags the
    strike. We still want to correct that.

    The previous implementation snapped **each beat independently** to
    the strongest drum onset within ±`max_distance`. That removed the
    lag but also absorbed the drummer's natural micro-timing into the
    grid: every beat's gap to its neighbours changed, so the per-bar
    tempo (`60 / mean(gap)` over a bar's 3-4 gaps) wobbled 5-10 BPM
    even on a dead-steady song, the LLM then emitted a `{{ bpm }}`
    change between nearly every bar.

    Instead, estimate ONE offset, the median over all beats of
    `(nearest strong onset − beat time)`, and shift every beat by it.
    The grid stays exactly as metrically regular as the DBN produced
    it (per-bar tempo is therefore stable), while the systematic lag is
    still removed. A uniform shift leaves inter-beat gaps unchanged, so
    a genuine accelerando the DBN tracked is preserved untouched.

    This is the FINE pass: it only searches ±`max_distance` (~50 ms), so
    it assumes the grid is already within ~1 slot of the true phase.
    `align_beats_to_envelope` runs first to guarantee that, it kills any
    larger multi-slot phase error this window can't see. The shift here is
    *added* to whatever the coarse pass already applied (`align_offset_sec`
    accumulates).

    The offset is only applied when enough beats actually had a nearby
    onset (`MIN_ALIGN_COVERAGE`); a handful of coincidental matches
    shouldn't drag the whole grid. `_rebuild_bar_fields` then refreshes
    per-bar `start_time` / `end_time` (which the shift moved) and the
    global tempo fields.
    """
    if not structure.beats or not onsets:
        return
    times = np.asarray([t for t, _ in onsets], dtype=np.float64)
    strengths = np.asarray([s for _, s in onsets], dtype=np.float64)
    order = np.argsort(times)
    times = times[order]
    strengths = strengths[order]

    # Strongest-not-closest: a quiet ghost hi-hat sitting nearer the
    # activation peak shouldn't outrank a louder kick/snare transient
    # slightly further out, the strong transient is the strike that
    # defines the beat.
    deltas: list[float] = []
    for beat in structure.beats:
        lo = int(np.searchsorted(times, beat.time - max_distance, side="left"))
        hi = int(np.searchsorted(times, beat.time + max_distance, side="right"))
        if lo >= hi:
            continue
        j = lo + int(np.argmax(strengths[lo:hi]))
        deltas.append(float(times[j] - beat.time))

    if not deltas:
        log.info(
            "beat alignment: no beats had an onset within ±%.0f ms; "
            "grid left unchanged",
            max_distance * 1000,
        )
        return

    coverage = len(deltas) / len(structure.beats)
    offset = float(np.median(deltas))

    if coverage < MIN_ALIGN_COVERAGE:
        log.info(
            "beat alignment: only %.0f%% of beats had a nearby onset "
            "(< %.0f%% required); offset %+.1f ms rejected, grid unchanged",
            coverage * 100, MIN_ALIGN_COVERAGE * 100, offset * 1000,
        )
        return

    for beat in structure.beats:
        beat.time += offset
    structure.align_offset_sec += offset
    structure.align_fine_offset_sec += offset
    log.info(
        "beat alignment: shifted all %d beats by %+.1f ms "
        "(median of %d beat→onset deltas, coverage %.0f%%)",
        len(structure.beats), offset * 1000, len(deltas), coverage * 100,
    )
    _rebuild_bar_fields(structure)
