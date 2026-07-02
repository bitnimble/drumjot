"""Odd-meter downbeat recovery + downbeat smoothing.

Two concerns, both operating on Beat This!'s raw beat/downbeat times before
they become a `BeatStructure`:

- **Meter recovery** (`_recover_bar_length_if_incoherent` + its autocorr
  helpers): when Beat This!'s DBN-free downbeat head sprays downbeats with no
  dominant per-bar period (its odd-meter failure), re-derive the true bar
  length from the autocorrelation of the beat-synchronous onset strength.
- **Downbeat smoothing** (`_beats_downbeats_to_raw` -> `_smooth_downbeats` ->
  `_merge_fragment_bars`): repair one-off mis-detections (merged / doubled /
  fragmented bars) that would fake a meter change, into the Nx2
  `(time, beat_pos_in_bar)` grid `_raw_to_structure` consumes.

Depends only on `beats_types` (+ numpy, and librosa lazily for the audio read).
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---- Meter recovery (odd-meter downbeat rescue) ----
#
# Beat This! is DBN-free, so its downbeat head carries no fixed-meter prior
# and reliably fails on odd meters (5/4, 7/4, 7/8): it sprays downbeats at a
# ~4-beat period instead of finding the true bar length, and no amount of
# `_smooth_downbeats` repair can recover a grouping that was never emitted.
# But the true bar length IS present in the audio, the onset-accent pattern
# repeats once per bar, so when Beat This!'s own downbeats are incoherent we
# re-derive the bar length from the autocorrelation of the beat-synchronous
# onset strength (which cleanly picks 5/7/… where the downbeat head could not).
# The beat *pulse* Beat This! tracks is left untouched; only the downbeat
# grouping is replaced.

# Fraction of Beat This! bars that must share the modal beat-count before its
# downbeats are trusted as-is. Below it the downbeats are judged incoherent
# (no dominant period) and the audio-autocorr bar length takes over. Confident
# 4/4 / 3/4 / 6/8 sit at ~0.85-1.0; the odd-meter failures sit at ~0.3-0.4, so
# a mid gate cleanly separates them without touching the common case.
_METER_TRUST_DOM_FRAC = 0.6

# Candidate bar lengths (in beats) the autocorr picker chooses among. Covers
# 2..9 beats/bar; the true fundamental wins argmax on real songs (its multiples
# 2B score lower), so no explicit octave/harmonic tie-break is needed.
_METER_CANDIDATES = (2, 3, 4, 5, 6, 7, 8, 9)

# Minimum beat-synchronous-accent autocorrelation the winning bar length must
# reach; below it there's no clear per-bar accent pattern, so keep Beat This!.
_METER_MIN_AUTOCORR = 0.12

# A divisor (≥3) of the argmax bar length is preferred as the true fundamental
# when its accent autocorr is at least this fraction of the argmax's (3/4 read
# as 6 → 3). Kept high so only a near-equal sub-period folds back.
_METER_FUNDAMENTAL_FRAC = 0.85

# Require this many bars of support (beats >= factor * B) before a candidate B
# is scorable, so a large B isn't chosen off a couple of noisy cycles.
_METER_MIN_BARS = 4

# Recovery only fires with at least this many beats of support. Autocorr meter
# detection needs enough bars to be reliable; on a short clip it picks spurious
# odd lengths (a compound 6/8 groove of ~35 beats reads as "5"), so below the
# floor we defer to Beat This!. Real songs sit far above it (hundreds of beats);
# only sub-30 s clips fall under.
_METER_MIN_BEATS = 64


def _recover_bar_length_if_incoherent(beats, downbeats, audio_path: Path):
    """Replace Beat This!'s downbeats with an autocorr-derived grouping when
    its own downbeats show no dominant per-bar period (the odd-meter failure).

    Returns a downbeat-times array: the original when Beat This! is coherent
    or nothing better can be found, else regular downbeats every `B` beats
    (phase = strongest-accent beat). Beat times are never modified.
    """
    beats = np.asarray(sorted(float(b) for b in beats), dtype=np.float64)
    downbeats = np.asarray(sorted(float(d) for d in downbeats), dtype=np.float64)
    if beats.size < _METER_MIN_BEATS or downbeats.size < 3:
        return downbeats

    dom_frac = _dominant_gap_fraction(beats, downbeats)
    if dom_frac >= _METER_TRUST_DOM_FRAC:
        return downbeats

    strength = _beat_sync_strength(beats, audio_path)
    bar_len = _bar_length_from_autocorr(strength)
    if bar_len is None:
        log.info(
            "meter recovery: downbeats incoherent (dom_frac %.2f) but no clear "
            "bar-length autocorr; keeping Beat This! downbeats", dom_frac,
        )
        return downbeats

    phase = _best_downbeat_phase(strength, bar_len)
    idx = np.arange(beats.size)
    new_downbeats = beats[idx[(idx % bar_len) == phase]]
    log.info(
        "meter recovery: Beat This! downbeats incoherent (dom_frac %.2f); "
        "re-derived %d-beat bars (phase %d) from onset-accent autocorr",
        dom_frac, bar_len, phase,
    )
    return new_downbeats


def _dominant_gap_fraction(beats: np.ndarray, downbeats: np.ndarray) -> float:
    """Fraction of inter-downbeat gaps (in beats) equal to the modal gap.

    ~1.0 for a cleanly-metered track, low when Beat This! scatters downbeats
    with no consistent bar length (the odd-meter failure signature).
    """
    db_idx = sorted({int(np.argmin(np.abs(beats - d))) for d in downbeats})
    gaps = np.diff(db_idx)
    if gaps.size == 0:
        return 1.0
    counts = Counter(int(g) for g in gaps)
    return counts.most_common(1)[0][1] / gaps.size


def _beat_sync_strength(beats: np.ndarray, audio_path: Path) -> np.ndarray:
    """Per-beat onset-accent: the onset-strength envelope summed in a ±70 ms
    window around each beat. One accent value per beat, so its autocorrelation
    exposes the bar-length periodicity."""
    import librosa

    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    except Exception as exc:
        log.warning("meter recovery: could not load %s (%s)", audio_path, exc)
        return np.zeros(beats.size, dtype=np.float64)
    if y.size == 0:
        return np.zeros(beats.size, dtype=np.float64)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    frame_times = librosa.frames_to_time(np.arange(env.size), sr=sr, hop_length=256)
    strength = np.empty(beats.size, dtype=np.float64)
    for i, t in enumerate(beats):
        lo = int(np.searchsorted(frame_times, t - 0.07))
        hi = int(np.searchsorted(frame_times, t + 0.07))
        strength[i] = float(env[lo:hi].sum()) if hi > lo else 0.0
    return strength


def _bar_length_from_autocorr(strength: np.ndarray) -> int | None:
    """Bar length (beats/bar) maximising the beat-accent autocorrelation.

    Returns the candidate `B` with the strongest lag-`B` autocorrelation of the
    per-beat accent, or None when no candidate has enough support or clears
    `_METER_MIN_AUTOCORR` (no coherent per-bar accent → don't override).

    A bar length also correlates at its multiples (a 3/4 bar repeats every 6
    beats too), so the raw argmax can land on a harmonic. We fold back to the
    fundamental: the smallest divisor ≥3 of the argmax whose accent autocorr is
    nearly as strong (a genuine 3/4 read as 6 collapses back to 3). Divisor 2 is
    excluded on purpose, a 4/4 backbeat gives a strong period-2 sub-correlation
    that would otherwise demote 4/4 to 2/4.
    """
    x = strength - strength.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return None
    scores: dict[int, float] = {}
    for B in _METER_CANDIDATES:
        if strength.size < _METER_MIN_BARS * B:
            continue
        scores[B] = float((x[:-B] * x[B:]).sum() / denom)
    if not scores:
        return None
    best_B = max(scores, key=lambda b: scores[b])
    if scores[best_B] < _METER_MIN_AUTOCORR:
        return None
    for d in _METER_CANDIDATES:
        if d >= best_B:
            break
        if d < 3:
            continue  # never demote to 2: that's the 4/4-backbeat harmonic
        if best_B % d == 0 and scores.get(d, 0.0) >= _METER_FUNDAMENTAL_FRAC * scores[best_B]:
            return d
    return best_B


def _best_downbeat_phase(strength: np.ndarray, bar_len: int) -> int:
    """Beat offset (0..bar_len-1) whose beats carry the most accent, the
    downbeat sits on the strongest recurring position (typically the kick)."""
    idx = np.arange(strength.size)

    def mean_accent(phase: int) -> float:
        vals = strength[(idx % bar_len) == phase]
        return float(vals.mean()) if vals.size else 0.0

    return max(range(bar_len), key=mean_accent)


def _beats_downbeats_to_raw(beats, downbeats, tol: float = 0.05) -> np.ndarray:
    """(beat times, downbeat times) -> Nx2 `(time, beat_pos_in_bar)`.

    `beat_pos_in_bar` is 1 at each downbeat and increments for the beats
    in between, matching the convention `_raw_to_structure` expects.
    Downbeats are a subset of the beat times, matched within `tol`.
    """
    beats = np.asarray(sorted(float(b) for b in beats), dtype=np.float64)
    if beats.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    db = np.asarray(sorted(float(d) for d in downbeats), dtype=np.float64)
    is_downbeat = np.zeros(beats.size, dtype=bool)
    for d in db:
        j = int(np.searchsorted(beats, d))
        nearest = min((k for k in (j - 1, j) if 0 <= k < beats.size),
                      key=lambda k: abs(beats[k] - d), default=None)
        if nearest is not None and abs(beats[nearest] - d) <= tol:
            is_downbeat[nearest] = True

    beats, is_downbeat = _smooth_downbeats(beats, is_downbeat)

    rows = np.empty((beats.size, 2), dtype=np.float64)
    pos = 0
    for k in range(beats.size):
        pos = 1 if (bool(is_downbeat[k]) or pos == 0) else pos + 1
        rows[k] = (beats[k], pos)
    return rows


def _smooth_downbeats(
    beats: np.ndarray, is_downbeat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Repair beat/downbeat mis-detections that would fake a meter change.

    Beat This! is DBN-free (no fixed-meter prior), so a stray downbeat or a
    local tempo flip shows up as a one-off odd bar. Against the *prevailing*
    meter P (majority interior bar length) and its typical duration D, a bar
    with an anomalous beat count `c` is repaired using its **duration** to
    tell the two failure modes apart:

    1. **Merged bars** (missed downbeat): `c == k·P` AND duration ≈ k·D, i.e.
       k real bars ran together. Split back into k bars of P (no 4/4→8/4, no
       3/4→6/4). Keeps every beat.
    2. **Local tempo/subdivision multiply** (a busy bar read at k× tempo, e.g.
       a 3/4 bar tracked as 6 fast beats → "6/8"): `c == k·P` AND duration ≈ D
       (same span as its neighbours). The bar boundary is correct, only the
       beat density is wrong, so **decimate** to P beats (keep every k-th) at
       the bar's true tempo, it stays one P bar, not split.
    3. **Fragmented bar** (extra downbeat): a run of consecutive sub-P bars
       whose lengths sum to exactly one P bar is merged (2+2 / 1+3 → 4).

    A *sustained* odd meter (≥2 bars that aren't a P-multiple and don't sum to
    one P bar, a real 3/4 or 6/8 section) is left untouched, so genuine
    mid-song changes survive. No-ops unless one meter holds a clear majority of
    the interior bars. A truly dropped/added *beat* in a lone bar that matches
    none of the above is preserved (can't fix without inventing beats).
    """
    n = int(beats.shape[0])
    db = [int(i) for i in np.flatnonzero(is_downbeat)]
    if len(db) < 3:
        return beats, is_downbeat
    # Prevailing meter P from the MEDIAN bar duration in beat-periods, NOT the
    # mode of per-bar counts: Beat This!'s downbeat jitter can make a half-bar
    # (e.g. 2 beats in a 4/4 song) the count plurality even when the typical
    # bar spans 4 beats. The median duration is robust to that.
    beat_period = float(np.median(np.diff(beats)))
    bar_durs = [float(beats[db[k + 1]] - beats[db[k]]) for k in range(len(db) - 1)]
    if beat_period <= 0 or not bar_durs:
        return beats, is_downbeat
    med_bar = float(np.median(bar_durs))
    p = int(round(med_bar / beat_period))
    # Skip unless there's a clean dominant bar length (median ≈ an integer
    # number of beats); a chaotically-varied grid is left as detected.
    if p < 2 or abs(med_bar - p * beat_period) > 0.15 * p * beat_period:
        return beats, is_downbeat
    bar_dur = p * beat_period

    out_t: list[float] = list(beats[:db[0]])          # leading anacrusis beats
    out_db: list[bool] = [False] * db[0]

    def emit(idxs: list[int], downbeat_every: int) -> None:
        for off, bi in enumerate(idxs):
            out_t.append(float(beats[bi]))
            out_db.append(off % downbeat_every == 0)

    for k in range(len(db) - 1):
        s, e = db[k], db[k + 1]
        c = e - s
        idxs = list(range(s, e))
        nbars = max(1, round((beats[e] - beats[s]) / bar_dur)) if bar_dur > 0 else 1
        if c >= 2 * p and c % p == 0 and c // p == nbars and nbars >= 2:
            emit(idxs, p)                              # merged k bars -> split
        elif c >= 2 * p and c % p == 0 and nbars == 1:
            emit(idxs[:: c // p][:p], p)               # tempo x k in one bar -> decimate
        else:
            emit(idxs, c if c > 0 else 1)              # keep as-is (one bar)
    out_t.extend(float(x) for x in beats[db[-1]:])     # trailing beats
    out_db.append(True)
    out_db.extend([False] * (n - db[-1] - 1))

    out_db = _merge_fragment_bars(out_db, p)
    return np.asarray(out_t, dtype=np.float64), np.asarray(out_db, dtype=bool)


def _merge_fragment_bars(db_flags: list[bool], p: int) -> list[bool]:
    """Drop interior downbeats of any run of consecutive sub-P bars whose
    lengths sum to exactly one P bar (an extra downbeat that fragmented a
    single bar, e.g. 2+2 or 1+3 -> 4). Flag-only; beats are untouched."""
    db = [i for i, f in enumerate(db_flags) if f]
    remove: set[int] = set()
    k = 0
    while k < len(db) - 1:
        if db[k + 1] - db[k] < p:
            run_sum, j = 0, k
            while j < len(db) - 1 and (db[j + 1] - db[j]) < p and run_sum < p:
                run_sum += db[j + 1] - db[j]
                j += 1
            if run_sum == p and j - k >= 2:
                remove.update(db[k + 1:j])
                k = j
                continue
        k += 1
    if not remove:
        return db_flags
    return [f and i not in remove for i, f in enumerate(db_flags)]
