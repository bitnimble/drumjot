"""Tom sub-classification: split the merged `t` lane into distinct toms.

Part of the model's onset post-processing (called from
`learned_onsets.detect_all_pitches_learned`): per song, cluster the tom onsets
by per-onset fundamental pitch and map the clusters (low->high) to GM tom notes,
so the transcriber distinguishes floor / low / mid / high toms instead of
emitting one merged tom. See `docs/tom-subclassification.md`.

Two layers:
  * `cluster_pitches` / `assign_keys` are PURE (numpy + scipy) and host-testable
    -- the valley-depth clustering that decides how many toms a kit has.
  * `subclassify` adds librosa-based per-onset pitch extraction (lazy import,
    run in the trainer image / transcriber venv).

Tuned + validated in `docs/tom-subclassification.md`; do not change the locked
constants without re-running that validation.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy.signal import argrelextrema

# --- locked clustering constants (see docs/tom-subclassification.md) ---
_BW = 0.5          # KDE bandwidth, semitones
_VALLEY_RATIO = 0.5  # valley density must be < this x the smaller adjacent peak
_MIN_SEP = 2.0     # min peak separation, semitones
_MIN_COUNT = 3     # min onsets per side of a split
_MIN_FIT = 8       # below this many voiced onsets, don't attempt to split

# Tom tiers low->high -> pipeline pitch key (-> GM note via onsets_midi).
# "f"=41 (floor), "tl"=45 (low), "tm"=47 (low-mid), "t"=50 (high).
_PALETTE = ("f", "tl", "tm", "t")


def _keys_for_k(k: int) -> list[str]:
    """Pitch keys low->high for `k` toms (single tom keeps the merged `t`)."""
    if k <= 1:
        return ["t"]
    table = {2: ["f", "t"], 3: ["f", "tl", "t"], 4: ["f", "tl", "tm", "t"]}
    if k in table:
        return table[k]
    return [_PALETTE[min(i, len(_PALETTE) - 1)] for i in range(k)]  # k>4 (rare)


def _density(grid: np.ndarray, xs: np.ndarray, h: float) -> np.ndarray:
    return np.exp(-0.5 * ((grid[:, None] - xs[None, :]) / h) ** 2).sum(axis=1)


def _split(xs: np.ndarray) -> list[np.ndarray]:
    """Recursively split sorted pitches at the deepest *real* density valley.

    The valley between two adjacent density peaks is taken as the *minimum*
    density point between them (argmin), not a strict local minimum: a clean
    gap (e.g. a floor tom an octave below the rack toms) flattens the density to
    ~0 in between, where strict-minimum detection finds nothing."""
    if len(xs) < 2 * _MIN_COUNT:
        return [xs]
    grid = np.linspace(xs[0] - 2, xs[-1] + 2, 600)
    d = _density(grid, xs, _BW)
    peaks = argrelextrema(d, np.greater)[0]
    if len(peaks) < 2:
        return [xs]
    best = None
    for lp, rp in zip(peaks, peaks[1:], strict=False):  # adjacent peak pairs
        if grid[rp] - grid[lp] < _MIN_SEP:
            continue
        v = lp + int(np.argmin(d[lp : rp + 1]))  # valley = min density between peaks
        ratio = d[v] / min(d[lp], d[rp])
        left = xs[xs <= grid[v]]
        right = xs[xs > grid[v]]
        if min(len(left), len(right)) < _MIN_COUNT:
            continue
        if best is None or ratio < best[0]:
            best = (ratio, left, right)
    if best is None or best[0] > _VALLEY_RATIO:
        return [xs]
    return _split(best[1]) + _split(best[2])


def cluster_pitches(semitones: Sequence[float]) -> list[float]:
    """Cluster a per-song tom-pitch population; return cluster centres (semitones,
    ascending). One centre => one tom. Pure (numpy/scipy), host-testable."""
    xs = np.sort(np.asarray([s for s in semitones], dtype=float))
    if len(xs) < _MIN_FIT:
        return [float(np.mean(xs))] if len(xs) else []
    clusters = _split(xs)
    return sorted(float(np.mean(c)) for c in clusters)


def assign_keys(pitches: Sequence[float | None]) -> list[str]:
    """Map per-onset pitches (semitones; None = unvoiced) to tom pitch keys
    (low->high). Falls back to all-`t` (merged) when there aren't enough voiced
    onsets or only one tom is found. Unvoiced onsets go to the modal tier."""
    voiced = [p for p in pitches if p is not None]
    if len(voiced) < _MIN_FIT:
        return ["t"] * len(pitches)
    centers = cluster_pitches(voiced)
    if len(centers) <= 1:
        return ["t"] * len(pitches)
    keys = _keys_for_k(len(centers))

    def tier(p: float) -> int:
        return int(np.argmin([abs(c - p) for c in centers]))

    modal = Counter(tier(p) for p in voiced).most_common(1)[0][0]
    return [keys[tier(p)] if p is not None else keys[modal] for p in pitches]


# --- audio-domain pitch extraction (lazy librosa; trainer/transcriber venv) ---
_SR = 22050
_HOP = 256


def _pyin_median(y, t: float, fmin: float, sr: int = _SR):
    import warnings

    import librosa

    a, b = int((t + 0.004) * sr), int((t + 0.18) * sr)
    if a < 0 or b > len(y) or b - a < 512 or np.max(np.abs(y[a:b])) < 1e-5:
        return None
    with warnings.catch_warnings():
        # benign "less than two periods of fmin" at our locked frame_length=1024
        warnings.simplefilter("ignore", UserWarning)
        f0, _, _ = librosa.pyin(
            y[a:b], fmin=fmin, fmax=400.0, sr=sr, frame_length=1024, hop_length=_HOP
        )
    v = f0[~np.isnan(f0)]
    return float(np.median(v)) if len(v) else None


def _kick_f0(kick_y, kick_times: Sequence[float]) -> float:
    """Median kick fundamental from the kick stem (default 50 Hz if absent)."""
    if kick_y is None or not len(kick_times):
        return 50.0
    vals = []
    for t in list(kick_times)[:40]:
        f = _pyin_median(kick_y, t, fmin=30.0)
        if f:
            vals.append(f)
    return float(np.median(vals)) if vals else 50.0


def _load(path, sr: int = _SR):
    import librosa

    if path is None:
        return None
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float64)


def tom_pitches(tom_audio, tom_times, kick_audio, kick_times) -> list[float | None]:
    """Per-onset tom fundamental in semitones (None = unvoiced), with the
    kick-aware low-cut applied. Lazy-imports librosa/scipy."""
    from scipy.signal import butter, sosfiltfilt

    y = _load(tom_audio)
    if y is None or not len(tom_times):
        return [None] * len(tom_times)
    kf = _kick_f0(_load(kick_audio), kick_times)
    hp = max(40.0, kf * 1.2)
    y = sosfiltfilt(butter(2, hp / (_SR / 2), btype="high", output="sos"), y)
    out: list[float | None] = []
    for t in tom_times:
        f = _pyin_median(y, t, fmin=hp)
        out.append(float(12 * np.log2(f)) if f else None)
    return out


def subclassify(tom_audio, tom_times, kick_audio, kick_times) -> list[str]:
    """Tom pitch key (`f`/`tl`/`tm`/`t`, low->high) per tom onset. Pure-fallback
    to all-`t` on any failure -- never breaks the merged-tom behaviour."""
    try:
        pitches = tom_pitches(tom_audio, tom_times, kick_audio, kick_times)
    except Exception:
        return ["t"] * len(tom_times)
    return assign_keys(pitches)
