"""ParaDB (Paradiddle) zip corpus -> per-stem training dataset + shared helpers.

The dataset layer over a folder of `maps__<id>.zip` ParaDB packs. `rlrr.py` is the
pure chart parser (events -> per-lane onsets); THIS module owns everything at the
zip/song level: discovering the referenced audio tracks, reconstructing the full
mix without double-counting drums (`build_mix`), the robust global chart->audio
offset (`global_offset`), and the per-stem dataset index consumed by training.

Single source of truth: `scripts/eval_paradb.py` (the benchmark), the corpus-cull
gate (`scripts/build_paradb_manifest.py`), and the separation step
(`scripts/separate_paradb_dataset.py`) all import the mix/offset/chart helpers
from here, so the cull, the eval, and training can't drift (the
`drumjot_dsp.peakpick` principle).

Per-stem layout produced by `separate_paradb_dataset.py` (mirrors enst-sep /
mdb-sep), consumed by `perstem_index`:

    <root>/perstem/<pitch>/<map_id>.flac   # pitch in k/s/h/c/t (MDX23C lanes)
    <root>/onsets/<map_id>.json            # offset-corrected 9-lane GT onsets

LICENSE / SCOPE: ParaDB songs are copyrighted recordings and the crowdsourced
charts are unlicensed. This whole pathway is RESEARCH-ONLY -- training
experiments + held-out evaluation. We do NOT intend to ship a model trained on
ParaDB data, and the held-out split is for measurement only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drumjot_training import clean, rlrr
from drumjot_training.lanes import LANES

_AUDIO_EXTS = {".ogg", ".mp3", ".wav", ".flac", ".m4a", ".aac"}
SEP_SR = 44100  # separate at full band so cymbal/hi-hat content survives

# MDX23C stem pitch -> the model lanes that legitimately belong to that isolated
# stem (a stem example is labelled with ONLY these lanes; onsets in any other
# lane are cross-instrument bleed the model must learn to suppress). Identical to
# enst.PERSTEM_TO_LANES / eval_paradb.STEM_TO_LANES.
PERSTEM_TO_LANES: dict[str, tuple[str, ...]] = {
    "k": ("k",),
    "s": ("s", "ss"),
    "h": ("hc", "hp", "ho"),
    "c": ("rd", "cr"),
    "t": ("t",),
}
PERSTEM_PITCHES: tuple[str, ...] = tuple(PERSTEM_TO_LANES)


# ---------------------------------------------------------------------------
# zip / chart discovery
# ---------------------------------------------------------------------------
def map_id_of_zip(zip_path: str | Path) -> str:
    """`.../maps__M000A2B.zip` -> `M000A2B` (the stable per-map id). Falls back to
    the bare stem for any zip not following the `maps__<id>` convention."""
    stem = Path(zip_path).stem
    return stem[len("maps__"):] if stem.startswith("maps__") else stem


def iter_zips(maps_dir: str | Path) -> list[Path]:
    """Sorted list of `*.zip` ParaDB packs under `maps_dir`."""
    return sorted(Path(maps_dir).glob("*.zip"))


def pick_chart(root: str | Path) -> Path | None:
    """Hardest `.rlrr` in an extracted map dir, chosen deterministically
    (complexity, then difficulty name, then path; see `rlrr.pick_hardest`). A
    complexity tie between e.g. Expert + Hard charts otherwise resolves by
    unstable rglob order, parsing a different GT per run.

    Excludes macOS AppleDouble junk: a pack zipped on macOS carries a
    `__MACOSX/.../._<name>.rlrr` resource-fork sibling for every real chart,
    which is binary (not JSON) and crashes the parser if `pick_hardest` reads it
    for complexity. Drop anything under `__MACOSX/` or starting with `._`."""
    charts = [
        p for p in Path(root).rglob("*.rlrr")
        if not p.name.startswith("._") and "__MACOSX" not in p.parts
    ]
    return rlrr.pick_hardest(charts)


# ---------------------------------------------------------------------------
# mix reconstruction (shared with eval_paradb)
# ---------------------------------------------------------------------------
def _sum_tracks(root: Path, names: list[str], sr: int):
    """Load + sum referenced audio tracks (mono, resampled). None if none found."""
    import librosa

    ys = []
    for name in names:
        base = Path(name).name
        hits = [p for p in root.rglob(base) if p.suffix.lower() in _AUDIO_EXTS]
        if not hits:
            print(f"    WARN track not found in zip: {name}", flush=True)
            continue
        y, _ = librosa.load(str(hits[0]), sr=sr, mono=True)
        ys.append(y.astype(np.float32))
    if not ys:
        return None
    n = max(len(y) for y in ys)
    out = np.zeros(n, dtype=np.float32)
    for y in ys:
        out[: len(y)] += y
    return out


def _pad_sum(a, b):
    n = max(len(a), len(b))
    out = np.zeros(n, dtype=np.float32)
    out[: len(a)] += a
    out[: len(b)] += b
    return out


def containment(song, drums, sr, max_seconds) -> float:
    """Raw-sample correlation of the drum track with the song mix.

    ~0 when the song is drumless backing (those drums are NOT in it); clearly
    positive when the song already contains those drums (a full mix = backing +
    drums, so corr ~= drum energy fraction). Unlike an onset-support test this
    isn't fooled by non-drum instruments hitting on the same beats.
    """
    n = min(len(song), len(drums))
    if max_seconds is not None:
        n = min(n, int(max_seconds * sr))
    a = song[:n].astype(np.float64) - float(np.mean(song[:n]))
    b = drums[:n].astype(np.float64) - float(np.mean(drums[:n]))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return abs(float(a @ b) / denom)


def build_mix(root, song_names, drum_names, sr, out_wav, max_seconds, corr_thresh):
    """Reconstruct the full original song without double-counting drums:

    - song tracks ONLY if they already contain the drums (full-mix map, even one
      that also ships redundant drum stems -> don't add them);
    - song + drum tracks if the song tracks are drumless backing (stems map).

    Decided by drum/song signal correlation (see `containment`), which is robust
    to the coincident-onset problem that breaks an onset-support test. Returns
    `(ok, case_label)`.
    """
    import soundfile as sf

    song = _sum_tracks(root, song_names, sr) if song_names else None
    drums = _sum_tracks(root, drum_names, sr) if drum_names else None

    if song is not None and drums is not None:
        corr = containment(song, drums, sr, max_seconds)
        if corr > corr_thresh:  # song already contains these drums (full mix)
            mix, case = song, f"song-only; drums already in song (corr {corr:.2f})"
        else:  # drumless backing -> add the drum track to rebuild the full song
            mix, case = _pad_sum(song, drums), f"backing+drums (corr {corr:.2f})"
    elif song is not None:
        mix, case = song, "song-only"
    elif drums is not None:
        mix, case = drums, "drums-only"
    else:
        return False, "no audio"

    peak = float(np.max(np.abs(mix)) or 1.0)
    sf.write(str(out_wav), mix / peak * 0.98, sr)  # float wav, headroom
    return True, case


def global_offset(gt, env, env_fps, floor, window_s, search_s):
    """Robust global chart->audio offset + chart-quality support at offset 0.

    offset = MEDIAN signed distance from each GT onset to its nearest envelope
    peak (>= floor) within +/-search_s. The median is robust to dense drum peaks
    and straggler onsets, unlike argmax-of-support (which, on a near-saturated
    support plateau, overshoots the true offset by chasing a few outliers).
    support@0 = fraction of onsets within +/-window_s of a qualifying peak at
    offset 0, a chart-accuracy / corruption signal. Returns `(offset, support@0)`.
    """
    half = round(search_s * env_fps)
    n = env.size
    deltas = []
    for ts in gt.values():
        for t in ts:
            c = int(round(t * env_fps))
            lo, hi = max(0, c - half), min(n, c + half + 1)
            if lo >= hi:
                continue
            idx = lo + int(np.argmax(env[lo:hi]))
            if float(env[idx]) >= floor:
                deltas.append(idx / env_fps - t)
    off = float(np.median(deltas)) if deltas else 0.0
    s0 = clean.support_score(gt, env, env_fps, window_s=window_s, support_floor=floor)["fraction"]
    return off, s0


def shift_onsets(onsets: dict[str, list[float]], offset: float) -> dict[str, list[float]]:
    """Shift every lane's onsets by `offset` seconds (no-op when offset == 0)."""
    if not offset:
        return {ln: list(ts) for ln, ts in onsets.items()}
    return {ln: [t + offset for t in ts] for ln, ts in onsets.items()}


# ---------------------------------------------------------------------------
# per-stem training dataset (consumes the separate_paradb_dataset.py output)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParadbPerstemClip:
    audio_path: Path   # <root>/perstem/<pitch>/<map_id>.flac
    onsets_path: Path  # <root>/onsets/<map_id>.json  (offset-corrected 9-lane GT)
    pitch: str         # k / s / h / c / t
    map_id: str


def perstem_index(root: str | Path) -> list[ParadbPerstemClip]:
    """Index per-instrument stems: one entry per (map, drum-piece pitch).

    Pairs each `perstem/<pitch>/<map_id>.flac` (written by
    `scripts/separate_paradb_dataset.py`) with its `onsets/<map_id>.json`. Stems
    that weren't produced are skipped; a map with no onsets json is skipped."""
    root = Path(root)
    onsets_dir = root / "onsets"
    perstem_dir = root / "perstem"
    clips: list[ParadbPerstemClip] = []
    for oj in sorted(onsets_dir.glob("*.json")):
        map_id = oj.stem
        for pitch in PERSTEM_TO_LANES:
            audio = perstem_dir / pitch / f"{map_id}.flac"
            if audio.exists():
                clips.append(ParadbPerstemClip(audio, oj, pitch, map_id))
    return clips


def onsets_by_lane(onsets_path: str | Path) -> dict[str, list[float]]:
    """Read a stored `onsets/<map_id>.json` -> all-lanes onset dict (sorted).

    Signature mirrors `rlrr.onsets_by_lane` / `enst.onsets_by_lane` so the pooled
    spec builder can use it as the `reader` for the parsed-onset cache."""
    raw = json.loads(Path(onsets_path).read_text())
    return {ln: sorted(float(t) for t in raw.get(ln, [])) for ln in LANES}


def restricted_onsets(onsets_path: str | Path, pitch: str) -> dict[str, list[float]]:
    """Stored onsets keeping ONLY the lanes that belong to `pitch`'s stem; all
    other lanes empty (so the isolated-stem example teaches bleed suppression).
    Always returns all lanes."""
    full = onsets_by_lane(onsets_path)
    keep = set(PERSTEM_TO_LANES.get(pitch, ()))
    return {ln: (full[ln] if ln in keep else []) for ln in LANES}


# ---------------------------------------------------------------------------
# corpus selection + held-out split (consumes build_paradb_manifest.py output)
# ---------------------------------------------------------------------------
def load_manifest(path: str | Path) -> dict:
    """Load `paradb_manifest.json` (map_id -> gate result entry)."""
    return json.loads(Path(path).read_text())


def kept_map_ids(
    manifest: dict, *, min_support: float, min_recall: float, min_onsets: int = 0
) -> list[str]:
    """Map ids that pass the cull: status 'ok' AND support_corr >= `min_support`
    AND recall >= `min_recall` (and >= `min_onsets` if set). Sorted (deterministic).
    Shared by separate_paradb_dataset.py (build the train tree) and the param
    dataset builder, so both agree on exactly which maps are 'kept'."""
    out = []
    for mid, e in manifest.items():
        if e.get("status") != "ok":
            continue
        if e.get("support_corr", 0.0) < min_support or e.get("recall", 0.0) < min_recall:
            continue
        if min_onsets and e.get("n_onsets", 0) < min_onsets:
            continue
        out.append(e.get("map_id", mid))
    return sorted(out)


def _hash_frac(map_id: str, salt: str) -> float:
    """Stable [0,1) hash of a map id (machine-independent; no RNG state)."""
    h = hashlib.sha1(f"{salt}:{map_id}".encode()).hexdigest()
    return (int(h[:8], 16) % 1_000_000) / 1_000_000.0


def perstem_for_split(clips, split: str, *, val_frac: float = 0.05, salt: str = "paradb-val"):
    """Split per-stem clips into train/validation by **map-id** hash (so all of a
    song's stems land in the same split -> no cross-stem leakage). This is the
    held-IN model val set (threshold tuning / early stop), distinct from the eval
    HOLDOUT (`holdout_split`), which is excluded from the tree entirely. `split` is
    'train' or 'validation'."""
    want_val = split in ("validation", "val")
    return [c for c in clips if (_hash_frac(c.map_id, salt) < val_frac) == want_val]


def holdout_split(
    map_ids, holdout_frac: float, *, salt: str = "paradb-eval"
) -> tuple[list[str], list[str]]:
    """Deterministic `(train_ids, eval_ids)` split by map-id hash: eval = the
    `holdout_frac` of ids whose hash falls below the cut. Stable across runs and
    machines (sha1, not RNG), so the held-out eval set is frozen + reproducible.
    Frozen to `paradb_eval_ids.json` by separate_paradb_dataset.py; the same call
    excludes those ids from BOTH the onset train tree and the param corpus."""
    ev = sorted(m for m in map_ids if _hash_frac(m, salt) < holdout_frac)
    evset = set(ev)
    tr = sorted(m for m in map_ids if m not in evset)
    return tr, ev
