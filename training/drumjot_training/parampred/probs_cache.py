"""Persistent per-(stem, augmentation) activation-curve cache.

The expensive step in building the param corpus is the MERT encode (GPU). This
caches the model's per-lane probability curves keyed by (stem path, augmentation
recipe, encoder/layer/in_dim, length + window caps), so re-runs -- more variants,
re-training, a resumed build -- never re-encode what's already done. Augmented
audio is reproducible (a deterministic per-(stem, variant) seed), so its recipe
and cache key are stable across runs.

Pure helpers (key / save / load / variant_audio) are host-testable; `probs_for`
lazily imports the torch inference path.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from drumjot_training.parampred import augment

CACHE_VERSION = "p1"  # bump if the stored representation changes


def probs_key(
    stem_path: str | Path, recipe: str, *, encoder: str, layer: int, in_dim: int,
    max_seconds: float | None, window_seconds: float | None,
) -> str:
    """Stable cache key for one (stem, augmentation) activation curve."""
    raw = (f"{Path(stem_path).expanduser().absolute()}|{recipe}|{encoder}|{layer}|{in_dim}"
           f"|{max_seconds}|{window_seconds}|{CACHE_VERSION}")
    return hashlib.sha1(raw.encode()).hexdigest()


def window_plan_key(audio_path: str | Path, window: float, search: float) -> str:
    """Key into train.py's `_window_plan.json` (path|window|search) -> the exact
    nudged [(start, length), ...] windows the MERT cache was built against."""
    return f"{audio_path}|{window}|{search}"


def window_onsets(
    gt: dict[str, list[float]], start: float, length: float
) -> dict[str, list[float]]:
    """Onsets falling in `[start, start+length)`, shifted to window-relative time."""
    return {ln: [t - start for t in ts if start <= t < start + length] for ln, ts in gt.items()}


def load_window_features(
    feature_cache_dir: str | Path, audio_path: str | Path, start: float, length: float,
    *, encoder: str, layer: int, variant: str,
) -> np.ndarray | None:
    """Load one window's cached MERT(+high-band) features `(T, in_dim)`, or None
    if not cached. Keyed exactly as `CachedClips._path` (window=length, start)."""
    from drumjot_training import embeddings

    key = embeddings.cache_key(audio_path, encoder, layer, window=length, variant=variant, start=start)
    f = Path(feature_cache_dir) / f"{key}.npy"
    return np.load(f) if f.exists() else None


def load_probs(cache_dir: str | Path, key: str) -> tuple[np.ndarray, float] | None:
    """Return `(probs (n_lanes, T) float32, fps)` if cached, else None."""
    f = Path(cache_dir) / f"{key}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return d["probs"].astype(np.float32), float(d["fps"])


def save_probs(cache_dir: str | Path, key: str, probs: np.ndarray, fps: float) -> None:
    """Persist `probs` (stored fp16 -- activations in [0,1] don't need more) atomically."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f".{key}.tmp.npz"
    np.savez(tmp, probs=probs.astype(np.float16), fps=np.float64(fps))
    tmp.replace(cache_dir / f"{key}.npz")


def variant_audio(
    stem_path: str | Path, variant: int, wave: np.ndarray, sr: int, *, codec: bool = True,
) -> tuple[np.ndarray, str]:
    """`(audio, recipe)` for one variant. Variant 0 is the untouched stem; higher
    variants apply a deterministic augmentation chain seeded by (stem, variant),
    so the same variant reproduces byte-for-byte (stable cache key)."""
    if variant == 0:
        return wave, "identity"
    seed = int(hashlib.sha1(f"{Path(stem_path).name}|{variant}".encode()).hexdigest()[:8], 16)
    aug, desc = augment.random_chain(wave, sr, np.random.default_rng(seed), use_codec=codec)
    return aug, f"v{variant}:{desc}"


def probs_for(
    stem_path, recipe, audio, sr, model, meta, encoder, cache_dir, *,
    max_seconds=None, window_seconds=30.0,
) -> tuple[np.ndarray, float]:
    """Cached activation curves for one (stem, recipe): load if present, else run
    the frozen model (encoding `audio`) and persist. `audio` is the (possibly
    augmented) waveform; identity uses the stem path directly."""
    import os
    import tempfile

    from drumjot_training import inference

    key = probs_key(stem_path, recipe, encoder=meta["encoder"], layer=meta["encoder_layer"],
                    in_dim=meta["in_dim"], max_seconds=max_seconds, window_seconds=window_seconds)
    hit = load_probs(cache_dir, key)
    if hit is not None:
        return hit

    tmp = None
    if recipe == "identity":
        audio_path = str(stem_path)
    else:
        import soundfile as sf

        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmp, audio, sr)
        audio_path = tmp
    try:
        probs, fps = inference.stitched_probs(audio_path, model, meta, encoder, max_seconds, window_seconds)
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)
    save_probs(cache_dir, key, probs, fps)
    return probs.astype(np.float32), fps
