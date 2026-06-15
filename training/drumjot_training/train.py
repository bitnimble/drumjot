"""Phase-0 smoke test: does a frozen MERT + per-lane heads learn onsets?

Per clip: frozen MERT features + per-lane Gaussian onset targets from the
MIDI -> train `MultiLaneHeads` with per-frame BCE. Milestones (design spec
§2): overfit one clip (wiring), then train loss over a few clips, then
held-out onset-F1. Scored on onset-F1, never frame accuracy.

Run in the CUDA sandbox (torch + MERT). The real-data path reads E-GMD via
`paths.dataset_path("egmd")`; `--synthetic` runs a dataset-free self-test
(random features + planted onsets) that verifies the training mechanics
(loss drops, the eval/peak-pick pipeline runs) with no data present.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drumjot_training import (
    checkpoint,
    egmd,
    embeddings,
    enst,
    metrics,
    midi_labels,
    paths,
    runtime,
    star,
)
from drumjot_training.config import Config
from drumjot_training.lanes import sibling_matrix
from drumjot_training.model import MultiLaneHeads
from drumjot_training.targets import (
    SUSTAINED_LANES,
    onsets_to_target,
    pos_weights_from_targets,
    ring_spans,
    spans_to_activity,
)


@dataclass
class Clip:
    """A training example: frozen features, per-lane targets, and the raw
    onset times (kept for onset-F1 eval, which scores against true onsets,
    not the smoothed target)."""

    features: np.ndarray  # (T, dim)
    targets: np.ndarray  # (n_lanes, T)
    onsets_by_lane: dict[str, list[float]]
    audio_path: str | None = None  # source clip, for envelope post-filtering
    # Targets used ONLY for sibling-aware loss weighting. None -> `targets`.
    # Differs in per-stem mode: `targets` are restricted to the stem's lanes,
    # but the weighting must still see the FULL kit's onsets so e.g. hat bleed
    # on the cymbals stem counts as a hard negative for ride.
    weight_targets: np.ndarray | None = None
    # Auxiliary ring-activity targets (n_lanes, T), nonzero only on the
    # sustained lanes (targets.SUSTAINED_LANES). None -> zeros.
    activity_targets: np.ndarray | None = None


def build_targets(
    onsets_by_lane: dict[str, list[float]], n_frames: int, cfg: Config
) -> np.ndarray:
    """Stack per-lane Gaussian target curves into (n_lanes, T)."""
    return np.stack(
        [
            onsets_to_target(onsets_by_lane.get(lane, []), n_frames, cfg.encoder_fps, cfg.sigma_frames)
            for lane in cfg.lanes
        ]
    )


def _build_clip(
    audio_path: Path,
    onsets: dict[str, list[float]],
    encoder: embeddings.MertEncoder,
    cfg: Config,
    cache_dir: Path | None = None,
    max_seconds: float | None = None,
) -> Clip:
    """Embed `audio_path` and build per-lane targets from precomputed onsets.

    `max_seconds` caps both the encoded audio (bounds MERT's sequence length
    on long clips) and the kept onsets, so targets line up with the features.
    Dataset-agnostic: callers supply onsets from MIDI (E-GMD) or .txt (STAR).
    """
    feat = embeddings.embed_clip(
        audio_path, encoder, cache_dir=cache_dir, max_seconds=max_seconds,
        cache_dtype=cfg.cache_dtype, high_band=cfg.high_band,
    )
    if max_seconds is not None:
        onsets = {ln: [t for t in ts if t < max_seconds] for ln, ts in onsets.items()}
    targets = build_targets(onsets, feat.shape[0], cfg)
    return Clip(
        features=feat, targets=targets, onsets_by_lane=onsets, audio_path=str(audio_path),
    )


def build_clip(
    audio_path: Path,
    midi_path: Path,
    encoder: embeddings.MertEncoder,
    cfg: Config,
    cache_dir: Path | None = None,
    max_seconds: float | None = None,
) -> Clip:
    """E-GMD: embed `audio_path` + per-lane targets from `midi_path` (MIDI)."""
    return _build_clip(
        audio_path, midi_labels.onsets_from_path(midi_path), encoder, cfg, cache_dir, max_seconds
    )


def _clip_probs(model, clip: Clip) -> np.ndarray:
    """Sigmoid activations (n_lanes, T) for one clip."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    x = torch.as_tensor(clip.features, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad(), runtime.autocast():
        return torch.sigmoid(model(x))[0].float().cpu().numpy()


def evaluate_clip(
    model, clip: Clip, cfg: Config, thresholds: dict[str, float] | None = None
) -> dict[str, float]:
    """Per-lane onset-F1 for one clip (peak-pick the sigmoid, match vs truth).

    `thresholds` overrides the per-lane peak height; defaults to
    `cfg.peak_threshold` for every lane."""
    probs = _clip_probs(model, clip)
    out: dict[str, float] = {}
    for i, lane in enumerate(cfg.lanes):
        thr = thresholds.get(lane, cfg.peak_threshold) if thresholds else cfg.peak_threshold
        est = metrics.pick_onsets_lane(probs[i], cfg.encoder_fps, lane, thr)
        ref = clip.onsets_by_lane.get(lane, [])
        out[lane] = metrics.onset_f1(ref, est, cfg.onset_tolerance_s)["f"]
    return out


def mean_f1(model, clips: Sequence[Clip], cfg: Config) -> float:
    """Macro onset-F1, averaged over clips, counting only lanes that ACTUALLY
    have reference onsets in each clip.

    Empty-reference lanes are skipped (not scored as 0): per-stem clips carry
    onsets for a single instrument's lanes, so averaging in the ~9 empty lanes
    made this metric meaningless (~0.05 even with kick at F1 0.96). Matches how
    `tune_thresholds` and the per-lane report aggregate. Full-mix clips (most
    lanes present) are essentially unchanged."""
    if not clips:
        return 0.0
    per_clip = []
    for clip in clips:
        per_lane = evaluate_clip(model, clip, cfg)
        present = [per_lane[ln] for ln in cfg.lanes if clip.onsets_by_lane.get(ln)]
        if present:
            per_clip.append(sum(present) / len(present))
    return sum(per_clip) / len(per_clip) if per_clip else 0.0


def tune_thresholds(
    model, val_clips: Sequence[Clip], cfg: Config, grid: Sequence[float] | None = None
) -> dict[str, float]:
    """Per-lane peak threshold maximizing mean held-out F1 on `val_clips`.

    Thresholds are hyperparameters tuned on validation (not test). Lanes with
    no val onsets keep `cfg.peak_threshold`. RARE lanes (fewer than
    `cfg.rare_lane_min_onsets` val onsets) only consider thresholds >=
    `cfg.rare_thr_floor`: tuning on a handful of clips once drove ride to 0.10
    and flooded real audio with false positives (see RESULTS.md)."""
    grid = grid or (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    # keep only probs + onsets per clip (not the feature-heavy Clip), so a
    # streaming val set isn't fully resident at once
    probs_onsets = [(_clip_probs(model, c), c.onsets_by_lane) for c in val_clips]
    best: dict[str, float] = {}
    for i, lane in enumerate(cfg.lanes):
        n_ref = sum(len(onsets.get(lane, [])) for _, onsets in probs_onsets)
        lane_grid = (
            grid if n_ref >= cfg.rare_lane_min_onsets
            else tuple(t for t in grid if t >= cfg.rare_thr_floor)
        )
        best_thr, best_f1 = max(cfg.peak_threshold, lane_grid[0] if lane_grid else 0.0), -1.0
        for thr in lane_grid:
            f1s = []
            for probs, onsets in probs_onsets:
                ref = onsets.get(lane, [])
                if not ref:
                    continue
                est = metrics.pick_onsets_lane(probs[i], cfg.encoder_fps, lane, thr)
                f1s.append(metrics.onset_f1(ref, est, cfg.onset_tolerance_s)["f"])
            if f1s and (mean := sum(f1s) / len(f1s)) > best_f1:
                best_f1, best_thr = mean, thr
        best[lane] = best_thr
    return best


def collate_clips(clips: Sequence[Clip]):
    """Pad a list of variable-length clips into batched CPU tensors + a frame
    mask. Used as a DataLoader `collate_fn`, so it builds on CPU (workers can't
    touch CUDA); `train_loop` moves the batch to the device.

    Returns (X (B, T, dim), Y (B, n_lanes, T), Yw (B, n_lanes, T),
    A (B, n_lanes, T), mask (B, T)); padded frames are zero everywhere and 0 in
    mask so the losses ignore them. `Yw` is the sibling-weighting target source
    (`clip.weight_targets`, falling back to `targets`); `A` is the auxiliary
    ring-activity target (zeros when absent). Clips are capped to a uniform
    `max_seconds` upstream, so most are full length and padding waste is small."""
    import torch

    dim = clips[0].features.shape[1]
    n_lanes = clips[0].targets.shape[0]
    lengths = [c.features.shape[0] for c in clips]
    t_max = max(lengths)
    b = len(clips)
    X = torch.zeros(b, t_max, dim, dtype=torch.float32)
    Y = torch.zeros(b, n_lanes, t_max, dtype=torch.float32)
    Yw = torch.zeros(b, n_lanes, t_max, dtype=torch.float32)
    A = torch.zeros(b, n_lanes, t_max, dtype=torch.float32)
    mask = torch.zeros(b, t_max, dtype=torch.float32)
    for i, clip in enumerate(clips):
        t = lengths[i]
        X[i, :t] = torch.as_tensor(clip.features, dtype=torch.float32)
        Y[i, :, :t] = torch.as_tensor(clip.targets, dtype=torch.float32)
        wt = clip.weight_targets if clip.weight_targets is not None else clip.targets
        Yw[i, :, :t] = torch.as_tensor(wt, dtype=torch.float32)
        if clip.activity_targets is not None:
            A[i, :, :t] = torch.as_tensor(clip.activity_targets, dtype=torch.float32)
        mask[i, :t] = 1.0
    return X, Y, Yw, A, mask


def sibling_weight(targets, sib_act, pos_w: float, neg_w: float):
    """Per-frame loss multiplier from sibling activity (lanes.CONFUSABLE).

    `sib_act` (B, n_lanes, T) is each lane's max confusable-sibling target.
    Frames where a sibling fires are the discriminative ones, so their loss is
    scaled up: negatives (this lane silent under sibling noise -> punish bleed
    triggers) toward `neg_w`, positives (true co-occurrence, the harder
    detection) toward `pos_w`. Smooth in both targets; 1 where no sibling is
    active or both weights are 1."""
    return 1.0 + sib_act * ((pos_w - 1.0) * targets + (neg_w - 1.0) * (1.0 - targets))


def masked_bce(logits, targets, mask, pos_weight, frame_weight=None):
    """Per-frame BCE averaged over valid (unpadded) frames and lanes.

    `pos_weight` is (n_lanes, 1), broadcasting over (B, n_lanes, T); `mask` is
    (B, T) and is broadcast across lanes so padded frames contribute nothing.
    `frame_weight` (B, n_lanes, T), e.g. `sibling_weight`, scales per-frame."""
    from torch.nn import functional as F

    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )  # (B, n_lanes, T)
    if frame_weight is not None:
        loss = loss * frame_weight
    m = mask.unsqueeze(1)  # (B, 1, T)
    denom = (m.sum() * logits.shape[1]).clamp_min(1.0)
    return (loss * m).sum() / denom


def masked_focal(logits, targets, mask, alpha: float = 2.0, beta: float = 4.0, frame_weight=None):
    """CenterNet-style penalty-reduced focal loss for soft Gaussian onset
    targets (peak 1.0), averaged over valid frames' positives.

    Positives are the exact peak frames (target == 1); elsewhere the negative
    penalty is reduced by `(1 - target) ** beta` so the Gaussian skirt around a
    peak is barely penalised, and `(.) ** alpha` focuses on hard frames. This
    replaces `pos_weight` reweighting (it targets the rare/hard frames directly)
    so no `pos_weight` is needed. Normalised by the positive count (CenterNet
    convention). `mask` is (B, T), broadcast across lanes; `frame_weight`
    (e.g. `sibling_weight`) scales per-frame."""
    import torch

    p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    pos = (targets >= 1.0).float()
    pos_loss = -((1.0 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1.0 - targets) ** beta) * (p**alpha) * torch.log(1.0 - p) * (1.0 - pos)
    loss = pos_loss + neg_loss
    if frame_weight is not None:
        loss = loss * frame_weight
    m = mask.unsqueeze(1)  # (B, 1, T)
    npos = (pos * m).sum().clamp_min(1.0)
    return (loss * m).sum() / npos


def _cap_onsets(onsets: dict[str, list[float]], max_seconds: float | None) -> dict[str, list[float]]:
    """Drop onsets past `max_seconds` so labels line up with the capped features."""
    if max_seconds is None:
        return onsets
    return {ln: [t for t in ts if t < max_seconds] for ln, ts in onsets.items()}


class CachedClips:
    """Lazy, RAM-bounded clip source backed by the on-disk MERT feature cache.

    Holds only lightweight specs (audio path, onset times, frame count) in RAM;
    each `__getitem__` reads that clip's features from its `.npy` cache file, so
    the full dataset never lives in memory at once (the whole point on a box
    where the feature set is larger than RAM). Indexable + sized, so it drops
    straight into a torch `DataLoader` (workers stream from the SSD in
    parallel). Build via `materialize`, which populates the cache first."""

    def __init__(self, specs: Sequence[tuple], cfg: Config, cache_dir, max_seconds: float | None):
        # specs: (audio_path, onsets_by_lane, weight_onsets_or_None, rings,
        # n_frames, start, length) -- one per WINDOW. onsets are already
        # window-relative ([0, length)). weight_onsets is the FULL-kit onsets
        # for sibling loss weighting (per-stem mode), None elsewhere. (start,
        # length) select the clip window; legacy single-window clips are (0,
        # max_seconds). See _window_specs / materialize.
        self._specs = list(specs)
        self._cfg = cfg
        self._cache_dir = Path(cache_dir)
        self._max_seconds = max_seconds

    def _path(self, audio_path, start: float, length: float | None) -> Path:
        variant = embeddings.feat_variant(self._cfg.high_band)
        key = embeddings.cache_key(
            audio_path, self._cfg.encoder, self._cfg.encoder_layer, length, variant, start
        )
        return self._cache_dir / f"{key}.npy"

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, i: int) -> Clip:
        audio_path, onsets, weight_onsets, rings, _n, start, length = self._specs[i]
        feat = np.load(self._path(audio_path, start, length))
        onsets = _cap_onsets(onsets, length)  # no-op (onsets pre-windowed); defensive
        targets = build_targets(onsets, feat.shape[0], self._cfg)
        wt = None
        if weight_onsets is not None:
            wt = build_targets(_cap_onsets(weight_onsets, length), feat.shape[0], self._cfg)
        act = None
        if rings:
            act = np.zeros_like(targets)
            for li, lane in enumerate(self._cfg.lanes):
                if rings.get(lane):
                    act[li] = spans_to_activity(rings[lane], feat.shape[0], self._cfg.encoder_fps)
        return Clip(
            features=feat, targets=targets, onsets_by_lane=onsets,
            audio_path=str(audio_path), weight_targets=wt, activity_targets=act,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def iter_targets(self):
        """Yield per-window targets WITHOUT loading features (uses the stored
        frame count), so `pos_weights` needn't re-read the whole feature set."""
        for _a, onsets, _w, _r, n_frames, _s, length in self._specs:
            yield build_targets(_cap_onsets(onsets, length), n_frames, self._cfg)


def _rings_for_clip(
    audio_path, onsets: dict[str, list[float]], cfg: Config, cache_dir: Path,
    max_seconds: float | None, start_seconds: float = 0.0, y_full=None,
) -> dict[str, list[tuple[float, float]]]:
    """Ring spans for the sustained lanes (aux activity targets), with a JSON
    side-cache next to the feature cache so re-runs never re-read audio.
    `onsets` are window-relative; (start_seconds, max_seconds) select the audio
    window so the ring envelope matches. Returns {} when no sustained-lane onset.

    `y_full` is the optional WHOLE-clip waveform already loaded at MERT_SR (=
    encoder.sr); when given, the per-window `load_audio` is skipped (batched encoder
    loads each clip once). Same slice -> same spans."""
    import json

    capped = _cap_onsets(onsets, max_seconds)
    if not any(capped.get(ln) for ln in SUSTAINED_LANES):
        return {}
    key = embeddings.cache_key(audio_path, cfg.encoder, cfg.encoder_layer, max_seconds,
                               start=start_seconds)
    rf = Path(cache_dir) / f"{key}.rings.json"
    if rf.exists():
        loaded = json.loads(rf.read_text())
        return {ln: [tuple(s) for s in spans] for ln, spans in loaded.items()}
    y = embeddings.load_audio(audio_path, sr=embeddings.MERT_SR) if y_full is None else y_full
    a = int(start_seconds * embeddings.MERT_SR)
    b = a + int(max_seconds * embeddings.MERT_SR) if max_seconds is not None else None
    y = y[a:b]
    rings = {
        ln: ring_spans(y, embeddings.MERT_SR, capped[ln], cfg.encoder_fps)
        for ln in SUSTAINED_LANES
        if capped.get(ln)
    }
    rf.write_text(json.dumps(rings))
    return rings


# Shortest standalone window we'll emit (seconds). A tail below this is merged into
# the previous window: MERT's conv feature extractor errors on a ~1-3s input, and
# 5s (~375 MERT frames) is a safe floor that still keeps genuine short clips.
MIN_WINDOW = 5.0


def plan_windows(
    audio_path, window: float, search: float, max_windows: int,
) -> list[tuple[float, float]]:
    """Split a clip into ~`window`-second pieces as [(start, length), ...].

    Each interior cut is nudged to the lowest-RMS point within +/- `search`
    seconds of the nominal `k*window` boundary, so a window edge lands in a quiet
    gap rather than bisecting a hit. Clips <= `window` return a single
    (0, window) window with NO audio read (uses sf.info duration). A final window
    shorter than `MIN_WINDOW` is merged into the previous one (MERT's conv feature
    extractor can't encode a ~1-3s sliver -> kernel-size error). `max_windows`
    > 0 caps the count (drops the tail); 0 = cover the whole clip."""
    import soundfile as sf

    dur = float(sf.info(str(audio_path)).duration)
    if dur <= window:
        return [(0.0, window)]
    import librosa

    full_n = int(np.ceil(dur / window))
    y, sr = librosa.load(str(audio_path), sr=embeddings.MERT_SR, mono=True)
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t = librosa.times_like(rms, sr=sr, hop_length=hop)
    cuts = [0.0]
    for k in range(1, full_n):
        b = k * window
        sel = np.where((t >= b - search) & (t <= b + search))[0]
        cuts.append(float(t[sel[np.argmin(rms[sel])]]) if sel.size else b)
    cuts.append(dur)
    wins = [(cuts[i], cuts[i + 1] - cuts[i]) for i in range(full_n)]
    # Full windowing spawns a tail sliver whenever dur isn't a clean multiple of
    # `window` (and the low-energy nudge can shrink it further). MERT's conv stack
    # errors on a ~1-3s input (kernel size > frames), so fold a sub-MIN_WINDOW tail
    # into the previous window (no audio lost; the last window just runs long).
    if len(wins) >= 2 and wins[-1][1] < MIN_WINDOW:
        s0, _ = wins[-2]
        wins[-2] = (s0, dur - s0)
        wins.pop()
    return wins[:max_windows] if max_windows else wins


def _window_specs(specs, window: float, search: float, max_windows: int) -> list[tuple]:
    """Expand (audio, onsets[, weight_onsets]) specs into per-window specs
    (audio, onsets_rel, weight_rel_or_None, start, length): onsets sliced to each
    window and shifted to window-relative time. `max_windows == 1` keeps the
    legacy single window (first `window`s only) with no audio read for planning."""
    out: list[tuple] = []

    def _slice(onsets, start, length):
        return {ln: [t - start for t in ts if start <= t < start + length]
                for ln, ts in onsets.items()}

    for spec in specs:
        audio, onsets = spec[0], spec[1]
        weight = spec[2] if len(spec) > 2 else None
        wins = [(0.0, window)] if max_windows == 1 else plan_windows(audio, window, search, max_windows)
        for start, length in wins:
            w = None if weight is None else _slice(weight, start, length)
            out.append((audio, _slice(onsets, start, length), w, start, length))
    return out


def materialize(
    specs: Sequence[tuple],
    encoder: embeddings.MertEncoder,
    cfg: Config,
    cache_dir,
    max_seconds: float | None,
    tag: str = "clips",
    log: Callable[[str], None] = print,
) -> CachedClips:
    """One-time encode pass: ensure every spec's features are in the `.npy`
    cache (encoded then discarded, never accumulated in RAM) and return a
    `CachedClips` over the specs that succeeded. Replaces an all-in-RAM build,
    so the train set can exceed available memory."""
    cache_dir = Path(cache_dir)
    ok: list[tuple] = []
    for i, spec in enumerate(specs):
        # spec: (audio, onsets_rel, weight_onsets_or_None, start, length) -- one
        # per WINDOW (see _window_specs). onsets already window-relative; weight
        # carries FULL-kit onsets for sibling loss weighting (per-stem mode).
        audio, onsets, weight_onsets, start, length = spec
        try:
            feat = embeddings.embed_clip(
                audio, encoder, cache_dir=cache_dir, max_seconds=length,
                cache_dtype=cfg.cache_dtype, high_band=cfg.high_band,
                start_seconds=start,
            )
            rings = _rings_for_clip(audio, onsets, cfg, cache_dir, length, start)
            ok.append((audio, onsets, weight_onsets, rings, int(feat.shape[0]), start, length))
        except Exception as e:  # noqa: BLE001
            log(f"  skip {Path(audio).name}@{start:.0f}s: {e!r}")
        if (i + 1) % 50 == 0:
            log(f"  {tag}: {i + 1}/{len(specs)} windows cached")
    log(f"{tag}: {len(ok)} windows")
    return CachedClips(ok, cfg, cache_dir, max_seconds)


def _fmt_eta(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _lane_converged(curve, window: int, slope_thr: float, jitter_thr: float) -> bool:
    """Has one lane's per-epoch val-F1 `curve` converged over its last `window`
    epochs? Two conditions, BOTH required:
    - **trend flat**: the least-squares slope is ~0 (|slope| < `slope_thr`, F1 per
      epoch) -- it "stopped increasing".
    - **low jitter**: the residual std *around that trend line* is small (<
      `jitter_thr`) -- it's not still bouncing up and down.
    Measuring jitter as the residual (not raw std) separates the two cleanly: a
    still-climbing lane fails on slope; an early lane oscillating around a flat mean
    fails on jitter (so we keep training until it settles). Returns False until at
    least `window` points exist."""
    if len(curve) < window:
        return False
    y = np.asarray(curve[-window:], dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    jitter = float(np.std(y - (slope * x + intercept)))
    return abs(float(slope)) < slope_thr and jitter < jitter_thr


def train_loop(
    model,
    clips,
    cfg: Config,
    *,
    epochs: int,
    pos_weight: float | np.ndarray = 1.0,
    batch_size: int = 8,
    num_workers: int = 0,
    val_clips=None,
    out_dir: str | None = None,
    checkpoint_every: int = 0,
    lr_schedule: str = "cosine",
    warmup_steps: int = 0,
    loss_fn: str = "bce",
    keep_best: bool = False,
    early_stop: bool = False,
    es_window: int = 8,
    es_slope: float = 0.002,
    es_jitter: float = 0.015,
    es_min_epochs: int = 20,
    log: Callable[[str], None] = print,
) -> dict:
    """Train `model` on `clips` in padded mini-batches of `batch_size` (clips
    are variable length; padded frames are masked out of the loss). Returns a
    history dict with per-epoch train loss and (if `val_clips`) val F1.

    `clips` is any indexable+sized clip source (an in-RAM `list[Clip]` or a
    streaming `CachedClips`); `num_workers` > 0 lets the DataLoader prefetch
    batches from the SSD in parallel with GPU compute. `pos_weight` may be a
    scalar or a per-lane array (length n_lanes).

    `early_stop` ends training once EVERY lane (with val onsets) has converged --
    `_lane_converged` over the last `es_window` epochs (|slope| < `es_slope` AND
    residual jitter < `es_jitter`) -- so a single still-climbing lane (e.g.
    open-hat) keeps the whole run going. `epochs` is the absolute cap, and nothing
    stops before `es_min_epochs`. Records `history["stopped_epoch"]`. Uses the
    per-lane val curves, which it computes when `early_stop` even without
    `keep_best`."""
    import math

    import torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pw = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    if pw.ndim == 1:
        pw = pw.view(-1, 1)  # (n_lanes, 1) broadcasts over (B, n_lanes, T)
    # sibling-aware weighting (lanes.CONFUSABLE): S (n_lanes, n_lanes) marks each
    # lane's confusable siblings; per batch, sib_act = max sibling target.
    sib_on = cfg.sib_neg_weight != 1.0 or cfg.sib_pos_weight != 1.0
    S = torch.as_tensor(sibling_matrix(cfg.lanes), dtype=torch.bool, device=device)
    # aux ring-activity rows + a unit pos_weight for the activity BCE
    sus_idx = [i for i, ln in enumerate(cfg.lanes) if ln in SUSTAINED_LANES]
    one_pw = torch.ones(len(sus_idx), 1, device=device)
    history: dict[str, list[float]] = {"train_loss": []}
    # keep_best: PER-LANE early stopping. Each lane's head is an independent
    # OnsetHead, so we snapshot each lane's head params at the epoch where THAT
    # lane's val F1 peaks and restore every head from its own best epoch. Lanes
    # overfit at different times (cymbals early, hats late), so a single global
    # best-epoch underserves some lanes. CPU clones avoid a 2nd GPU copy.
    best_lane_f1 = {ln: -1.0 for ln in cfg.lanes}
    best_lane_epoch = {ln: -1 for ln in cfg.lanes}
    best_lane_state: dict = {}

    gen = torch.Generator().manual_seed(0)  # reproducible per-epoch shuffle
    loader = DataLoader(
        clips,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_clips,
        persistent_workers=bool(num_workers),
        generator=gen,
    )
    t0 = time.perf_counter()
    expected_batches = (len(clips) + batch_size - 1) // batch_size
    # warmup -> cosine LR schedule (stepped per optimizer step). Cosine decays to
    # ~0 over the run so the final (saved + threshold-tuned) epoch settles at the
    # LR minimum; warmup_steps>0 ramps in, which matters for large-batch/high-LR
    # runs. lr_schedule="none" keeps a constant LR (legacy behaviour).
    sched = None
    if lr_schedule == "cosine":
        total_steps = max(1, epochs * expected_batches)

        def _lr_mult(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, prog))))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_mult)
    for epoch in range(epochs):
        ep_start = time.perf_counter()
        model.train()
        total, n_batches = 0.0, 0
        for bi, (X, Y, Yw, A, mask) in enumerate(loader):
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            fw = None
            if sib_on:
                Yw = Yw.to(device, non_blocking=True)
                # per lane l: max activity over its confusable OUTPUT siblings
                # (Yw[S]) at each frame -- a hard negative where l is silent there.
                parts = []
                for li in range(len(cfg.lanes)):
                    parts.append(
                        Yw[:, S[li]].amax(dim=1) if bool(S[li].any()) else torch.zeros_like(Yw[:, 0])
                    )
                sib_act = torch.stack(parts, dim=1)  # (B, n_lanes, T)
                fw = sibling_weight(Y, sib_act, cfg.sib_pos_weight, cfg.sib_neg_weight)
            opt.zero_grad()
            with runtime.autocast():  # bf16 fwd on Ampere+; FP32 no-op elsewhere
                logits, act_logits = model.forward_all(X)
                loss = (
                    masked_focal(logits, Y, mask, frame_weight=fw)
                    if loss_fn == "focal"
                    else masked_bce(logits, Y, mask, pw, frame_weight=fw)
                )
                if cfg.aux_act_weight > 0.0:
                    # auxiliary ring-activity BCE, sustained lanes only (the
                    # open-hat/cymbal tail that defines those classes)
                    A = A.to(device, non_blocking=True)
                    loss = loss + cfg.aux_act_weight * masked_bce(
                        act_logits[:, sus_idx], A[:, sus_idx], mask, one_pw
                    )
            loss.backward()  # bf16 keeps FP32 range, so no GradScaler needed
            opt.step()
            if sched is not None:
                sched.step()
            total += float(loss.detach())
            n_batches += 1
            # show movement within epoch 0 (before any epoch line prints), so a
            # fresh run on new hardware confirms throughput immediately
            if epoch == 0 and expected_batches >= 4 and (bi + 1) % (expected_batches // 4) == 0:
                log(f"  epoch 0  batch {bi + 1}/{expected_batches}  ({time.perf_counter() - ep_start:.0f}s)")
        avg = total / max(1, n_batches)
        history["train_loss"].append(avg)
        if val_clips:
            if keep_best or early_stop:
                # one eval pass -> per-clip per-lane F1; derive the clip-macro (for
                # the log, matching mean_f1), the per-lane curves (early-stop +
                # keep_best read them), AND track each lane's own best epoch.
                per = [evaluate_clip(model, c, cfg) for c in val_clips]
                clip_macros = []
                for c, pl in zip(val_clips, per, strict=True):
                    present = [pl[ln] for ln in cfg.lanes if c.onsets_by_lane.get(ln)]
                    if present:
                        clip_macros.append(sum(present) / len(present))
                vf1 = sum(clip_macros) / len(clip_macros) if clip_macros else 0.0
                sd = model.state_dict() if keep_best else None
                for lane in cfg.lanes:
                    vals = [per[i][lane] for i, c in enumerate(val_clips) if c.onsets_by_lane.get(lane)]
                    if not vals:
                        continue
                    lf = sum(vals) / len(vals)
                    history.setdefault(f"vf1_{lane}", []).append(lf)  # per-epoch per-lane curve
                    if keep_best and lf > best_lane_f1[lane]:
                        best_lane_f1[lane], best_lane_epoch[lane] = lf, epoch
                        pref = f"heads.{lane}."
                        for k, v in sd.items():
                            if k.startswith(pref):
                                best_lane_state[k] = v.detach().cpu().clone()
            else:
                vf1 = mean_f1(model, val_clips, cfg)
            history.setdefault("val_f1", []).append(vf1)
        dt = time.perf_counter() - ep_start
        # print the first 3 epochs (immediate speed read), then every 10th + last
        if epoch < 3 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            eta = (time.perf_counter() - t0) / (epoch + 1) * (epochs - epoch - 1)
            msg = f"epoch {epoch:3d}  train_loss {avg:.4f}"
            if val_clips:
                msg += f"  val_macro_f1 {history['val_f1'][-1]:.3f}"
            msg += f"  {dt:.1f}s/ep  eta {_fmt_eta(eta)}"
            log(msg)
        # periodic safety checkpoint (untuned thresholds) for long unattended
        # runs; the final main() save overwrites this with tuned thresholds
        if out_dir and checkpoint_every and (epoch + 1) % checkpoint_every == 0 and epoch != epochs - 1:
            checkpoint.save(out_dir, model, cfg, {ln: cfg.peak_threshold for ln in cfg.lanes},
                            in_dim=embeddings.feat_dim(cfg.high_band))
            log(f"  checkpoint saved @ epoch {epoch} (untuned) -> {out_dir}")
        # convergence early-stop: stop once EVERY lane with val onsets is converged
        # (so a still-climbing lane blocks it); `epochs` is the absolute cap and we
        # never stop before `es_min_epochs`.
        if early_stop and val_clips and (epoch + 1) >= es_min_epochs:
            tracked = [ln for ln in cfg.lanes if history.get(f"vf1_{ln}")]
            if tracked and all(
                _lane_converged(history[f"vf1_{ln}"], es_window, es_slope, es_jitter)
                for ln in tracked
            ):
                history["stopped_epoch"] = float(epoch)
                log(f"  early stop @ epoch {epoch}: all {len(tracked)} lanes converged "
                    f"(|slope|<{es_slope}, jitter<{es_jitter}, last {es_window} ep)")
                break
    if keep_best and best_lane_state:
        # load each lane's head from its own best epoch (strict=False: lanes with no
        # val onsets aren't in best_lane_state and keep their final-epoch weights).
        model.load_state_dict(best_lane_state, strict=False)
        history["best_epoch_by_lane"] = [float(best_lane_epoch[ln]) for ln in cfg.lanes]
        restored = ", ".join(
            f"{ln}@{best_lane_epoch[ln]}({best_lane_f1[ln]:.2f})"
            for ln in cfg.lanes if best_lane_epoch[ln] >= 0
        )
        log(f"  keep_best: restored per-lane best epochs -> {restored}")
    return history


# --- synthetic self-test (no dataset required) ---------------------------


def synthetic_clip(n_frames: int = 300, dim: int = 32, fps: float = 100.0, seed: int = 0) -> Clip:
    """Random features with a few planted onsets per lane, for a dataset-free
    wiring/overfit check. Random features carry no real onset signal, so this
    proves the head can *fit* (memorize) a fixed clip, exactly the spec's
    'overfit one clip' wiring milestone."""
    rng = np.random.default_rng(seed)
    feat = rng.standard_normal((n_frames, dim)).astype(np.float32)
    onsets = {
        "k": [0.5, 1.0, 1.5, 2.0],
        "s": [1.0, 2.0],
        "hc": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        "rd": [0.0, 0.5, 1.0],
        "ss": [1.5],
    }
    cfg = Config(encoder_fps=fps)
    targets = build_targets(onsets, n_frames, cfg)
    return Clip(features=feat, targets=targets, onsets_by_lane=onsets)


def synthetic_smoke(epochs: int = 80, loss_fn: str = "bce", log: Callable[[str], None] = print) -> dict:
    """Overfit one synthetic clip; train loss should fall sharply."""
    clip = synthetic_clip()
    cfg = Config(encoder_fps=100.0)
    model = MultiLaneHeads(in_dim=clip.features.shape[1], hidden=64, num_layers=1)
    return train_loop(model, [clip], cfg, epochs=epochs, val_clips=[clip], loss_fn=loss_fn, log=log)


# --- real-data entry point ----------------------------------------------


def _report(model, val_clips: Sequence[Clip], cfg: Config, thresholds: dict[str, float]) -> None:
    """Print held-out per-lane F1 (with tuned thresholds + onset counts)."""
    from collections import defaultdict

    lane_f1: dict[str, list[float]] = defaultdict(list)
    lane_n: dict[str, int] = defaultdict(int)
    for clip in val_clips:
        f1 = evaluate_clip(model, clip, cfg, thresholds)
        for lane in cfg.lanes:  # output lanes only; onsets_by_lane may carry the `x` ghost lane
            ts = clip.onsets_by_lane.get(lane)
            if ts:
                lane_f1[lane].append(f1[lane])
                lane_n[lane] += len(ts)
    def _mean_f1(lane: str) -> float:  # no-onset lanes sort to the bottom
        return sum(lane_f1[lane]) / len(lane_f1[lane]) if lane_f1[lane] else -1.0

    print("\nheld-out per-lane F1 (tuned thresholds, sorted by F1):", flush=True)
    for lane in sorted(cfg.lanes, key=_mean_f1, reverse=True):
        if lane_f1[lane]:
            print(
                f"  {lane:3s} onsets={lane_n[lane]:6d} clips={len(lane_f1[lane]):3d} "
                f"thr={thresholds.get(lane, cfg.peak_threshold):.2f} F1={_mean_f1(lane):.3f}",
                flush=True,
            )
        else:
            print(f"  {lane:3s} (no onsets in val subset)", flush=True)


def _report_compare(model, val_clips, cfg: Config, thresholds: dict[str, float]) -> None:
    """Per-lane bare peak-pick vs the shared per-lane deterministic picker.

    Shows whether the shared picker (`drumjot_dsp.peakpick` via
    `metrics.pick_onsets_lane`: per-lane min-distance + prominence + decay-reset)
    earns its place over a bare height+min-distance pick. Replaces the old
    onset-envelope support gate, which was a measured no-op (dF ~ 0). Needs only
    cached features, so it runs on every val clip."""
    from collections import defaultdict

    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    n = 0
    for clip in val_clips:
        probs = _clip_probs(model, clip)
        n += 1
        for i, lane in enumerate(cfg.lanes):
            ref = clip.onsets_by_lane.get(lane, [])
            if not ref:
                continue
            thr = thresholds.get(lane, cfg.peak_threshold)
            bare = metrics.onset_f1(
                ref, metrics.pick_onsets(probs[i], cfg.encoder_fps, thr, cfg.peak_min_distance_s),
                cfg.onset_tolerance_s,
            )
            pick = metrics.onset_f1(
                ref, metrics.pick_onsets_lane(probs[i], cfg.encoder_fps, lane, thr),
                cfg.onset_tolerance_s,
            )
            for k, v in (("f_bare", bare["f"]), ("f_pick", pick["f"]),
                         ("p_bare", bare["p"]), ("p_pick", pick["p"]),
                         ("r_bare", bare["r"]), ("r_pick", pick["r"])):
                agg[lane][k].append(v)

    if not n:
        print("\n(no val clips; skipping picker comparison)", flush=True)
        return
    print(f"\nbare peak-pick vs +shared deterministic picker  ({n} val clips):", flush=True)
    print("  lane  F_bare F_pick    dF   P_bare>P_pick  R_bare>R_pick", flush=True)

    def _m(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    for lane in sorted(cfg.lanes, key=lambda ln: _m(agg[ln]["f_pick"]), reverse=True):
        a = agg[lane]
        if not a["f_bare"]:
            continue
        fb, fp = _m(a["f_bare"]), _m(a["f_pick"])
        print(
            f"  {lane:4s} {fb:6.3f} {fp:6.3f} {fp - fb:+6.3f}   "
            f"{_m(a['p_bare']):.3f}>{_m(a['p_pick']):.3f}   "
            f"{_m(a['r_bare']):.3f}>{_m(a['r_pick']):.3f}",
            flush=True,
        )


def _egmd_specs(args) -> tuple[list, list, Path]:
    """(train_specs, val_specs, cache_dir) for E-GMD; specs = (audio, onsets)."""
    root = paths.dataset_path("egmd")
    meta = egmd.read_index(root / "e-gmd-v1.0.0.csv", root)
    tr = egmd.take_duration(egmd.for_split(meta, "train"), args.train_min * 60)
    va = egmd.take_duration(egmd.for_split(meta, "validation"), args.val_min * 60)
    spec = lambda m: (m.audio_path, midi_labels.onsets_from_path(m.midi_path))  # noqa: E731
    return [spec(m) for m in tr], [spec(m) for m in va], root / "_cache_mert"


def _egmd_perstem_specs(args) -> tuple[list, list, Path]:
    """(train_specs, val_specs, cache_dir) for E-GMD per-instrument stems.

    One example per (clip, drum-piece stem), labelled with ONLY that stem's lanes
    (matches the per-instrument eval/inference). Point `DRUMJOT_EGMD` at a
    separation-aware tree built by scripts/separate_egmd_dataset.py -- that tree
    is already a balanced, duration-capped subset, so every stem in it is used
    (split by the tree's own train/validation labels)."""
    root = paths.dataset_path("egmd")
    per = egmd.perstem_index(root)
    tr = [c for c in per if c.split == "train"]
    va = [c for c in per if c.split == "validation"]
    # targets = the stem's own lanes only; FULL onsets ride along as the
    # sibling-weighting source (bleed from other instruments = a hard negative).
    spec = lambda c: (  # noqa: E731
        c.audio_path,
        egmd.restricted_onsets(c.midi_path, c.pitch),
        midi_labels.onsets_from_path(c.midi_path),
    )
    return [spec(c) for c in tr], [spec(c) for c in va], root / "_cache_mert"


def _star_specs(args) -> tuple[list, list, Path]:
    """(train_specs, val_specs, cache_dir) for STAR; specs = (audio, onsets)."""
    root = paths.dataset_path("star")
    clips = star.index(root)
    tr = star.for_split(clips, "training")[: args.train_clips]
    # eval on validation + test (both held out, song-disjoint from training):
    # STAR's validation mix audio is sparse, so test supplements it.
    held_out = star.for_split(clips, "validation") + star.for_split(clips, "test")
    va = held_out[: args.val_clips]
    spec = lambda c: (c.audio_path, star.onsets_by_lane(c.annotation_path))  # noqa: E731
    return [spec(c) for c in tr], [spec(c) for c in va], root / "_cache_mert"


def _star_perstem_specs(args) -> tuple[list, list, Path]:
    """(train_specs, val_specs, cache_dir) for STAR per-instrument stems.

    One example per (song, drum-piece stem), labelled with ONLY that stem's lanes
    so the model learns to ignore cross-instrument bleed (matches the
    per-instrument eval/inference). `--train-clips`/`--val-clips` cap the number
    of SONGS (each expands to up to 5 stem examples). Point `DRUMJOT_STAR` at a
    separation-aware dataset built by scripts/separate_star_dataset.py."""
    root = paths.dataset_path("star")
    songs = star.index(root)
    tr_songs = {c.annotation_path for c in star.for_split(songs, "training")[: args.train_clips]}
    held = star.for_split(songs, "validation") + star.for_split(songs, "test")
    va_songs = {c.annotation_path for c in held[: args.val_clips]}
    per = star.perstem_index(root)
    tr = [c for c in per if c.annotation_path in tr_songs]
    va = [c for c in per if c.annotation_path in va_songs]
    # targets = the stem's own lanes only; the FULL onsets ride along as the
    # sibling-weighting source, so bleed from other instruments on this stem
    # (e.g. hats on the cymbals stem) counts as a hard negative.
    spec = lambda c: (  # noqa: E731
        c.audio_path,
        star.restricted_onsets(c.annotation_path, c.pitch),
        star.onsets_by_lane(c.annotation_path),
    )
    return [spec(c) for c in tr], [spec(c) for c in va], root / "_cache_mert"


def _enst_specs(args) -> tuple[list, list, Path]:
    """(train_specs, val_specs, cache_dir) for ENST-Drums; specs = (audio, onsets).

    Real acoustic-drum recordings; split holds out a whole drummer/kit (so eval
    is on an unseen player+kit). `--enst-mix` picks the audio variant."""
    root = paths.dataset_path("enst")
    clips = enst.index(root, mix=args.enst_mix)
    tr = enst.for_split(clips, "train")[: args.train_clips]
    va = enst.for_split(clips, "validation")[: args.val_clips]
    spec = lambda c: (c.audio_path, enst.onsets_by_lane(c.annotation_path))  # noqa: E731
    return [spec(c) for c in tr], [spec(c) for c in va], root / "_cache_mert"


def _enst_perstem_specs(args) -> tuple[list, list, Path]:
    """(train_specs, val_specs, cache_dir) for ENST per-instrument stems.

    One example per (take, drum-piece stem), labelled with ONLY that stem's lanes
    so the model learns to ignore cross-instrument bleed (matches the
    per-instrument eval/inference). Split holds out a whole drummer/kit;
    `--train-clips`/`--val-clips` cap the number of TAKES (each expands to up to 5
    stem examples). Point `DRUMJOT_ENST` at a separation-aware tree built by
    scripts/separate_enst_dataset.py."""
    root = paths.dataset_path("enst")
    takes = enst.index(root, mix="sep_drum")  # one handle per separated take
    tr_takes = {c.annotation_path for c in enst.for_split(takes, "train")[: args.train_clips]}
    va_takes = {c.annotation_path for c in enst.for_split(takes, "validation")[: args.val_clips]}
    per = enst.perstem_index(root)
    tr = [c for c in per if c.annotation_path in tr_takes]
    va = [c for c in per if c.annotation_path in va_takes]
    # targets = the stem's own lanes only; the FULL onsets ride along as the
    # sibling-weighting source, so bleed from other instruments on this stem
    # (e.g. hats on the cymbals stem) counts as a hard negative.
    spec = lambda c: (  # noqa: E731
        c.audio_path,
        enst.restricted_onsets(c.annotation_path, c.pitch),
        enst.onsets_by_lane(c.annotation_path),
    )
    return [spec(c) for c in tr], [spec(c) for c in va], root / "_cache_mert"


def _cap_by_clip(perstem_clips, keyfn, cap: int):
    """Keep all per-stem examples for the first `cap` distinct source clips
    (songs/takes), in order. `cap<=0` keeps everything."""
    if cap <= 0:
        return list(perstem_clips)
    keys: set = set()
    out = []
    for c in perstem_clips:
        k = keyfn(c)
        if k in keys:
            out.append(c)
        elif len(keys) < cap:
            keys.add(k)
            out.append(c)
    return out


def _pooled_specs(args) -> tuple[list, list, Path]:
    """(train_specs, val_specs, cache_dir) pooling several SEPARATION-AWARE
    per-stem trees into one training set.

    All sources fold to the same 10-lane vocab and the same 5-stem (k/s/h/c/t)
    routing, so each contributes `(audio, restricted_onsets, full_onsets)` tuples
    exactly as its own `*_perstem` mode -- they pool directly.

    Class coverage: absent lanes are treated as valid negatives. That is exact
    for STAR (ADT) and E-GMD (GM MIDI), whose audio is derived FROM the labels so
    nothing is unlabelled; ENST is hand-labelled real audio covering every lane
    except pedal-hat (`hp`, 0 onsets) and misc-cymbal (`mc`, ~0), both supplied
    richly by STAR+E-GMD, so ENST's negatives there are a minor, mostly-correct
    signal. (If `hp`/`mc` regress, add per-source lane masking for ENST.)

    `DRUMJOT_STAR/ENST/EGMD` must point at the sep trees (star_balanced_sep,
    enst-sep, egmd-sep). `--pool-sources` selects which; `--pool-cap` caps clips
    per source; `--pool-balance` oversamples smaller sources so none dominates.
    """
    import json
    import os

    from drumjot_training.lanes import LANES

    sources = [s.strip() for s in args.pool_sources.split(",") if s.strip()]

    # Index each source's per-stem clips (file pairing only -- NO onset parsing
    # here) + how to read its labels: ann_of -> the clip's label file, reader ->
    # the parser. --pool-cap dedups by label file (one song per file).
    info: dict[str, tuple] = {}
    roots: list[str] = []
    for name in sources:
        root = paths.dataset_path(name)
        roots.append(str(root))
        if name == "star":
            allper = star.perstem_index(root)
            tr = [c for c in allper if c.split == "training"]
            va = [c for c in allper if c.split in ("validation", "test")]
            ann_of, reader, p2l = (lambda c: c.annotation_path), star.onsets_by_lane, star.PERSTEM_TO_LANES
        elif name == "enst":
            allper = enst.perstem_index(root)
            tr = enst.perstem_for_split(allper, "train")
            va = enst.perstem_for_split(allper, "validation")
            ann_of, reader, p2l = (lambda c: c.annotation_path), enst.onsets_by_lane, enst.PERSTEM_TO_LANES
        elif name == "egmd":
            allper = egmd.perstem_index(root)
            tr = [c for c in allper if c.split == "train"]
            va = [c for c in allper if c.split == "validation"]
            ann_of, reader, p2l = (lambda c: c.midi_path), midi_labels.onsets_from_path, egmd.PERSTEM_TO_LANES
        else:
            raise SystemExit(f"--pool-sources: unknown source {name!r} (use star/enst/egmd)")
        # use each SOURCE's own pitch->lanes map (not STAR's) so a source whose
        # stem-pitch vocab ever diverges can't silently yield all-empty restricted
        # onsets. (They're identical today; this keeps it correct if one changes.)
        info[name] = (tr, va, ann_of, reader, p2l)

    # Feature cache: default beside the sep trees (NFS), but --pool-cache should
    # point it at LOCAL NVMe -- the .npy features are re-encodable scratch (~50 GB
    # for --pool-cap 1000) and local reads keep the GPU compute-bound instead of
    # NFS-throttled, with no large-RAM/page-cache requirement.
    cache = (
        Path(args.pool_cache) if getattr(args, "pool_cache", None)
        else Path(os.path.commonpath(roots)) / "_cache_mert_pooled"
    )
    cache.mkdir(parents=True, exist_ok=True)

    # Parsed-onset cache: each label file is parsed ONCE -- a 5-stem song was
    # otherwise parsed 10x (5 pitches x {restricted, full}) -- memoized in-run and
    # persisted so reruns/sweeps skip parsing entirely (the dominant spec-build
    # cost). restricted onsets are DERIVED by filtering full to the stem's lanes
    # (matches *.restricted_onsets). Keyed by absolute label path; delete
    # `_onsets.json` if the labels are regenerated.
    ocp = cache / "_onsets.json"
    try:
        onsets_cache = json.loads(ocp.read_text()) if ocp.exists() else {}
    except Exception:  # noqa: BLE001  corrupt cache -> rebuild
        onsets_cache = {}
    dirty = False

    def _full(path, reader):
        nonlocal dirty
        v = onsets_cache.get(str(path))
        if v is None or any(ln not in v for ln in LANES):
            r = reader(path)
            v = {ln: list(r.get(ln, [])) for ln in LANES}
            onsets_cache[str(path)] = v
            dirty = True
        return v

    def _spec(c, ann_of, reader, p2l):
        full = _full(ann_of(c), reader)  # all output lanes + the `x` negative lane
        keep = set(p2l.get(c.pitch, ()))
        restricted = {ln: (full[ln] if ln in keep else []) for ln in LANES}
        return (c.audio_path, restricted, full)

    per_train: dict[str, list] = {}
    per_val: dict[str, list] = {}
    for name in sources:
        tr, va, ann_of, reader, p2l = info[name]
        per_train[name] = [_spec(c, ann_of, reader, p2l) for c in _cap_by_clip(tr, ann_of, args.pool_cap)]
        per_val[name] = [_spec(c, ann_of, reader, p2l) for c in va]
    if dirty:  # atomic so a ctrl-C can't leave a half-written cache
        tmp = ocp.with_name(ocp.name + ".tmp")
        tmp.write_text(json.dumps(onsets_cache))
        os.replace(tmp, ocp)

    # Oversample (repeat) smaller sources up to the largest so a big synthetic
    # source can't drown the small real-acoustic one (ENST). Capped at 5x.
    if args.pool_balance and per_train:
        target = max((len(v) for v in per_train.values()), default=0)
        for name, specs in per_train.items():
            if specs and len(specs) < target:
                per_train[name] = specs * min(5, max(1, round(target / len(specs))))

    print("pooled (per-stem examples, absent lanes = negatives):", flush=True)
    for name in sources:
        print(f"  {name:5} train={len(per_train[name]):6d}  val={len(per_val[name]):5d}", flush=True)

    train_specs = [s for name in sources for s in per_train[name]]
    val_specs = [s for name in sources for s in per_val[name]]
    return train_specs, val_specs, cache


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Drum-onset training (frozen MERT + per-lane heads)")
    ap.add_argument(
        "--dataset",
        choices=("egmd", "egmd_perstem", "star", "star_perstem", "enst", "enst_perstem", "pooled"),
        default="egmd",
    )
    ap.add_argument("--pool-sources", default="star,enst,egmd",
                    help="pooled mode: comma-list of sep-tree sources to combine (DRUMJOT_<SRC> "
                    "must point at each sep tree, e.g. star_balanced_sep / enst-sep / egmd-sep)")
    ap.add_argument("--pool-cap", type=int, default=0,
                    help="pooled mode: max source-clips per dataset (0 = all); each expands to ~5 stems")
    ap.add_argument("--pool-balance", action="store_true",
                    help="pooled mode: oversample smaller sources up to the largest (<=5x) so the big "
                    "synthetic sets don't drown the small real-acoustic one (ENST)")
    ap.add_argument("--pool-cache", default=None,
                    help="pooled mode: feature-cache dir; point at LOCAL NVMe (not the NFS sep trees) "
                    "to keep training compute-bound. Default: <common parent of sources>/_cache_mert_pooled")
    ap.add_argument("--enst-mix", choices=("wet_mix", "dry_mix", "accompaniment", "sep_drum"),
                    default="wet_mix",
                    help="ENST-Drums audio variant (wet_mix = realistic isolated kit; default; "
                    "sep_drum = separated drum stem from separate_enst_dataset.py)")
    ap.add_argument("--synthetic", action="store_true", help="dataset-free self-test")
    ap.add_argument("--overfit-one", action="store_true", help="train on a single clip")
    ap.add_argument("--train-min", type=float, default=240.0, help="E-GMD train minutes")
    ap.add_argument("--val-min", type=float, default=30.0, help="E-GMD val minutes")
    ap.add_argument("--train-clips", type=int, default=400, help="STAR train clip count")
    ap.add_argument("--val-clips", type=int, default=80, help="STAR val clip count")
    ap.add_argument("--max-seconds", type=float, default=30.0, help="per-clip encode cap")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=Config.batch_size, help="clips per step")
    ap.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader prefetch workers; >0 needs docker --shm-size=2g (else it hangs). "
        "0 still streams from the SSD cache (RAM stays bounded), just no prefetch overlap.",
    )
    ap.add_argument("--layer", type=int, default=10, help="MERT hidden layer (0-24)")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch init seed for reproducible head weights (multi-seed ablations)")
    ap.add_argument(
        "--high-band", default=True, action=argparse.BooleanOptionalAction,
        help="append the 6-20 kHz high-band block to MERT features (default on); "
        "--no-high-band trains on raw MERT only (high-band ablation). Separate cache key.",
    )
    # Windowing is unconditional: every clip (train AND val) is sliced into as many
    # ~max-seconds windows as fit, recovering ALL the separated audio instead of just
    # the first window. (Was a `--max-windows` flag defaulting to first-window-only;
    # removed -- using the whole clip is always right, and the flag was only kept for
    # legacy single-window reproducibility.) Cuts land in low-energy gaps via
    # --window-search. NB: the feature cache is keyed per (start, length), so the
    # first run after this re-encodes every clip's later windows (one-time cost,
    # proportional to total audio duration).
    ap.add_argument(
        "--window-search", type=float, default=3.0,
        help="nudge each window cut to the lowest-RMS point within +/- this many seconds "
        "of the nominal boundary, to avoid bisecting a hit.",
    )
    ap.add_argument(
        "--cache-dtype", choices=("float16", "float32"), default=Config.cache_dtype,
        help="on-disk feature cache precision (float16 halves size + I/O)",
    )
    ap.add_argument("--pos-weight-cap", type=float, default=50.0)
    ap.add_argument("--lr", type=float, default=Config.lr, help="learning rate (lower for warm-start fine-tune)")
    ap.add_argument("--weight-decay", type=float, default=Config.weight_decay,
                    help="AdamW decoupled weight decay")
    ap.add_argument("--lr-schedule", choices=("cosine", "none"), default="cosine",
                    help="warmup->cosine LR decay (default) or constant LR")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="linear LR warmup steps before cosine (helps large-batch / high-LR runs)")
    ap.add_argument("--loss", choices=("bce", "focal"), default="bce",
                    help="pos-weighted BCE (default) or CenterNet penalty-reduced focal "
                    "(focal ignores pos_weight; A/B it before committing)")
    ap.add_argument("--keep-best", default=True, action=argparse.BooleanOptionalAction,
                    help="restore each lane's head from the epoch where THAT lane's val F1 "
                    "peaked (per-lane early stopping; lanes overfit at different times -- "
                    "cymbals early, hats late); --no-keep-best keeps the final epoch (legacy)")
    ap.add_argument("--early-stop", default=True, action=argparse.BooleanOptionalAction,
                    help="end training once EVERY lane's val F1 has converged (flat trend + low "
                    "jitter over --es-window epochs); --epochs is the absolute cap. "
                    "--no-early-stop trains the full --epochs (reproduces fixed-length runs)")
    ap.add_argument("--es-window", type=int, default=8, help="epochs in the convergence window")
    ap.add_argument("--es-slope", type=float, default=0.002, help="max |val-F1 slope|/epoch to be 'flat'")
    ap.add_argument("--es-jitter", type=float, default=0.015, help="max residual std around the trend")
    ap.add_argument("--es-min-epochs", type=int, default=20, help="never stop before this many epochs")
    ap.add_argument("--sib-neg-weight", type=float, default=Config.sib_neg_weight,
                    help="loss multiplier on hard negatives (confusable sibling active, "
                    "this lane silent); 1 disables")
    ap.add_argument("--sib-pos-weight", type=float, default=Config.sib_pos_weight,
                    help="loss multiplier on co-occurring positives (hit under sibling "
                    "noise, the harder detection); 1 disables")
    ap.add_argument("--no-filter-report", action="store_true", help="skip the deterministic-filter F1 comparison")
    ap.add_argument("--out", type=str, default=None, help="save model.pt + meta.json here")
    ap.add_argument("--resume", type=str, default=None, help="checkpoint dir to warm-start weights from")
    args = ap.parse_args(argv)

    if args.synthetic:
        synthetic_smoke(epochs=args.epochs)
        return

    import torch

    runtime.configure_backends()  # TF32 + sets up the bf16 autocast path
    cfg = Config(
        encoder=embeddings.MERT_NAME, encoder_fps=embeddings.MERT_FPS,
        encoder_layer=args.layer, cache_dtype=args.cache_dtype,
        high_band=args.high_band,
        lr=args.lr, weight_decay=args.weight_decay,
        sib_neg_weight=args.sib_neg_weight, sib_pos_weight=args.sib_pos_weight,
    )
    train_specs, val_specs, cache = (
        _star_specs(args) if args.dataset == "star"
        else _star_perstem_specs(args) if args.dataset == "star_perstem"
        else _enst_perstem_specs(args) if args.dataset == "enst_perstem"
        else _enst_specs(args) if args.dataset == "enst"
        else _egmd_perstem_specs(args) if args.dataset == "egmd_perstem"
        else _pooled_specs(args) if args.dataset == "pooled"
        else _egmd_specs(args)
    )
    if args.overfit_one:
        train_specs = train_specs[:1]

    encoder = embeddings.make_encoder(cfg.encoder, cfg.encoder_layer)
    # Fail fast on an out-of-range layer (MERT exposes 25 hidden states) instead of
    # an opaque IndexError deep into the encode pass.
    nhs = encoder.n_hidden_states()
    if not 0 <= cfg.encoder_layer < nhs:
        raise SystemExit(f"--layer {cfg.encoder_layer} out of range: valid 0..{nhs - 1}")

    # Expand into per-window specs. Both train and val are sliced into as many
    # ~max-seconds windows as fit the whole clip (max_windows=0 = unlimited),
    # recovering ALL the separated audio rather than just the first window. The
    # split is deterministic (low-energy cuts via --window-search), so runs stay
    # comparable to each other, and it multiplies val onsets, tightening the noisy
    # rare-lane F1.
    win = args.max_seconds or 30.0
    train_wspecs = _window_specs(train_specs, win, args.window_search, 0)
    val_wspecs = _window_specs(val_specs, win, args.window_search, 0)
    extra = len(train_wspecs) - len(train_specs)
    print(
        f"dataset={args.dataset}  {len(train_specs)} train clips -> {len(train_wspecs)} windows "
        f"(+{extra} from segmenting, full windowing) / "
        f"{len(val_specs)} val -> {len(val_wspecs)} windows  (cache {cache}) ...",
        flush=True,
    )
    # Encode-once into the cache, then stream from disk: features are never all
    # held in RAM, so the train set can exceed available memory.
    log_p = lambda s: print(s, flush=True)  # noqa: E731
    train_clips = materialize(train_wspecs, encoder, cfg, cache, args.max_seconds, "train", log_p)
    val_clips = materialize(val_wspecs, encoder, cfg, cache, args.max_seconds, "val", log_p)

    pos_w = pos_weights_from_targets(train_clips.iter_targets(), cap=args.pos_weight_cap)
    print("pos_weights:", {ln: round(float(w), 1) for ln, w in zip(cfg.lanes, pos_w, strict=True)}, flush=True)

    torch.manual_seed(args.seed)  # reproducible head init (multi-seed ablations)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    in_dim = embeddings.feat_dim(cfg.high_band)
    model = MultiLaneHeads(in_dim=in_dim, hidden=cfg.head_hidden, num_layers=cfg.head_layers)
    if args.resume:
        sd = torch.load(Path(args.resume) / "model.pt", map_location="cpu")
        model.load_state_dict(sd)
        print(f"resumed weights from {args.resume}", flush=True)
    if torch.cuda.is_available():
        model = model.cuda()
        print("device: cuda", flush=True)

    train_loop(
        model, train_clips, cfg, epochs=args.epochs, pos_weight=pos_w,
        batch_size=args.batch_size, num_workers=args.num_workers, val_clips=val_clips,
        out_dir=args.out, checkpoint_every=10,
        lr_schedule=args.lr_schedule, warmup_steps=args.warmup_steps, loss_fn=args.loss,
        keep_best=args.keep_best,
        early_stop=args.early_stop, es_window=args.es_window, es_slope=args.es_slope,
        es_jitter=args.es_jitter, es_min_epochs=args.es_min_epochs,
        log=lambda s: print(s, flush=True),
    )
    thresholds = tune_thresholds(model, val_clips, cfg)
    _report(model, val_clips, cfg, thresholds)
    if not args.no_filter_report:
        _report_compare(model, val_clips, cfg, thresholds)

    if args.out:
        saved = checkpoint.save(args.out, model, cfg, thresholds, in_dim=in_dim)
        print(f"\nsaved model + meta to {saved}", flush=True)


if __name__ == "__main__":
    main()
