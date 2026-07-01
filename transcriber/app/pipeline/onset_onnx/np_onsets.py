"""Torch-free ONNX inference for the learned onset model.

`OnnxOnsetModel.stitched_probs` reproduces
`drumjot_training.inference.stitched_probs` (the default single-layer,
non-overlapping `plan_windows` path) with no torch: the MERT encoder and the
per-lane heads run on onnxruntime, and the surrounding glue (window planning,
the 6-20 kHz high-band block, the sigmoid/cymbal-softmax activation, per-window
stitching) is numpy/librosa. The per-window batch=1 head forward is numerically
identical to the torch padded+packed batch.

The two `.onnx` graphs are exported once (cached next to the checkpoint; that
step needs torch, see `export.py`), after which inference is torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


# `drumjot_training.train` pulls in torch (via model.py) at import, which would
# break the torch-free ONNX runtime, so plan_windows is copied here (torch-free:
# soundfile + librosa + numpy). MUST stay in sync with train.plan_windows.
_MIN_WINDOW = 5.0  # a sub-5s tail can't feed MERT's conv stack (kernel > frames)


def _plan_windows(audio_path, window: float, search: float, max_windows: int):
    """Split a clip into ~`window`-second pieces `[(start, length), ...]`, each
    interior cut nudged to the lowest-RMS point within ±`search` s so an edge
    never bisects a hit. A sub-`_MIN_WINDOW` tail is folded into the prior window."""
    import soundfile as sf
    from drumjot_training import embeddings

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
    if len(wins) >= 2 and wins[-1][1] < _MIN_WINDOW:
        s0, _ = wins[-2]
        wins[-2] = (s0, dur - s0)
        wins.pop()
    return wins[:max_windows] if max_windows else wins


def activate(logits: np.ndarray, lane_names, cymbal_softmax: bool) -> np.ndarray:
    """Numpy port of `model.activate_onsets`: per-lane sigmoid, except (when
    `cymbal_softmax`) the rd/cr rows become a joint 3-way softmax {none, ride,
    crash}. `logits` is (n_lanes, T); returns the same shape."""
    probs = _sigmoid(logits)
    if cymbal_softmax and "rd" in lane_names and "cr" in lane_names:
        rd, cr = lane_names.index("rd"), lane_names.index("cr")
        stack = np.stack([np.zeros_like(logits[rd]), logits[rd], logits[cr]], axis=0)
        stack = stack - stack.max(axis=0, keepdims=True)
        sm = np.exp(stack)
        sm /= sm.sum(axis=0, keepdims=True)
        probs[rd], probs[cr] = sm[1], sm[2]
    return probs


def _session(onnx_path, providers):
    import onnxruntime as ort

    if providers is None:
        providers = ort.get_available_providers()
    try:
        return ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


class OnnxOnsetModel:
    """MERT encoder + per-lane heads on onnxruntime; `stitched_probs` matches the
    torch path's `(probs (n_lanes, T), fps)`."""

    def __init__(self, mert_onnx, heads_onnx, meta: dict, providers=None) -> None:
        self.meta = meta
        self.mert = _session(mert_onnx, providers)
        self.heads = _session(heads_onnx, providers)

    def stitched_probs(self, audio_path):
        from drumjot_training import embeddings

        meta = self.meta
        sr = embeddings.MERT_SR
        fps = meta["encoder_fps"]
        use_hb = meta.get("high_band", int(meta.get("in_dim", embeddings.MERT_DIM)) > embeddings.MERT_DIM)
        lanes = meta["lanes"]
        cymbal_softmax = meta.get("cymbal_softmax", False)

        y = embeddings.load_audio(audio_path, sr=sr)
        y44 = None
        if use_hb:
            import librosa

            y44, _ = librosa.load(str(audio_path), sr=embeddings.HB_SR, mono=True)

        wins = _plan_windows(audio_path, 30.0, 3.0, 0)
        total = int(np.ceil(len(y) / sr * fps)) + 2
        out = np.zeros((len(lanes), total), dtype=np.float32)
        written = 0
        for start, length in wins:
            a = int(start * sr)
            seg = y[a : a + int(length * sr)]
            mert = self.mert.run(None, {"input_values": seg[None].astype(np.float32)})[0][0]  # (T, 1024)
            if use_hb:
                hb = embeddings.highband_features(audio_path, mert.shape[0], length, start, fps, y44)
                feat = np.concatenate([mert, hb], axis=1).astype(np.float32)
            else:
                feat = mert.astype(np.float32)
            logits = self.heads.run(None, {"features": feat[None]})[0][0]  # (n_lanes, T)
            probs = activate(logits, lanes, cymbal_softmax)
            f0 = int(round(start * fps))
            gh = min(total, f0 + probs.shape[1])
            out[:, f0:gh] = probs[:, : gh - f0]
            written = max(written, gh)
        return out[:, :written], fps


def _onnx_paths(checkpoint_dir: Path, meta: dict) -> tuple[Path, Path]:
    d = Path(checkpoint_dir)
    return d / f"mert_L{meta['encoder_layer']}.onnx", d / "heads.onnx"


def load_onnx_onset(checkpoint_dir, *, providers=None):
    """Build the torch-free `OnnxOnsetModel` from a saved run dir, exporting the
    MERT + heads `.onnx` once (cached next to the checkpoint) if absent. Returns
    `(model, meta)`."""
    from app.pipeline.provision import provisioned_file, shipped_onnx

    checkpoint_dir = Path(checkpoint_dir)
    # meta.json: the provisioned sidecar (shipped app), else the checkpoint dir (dev).
    meta_path = provisioned_file("onset_meta.json") or (checkpoint_dir / "meta.json")
    meta = json.loads(meta_path.read_text())
    if meta.get("lane_layers"):
        # Per-lane-layer checkpoints route each head to its own MERT layer; this
        # path exports a single layer and would feed every head the wrong
        # features (silently wrong, not a crash). Callers route these to torch.
        raise NotImplementedError("ONNX onset path supports single-layer checkpoints only")

    layer = meta["encoder_layer"]
    mert_onnx = shipped_onnx(f"mert_L{layer}")  # provisioned fp16 (shared encoder)
    heads_onnx = shipped_onnx("onset_heads")  # provisioned fp16 (this checkpoint's heads)
    if mert_onnx is None or heads_onnx is None:
        # Dev fallback: export both next to the checkpoint (needs torch).
        mert_onnx, heads_onnx = _onnx_paths(checkpoint_dir, meta)
        if not mert_onnx.exists():
            from app.pipeline.onset_onnx.export import export_mert

            export_mert(mert_onnx, layer, name=meta["encoder"])
        if not heads_onnx.exists():
            from app.pipeline.onset_onnx.export import export_heads

            export_heads(checkpoint_dir, heads_onnx)
    return OnnxOnsetModel(mert_onnx, heads_onnx, meta, providers=providers), meta
