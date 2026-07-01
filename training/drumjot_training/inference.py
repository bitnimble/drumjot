"""Inference + transcriber handoff.

Loads a trained checkpoint (`checkpoint.load`), transcribes audio to per-lane
onset times, and folds the 11 training lanes to the transcriber's DSL pitch
letters so the output drops straight into the existing pipeline
(`onsets_midi.py` / `from_midi.ts` consume these).

`to_dsl_onsets` + `LANE_TO_DSL` are pure (host-testable); `transcribe` /
`load_model` lazily import torch (run in the trainer image / sandbox).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from drumjot_training import checkpoint, runtime
from drumjot_training.lanes import LANES

# Training lane -> transcriber pitch key, INJECTIVE: every trained class keeps
# a distinct pitch (and a distinct GM note via onsets_midi.PITCH_TO_MIDI), so
# nothing is merged back down. Existing pipeline letters are reused where they
# line up (h=closed hat, H=open hat, d=ride, c=crash); the rest get new pitch
# keys (ss=side stick). Any folding for display (e.g. side stick
# onto the snare track) happens later, in the frontend's MIDI->Jot conversion,
# not here.
LANE_TO_PITCH: dict[str, str] = {
    "k": "k",
    "s": "s",
    "ss": "ss",
    "t": "t",
    "hc": "h",
    "ho": "H",
    "rd": "d",
    "cr": "c",
}

assert set(LANE_TO_PITCH) == set(LANES), "LANE_TO_PITCH must map every lane"
assert len(set(LANE_TO_PITCH.values())) == len(LANE_TO_PITCH), "mapping must be injective"


def to_pitch_onsets(
    lane_onsets: Mapping[str, Sequence[float]],
    mapping: Mapping[str, str] = LANE_TO_PITCH,
) -> dict[str, list[float]]:
    """Rekey per-lane onsets by pipeline pitch (sorted). Injective, so every
    class is preserved as its own pitch; empty lanes are dropped."""
    out: dict[str, list[float]] = {}
    for lane, ts in lane_onsets.items():
        pitch = mapping.get(lane)
        if pitch is None or not ts:
            continue
        out.setdefault(pitch, []).extend(float(t) for t in ts)
    return {pitch: sorted(v) for pitch, v in out.items()}


def load_model(checkpoint_dir, device: str = "cpu"):
    """Load `(model, meta)` from a saved run directory."""
    return checkpoint.load(checkpoint_dir, device)


def _embed_layers(audio_path, enc, layers, *, max_seconds, start_seconds, high_band,
                  cache_dtype, y_full, y44_full):
    """{layer: features} for one window, one `embed_clip` per distinct MERT layer (a
    cache hit if that layer was encoded in training). Transiently sets `enc.layer`
    (the cache key + which hidden state to read); restored before returning. Used by
    the per-lane-layer inference path -- the heads each read their layer's tensor."""
    from drumjot_training import embeddings

    orig = enc.layer
    out: dict = {}
    try:
        for layer in layers:
            enc.layer = layer
            out[layer] = embeddings.embed_clip(
                audio_path, enc, max_seconds=max_seconds, start_seconds=start_seconds,
                high_band=high_band, cache_dtype=cache_dtype, y_full=y_full, y44_full=y44_full)
    finally:
        enc.layer = orig
    return out


def lane_probs(audio_path, model, meta: dict, encoder=None, max_seconds: float | None = None):
    """Encode `audio_path` and run the heads: returns (probs (n_lanes, T), fps).

    `probs` are the sigmoid activations; callers peak-pick per lane. Shared by
    `transcribe` and the transcriber pipeline stage (which also reads the
    per-onset activation as a confidence strength). `max_seconds` caps the
    audio before encoding (bounds MERT's sequence length on full songs)."""
    import torch

    from drumjot_training import embeddings

    runtime.configure_backends()
    enc = encoder or embeddings.MertEncoder(name=meta["encoder"], layer=meta["encoder_layer"])
    use_hb = meta.get("high_band", int(meta.get("in_dim", embeddings.MERT_DIM)) > embeddings.MERT_DIM)
    y = embeddings.load_audio(audio_path, sr=enc.sr)
    device = next(model.parameters()).device
    lane_layers = meta.get("lane_layers")
    if lane_layers:  # per-lane-layer: one encode per distinct layer -> {layer: (1,T,dim)}
        feats = _embed_layers(audio_path, enc, sorted(set(lane_layers.values())),
                              max_seconds=max_seconds, start_seconds=0.0, high_band=use_hb,
                              cache_dtype="float32", y_full=y, y44_full=None)
        x = {L: torch.as_tensor(f, dtype=torch.float32, device=device).unsqueeze(0)
             for L, f in feats.items()}
    else:
        # cached encode (+ high-band) via the shared project MERT cache (fp32, byte-identical).
        # embed_clip applies max_seconds itself, so pass the full y (no manual truncation).
        feat = embeddings.embed_clip(
            audio_path, enc, max_seconds=max_seconds, high_band=use_hb,
            cache_dtype="float32", y_full=y)
        x = torch.as_tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad(), runtime.autocast():
        from drumjot_training.model import activate_onsets
        return activate_onsets(
            model(x), meta["lanes"], meta.get("cymbal_softmax", False)
        )[0].float().cpu().numpy(), meta["encoder_fps"]


# How many windows to push through the heads in one padded+packed batch. The BiGRU
# parallelises over the batch, so batching is the main inference-speed lever (the
# head forward is the GPU cost once the MERT features are cached); bounded so a long
# song's many windows don't blow up VRAM (especially under parallel eval).
WINDOW_BATCH = 16


def _stitched_probs_multilayer(audio_path, model, meta, enc, layers, *, y, y44, use_hb,
                               fps, max_seconds, window_seconds):
    """`stitched_probs` for a PER-LANE-LAYER checkpoint: the same default (non-
    overlapping `plan_windows`) tiling + padded/packed batched heads as the single-
    layer path, but every window is encoded at each distinct MERT `layer` and the
    model receives a `{layer: (B, T, dim)}` batch so each head reads its own layer.
    fp16 features, so it shares the training cache (encoded once, ever)."""
    import numpy as np
    import torch

    from drumjot_training.model import activate_onsets
    from drumjot_training.train import plan_windows

    device = next(model.parameters()).device
    wins = plan_windows(audio_path, window_seconds or 30.0, 3.0, 0)
    if max_seconds is not None:
        wins = [(s, min(length, max_seconds - s)) for s, length in wins if s < max_seconds]
    total = int(np.ceil(len(y) / enc.sr * fps)) + 2
    out = np.zeros((len(meta["lanes"]), total), dtype=np.float32)
    written = 0
    per_win = [
        _embed_layers(audio_path, enc, layers, max_seconds=length, start_seconds=start,
                      high_band=use_hb, cache_dtype="float16", y_full=y, y44_full=y44)
        for start, length in wins
    ]
    for b0 in range(0, len(per_win), WINDOW_BATCH):
        batch, bw = per_win[b0:b0 + WINDOW_BATCH], wins[b0:b0 + WINDOW_BATCH]
        bmax = max(f[layers[0]].shape[0] for f in batch)
        x = {L: torch.zeros((len(batch), bmax, batch[0][L].shape[1]), dtype=torch.float32, device=device)
             for L in layers}
        mask = torch.zeros((len(batch), bmax), dtype=torch.bool, device=device)
        for j, f in enumerate(batch):
            t = f[layers[0]].shape[0]
            for L in layers:
                x[L][j, :t] = torch.as_tensor(f[L], dtype=torch.float32, device=device)
            mask[j, :t] = True
        with torch.no_grad(), runtime.autocast():
            probs = activate_onsets(model(x, mask=mask, pack=True), meta["lanes"],
                                    meta.get("cymbal_softmax", False)).float().cpu().numpy()  # (B, L, bmax)
        for j, (start, _length) in enumerate(bw):
            pj = probs[j, :, : batch[j][layers[0]].shape[0]]  # drop pad frames
            f0 = int(round(start * fps))
            gh = min(total, f0 + pj.shape[1])
            out[:, f0:gh] = pj[:, : gh - f0]
            written = max(written, gh)
    return out[:, :written], fps


def stitched_probs(
    audio_path, model, meta: dict, encoder=None,
    max_seconds: float | None = None, window_seconds: float | None = 30.0,
    overlap_seconds: float = 2.0, legacy_overlap: bool = False,
):
    """One global `(probs (n_lanes, T_total), fps)` for the whole clip.

    Default: the SAME windowing + precision the model trained on -- non-
    overlapping `plan_windows` cuts (~`window_seconds`, nudged to low-RMS gaps so
    a window edge never bisects a hit), each window scored in FULL and
    concatenated, fp16 features. This matches the trained regime (an A/B showed no
    F1 difference vs the old overlapping stitch -- see RESULTS) AND shares the MERT
    cache with training (same resolved path / window / dtype), so stems encoded
    during training are reused for free.

    `legacy_overlap=True` restores the prior scheme: overlapping ~`window_seconds`
    chunks at fp32, STITCHED center-crop (half the overlap trimmed per interior
    edge). If the checkpoint was trained with the high-band block
    (`meta["in_dim"] > MERT_DIM`), the 6-20 kHz features are appended per chunk."""
    import numpy as np
    import torch

    from drumjot_training import embeddings
    from drumjot_training.model import activate_onsets

    runtime.configure_backends()
    enc = encoder or embeddings.MertEncoder(name=meta["encoder"], layer=meta["encoder_layer"])
    device = next(model.parameters()).device
    use_hb = meta.get("high_band", int(meta.get("in_dim", embeddings.MERT_DIM)) > embeddings.MERT_DIM)
    y = embeddings.load_audio(audio_path, sr=enc.sr)
    if max_seconds is not None:
        y = y[: int(max_seconds * enc.sr)]
    fps = meta["encoder_fps"]
    y44 = None
    if use_hb:
        import librosa

        y44, _ = librosa.load(str(audio_path), sr=embeddings.HB_SR, mono=True)
        if max_seconds is not None:
            y44 = y44[: int(max_seconds * embeddings.HB_SR)]

    # Default: the TRAINING windowing -- non-overlapping `plan_windows` cuts
    # (nudged to low-RMS gaps so an edge never bisects a hit), each window scored
    # in FULL (no overlap, no centre-crop), fp16 features. Matches the trained
    # regime (A/B = parity, see RESULTS) and shares the MERT cache with training.
    # Windows are contiguous, so concatenating their probs tiles the song.
    lane_layers = meta.get("lane_layers")
    if lane_layers:  # per-lane-layer model: route each head to its MERT layer's features
        if legacy_overlap:
            raise NotImplementedError(
                "per-lane-layer stitching only supports the default (non-legacy) windowing")
        return _stitched_probs_multilayer(
            audio_path, model, meta, enc, sorted(set(lane_layers.values())),
            y=y, y44=y44, use_hb=use_hb, fps=fps,
            max_seconds=max_seconds, window_seconds=window_seconds)

    if not legacy_overlap:
        from drumjot_training.train import plan_windows

        wins = plan_windows(audio_path, window_seconds or 30.0, 3.0, 0)
        if max_seconds is not None:  # cap the planned windows to the analysed span
            wins = [(s, min(length, max_seconds - s)) for s, length in wins if s < max_seconds]
        total = int(np.ceil(len(y) / enc.sr * fps)) + 2
        out = np.zeros((len(meta["lanes"]), total), dtype=np.float32)
        written = 0
        # Encode every window (a cache hit is instant), then run the heads on the
        # windows as PADDED+PACKED batches: the BiGRU parallelises across the batch,
        # so N windows cost ~one forward instead of N. pack_padded_sequence makes the
        # batched output numerically identical to per-window forwards.
        feats = [
            embeddings.embed_clip(
                audio_path, enc, max_seconds=length, start_seconds=start,
                high_band=use_hb, cache_dtype="float16", y_full=y, y44_full=y44)
            for start, length in wins
        ]
        for b0 in range(0, len(feats), WINDOW_BATCH):
            batch, bw = feats[b0 : b0 + WINDOW_BATCH], wins[b0 : b0 + WINDOW_BATCH]
            bmax = max(f.shape[0] for f in batch)
            x = torch.zeros((len(batch), bmax, batch[0].shape[1]), dtype=torch.float32, device=device)
            mask = torch.zeros((len(batch), bmax), dtype=torch.bool, device=device)
            for j, f in enumerate(batch):
                x[j, : f.shape[0]] = torch.as_tensor(f, dtype=torch.float32, device=device)
                mask[j, : f.shape[0]] = True
            with torch.no_grad(), runtime.autocast():
                probs = activate_onsets(model(x, mask=mask, pack=True), meta["lanes"],
                                        meta.get("cymbal_softmax", False)).float().cpu().numpy()  # (B, L, bmax)
            for j, (start, _length) in enumerate(bw):
                pj = probs[j, :, : batch[j].shape[0]]  # drop pad frames
                f0 = int(round(start * fps))
                gh = min(total, f0 + pj.shape[1])
                out[:, f0:gh] = pj[:, : gh - f0]
                written = max(written, gh)
        return out[:, :written], fps

    chunk = int(window_seconds * enc.sr) if window_seconds else 0
    if not chunk or len(y) <= chunk:
        starts = [0]
    else:
        step = max(1, chunk - int(overlap_seconds * enc.sr))
        starts = list(range(0, len(y) - int(overlap_seconds * enc.sr), step))
    margin = int(round(overlap_seconds / 2.0 * fps))
    total = int(np.ceil(len(y) / enc.sr * fps)) + 2
    out = np.zeros((len(meta["lanes"]), total), dtype=np.float32)
    written = 0
    for si, s in enumerate(starts):
        seg = y[s : s + chunk] if chunk else y
        if chunk and len(seg) < enc.sr // 10:  # skip a <0.1s tail
            continue
        # cached encode (+ high-band) via the shared project MERT cache: each window
        # is encoded once, ever (embeddings.MERT_CACHE_DIR). fp32 keeps it
        # byte-identical to the un-cached path; y/y44 are already loaded so embed_clip
        # slices them in-memory (no re-decode). Window starts are exact multiples of
        # the step, so the [start, start+len] slice == the old y[s:s+chunk].
        feat = embeddings.embed_clip(
            audio_path, enc, max_seconds=len(seg) / enc.sr, start_seconds=s / enc.sr,
            high_band=use_hb, cache_dtype="float32", y_full=y, y44_full=y44)
        x = torch.as_tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad(), runtime.autocast():
            probs = activate_onsets(model(x), meta["lanes"], meta.get("cymbal_softmax", False))[0].float().cpu().numpy()  # (L, Tc)
        tc = probs.shape[1]
        lo = 0 if si == 0 else margin
        hi = tc if si == len(starts) - 1 else max(lo, tc - margin)
        f0 = int(round(s / enc.sr * fps))
        gh = min(total, f0 + hi)
        out[:, f0 + lo : gh] = probs[:, lo : lo + (gh - f0 - lo)]
        written = max(written, gh)
    return out[:, :written], fps


def transcribe(
    audio_path, model, meta: dict, encoder=None,
    max_seconds: float | None = None, window_seconds: float | None = 30.0,
) -> dict[str, list[float]]:
    """Per-lane onset times using the saved tuned thresholds + the shared per-lane
    deterministic picker (`metrics.pick_onsets_lane`), over the globally stitched
    probability curves (no window-boundary artifacts). Returns onsets keyed by
    training lane; pass through `to_pitch_onsets` for the transcriber's letters."""
    from drumjot_training import metrics

    probs, fps = stitched_probs(audio_path, model, meta, encoder, max_seconds, window_seconds)
    thresholds = meta["thresholds"]
    out: dict[str, list[float]] = {}
    for i, lane in enumerate(meta["lanes"]):
        thr = thresholds.get(lane, meta["peak_threshold"])
        out[lane] = [float(t) for t in metrics.pick_onsets_lane(probs[i], fps, lane, thr)]
    return out


def transcribe_dual(
    audio_path, model, meta: dict, encoder=None,
    max_seconds: float | None = None, window_seconds: float | None = 30.0,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """One encode pass -> `(bare, full)` per-lane onset times for A/B comparison:
    `bare` = height + flat `peak_min_distance_s` only (raw model peaks); `full` =
    the shared per-lane deterministic picker (per-lane min-distance + prominence +
    decay-reset). `full` is what we deploy; `bare` is the ablation baseline."""
    from drumjot_training import metrics

    probs, fps = stitched_probs(audio_path, model, meta, encoder, max_seconds, window_seconds)
    thresholds = meta["thresholds"]
    md = meta["peak_min_distance_s"]
    bare: dict[str, list[float]] = {}
    full: dict[str, list[float]] = {}
    for i, lane in enumerate(meta["lanes"]):
        thr = thresholds.get(lane, meta["peak_threshold"])
        bare[lane] = [float(t) for t in metrics.pick_onsets(probs[i], fps, thr, md)]
        full[lane] = [float(t) for t in metrics.pick_onsets_lane(probs[i], fps, lane, thr)]
    return bare, full
