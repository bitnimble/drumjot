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
# keys (ss=side stick, hp=pedal hat, mc=misc cymbals, mp=misc percussion). Any
# folding for display (e.g. side stick onto the snare track) happens later, in
# the frontend's MIDI->Jot conversion, not here.
LANE_TO_PITCH: dict[str, str] = {
    "k": "k",
    "s": "s",
    "ss": "ss",
    "t": "t",
    "hc": "h",
    "hp": "hp",
    "ho": "H",
    "rd": "d",
    "cr": "c",
    "mc": "mc",
    "mp": "mp",
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
    y = embeddings.load_audio(audio_path, sr=enc.sr)
    if max_seconds is not None:
        y = y[: int(max_seconds * enc.sr)]
    feat = enc.encode(y, enc.sr)
    device = next(model.parameters()).device
    x = torch.as_tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad(), runtime.autocast():
        return torch.sigmoid(model(x))[0].float().cpu().numpy(), meta["encoder_fps"]


def transcribe(
    audio_path, model, meta: dict, encoder=None,
    max_seconds: float | None = None, window_seconds: float | None = 30.0,
) -> dict[str, list[float]]:
    """Per-lane onset times for `audio_path` using a loaded model + its meta.

    Encodes with the meta's encoder/layer, peak-picks each lane with the saved
    tuned thresholds. Returns onsets keyed by training lane; pass through
    `to_pitch_onsets` for the transcriber's pitch letters.

    `window_seconds` chunks long audio (full songs would otherwise blow up
    MERT's O(n^2) attention / VRAM): each chunk is encoded + peak-picked
    independently and its onsets are offset by the chunk's true start time, so
    there is no frame-count drift across chunks. `max_seconds` caps the audio.
    """
    import torch

    from drumjot_training import embeddings, metrics

    runtime.configure_backends()
    enc = encoder or embeddings.MertEncoder(name=meta["encoder"], layer=meta["encoder_layer"])
    device = next(model.parameters()).device
    y = embeddings.load_audio(audio_path, sr=enc.sr)
    if max_seconds is not None:
        y = y[: int(max_seconds * enc.sr)]
    fps, thresholds = meta["encoder_fps"], meta["thresholds"]
    chunk = int(window_seconds * enc.sr) if window_seconds else 0
    starts = list(range(0, len(y), chunk)) if (chunk and len(y) > chunk) else [0]

    out: dict[str, list[float]] = {lane: [] for lane in meta["lanes"]}
    for s in starts:
        seg = y[s : s + chunk] if chunk else y
        if chunk and len(seg) < enc.sr // 10:  # skip a <0.1s tail
            continue
        feat = enc.encode(seg, enc.sr)
        x = torch.as_tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad(), runtime.autocast():
            probs = torch.sigmoid(model(x))[0].float().cpu().numpy()
        t0 = s / enc.sr
        for i, lane in enumerate(meta["lanes"]):
            thr = thresholds.get(lane, meta["peak_threshold"])
            for t in metrics.pick_onsets_lane(probs[i], fps, lane, thr):
                out[lane].append(t0 + float(t))
    for ts in out.values():
        ts.sort()
    return out
