"""Export the learned onset model (MERT encoder + per-lane heads) to ONNX.

Two graphs, mirroring the torch inference split (`inference.stitched_probs`):

  * **mert.onnx**: raw 24 kHz waveform `(1, samples)` -> layer-`L` features
    `(1, T, 1024)`. The encoder is truncated to its first `L` transformer blocks
    with the final stable-layer-norm neutralised, so `last_hidden_state` equals
    the full model's `hidden_states[L]` BIT-EXACT (~L/24 the compute, one clean
    output instead of an `output_hidden_states` tuple). No feature-extractor
    normalisation (`Wav2Vec2FeatureExtractor.do_normalize` is False -> raw
    waveform straight in).
  * **heads.onnx**: features `(1, T, in_dim)` -> per-lane onset logits
    `(1, n_lanes, T)`. The per-window (batch=1, no mask/pack) head forward is
    numerically identical to the torch padded+packed batch (packing exists
    precisely to make the two match).

Torch is needed only here (one-time export, cached); inference runs on
onnxruntime via `np_onsets`.
"""

from __future__ import annotations

from pathlib import Path


def export_mert(out_path: str | Path, layer: int, *, name: str = "m-a-p/MERT-v1-330M",
                opset: int = 17, fp16: bool = False) -> Path:
    """Export the truncated MERT encoder to `out_path`. Returns the path."""
    import torch
    from drumjot_training import embeddings
    from torch import nn

    enc = embeddings.MertEncoder(name=name, layer=layer, device="cpu")
    enc._ensure_model()
    mert = enc._model
    # hidden_states[layer] is the input to block `layer` (collected PRE the
    # encoder's final stable-layer-norm). Truncating to the first `layer` blocks
    # makes that the last block; neutralising the final norm yields it unchanged.
    del mert.encoder.layers[layer:]
    mert.encoder.layer_norm = nn.Identity()
    mert.eval()

    class MertBody(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_values):
            return self.m(input_values).last_hidden_state

    body = MertBody(mert).eval()
    dummy = torch.zeros(1, int(enc.sr * 8.0))
    out_path = Path(out_path)
    with torch.no_grad():
        torch.onnx.export(
            body, (dummy,), str(out_path),
            input_names=["input_values"], output_names=["features"],
            dynamic_axes={"input_values": {1: "samples"}, "features": {1: "frames"}},
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
    if fp16:
        from app.pipeline.onnx_fp16 import to_fp16

        to_fp16(out_path)
    return out_path


def export_heads(checkpoint_dir: str | Path, out_path: str | Path, *, opset: int = 17,
                 fp16: bool = False):
    """Export the `MultiLaneHeads` to `out_path`. Returns `(path, meta)`."""
    import torch
    from drumjot_training import inference
    from torch import nn

    model, meta = inference.load_model(checkpoint_dir, "cpu")
    model.eval()

    class HeadsBody(nn.Module):
        def __init__(self, heads):
            super().__init__()
            self.heads = heads

        def forward(self, x):
            return self.heads(x)  # (1, n_lanes, T) logits; mask=None, pack=False

    body = HeadsBody(model).eval()
    dummy = torch.zeros(1, 200, int(meta["in_dim"]))
    out_path = Path(out_path)
    with torch.no_grad():
        torch.onnx.export(
            body, (dummy,), str(out_path),
            input_names=["features"], output_names=["logits"],
            dynamic_axes={"features": {1: "frames"}, "logits": {2: "frames"}},
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
    if fp16:
        from app.pipeline.onnx_fp16 import to_fp16

        to_fp16(out_path)
    return out_path, meta
