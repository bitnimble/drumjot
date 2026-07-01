"""Export the ADTOF Frame_RNN to ONNX + a torch-free onnxruntime session.

The audio frontend (STFT + Bark filterbank + log) is already numpy in
`adtof_pytorch.audio` (used by `adtof_onsets._features`), so only the CRNN body
runs on torch today. Exporting it lets inference go through onnxruntime:
input `[1, T, 168, 1]` float32 -> `[1, T, 5]` sigmoid activations @ 100 fps.

Torch is needed only for the one-time export (cached); inference is torch-free.
"""

from __future__ import annotations

from pathlib import Path


def export_adtof(out_path: str | Path, *, opset: int = 17, fp16: bool = False) -> Path:
    """Export the pretrained Frame_RNN to `out_path`. Returns the path."""
    import torch
    from adtof_pytorch import (
        calculate_n_bins,
        create_frame_rnn_model,
        get_default_weights_path,
        load_pytorch_weights,
    )

    n_bins = calculate_n_bins()
    model = create_frame_rnn_model(n_bins)
    model.eval()
    model = load_pytorch_weights(model, str(get_default_weights_path()), strict=False)
    dummy = torch.zeros(1, 100, n_bins, 1)
    out_path = Path(out_path)
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy,), str(out_path),
            input_names=["features"], output_names=["activations"],
            dynamic_axes={"features": {1: "frames"}, "activations": {1: "frames"}},
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
    if fp16:
        from app.pipeline.onnx_fp16 import to_fp16

        to_fp16(out_path)
    return out_path


def load_adtof_session(models_dir: str | Path, *, providers=None):
    """Build the ADTOF onnxruntime session, exporting the `.onnx` once (cached in
    `models_dir`) if absent. `providers=None` -> onnxruntime's available set with
    a CPU fallback."""
    import onnxruntime as ort

    onnx_path = Path(models_dir) / "adtof_frame_rnn.onnx"
    if not onnx_path.exists():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        export_adtof(onnx_path)
    if providers is None:
        providers = ort.get_available_providers()
    try:
        return ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
