"""Export a loaded separation model's STFT-free body to ONNX, keeping the
complex STFT/iSTFT (and, for BS-Roformer, the complex mask multiply) OUT of the
graph so the body runs cleanly on any onnxruntime execution provider.

The graph is fixed-shape at the model's real chunk length: the runner always
feeds exactly one `chunk_size` window per call, so a fixed time axis is correct
and sidesteps the rotary-embedding cache's trace-time specialisation that a
dynamic axis would risk.

  * MDX23C: exports `forward_spec` (spectrogram -> spectrogram, all-real conv).
  * BS-Roformer: exports `forward_mask` (spectrogram -> real mask); the numpy
    inference path applies the complex mask + iSTFT around it
    (np_inference.bs_apply_mask / bs_unpack).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .loader import LoadedModel


class _MdxBody(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_spec(x)


class _BsBody(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_mask(x)


def _mdx_example(loaded: LoadedModel) -> torch.Tensor:
    cfg = loaded.config
    chunk = cfg.audio.hop_length * (cfg.inference.dim_t - 1)
    dummy = torch.randn(1, cfg.audio.num_channels, chunk)
    with torch.no_grad():
        return loaded.model.stft(dummy)


def _bs_example(loaded: LoadedModel) -> torch.Tensor:
    model = loaded.model
    chunk = model.stft_kwargs["hop_length"] * (loaded.config.inference.dim_t - 1)
    dummy = torch.randn(1, model.audio_channels, chunk)
    with torch.no_grad():
        stft_repr, _ = model._stft_prep(dummy)
    return stft_repr


def _to_fp16(out_path: Path) -> None:
    """Convert an fp32 ONNX graph in place to fp16, keeping the graph inputs and
    outputs fp32 (Cast nodes at the boundary) so the numpy inference path feeds
    fp32 and reads fp32 unchanged. Halves the file and unlocks the GPU tensor /
    NPU fp16 path; the STFT/iSTFT stay fp32 in numpy outside this graph."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(out_path))
    onnx.save(convert_float_to_float16(model, keep_io_types=True), str(out_path))


def export_body(
    loaded: LoadedModel, out_path: str | Path, *, opset: int = 17, fp16: bool = False
) -> Path:
    """Export `loaded`'s body to `out_path` (.onnx). Returns the path.

    Exports on CPU (the example tensors are built on CPU and the exported graph
    is device-agnostic), restoring the model's original device afterwards, so it
    works whether the model was loaded on CPU or CUDA. `fp16=True` converts the
    graph to fp16 weights (fp32 I/O preserved)."""
    out_path = Path(out_path)
    model = loaded.model
    orig_device = next(model.parameters()).device
    model.cpu().eval()
    try:
        if loaded.kind == "mdx23c":
            body, example, in_name, out_name = _MdxBody(model), _mdx_example(loaded), "spec", "out"
        else:
            body, example, in_name, out_name = _BsBody(model), _bs_example(loaded), "stft_repr", "mask"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(
                body,
                example,
                str(out_path),
                input_names=[in_name],
                output_names=[out_name],
                opset_version=opset,
                dynamo=False,
            )
    finally:
        # Restore device AND eval: torch.onnx.export can leave the module in
        # train mode, which would re-enable BS-Roformer's attn/ff dropout on any
        # later torch forward (the separators are always eval / inference).
        model.to(orig_device).eval()
    if fp16:
        _to_fp16(out_path)
    return out_path
