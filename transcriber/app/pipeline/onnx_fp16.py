"""Shared fp16 conversion for exported ONNX graphs.

fp16 halves the file (download) and unlocks GPU tensor-core / NPU fp16 execution.
`keep_io_types=True` leaves the graph inputs/outputs fp32 (Cast nodes at the
boundary), so numpy callers feed and read fp32 unchanged; only the internal
weights/compute go fp16. Validated on CUDA at corr >= 0.99998 vs fp32 across the
separation, onset (MERT + GRU heads), ADTOF, beat, and CTC models -- fp16 is the
shipping format (GPU EPs only; ORT's CPU EP lacks fp16 GRU kernels).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def to_fp16(onnx_path: str | Path, op_block_list: Sequence[str] | None = None) -> Path:
    """Convert the fp32 ONNX graph at `onnx_path` to fp16 in place. Returns it.

    `op_block_list` keeps the named op types in fp32 (on top of the converter's
    defaults) -- for graphs with a numerically sensitive op whose fp16 rounding
    would move a decision boundary (e.g. the onset heads' per-clip calibration
    `Exp`, which amplifies the logit before the peak-pick threshold)."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    onnx_path = Path(onnx_path)
    kwargs: dict[str, object] = {"keep_io_types": True}
    if op_block_list is not None:
        kwargs["op_block_list"] = list(op_block_list)
    onnx.save(convert_float_to_float16(onnx.load(str(onnx_path)), **kwargs), str(onnx_path))
    return onnx_path
