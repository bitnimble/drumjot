"""Vendored Beat Transformer model code.

Upstream: https://github.com/zhaojw1998/Beat-Transformer (Zhao et al., 2022,
"Beat Transformer: Demixed Beat and Downbeat Tracking with Dilated
Self-Attention", ISMIR 2022). Apache 2.0 / MIT license — see upstream repo.

Imports rewritten to package-relative form; `__main__` test blocks
stripped. No functional changes to the model architecture.
"""
from .layer import DilatedTransformerLayer
from .model import Demixed_DilatedTransformerModel

__all__ = ["DilatedTransformerLayer", "Demixed_DilatedTransformerModel"]
