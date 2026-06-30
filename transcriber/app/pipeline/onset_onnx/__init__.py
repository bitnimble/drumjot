"""Torch-free ONNX inference path for the learned drum-onset model.

`export.py` (build-time, torch) exports the MERT encoder + per-lane heads to two
`.onnx` graphs; `np_onsets.py` (runtime, torch-free) runs them through
onnxruntime + numpy/librosa, producing the same per-lane probability curves as
`drumjot_training.inference.stitched_probs`. Import from `.export` / `.np_onsets`
directly; this package intentionally exposes no re-export barrel.
"""
