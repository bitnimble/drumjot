"""Thin torch-only separation wrapper, a drop-in for `audio-separator` for the
two models the drum pipeline uses (BS-Roformer SW and MDX23C DrumSep).

`loader.load_model` builds a vendored `nn.Module` from a (ckpt, yaml) pair and
`runner.SeparationRunner.separate` runs the chunked overlap-add inference,
returning stems in memory. No ONNX yet (torch only); the vendored models expose
a `forward_spec` cut point for a later ONNX export. Import from `.loader` /
`.runner` directly; this package intentionally exposes no re-export barrel.
"""
