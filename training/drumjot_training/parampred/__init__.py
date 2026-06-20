"""Adaptive per-song peak-picking parameters for the learned onset model.

A decoupled post-hoc predictor that sits after the frozen MERT+heads model and
before the peak-picker: it reads label-free signal features and emits per-song,
per-lane peakpick params, replacing today's single global-tuned value per lane.

See docs/superpowers/specs/2026-06-20-adaptive-peakpick-params-design.md.
"""
