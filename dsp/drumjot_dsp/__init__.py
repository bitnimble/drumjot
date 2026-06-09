"""Drumjot shared DSP helpers (numpy/scipy only; no torch, no domain deps).

Single source of truth for signal-processing code shared by the transcriber
(`app/pipeline`) and the trainer (`drumjot_training`). Keep this package tiny
and dependency-light so both consumers can depend on it without resolution
conflicts.
"""
