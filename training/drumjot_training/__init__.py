"""Drumjot drum-onset model training.

Frozen music-SSL encoder (MERT / MusicFM) + small per-lane onset heads.
See `docs/superpowers/specs/2026-06-07-drum-onset-frozen-ssl-design.md`.
"""
import os as _os

# HERMETIC BY DEFAULT: the training/eval code never reaches the internet. Hugging
# Face libs are forced offline so a runtime call can only ever hit the LOCAL
# cache -- this both makes runs reproducible and prevents the unauthenticated-Hub
# rate-limiting that intermittently killed batch jobs (e.g. the cym ablation lost
# 4/6 runs to a Hub "can't load feature extractor" throttle). Models are fetched
# ONCE, explicitly, by `training/scripts/fetch_models.py`; if a model isn't
# cached, the loader raises a clear "run fetch_models" error instead of
# downloading. `setdefault` so the fetch script (which sets these to "0") and any
# deliberate override still win. Must run before transformers/huggingface_hub are
# imported (they read these at import time); this package init is that chokepoint.
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
