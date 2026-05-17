"""Per-dataset loaders that yield `LoadedTrack`s of (audio_path, reference events).

Each loader hides the dataset-specific layout and annotation format
behind a uniform `iter_tracks()` generator, so `run_benchmark.py`
doesn't care which dataset it's iterating.
"""
from .base import DatasetLoader, LoadedTrack, get_loader

__all__ = ["DatasetLoader", "LoadedTrack", "get_loader"]
