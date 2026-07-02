"""Tests for `_pipeline_options_from_params`: the shared /transcribe(/resume)
form-param -> PipelineOptions translation.

Pins the two fields the HTTP layer resolves: an empty `llm_model` falls back
to `settings.llm_model`, and `onset_backend` overrides
`settings.use_learned_onsets` only when supplied. Imports `app.main` (cheap;
torch is lazy-loaded inside pipeline methods, not at import time).
"""
from __future__ import annotations

import app.main as main
from app.config import settings


def test_empty_llm_model_falls_back_to_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_model", "claude-default")
    opts = main._pipeline_options_from_params(
        beat_input="full_mix",
        quantise=True,
        quantise_use_llm=True,
        llm_model="",
        onset_backend="",
    )
    assert opts.llm_model == "claude-default"


def test_explicit_llm_model_wins_over_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_model", "claude-default")
    opts = main._pipeline_options_from_params(
        beat_input="full_mix",
        quantise=True,
        quantise_use_llm=True,
        llm_model="claude-explicit",
        onset_backend="",
    )
    assert opts.llm_model == "claude-explicit"


def test_empty_onset_backend_keeps_settings_default(monkeypatch) -> None:
    """Empty form field -> configured default (both truthy and falsey)."""
    monkeypatch.setattr(settings, "use_learned_onsets", True)
    assert (
        main._pipeline_options_from_params(
            beat_input="full_mix",
            quantise=True,
            quantise_use_llm=True,
            llm_model="m",
            onset_backend="",
        ).use_learned_onsets
        is True
    )
    monkeypatch.setattr(settings, "use_learned_onsets", False)
    assert (
        main._pipeline_options_from_params(
            beat_input="full_mix",
            quantise=True,
            quantise_use_llm=True,
            llm_model="m",
            onset_backend="",
        ).use_learned_onsets
        is False
    )


def test_onset_backend_learned_forces_learned(monkeypatch) -> None:
    """A non-empty backend overrides the default. 'learned' (any case /
    surrounding whitespace) -> True; anything else -> False, even when the
    configured default is True."""
    monkeypatch.setattr(settings, "use_learned_onsets", False)
    assert (
        main._pipeline_options_from_params(
            beat_input="full_mix",
            quantise=True,
            quantise_use_llm=True,
            llm_model="m",
            onset_backend="  LEARNED  ",
        ).use_learned_onsets
        is True
    )
    monkeypatch.setattr(settings, "use_learned_onsets", True)
    assert (
        main._pipeline_options_from_params(
            beat_input="full_mix",
            quantise=True,
            quantise_use_llm=True,
            llm_model="m",
            onset_backend="adtof",
        ).use_learned_onsets
        is False
    )


def test_passthrough_fields(monkeypatch) -> None:
    monkeypatch.setattr(settings, "learned_onsets_checkpoint", "/ckpt/dir")
    opts = main._pipeline_options_from_params(
        beat_input="drum_stem",
        quantise=False,
        quantise_use_llm=False,
        llm_model="m",
        onset_backend="",
    )
    assert opts.beat_input == "drum_stem"
    assert opts.quantise is False
    assert opts.quantise_use_llm is False
    assert opts.learned_onsets_checkpoint == "/ckpt/dir"
