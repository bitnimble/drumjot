"""Runtime configuration loaded from environment variables.

Set via `.env` file in the transcriber/ directory (gitignored) or by passing
real environment variables when running under Docker / IaaS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings. 12-factor: everything overridable via env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-4-7"
    llm_max_tokens: int = 8192
    # Cheaper model used by the refinement critic (issue triage). Set to
    # empty string to disable the critic call entirely (fall back to
    # deterministic confidence ranking).
    critic_model: str = "claude-haiku-4-5-20251001"

    # --- Refinement defaults ---
    refine_by_default: bool = True
    # Lint pass is a separate toggle from the F1-gated refinement levels.
    # It runs first when enabled (so the F1-gated levels see a chart
    # that's at least musically well-formed) and is independently
    # toggleable so cost-sensitive callers can skip it.
    lint_by_default: bool = True
    # Transcription is per-instrument, so best-of-K is applied PER
    # INSTRUMENT: K candidates per drum pitch, each scored on that
    # pitch's onset F1, best kept. Cost ≈ K × (#instruments) LLM calls,
    # but the calls are tiny and run in parallel (`instrument_concurrency`).
    best_of_k_default: int = 1
    # Max concurrent per-instrument LLM calls in the transcribe stage
    # (and the per-instrument refinement loop). Each call is small; the
    # ceiling mostly guards Anthropic rate limits, not local resources.
    instrument_concurrency: int = 4

    # --- Separation models ---
    # `htdemucs_ft` gives drums stem; community drum-piece separator turns
    # that into kick/snare/hat/ride/crash/toms.
    demucs_model: str = "htdemucs_ft.yaml"
    # Filename in audio-separator's model registry. The project originally
    # pinned this to `aufr33-jarredou_MDX23C_DrumSep_model_v0.1.ckpt` —
    # upstream normalized the name in the 0.31+ line. Same underlying
    # checkpoint, different registry key.
    drum_pieces_model: str = "MDX23C-DrumSep-aufr33-jarredou.ckpt"

    # --- Onset detector tuning (librosa) ---
    # Windows are tight because we run on per-instrument stems, not the
    # full mix — each stem's transients are well isolated, so the
    # ±35ms local-max window (pre_max=post_max=3 at hop=512 / sr=44100)
    # still suppresses double-counting one transient while preserving
    # back-to-back hits like kick doubles or hi-hat 16ths (~125ms at
    # 120bpm). The pre_avg / post_avg window is also halved relative
    # to librosa's defaults to keep the running threshold responsive
    # to short loud passages.
    onset_delta: float = 0.05
    onset_wait: int = 3
    onset_pre_max: int = 3
    onset_post_max: int = 3
    onset_pre_avg: int = 50
    onset_post_avg: int = 50

    # --- Beat tracker ---
    # `madmom`         = the legacy RNN+DBN downbeat tracker (default).
    # `beat_transformer` = vendored Beat Transformer (Zhao et al. 2022)
    #                      activations -> shared DBN postprocessor.
    # The DBN stage is shared between both — the toggle only changes
    # which network produces the per-frame (beat, downbeat) activations.
    beat_tracker: Literal["madmom", "beat_transformer"] = "madmom"
    # Path to a pretrained Beat Transformer checkpoint (`fold_N_trf_param.pt`
    # from upstream). Baked into the image at build time from
    # `transcriber/checkpoints/`. The released checkpoints all share the
    # same dmodel=256 / nhead=8 / d_hid=1024 / nlayers=9 architecture
    # (`Demixed_DilatedTransformerModel` with `instr=5`).
    beat_transformer_checkpoint: Path = Path("/app/checkpoints/beat_transformer.pt")
    # Which audio to feed into the beat tracker:
    # - `full_mix`  = the original upload (madmom's training distribution;
    #                 also BT's "non-demixed" baseline).
    # - `drum_stem` = the Demucs Stage 1 drum stem (no melody/bass cues,
    #                 but cleaner transients — sometimes helps BT on tracks
    #                 with heavy syncopation in non-drum stems).
    # Overridable per-request via the `beat_input` form parameter.
    beat_input_default: Literal["full_mix", "drum_stem"] = "full_mix"

    # --- Paths (Docker volumes mount these) ---
    models_dir: Path = Path("/models")
    spec_path: Path = Path("/app/SPEC.md")

    # Where the user-facing stem deliverables (drumless + drum-only mixes)
    # are written. Unlike `debug_dir`, these are produced every run, served
    # by the FastAPI app under `/outputs/...`, and surfaced as URLs on the
    # /transcribe response so the caller can play them back in a browser.
    outputs_dir: Path = Path("/outputs")

    # --- Debug artifact persistence ---
    # If set, every /transcribe request persists its intermediate files
    # (input audio, drum stem, per-instrument stems, beats.json,
    # onsets.json, initial.jot, final.jot, refinement.json) into a
    # per-request subdir under this path. Leave unset (or empty) to use
    # ephemeral tempdirs that are deleted on request completion.
    #
    # Per-request override: pass `debug=true` on /transcribe. When debug
    # is on but `debug_dir` is unset, the service falls back to
    # `/debug` so the docker-compose volume mount still works.
    debug_dir: Path | None = None

    # --- HTTP ---
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # --- GPU ---
    # `auto` = detect CUDA / MPS / CPU; `cuda`, `cpu`, `mps` for explicit.
    device: str = "auto"


settings = Settings()
