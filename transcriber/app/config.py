"""Runtime configuration loaded from environment variables.

Set via `.env` file in the transcriber/ directory (gitignored) or by passing
real environment variables when running under Docker / IaaS.
"""
from __future__ import annotations

from pathlib import Path

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
    best_of_k_default: int = 1

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
    onset_delta: float = 0.05
    onset_wait: int = 5
    onset_pre_max: int = 20
    onset_post_max: int = 20
    onset_pre_avg: int = 100
    onset_post_avg: int = 100

    # --- Paths (Docker volumes mount these) ---
    models_dir: Path = Path("/models")
    spec_path: Path = Path("/app/SPEC.md")

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
