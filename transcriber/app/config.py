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

    # Which transcribe pathway to run:
    # - `dsl`    = the LLM emits a Drumjot DSL line per instrument,
    #              recomposed into a Jot, then F1-gated refinement.
    # - `filter` = the LLM only *filters* the detected onsets (rejects
    #              separation/detection artifacts); the kept onsets are
    #              rendered straight to a MIDI file with their original
    #              un-quantized times. No Jot, no recompose, no refine.
    # Overridable per-request via the `transcribe_mode` form parameter.
    transcribe_mode: Literal["dsl", "filter"] = "dsl"
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
    # Both Stage-1 and Stage-2 models are NOT in audio-separator's
    # registry; `pipeline/provision.py` injects them and downloads their
    # weights on startup. These values are the *local* filenames it writes
    # into `models_dir` and must stay in sync with `provision._MODELS`
    # (the field name `demucs_model` is historical — Stage 1 is a Roformer
    # now, not Demucs). Stage-1 `model_bs_roformer_sw.ckpt` = BS-Roformer
    # SW (6-stem; we consume only its drums stem). Stage-2
    # `drumsep_5stems_mdx23c_jarredou.ckpt` = jarredou 5-stem MDX23C
    # DrumSep (kick/snare/toms/hh/cymbals; ride+crash merged).
    demucs_model: str = "model_bs_roformer_sw.ckpt"
    drum_pieces_model: str = "drumsep_5stems_mdx23c_jarredou.ckpt"

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

    # --- Onset backend selection ---
    # Which mechanism produces the per-stem onset array:
    # - `librosa` = the high-recall spectral-flux detector above (default).
    # - `adtof`   = ADTOF CRNN run per stem, reading only that stem's
    #               matching class lane. Out-of-distribution (ADTOF is
    #               trained on full mixes, we run it on isolated stems) so
    #               it auto-falls-back to librosa if unavailable/erroring.
    # This is ONLY the default for the `onset_backend` /transcribe form
    # parameter — callers switch backend per request, not via the env.
    onset_backend: Literal["librosa", "adtof"] = "librosa"
    # ADTOF (xavriley/ADTOF-pytorch Frame_RNN) is a fixed pretrained
    # model whose weights ship inside the package — there is no model /
    # scenario / fold / weights-dir to select. We only tune a
    # deterministic peak-pick over its per-frame activations. The
    # threshold is kept low on purpose: same "high-recall, the LLM
    # prunes" contract the librosa detector uses.
    adtof_peak_threshold: float = 0.10
    adtof_peak_min_distance_s: float = 0.020
    # The hihat + merged-cymbal ADTOF lanes are OOD-compressed and
    # bleed-heavy on isolated stems (see pipeline/adtof_onsets.py), so a
    # global fixed threshold over-triggers them. For those lanes only we
    # (a) RMS-normalize the stem before inference — deterministic and
    # robust to bleed peaks, unlike the package's internal peak-norm
    # whose default we can't rely on — and (b) use a per-stem ADAPTIVE
    # peak threshold. Kick/snare/toms keep the fixed `adtof_peak_threshold`
    # since their lanes aren't OOD-compressed the same way. Set the bools
    # False to fall back to the old global fixed behaviour.
    adtof_rms_normalize: bool = True
    adtof_rms_target_dbfs: float = -20.0
    adtof_adaptive_threshold: bool = True
    # threshold = max(floor, k * percentile(lane_activation, pct)).
    # floor/k raised from 0.15/0.35 after open-hihat over-triggering.
    adtof_adaptive_threshold_floor: float = 0.22
    adtof_adaptive_threshold_k: float = 0.50
    adtof_adaptive_threshold_pct: float = 95.0
    # Open hi-hats / sustained cymbals make the activation a high, mushy
    # PLATEAU (low peak-to-trough contrast), so a height threshold alone
    # still picks plateau ripples — and the adaptive threshold rises with
    # the plateau, defeating itself. For the noisy lanes only we also
    # require a minimum peak PROMINENCE (rise above the local baseline —
    # targets the low-contrast case directly, independent of level) and a
    # wider min-distance than kick/snare. Set prominence to 0 to disable.
    adtof_noisy_peak_prominence: float = 0.2
    adtof_noisy_peak_min_distance_s: float = 0.070
    # Prominence/height still can't tell ONE sustained open-hihat (whose
    # activation wobbles > prominence across its ring) from a real stream
    # of hits. The decay-reset post-filter keeps a peak only if, since the
    # previous accepted peak, the activation fell back below
    # `max(floor, frac * prev_peak_height)` — i.e. the prior ring actually
    # decayed before this onset. Continuous sustain collapses to a single
    # onset; genuinely separate hits (activation plunges between them) all
    # survive. Noisy lanes only. Set frac to 0 to disable.
    adtof_noisy_decay_reset_frac: float = 0.6
    adtof_noisy_decay_reset_floor: float = 0.05

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
