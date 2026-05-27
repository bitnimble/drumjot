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
    # Used by the filter stage (the only LLM call in the live pipeline)
    # to reject artifact onsets per instrument before the kept onsets
    # render to MIDI.
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-4-7"
    llm_max_tokens: int = 8192

    # --- Filter stage ---
    # Max concurrent per-instrument filter LLM calls. Each call is small;
    # the ceiling mostly guards Anthropic rate limits, not local resources.
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
    # Used ONLY by /lyrics/align when given a full mix. A fast 2-stem
    # (vocals / instrumental) MDX-Net, ~8× faster than running the drum
    # pipeline's BS-Roformer SW just to throw away five of its six stems.
    # Vocal SDR doesn't need to be pristine for whisperx. Whisper is
    # robust to bleed; the downstream wav2vec2 forced aligner cares more
    # about *lead vocal preservation* than separation purity, so we pick
    # the throughput-leaning MDX-Net variant rather than Kim_Vocal_2: in
    # practice ~2× faster on GPU with no observable hit to word-level
    # alignment quality on our inputs.
    #
    # Loaded lazily on first /lyrics/align mix call. The vocals cache
    # key includes the model name (see `app/main.py::_vocals_model_id`),
    # so swapping models via env auto-invalidates previously cached
    # stems.
    vocals_model: str = "UVR-MDX-NET-Voc_FT.onnx"

    # --- ADTOF onset detector tuning ---
    # ADTOF (xavriley/ADTOF-pytorch Frame_RNN) is a fixed pretrained
    # model whose weights ship inside the package — there is no model /
    # scenario / fold / weights-dir to select. We only tune a
    # deterministic peak-pick over its per-frame activations. The
    # threshold is kept low on purpose: high-recall, the filter LLM
    # prunes.
    adtof_peak_threshold: float = 0.10
    adtof_peak_min_distance_s: float = 0.020
    # Universal peak PROMINENCE gate, applied to ALL lanes (previously
    # only the noisy lanes had prominence — see
    # `adtof_noisy_peak_prominence` below). Prominence measures how far
    # a peak rises above its surrounding baseline; far more robust than
    # a fixed-height threshold for rejecting decay-tail wobbles, since
    # those wobbles can clear `height` but never rise above their local
    # baseline. Set to 0 to disable on non-noisy lanes (kick/snare/toms)
    # and fall back to height-only.
    adtof_peak_prominence: float = 0.10
    # After find_peaks emits activation-domain peaks, refine each peak's
    # time to the local maximum of the AUDIO's onset-strength envelope
    # within ±this window. The NN's activation peak doesn't always sit
    # on the audio transient (OOD on isolated stems smears the response;
    # the BiGRU smears it too); snapping to where the actual audio's
    # onset envelope peaks within a tight window pins onsets to
    # sample-level accuracy regardless of how messy the activation
    # shape is. Set to 0 to disable refinement and use raw activation-
    # peak times. Replaces the older `librosa.onset.onset_backtrack`
    # step, which could overshoot 200+ ms when the activation's rising
    # edge had no local minima.
    adtof_audio_refine_window_s: float = 0.030
    # Median-of-non-silent amplitude normalization applied to EVERY stem
    # before ADTOF inference. Mirrors the frontend waveform's per-track
    # scaling (`src/playback/waveform_compute.ts::computeTrackAmpScale`):
    # take the median of |sample| above `silence_floor`, scale so that
    # median lands at `target`. Robust to bleed spikes (the median ignores
    # the tail) and to separator output-gain variance, so the fixed
    # `adtof_peak_threshold` stays meaningful across stems. Set
    # `adtof_median_normalize` False to feed ADTOF the raw stem.
    adtof_median_normalize: bool = True
    adtof_median_target: float = 0.3
    adtof_median_silence_floor: float = 0.05
    # The hihat + merged-cymbal ADTOF lanes are OOD-compressed on isolated
    # stems (see pipeline/adtof_onsets.py), so a global fixed threshold
    # over-triggers them. For those lanes only we use a per-stem ADAPTIVE
    # peak threshold; kick/snare/toms keep the fixed `adtof_peak_threshold`
    # since their lanes aren't OOD-compressed the same way. Set False to
    # fall back to the old global fixed behaviour.
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

    # Where the user-facing stem deliverables (drumless + drum-only mixes)
    # are written. Unlike `debug_dir`, these are produced every run, served
    # by the FastAPI app under `/outputs/...`, and surfaced as URLs on the
    # /transcribe response so the caller can play them back in a browser.
    outputs_dir: Path = Path("/outputs")

    # Content-addressed cache for the /lyrics/align pipeline. The
    # `vocals/` subdir holds opus-encoded separated vocals keyed by
    # SHA-256 of the input mix + the vocals-separator model id, so a
    # repeat alignment of the same mix skips the separator. Bounded by
    # `cache_vocals_cap_bytes` with LRU-by-last-access eviction; safe to
    # nuke at any time, entries refill on demand. See `app/cache.py`.
    cache_dir: Path = Path("/cache")
    cache_vocals_cap_bytes: int = 5 * 1024 * 1024 * 1024  # 5 GB

    # --- Debug artifact persistence ---
    # If set, every /transcribe request persists its intermediate files
    # (input audio, drum stem, per-instrument stems, beats.json,
    # onsets.json, prediction.mid, note_provenance.json) into a
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

    # Which role this process is playing inside the multi-process Docker
    # image (see transcriber/entrypoint.sh + transcriber/Caddyfile):
    # - `pipeline` (default) = eager-loads the separation models and
    #                          serves `/transcribe` + `/transcribe/resume`.
    #                          Single-process local runs leave it here.
    # - `api`                = no model load; serves the lightweight
    #                          control endpoints (`/health`,
    #                          `/transcribe/list`, `/outputs/*`) so they
    #                          stay responsive while the pipeline worker's
    #                          GIL is pinned by a transcription.
    # Caddy fans incoming requests across the two roles by method+path.
    worker_role: Literal["pipeline", "api"] = "pipeline"

    # --- GPU ---
    # `auto` = detect CUDA / MPS / CPU; `cuda`, `cpu`, `mps` for explicit.
    device: str = "auto"

    # --- Lyrics alignment (whisperx) ---
    # Model size for the lyrics-alignment endpoint. `medium` +
    # `int8_float16` (the default below) uses ~700 MB VRAM peak and gives
    # near-large word alignment accuracy; comfortable on a 6 GB GPU even
    # alongside the separator's eager load. `large-v3` uses ~1.5 GB int8
    # for higher accuracy on noisy/accented vocals; `small` / `tiny` exist
    # for CPU-only fallback boxes. Loaded lazily on the first
    # `/lyrics/align` call; the weights cache lives under
    # `<models_dir>/whisperx/` and survives container restarts via the
    # standard models-volume mount.
    whisper_model: str = "medium"
    # CTranslate2 compute type. `int8_float16` is the standard
    # low-VRAM-with-CUDA setting; `float16` is fp16 (less accuracy loss,
    # ~2x VRAM); `int8` for CPU-only boxes. The aligner forces float32 on
    # CPU regardless because CT2 can't run int8 on CPU.
    whisper_compute_type: str = "int8_float16"
    # ISO-639-1 language hint for transcription. Empty string =
    # auto-detect on the first 30 s of audio (whisperx default); set
    # explicitly for higher accuracy on short / noisy clips.
    whisper_language: str = ""


settings = Settings()
