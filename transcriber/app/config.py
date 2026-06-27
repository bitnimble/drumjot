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
    # Vocal SDR doesn't need to be pristine for forced alignment, which
    # is robust to bleed; the downstream wav2vec2 forced aligner cares more
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

    # --- Onset backend ---
    # The trained frozen-MERT + per-lane-heads model (training/, run PER STEM via
    # pipeline/learned_onsets.py) is the DEFAULT onset detector. A request can fall
    # back to the ADTOF detector (below) with the `onset_backend=adtof` form param.
    # `learned_onsets_checkpoint` is a run dir (model.pt + meta.json with tuned
    # per-lane thresholds, e.g. the patched ab3_prev). Override per-machine via env.
    use_learned_onsets: bool = True
    learned_onsets_checkpoint: Path = Path("/codebox-workspace/datasets/ab3_prev")

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

    # --- Hi-hat lane recall (band-limit compensation) ---
    # Hi-hat is the highest-frequency drum, and the MP3 source lowpasses
    # ~14 kHz (the separator doesn't restore it), so ADTOF, trained on
    # full-band audio, gets a weak/smeared HH activation and the standard
    # noisy-lane gates (tuned for cymbals) cull a large fraction of real
    # hits. The hi-hat lane (`h`) therefore uses its OWN, looser gates;
    # cymbals (`d`/`c`) keep the stricter `adtof_noisy_*` /
    # `adtof_adaptive_threshold_floor` values above. These override only
    # the floor / prominence / min-distance for `h`; the adaptive `k * pXX`
    # term and the decay-reset filter are shared.
    adtof_hihat_adaptive_threshold_floor: float = 0.12
    adtof_hihat_peak_prominence: float = 0.10
    adtof_hihat_peak_min_distance_s: float = 0.050
    # After ADTOF peak-picking the hi-hat lane, also detect onsets directly
    # from the isolated hat-stem AUDIO (onset-strength peaks) and UNION them
    # into the peak set. The band-limit starves ADTOF specifically on
    # hi-hat; the clean isolated stem's audio transients recover hits ADTOF
    # never activated on (no peak-pick threshold can recover a peak that
    # isn't there). Audio onsets within `_dedup_s` of an existing ADTOF
    # peak are dropped (same hit); audio-only onsets take the ADTOF
    # activation at their frame as `strength` (honest: low where ADTOF was
    # unsure). High-recall by design, the split + filter LLM prune. Set
    # False to disable.
    adtof_hihat_audio_supplement: bool = True
    adtof_hihat_audio_supplement_dedup_s: float = 0.040
    # Onset-strength peak-pick params for the audio supplement (librosa
    # peak_pick over the hat stem's onset envelope, hop 512 -> ~11.6ms).
    adtof_hihat_audio_supplement_delta: float = 0.06
    adtof_hihat_audio_supplement_wait_s: float = 0.045
    # Sizzle rejection at the source: keep only peaks whose onset-strength
    # value is >= this multiple of the stem's median onset-strength. A real
    # strike makes a tall flux spike; open-hat sizzle is low ripple. Without
    # this floor the supplement over-segments a ring into a stream of phantom
    # 16ths (raw peak-pick yields ~5x as many). Tuned to land near the count
    # of clear audio transients (~4x median on the validation track). LOWER
    # for more recall (risks sizzle the split must then prune); RAISE if
    # sizzle survives.
    adtof_hihat_audio_supplement_min_strength_mult: float = 4.0
    # Energy floor for the hi-hat lane: drop any detected onset (ADTOF or
    # supplement) whose |sample| amplitude is below this fraction of the
    # median detected-onset amplitude. A real strike is ~1x the median;
    # phantom onsets on the noise floor / a previous hit's decay sit <0.2x
    # (validated on Cold-Hard-Bitch: real hats >=0.95x, phantom 8ths <=0.17x).
    # Normalized to the song's own hits, so it's loudness/kit-invariant.
    # Catches phantoms at the source, before a near-zero peak makes the
    # split's `pre_rms` explode. RAISE for stricter culling (risks soft real
    # hats); set 0 to disable. Only applied with >= 8 onsets (stable median).
    adtof_hihat_min_amplitude_frac: float = 0.25
    # Energy floor for the cymbal lane (`c`), same idea as the hi-hat one
    # but with TWO cymbal-specific differences. (1) The amplitude is
    # measured over a forward "bloom" window, not the ±20ms attack: a crash
    # peaks ~100ms AFTER the strike, so the attack window scores full-volume
    # crashes as quiet (see `_bloom_amplitude`). (2) The floor is lower:
    # cymbals have a wide dynamic range so the median sits higher relative
    # to soft hits. Validated on itte: real crashes read 1.3-2.0x the bloom
    # median, silent phantoms <=0.06x, with a clean gap, so 0.10x drops the
    # phantoms and keeps every real hit. RAISE for stricter culling; set 0
    # to disable. Only applied with >= 8 onsets (stable median).
    adtof_cymbal_min_amplitude_frac: float = 0.10
    # Crash-shadow filter for the cymbal lane: drop an onset that rides the
    # decay of a recent MUCH-louder hit without injecting fresh energy (a
    # crash's sustain re-triggering the detector ~0.3-1.5s after the
    # strike). Distinct from the amplitude floor, which can't catch these
    # (they carry the crash-tail's real energy), and from the decay-reset
    # (the crash decays enough to pass it). An onset is dropped only when
    # BOTH hold: a prior onset within `_SHADOW_WINDOW_S` is at least this
    # many times louder, AND the onset's RMS is not rising (energy injection
    # < `_SHADOW_INJECT_MAX`). Both conditions spare real soft hits (they
    # inject energy) and dense ride streams (uniform loudness, no shadow).
    # Set 0 to disable.
    adtof_cymbal_shadow_louder_mult: float = 3.0

    # --- Filter-LLM double-trigger guardrail ---
    # The filter LLM may flag an onset as a `double_trigger` (the detector
    # firing twice for one physical strike). A real detector double-trigger
    # is a NEAR-SIMULTANEOUS re-fire; two hits separated by more than a few
    # tens of ms are two real strikes (a roll, drag, flam, or fast double),
    # not an artifact. So after the LLM returns, any `double_trigger`
    # rejection whose gap to the strike it claims to duplicate (the LLM's
    # `double_of` index) is >= the per-lane window below is OVERTURNED, the
    # onset is kept. The guardrail only ever RESTORES hits (recall-positive);
    # it never adds rejections. A double_trigger rejection that omits a
    # usable `double_of` is also overturned (unverifiable -> keep).
    #
    # Snare/toms/clap/cowbell play fast repeated hits as a matter of course
    # (a 32nd-note single-stroke roll at 180 BPM is ~42 ms apart), so their
    # window sits just above the detector's 20 ms min-distance: only the
    # closest re-fires can be culled. Kick beaters can't re-strike as fast as
    # a stick bounce, so kick gets a wider window, but fast double-pedal
    # (16ths/32nds) still clears it at realistic tempos and stays protected.
    # Crash/ride never reach the filter LLM (cymbal_split vets them upstream).
    double_trigger_refractory_default_s: float = 0.030
    double_trigger_refractory_kick_s: float = 0.055

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

    # Content-addressed cache for the /lyrics/align pipeline, two subdirs:
    #   - `vocals/`: opus-encoded separated vocals keyed by SHA-256 of the
    #     input mix + the vocals-separator model id, so a repeat alignment
    #     of the same mix skips the separator.
    #   - `alignment/`: the forced-alignment result JSON keyed by SHA-256
    #     of the input audio + the aligner version + a hash of the caller's
    #     lyrics text + language, so an identical repeat request skips the
    #     GPU entirely.
    # Both are bounded by their `*_cap_bytes` with LRU-by-last-access
    # eviction; safe to nuke at any time, entries refill on demand. See
    # `app/cache.py`.
    cache_dir: Path = Path("/cache")
    cache_vocals_cap_bytes: int = 5 * 1024 * 1024 * 1024  # 5 GB
    # Alignment JSON is small (KB per song); a modest cap holds many
    # thousands of results.
    cache_alignment_cap_bytes: int = 256 * 1024 * 1024  # 256 MB

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
    # image (see docker/entrypoint.sh + docker/Caddyfile):
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

    # --- Lyrics alignment ---
    # ISO-639-1 language hint for the /lyrics/align endpoint. Empty
    # string = detect from the caller's lyric text
    # (`_detect_language_from_text`); set explicitly to override that
    # (e.g. to force a specific same-script language uroman would
    # otherwise guess).
    whisper_language: str = ""


settings = Settings()
