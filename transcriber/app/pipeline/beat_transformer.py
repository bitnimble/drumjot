"""Beat Transformer (Zhao et al. 2022) activation extractor.

This module is the analogue of madmom's `RNNDownBeatProcessor` for the
vendored Beat Transformer model — it takes an audio file path and
returns a `(T, 2)` array of `[beat_activation, downbeat_activation]`
values that downstream `DBNDownBeatTrackingProcessor` can decode into
beats + downbeats. Sharing the DBN postprocessor between madmom and
BT means the rest of the pipeline (`_from_madmom_raw`, BarInfo
construction, feel detection, etc.) is identical in both modes.

## Preprocessing

Combining upstream `preprocessing/demixing.py` (extraction) and
`code/spectrogram_dataset.py::__getitem__` (training-time load):

- Sample rate: 44100 Hz, mono.
- STFT: `n_fft=4096`, `hop_length=1024` → ~43.07 fps.
- Mel filterbank: 128 mels, fmin=30 Hz, fmax=11000 Hz, computed on the
  power spectrogram (`|STFT|^2`).
- **Log-scale via `librosa.power_to_db(mel, ref=np.max)`**. Upstream
  saves raw power-mel to disk but the data loader applies this dB
  transform every time a sample is fetched, so the model was trained
  on log-scaled, max-normalized mels (all values ≤ 0 dB). Feeding raw
  power values produces saturated sigmoid output (every frame ≈ 1.0)
  because the conv stack never saw values in that range.

## Demixing

Upstream uses Spleeter to produce 5 stems (vocals, drums, bass, piano,
other) and feeds them as 5 instrument channels of the model input.

We take the simpler non-demixed baseline: replicate the full-mix mel
spectrogram across all 5 input channels. The model still runs (the
inter-instrument attention layers operate on identical channels), and
the paper's ablations report this baseline still outperforms madmom on
most benchmarks — just by a smaller margin than the full demixed
configuration. Swapping in true 5-stem demixing is a future
enhancement and would require adding Spleeter (TensorFlow) or
remapping our existing Demucs stems onto BT's instrument set.

## Model

The released `fold_N_trf_param.pt` checkpoints all use:
    dmodel=256, nhead=8, d_hid=1024, nlayers=9, instr=5, ntoken=2,
    attn_len=5, norm_first=True.

These are hardcoded here — they MUST match the checkpoint or
`load_state_dict` will refuse with shape errors.

## Output

Returns a `(combined_activations, predicted_bpm)` tuple:

- `combined_activations`: `(T, 2)` `np.float32` array shaped like
  madmom's `RNNDownBeatProcessor` output, ready for
  `DBNDownBeatTrackingProcessor`.
- `predicted_bpm`: scalar BPM from the model's tempo classifier head
  (`out_linear_t`, 300-way softmax). Upstream training quantizes
  tempo as `bin_index = round(bpm)` so the argmax bin IS the
  predicted BPM. `None` if the prediction looks degenerate
  (≤ 30 or ≥ 240 BPM — outside the realistic range for popular
  music, more likely a model failure than ground truth). Downstream
  uses this to tighten the DBN's `min_bpm`/`max_bpm` search,
  killing half-time / double-time lock-ons that the DBN can't
  break on activations alone.

Following upstream's `eight_fold_test.py` postprocessing:

    combined[:, 0] = max(sigmoid(beat_logit) - sigmoid(downbeat_logit), 0)
    combined[:, 1] = sigmoid(downbeat_logit)

The subtraction matters because BT was trained with the convention that
every downbeat is ALSO a beat — at a downbeat moment both channels fire
together. madmom's DBN, conversely, treats them as mutually exclusive
states. Without the subtraction, column 0 hits 1.0 at every downbeat
and the DBN can't pick out the bar phase (symptom: "tracker returned
no beats" because no clean peaks survive thresholding).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import torch

from app.config import settings
from app.vendor.beat_transformer import Demixed_DilatedTransformerModel

log = logging.getLogger(__name__)

# Preprocessing constants — see module docstring for provenance. Changing
# any of these will silently break the model since the released
# checkpoints were trained against this exact spectrogram pipeline.
SAMPLE_RATE = 44100
N_FFT = 4096
HOP_LENGTH = 1024
N_MELS = 128
FMIN = 30.0
FMAX = 11000.0
FPS = SAMPLE_RATE / HOP_LENGTH  # ~43.066

# BPM range we trust from the tempo head. Anything outside is treated
# as a model failure rather than a real estimate, and downstream falls
# back to the DBN's own tempo search instead of constraining it.
MIN_TRUSTED_BPM = 30.0
MAX_TRUSTED_BPM = 240.0

# Checkpoint-fixed architecture. Do not change unless retraining.
_MODEL_KWARGS = dict(
    attn_len=5,
    instr=5,
    ntoken=2,
    dmodel=256,
    nhead=8,
    d_hid=1024,
    nlayers=9,
    norm_first=True,
    dropout=0.1,
)


def _resolve_device() -> torch.device:
    """Match the rest of the service's device resolution logic.

    `settings.device == 'auto'` => prefer CUDA, fall back to CPU. We
    don't bother with MPS here because BT is only useful with GPU
    in practice (a 4-minute song takes ~minutes on CPU).
    """
    pref = (settings.device or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Beat Transformer is MIT (zhaojw1998/Beat-Transformer); the 8 CV-fold checkpoints
# are ~37 MB each and any one is a valid beat tracker. Fetched on first use rather
# than bundled (no gitignored build asset; smaller installer). Fold + source are
# overridable via BEAT_TRANSFORMER_CHECKPOINT_URL.
_DEFAULT_CHECKPOINT_URL = (
    "https://github.com/zhaojw1998/Beat-Transformer/raw/main/checkpoint/fold_4_trf_param.pt"
)


def _download_checkpoint(dest: Path) -> None:
    """Fetch the Beat Transformer checkpoint to `dest`, atomically. Docker ships it
    locally (no download); the desktop app points BEAT_TRANSFORMER_CHECKPOINT at a
    writable cache dir, where this lands it on first transcribe."""
    import os

    import httpx

    url = os.environ.get("BEAT_TRANSFORMER_CHECKPOINT_URL", _DEFAULT_CHECKPOINT_URL)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f"{dest.suffix}.{os.getpid()}.part")
    log.info("Beat Transformer checkpoint absent; downloading from %s -> %s", url, dest)
    try:
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=httpx.Timeout(30.0, read=None)
        ) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        os.replace(tmp, dest)  # atomic; a racing sibling process just re-wins harmlessly
    finally:
        tmp.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _load_model() -> tuple[Demixed_DilatedTransformerModel, torch.device]:
    """Load Beat Transformer once per process and cache.

    Same lifetime story as the Separator: model load is multi-second
    and we don't want to pay it on every request. `lru_cache` gives us
    a thread-safe lazy singleton without having to plumb yet another
    object through `lifespan`.
    """
    ckpt_path = Path(settings.beat_transformer_checkpoint)
    if not ckpt_path.exists():
        _download_checkpoint(ckpt_path)

    device = _resolve_device()
    log.info(
        "Loading Beat Transformer checkpoint from %s onto %s",
        ckpt_path,
        device,
    )
    model = Demixed_DilatedTransformerModel(**_MODEL_KWARGS)
    state = torch.load(str(ckpt_path), map_location=device)
    # Upstream eight_fold_test.py wraps state_dict in a `{state_dict: ...}`
    # dict; some forks save it bare. Tolerate both.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # DataParallel-trained checkpoints prefix every key with `module.`;
    # strip that so the keys match our (non-wrapped) module hierarchy.
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log.warning(
            "Beat Transformer checkpoint load: %d missing keys, %d unexpected. "
            "First few missing=%s, unexpected=%s",
            len(missing), len(unexpected),
            list(missing)[:5], list(unexpected)[:5],
        )
    else:
        log.info("Beat Transformer checkpoint: all keys matched exactly.")
    model.to(device).eval()
    return model, device


def park_model() -> None:
    """Move the cached Beat Transformer model to CPU. No-op when the
    lru_cache hasn't been hit yet (model has never loaded). Callers
    must hold the process-wide GPU lock; see `app.pipeline.gpu_park`."""
    if _load_model.cache_info().currsize == 0:
        return
    from app.pipeline.gpu_park import park_module

    model, _ = _load_model()
    park_module(model, "beat_transformer")


def unpark_model() -> None:
    """Move the cached Beat Transformer model back to CUDA. No-op when
    the lru_cache hasn't been hit yet."""
    if _load_model.cache_info().currsize == 0:
        return
    from app.pipeline.gpu_park import unpark_module

    model, _ = _load_model()
    unpark_module(model, "beat_transformer")


def _audio_to_mel(audio_path: Path) -> np.ndarray:
    """Compute the BT-format mel spectrogram for one audio file.

    Returns shape `(T, N_MELS)` float32.
    """
    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        return np.zeros((0, N_MELS), dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )  # (n_mels, T)
    # Match training-time normalization (see module docstring "Preprocessing").
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T.astype(np.float32, copy=False)


def extract_activations(audio_path: Path) -> tuple[np.ndarray, float | None]:
    """Run Beat Transformer over `audio_path` and return per-frame activations.

    Returns `(combined_activations, predicted_bpm)`. See module
    docstring "Output" for the shape and BPM semantics.

    The full mel spec is replicated across the model's 5 instrument
    channels — see module docstring "Demixing" for rationale.
    """
    mel = _audio_to_mel(audio_path)  # (T, n_mels)
    if mel.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), None

    log.info(
        "BT mel-spec: shape=%s, min=%.3e, max=%.3e, mean=%.3e, median=%.3e",
        mel.shape,
        float(mel.min()),
        float(mel.max()),
        float(mel.mean()),
        float(np.median(mel)),
    )

    model, device = _load_model()

    # (T, mel) -> (1, instr=5, T, mel)
    instr = _MODEL_KWARGS["instr"]
    x = np.broadcast_to(mel[None, :, :], (instr, mel.shape[0], N_MELS)).copy()
    x = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, 5, T, 128)

    with torch.no_grad():
        beat_logits, tempo_logits = model(x)  # (1, T, 2), (1, 300)
        beat_probs = torch.sigmoid(beat_logits).squeeze(0).cpu().numpy()
        tempo_bin = int(tempo_logits.squeeze(0).argmax().item())

    # tempo_bin = predicted BPM (upstream training rounds BPM to int).
    # Only trust values inside a sane musical range; outside that, the
    # head is probably outputting garbage and we let the DBN search
    # without a constraint.
    if MIN_TRUSTED_BPM <= tempo_bin <= MAX_TRUSTED_BPM:
        predicted_bpm: float | None = float(tempo_bin)
    else:
        log.warning(
            "BT tempo head returned %d BPM (outside [%g, %g]); "
            "ignoring and letting the DBN search unconstrained.",
            tempo_bin,
            MIN_TRUSTED_BPM,
            MAX_TRUSTED_BPM,
        )
        predicted_bpm = None

    # See module docstring "Output" — subtract downbeat from beat so the
    # DBN sees mutually-exclusive [non-downbeat-beat, downbeat] columns
    # instead of the redundant [beat, downbeat] convention BT trained on.
    beat = beat_probs[:, 0]
    downbeat = beat_probs[:, 1]
    combined = np.stack(
        [np.maximum(beat - downbeat, 0.0), downbeat],
        axis=-1,
    ).astype(np.float32, copy=False)

    # Full distribution stats — if the model is saturating universally
    # (every frame near 1.0) we'll see median≈1 and the DBN has no peaks
    # to pick from. Healthy output looks like median≈0 with sparse peaks.
    log.info(
        "Beat Transformer activations: %d frames @ %.2f fps. "
        "raw beat: min=%.3f median=%.3f mean=%.3f max=%.3f frac>0.5=%.2f. "
        "raw downbeat: min=%.3f median=%.3f mean=%.3f max=%.3f frac>0.5=%.2f. "
        "combined beat max=%.3f / downbeat max=%.3f. "
        "predicted BPM=%s (raw argmax=%d)",
        combined.shape[0],
        FPS,
        float(beat.min()), float(np.median(beat)), float(beat.mean()), float(beat.max()),
        float((beat > 0.5).mean()),
        float(downbeat.min()), float(np.median(downbeat)), float(downbeat.mean()),
        float(downbeat.max()),
        float((downbeat > 0.5).mean()),
        float(combined[:, 0].max()),
        float(combined[:, 1].max()),
        f"{predicted_bpm:.1f}" if predicted_bpm is not None else "<rejected>",
        tempo_bin,
    )
    return combined, predicted_bpm
