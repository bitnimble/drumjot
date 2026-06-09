# Drumjot drum-onset trainer image.
#
# Self-contained: a LEAN dep stack (torch cu128 + transformers + librosa +
# mir_eval + mido), the `drumjot_training` package, and the frozen
# MERT-v1-330M weights baked in. Datasets are NOT included, mount E-GMD (or
# STAR) and point DRUMJOT_EGMD at it.
#
# Build (context = this directory, training/):
#   docker build -f Dockerfile -t drumjot-trainer .
#
# Run on a CUDA box (e.g. RTX 3080; needs a driver new enough for CUDA 12.8):
#   docker run --rm --gpus all \
#       -v /mnt/ssd/e-gmd-v1.0.0:/data/e-gmd \
#       -e DRUMJOT_EGMD=/data/e-gmd \
#       drumjot-trainer --train-min 240 --val-min 30 --epochs 80
#
# The MERT feature cache is written to <DRUMJOT_EGMD>/_cache_mert, so keep
# the mounted dataset on the SSD (fast caching + reads). RTX 3080 (10 GB)
# comfortably fits frozen MERT-330M + the small per-lane heads.

ARG UV_VERSION=latest
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libsndfile + ffmpeg for librosa/soundfile; build-essential for any wheels
# needing a compiler; curl/git/ca-certificates for fetches.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common build-essential \
        ffmpeg libsndfile1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11 (Ubuntu 22.04 ships 3.10; we need >=3.11).
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300
RUN uv venv /opt/venv --python python3.11

# Cacheable dep layers: numpy/cython, then the heavy torch fetch, then the
# package + remaining deps. NOTE: cu130 (CUDA 13.x) torch, must match the host
# driver's CUDA major. On WSL the driver is exposed as a forwarding `libcuda`
# shim (via libdxcore); a cu128 torch on a CUDA-13 driver fails CUDA init with
# `Error 500: named symbol not found`. Use the wheel matching the driver.
RUN uv pip install 'numpy>=1.26' cython
RUN uv pip install --index-url https://download.pytorch.org/whl/cu130 \
        'torch>=2.9,<2.12' 'torchaudio>=2.9,<2.12'

WORKDIR /app
COPY pyproject.toml ./
COPY drumjot_training ./drumjot_training
RUN uv pip install --extra-index-url https://download.pytorch.org/whl/cu130 .

# Bake the frozen MERT-v1-330M weights so the image runs offline and is fast
# on first launch. --build-arg BAKE_MERT=0 skips it (then weights download at
# runtime to HF_HOME, which you can mount as a cache volume).
ENV HF_HOME=/opt/models/hf
ARG BAKE_MERT=1
RUN if [ "$BAKE_MERT" = "1" ]; then \
        python3 -c "from transformers import AutoModel, Wav2Vec2FeatureExtractor; n='m-a-p/MERT-v1-330M'; Wav2Vec2FeatureExtractor.from_pretrained(n, trust_remote_code=True); AutoModel.from_pretrained(n, trust_remote_code=True)" ; \
    fi

# Baked weights mean no HF network at runtime, so default to offline (fast,
# reproducible, air-gap friendly). If you built with BAKE_MERT=0, override:
# `-e TRANSFORMERS_OFFLINE=0 -e HF_HUB_OFFLINE=0`.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# Run as a non-root user (uid/gid 1000 by default) so files written to mounted
# volumes (the MERT feature cache under the dataset dir, --out checkpoints) are
# owned by the host user, not root. Override at build with
# `--build-arg APP_UID=<uid> --build-arg APP_GID=<gid>` if your host user
# differs. /opt/models + /app are chowned so the baked weights stay readable.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} app \
    && useradd -u ${APP_UID} -g ${APP_GID} -m -s /bin/bash app \
    && chown -R app:app /opt/models /app
USER app:app

# `docker run drumjot-trainer <args>` -> the trainer CLI. No args -> help.
ENTRYPOINT ["python3", "-m", "drumjot_training.train"]
CMD ["--help"]
