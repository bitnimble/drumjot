# Drumjot sandbox container.
#
# A bare CUDA + Python environment that carries the *exact same*
# dependency stack as the transcriber runtime (docker/Dockerfile) but
# runs NOTHING on its own. No Caddy, no uvicorn workers, no frontend
# bundle, no entrypoint - just an idle container an agent can exec into
# to run Python scripts freely, poke at CUDA, and read/write at will.
#
# The intended use is exploratory: mount /codebox-workspace (where the
# transcriber writes debug artifacts, separated stems, benchmark
# results, etc.) read-only or read-write, inspect the output of previous
# transcribe runs, and prototype new pipeline code against the same
# torch / audio-separator / librosa / madmom stack the real service uses
# - without rebuilding or touching the production image.
#
# Deps are managed with uv (the transcriber's dev loop uses uv, not
# pip), installed into a venv at /opt/venv that's on PATH - so `python`,
# `python3`, and any console scripts resolve to it automatically, and
# `uv pip install <more>` Just Works for the agent.
#
# Base: CUDA runtime on Ubuntu 22.04, identical to docker/Dockerfile, so
# anything that runs here runs there. Works on any host with NVIDIA
# driver 555+ and the NVIDIA Container Toolkit. Falls back to CPU if no
# GPU is exposed to the container.
#
# Build (from the repo root, NOT from sandbox/ - the build context needs
# transcriber/pyproject.toml):
#
#   docker build -f sandbox/Dockerfile -t drumjot-sandbox .
#
# Run (detached + idle, then exec a shell in):
#
#   docker run -d --name drumjot-sandbox \
#       --gpus all --shm-size=2g \
#       -v /codebox-workspace:/codebox-workspace \
#       drumjot-sandbox
#   docker exec -it drumjot-sandbox bash
#
# (--shm-size=2g: default 64MB /dev/shm hangs DataLoader num_workers>0.)
#
# Drop `--gpus all` to run CPU-only. Mount /codebox-workspace read-only
# (append `:ro`) if you only want to inspect prior output without risking
# a stray write.

# uv binary source. Aliased as a build stage (rather than referenced
# inline in the COPY below) because an ARG only expands in a stage's
# image reference, not in `COPY --from=<image>`. Pin UV_VERSION (e.g.
# 0.9.0) for reproducibility; it defaults to `latest` so the image
# builds out of the box - bump deliberately, like the CADDY_VERSION pin
# in docker/Dockerfile.
# Both version ARGs must precede the first FROM to be usable in a stage's
# image reference (an ARG declared after a FROM is stage-scoped, not
# global). Pin UV_VERSION (defaults to `latest`) for reproducibility;
# bump deliberately, like the CADDY_VERSION pin in docker/Dockerfile.
ARG UV_VERSION=latest
ARG BUN_VERSION=1.3.5
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# bun binary source. The transcriber runtime is pure Python (no bun);
# this is a sandbox-only convenience so an agent can run JS/TS one-liners
# (`bun -e`) alongside Python. Version matches package.json's
# `packageManager`.
FROM oven/bun:${BUN_VERSION} AS bun

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies, same set as the transcriber image: libsndfile +
# ffmpeg for librosa/soundfile; build-essential for madmom's C
# extensions; curl/ca-certificates for the in-image downloads below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        ffmpeg libsndfile1 \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11 from deadsnakes (Ubuntu 22.04 ships 3.10; the transcriber
# needs >=3.11). Installed as `python3.11` explicitly rather than
# rewriting /usr/bin/python3, to avoid breaking apt hooks. No pip is
# bootstrapped here - uv (below) owns package management and builds the
# venv against this interpreter.
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# uv: the project's package manager. Copied as the official static
# binary from the aliased stage above so the install doesn't drag in a
# bootstrap pip.
COPY --from=uv /uv /uvx /usr/local/bin/

# bun: copied as the static binary from the aliased stage above (relies
# on libstdc++6, already present via build-essential). `bunx` is just a
# bun alias, symlinked for parity with a normal bun install.
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
RUN ln -s /usr/local/bin/bun /usr/local/bin/bunx

# A uv-managed venv at /opt/venv, put on PATH so the interpreter and any
# installed console scripts resolve to it without activation. VIRTUAL_ENV
# lets `uv pip install` target it implicitly. UV_LINK_MODE=copy silences
# the hardlink warning on the image's layered filesystem; the generous
# UV_HTTP_TIMEOUT covers the multi-GB torch wheel fetches.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300
RUN uv venv /opt/venv --python python3.11

# Runtime user. Matches the typical host UID 1000:1000 so files written
# into the bind-mounted /codebox-workspace end up owned by the host user
# rather than root. Override at build time with
# --build-arg APP_UID=<uid> --build-arg APP_GID=<gid> if your host user
# differs. Need to apt-install from inside the sandbox? Re-exec as root:
# `docker exec -u 0 ... bash`.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} app \
    && useradd -u ${APP_UID} -g ${APP_GID} -m -s /bin/bash app

# Python deps in three independently-cacheable layers, mirroring
# docker/Dockerfile so the heavy torch fetch stays cached across edits.

# Layer 1: numpy/cython bedrock. madmom's Cython extensions want these
# present at build time; isolating the layer also seeds the dep cache.
RUN uv pip install 'numpy>=1.26' cython

# Layer 2: torch + torchaudio from the cu128 wheel index (CUDA 12.8 =
# highest most consumer drivers handle). Pinned to the same range as
# pyproject.toml so the editable install below won't try to upgrade it.
RUN uv pip install --index-url https://download.pytorch.org/whl/cu128 \
        'torch>=2.8,<2.12' 'torchaudio>=2.8,<2.12'

# Layer 3: the full transcriber dependency set. Same trick as
# docker/Dockerfile: drop in ONLY pyproject.toml and `uv pip install -e .`
# with no package source present - setuptools finds nothing to build but
# uv still resolves and installs every declared dependency
# (audio-separator, madmom, adtof_pytorch, ctc-forced-aligner, librosa,
# anthropic, ...), keeping the sandbox's deps in lockstep with the real
# service. Installed into /opt/deps so the editable .pth points there and
# /sandbox stays pristine. --extra-index-url keeps any transitive torch
# resolution on the cu128 wheels. The app source is deliberately NOT
# copied - this stays an empty sandbox; bring your own scripts.
WORKDIR /opt/deps
COPY transcriber/pyproject.toml ./
# `.[dev]` adds ruff + pytest so the agent can lint/test in-sandbox (e.g.
# the repo-root `training/` package, which reuses this same dep stack).
RUN uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e '.[dev]'

# Models / weights caches. Pointed at /models (and HF/torch subdirs) so
# audio-separator, HF transformers, and torch.hub reuse the same cache
# layout as the transcriber. Bind-mount the host's existing models cache
# here (e.g. -v /codebox-workspace/drumjot/models-cache:/models) to avoid
# re-downloading multi-GB weights.
ENV AUDIO_SEPARATOR_MODEL_FILE_DIR=/models \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    TORCH_HOME=/models/torch

# Writable scratch + venv + caches + mount point, all owned by the app
# user. /opt/venv and /opt/deps are chowned so the agent can `uv pip
# install` more packages without re-execing as root. /sandbox is the
# working dir for throwaway scripts and output.
RUN mkdir -p /sandbox /models /codebox-workspace \
    && chown -R app:app /sandbox /models /codebox-workspace /opt/venv /opt/deps

WORKDIR /sandbox
USER app:app

# Idle by default so the container stays up for `docker exec`. Nothing
# Drumjot-related is launched. Override the command for a one-shot run,
# e.g. `docker run --rm ... drumjot-sandbox python my_script.py`.
CMD ["sleep", "infinity"]
