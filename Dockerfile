# Drumjot all-in-one container.
#
# Bundles the React/Vite frontend (served as a static SPA) and the
# Python transcriber service (FastAPI under Caddy) into a single image.
# Caddy fans out requests:
#
#   :5173  -> static `/app/web` (the built frontend) with SPA fallback,
#            plus `/api/*` reverse-proxied to the transcriber router
#            on :8001 (with `/api` stripped, mirroring the Vite dev
#            proxy so the browser bundle stays origin-agnostic).
#   :8001  -> transcriber router. Splits POST /transcribe* / /lyrics/align
#            into the pipeline worker (:8002) and everything else into
#            the api worker (:8003). See transcriber/Caddyfile.
#
# Base: CUDA runtime on Ubuntu 22.04. Works on any host with NVIDIA driver
# 555+ and the NVIDIA Container Toolkit installed. Falls back to CPU
# automatically if no GPU is exposed to the container (audio-separator
# handles the device selection).

# -----------------------------------------------------------------------------
# Stage 1: build the frontend with bun.
#
# Lives in a separate stage so the multi-GB CUDA runtime doesn't carry
# node_modules into the final image, and so a frontend-only edit
# doesn't invalidate the Python/torch layers below it (and vice
# versa). Bun version matches `packageManager` in package.json.
# -----------------------------------------------------------------------------
FROM oven/bun:1.3.5 AS frontend-builder
WORKDIR /build

# Dep manifest first so the install layer caches across source edits.
COPY package.json bun.lock bunfig.toml ./
RUN bun install --frozen-lockfile

# Build inputs. `bun run build` runs stylelint + tsc --noEmit + vite
# build, so we need the stylelint config alongside the source. Public
# assets (favicon, etc.) are copied so vite's `public/` pass-through
# works at build time.
COPY tsconfig.json vite.config.ts index.html .stylelintrc.json ./
COPY src/ ./src/
COPY public/ ./public/
RUN bun run build

# -----------------------------------------------------------------------------
# Stage 2: transcriber runtime (Python + Caddy + frontend static bundle).
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies. libsndfile + ffmpeg for librosa/soundfile;
# build-essential is required by madmom's C extensions; curl/ca-certificates
# for the in-image downloads below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        ffmpeg libsndfile1 \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# transcriber/pyproject.toml requires Python >= 3.11 (StrEnum etc.).
# Ubuntu 22.04's `python3` is 3.10, so install 3.11 from the deadsnakes
# PPA and bootstrap pip for it. We use `python3.11` explicitly below
# rather than rewriting `/usr/bin/python3`, which would break apt hooks.
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Caddy: in-container reverse proxy. Now does double duty; fronts the
# transcriber workers on :8001 (the original split) AND serves the
# bundled frontend on :5173 with `/api/*` proxied back through the
# router. Pulled as the official static binary so the install doesn't
# drag in apt repository keys / systemd units we don't need. Version
# pinned for reproducibility; bump deliberately, not by floating tag.
ARG CADDY_VERSION=2.8.4
RUN curl -fsSL "https://github.com/caddyserver/caddy/releases/download/v${CADDY_VERSION}/caddy_${CADDY_VERSION}_linux_amd64.tar.gz" \
        | tar -xz -C /usr/local/bin caddy \
    && chmod +x /usr/local/bin/caddy

# Runtime user. Matches the typical host UID 1000:1000 so bind-mounted
# files written by the container (debug artifacts, benchmark results,
# the models cache volume) end up owned by the host user rather than
# root. If your host user is a different UID, override at build time
# with --build-arg APP_UID=<uid> --build-arg APP_GID=<gid>.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} app \
    && useradd -u ${APP_UID} -g ${APP_GID} -m -s /bin/bash app

WORKDIR /app

# Python deps in three independently-cacheable layers so a pyproject.toml
# edit only re-runs the editable install at the bottom, not the ~3 GB
# torch/torchaudio fetch above it.

# Layer 1: pip/setuptools + numpy/cython. Invalidated only when this
# RUN string itself changes, which is essentially never; serves as the
# bedrock for both the torch layer and the editable install below.
RUN python3.11 -m pip install --upgrade pip setuptools wheel \
    && python3.11 -m pip install --no-cache-dir 'numpy>=1.26' cython

# Layer 2: torch + torchaudio from the cu128 wheel index. Installed
# BEFORE the editable install so audio-separator's `torch` dep is
# satisfied by this CUDA-12.8 build rather than the default PyPI wheel
# (which now ships against CUDA 13.0 and refuses to run on drivers
# older than ~580). The torch pin matches what cu128 carries so the
# subsequent `pip install -e .` won't try to upgrade torch out of range.
# Pinned versions live in this RUN string, not pyproject.toml, so an
# unrelated dep bump on the editable install doesn't blow away the
# multi-GB torch wheel cache.
RUN python3.11 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        'torch>=2.8,<2.9' 'torchaudio>=2.8,<2.9'

# Layer 3: editable install of the transcriber package. This is the
# only layer that re-runs when pyproject.toml changes; torch/torchaudio
# above stay cached. `--extra-index-url` on this install too so any
# transitive torchaudio resolution also looks at cu128 wheels. The
# ADTOF onset backend (xavriley/ADTOF-pytorch) is a core dep installed
# by this same step, it's torch-only and bundles its own weights, so
# the default image carries it with no extra build arg.
COPY transcriber/pyproject.toml ./
RUN python3.11 -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -e .

# `faster-whisper` (transitive dep of `whisperx`, which we use for the
# /lyrics/align language-detect fallback) declares plain `onnxruntime`
# as a hard install_requires for its Silero VAD; that lands alongside
# the `onnxruntime-gpu` brought in by `audio-separator[gpu]`. Both
# wheels install into the same `onnxruntime/` package directory and
# share `onnxruntime_pybind11_state.so`; whichever was installed last
# wins, and in our resolution order the CPU build ends up shadowing
# the GPU one. `get_available_providers()` then returns just CPU +
# Azure and the vocal separator silently runs on CPU at 100% util
# with multi-GB RSS growth. There is no env var or import-time
# mechanism to force the GPU build (microsoft/onnxruntime#7313,
# qdrant/fastembed#608). Uninstall both wheels (they share files -
# removing only one corrupts the directory) and reinstall just the
# GPU wheel, which exposes the same `onnxruntime` import name and
# provides BOTH CUDA and CPU ExecutionProviders, so consumers that
# only know the plain name (faster-whisper's VAD path) still work.
RUN python3.11 -m pip uninstall -y onnxruntime onnxruntime-gpu \
    && python3.11 -m pip install --no-cache-dir onnxruntime-gpu

# ADTOF pretrained weights are NOT provisioned separately: the
# adtof_pytorch wheel bundles its Frame_RNN weights, so they come in
# with the editable install above. `_load_model()` still fails loud
# (then the request falls back to librosa) if the package or its
# bundled weights are somehow unavailable.

# whisperx ships a Lightning v1.5.4-era checkpoint
# (`whisperx/assets/pytorch_model.bin`). Lightning 2.x migrates it
# in-memory on every load and prints a noisy "automatically upgraded"
# warning. Persist the migrated format inside the image so the runtime
# loader has nothing to do. Path is resolved from the package itself so
# the step doesn't bake in a hard-coded site-packages location that
# could drift across Python versions / install layouts.
#
# We bypass Lightning's `upgrade_checkpoint` CLI and call its migration
# function directly for two reasons the CLI can't handle:
#
#   1. CUDA-device tensors: the bundled checkpoint was saved on CUDA;
#      we force `map_location='cpu'` because `docker build` has no GPU.
#   2. PyTorch 2.6 flipped `torch.load`'s default to `weights_only=True`,
#      and the checkpoint contains an `omegaconf.ListConfig` instance
#      that isn't in the safe-globals allowlist. The CLI calls
#      `torch.load(...)` without `weights_only=`, so it hits the new
#      restrictive default with no override knob. The whisperx asset
#      is a trusted bundled file (shipped via pip in the layer above),
#      so disabling weights_only here is exactly the intended escape
#      hatch; not a security regression.
RUN WHISPERX_DIR=$(python3.11 -c "import whisperx, os; print(os.path.dirname(whisperx.__file__))") \
    && python3.11 -c "import sys, torch; from lightning.pytorch.utilities.migration import migrate_checkpoint; p = sys.argv[1]; ckpt = torch.load(p, map_location='cpu', weights_only=False); ckpt, _ = migrate_checkpoint(ckpt); torch.save(ckpt, p); print(f'upgraded {p}')" \
        "$WHISPERX_DIR/assets/pytorch_model.bin"

# Python app + prompts.
COPY transcriber/app/ ./app/
COPY transcriber/prompts/ ./prompts/

# Caddy config + the entrypoint that launches both uvicorn workers and
# Caddy as siblings. See the file comments for the routing split. The
# frontend snippet is a separate file so entrypoint.sh can conditionally
# include it via DISABLE_FRONTEND (lets the host-side Vite dev server
# claim :5173 without colliding with the bundled SPA).
COPY transcriber/Caddyfile /etc/caddy/Caddyfile
COPY transcriber/Caddyfile.frontend /etc/caddy/Caddyfile.frontend
COPY transcriber/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Built frontend bundle from stage 1. Served as static files by Caddy
# on :5173 with SPA fallback to index.html (see Caddyfile).
COPY --from=frontend-builder /build/dist /app/web

# Beat Transformer checkpoint (optional). If the directory is empty
# the image still builds; BT just fails-loud at first use when
# BEAT_TRACKER=beat_transformer is set. See transcriber/checkpoints/README.md.
COPY transcriber/checkpoints/ ./checkpoints/

# Benchmark harness code (datasets/ and results/ are intentionally NOT
# copied; they're host-mounted at runtime via docker-compose volumes
# so the user can paste large fixture files in without rebuilding).
COPY transcriber/benchmarks/__init__.py \
     transcriber/benchmarks/README.md \
     transcriber/benchmarks/run_benchmark.py \
     ./benchmarks/
COPY transcriber/benchmarks/core/ ./benchmarks/core/
COPY transcriber/benchmarks/loaders/ ./benchmarks/loaders/

# Models cache lives on a Docker volume so weights persist across rebuilds.
# Pre-create the runtime-writable paths and chown them to the `app` user.
# A named-volume mount onto a directory the image owns at app:app will
# inherit that ownership on FIRST creation; if you already have a
# models-cache volume from a previous root-owned run, you'll need to
# `docker compose down -v` once for the new ownership to stick.
RUN mkdir -p /models /debug /app/benchmarks/datasets /app/benchmarks/results \
    && chown -R app:app /app /models /debug
ENV AUDIO_SEPARATOR_MODEL_FILE_DIR=/models \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    TORCH_HOME=/models/torch

USER app:app

EXPOSE 5173 8001

# Healthcheck so docker-compose / orchestrators can detect readiness.
# `start-period` is generous because the first container startup eagerly
# loads both separation models, and may also download them on demand
# (~3 GB) the first time the models volume is empty.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["/app/entrypoint.sh"]
