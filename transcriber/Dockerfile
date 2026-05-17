# Drumjot transcriber container.
#
# Base: CUDA runtime on Ubuntu 22.04. Works on any host with NVIDIA driver
# 555+ and the NVIDIA Container Toolkit installed. Falls back to CPU
# automatically if no GPU is exposed to the container (audio-separator
# handles the device selection).
#
# Build context: the repo root (see transcriber/docker-compose.yml). We
# need both `transcriber/` (Python service) and `src/` (canonical TS
# parser and code-share with the Vite frontend) in the same image, so the
# refinement loop can use the same parser via a `bun` subprocess.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies. libsndfile + ffmpeg for librosa/soundfile;
# build-essential is required by madmom's C extensions; unzip + curl
# install bun.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        ffmpeg libsndfile1 \
        git curl ca-certificates unzip \
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

# Bun for the TS parser bridge (jot_to_onsets.ts). Pinned to keep
# behaviour reproducible across rebuilds. Installed under /usr/local so
# the non-root runtime user (created below) can also invoke it without
# needing access to /root.
RUN curl -fsSL https://bun.sh/install | BUN_INSTALL=/usr/local/bun bash -s "bun-v1.3.5"
ENV PATH=/usr/local/bun/bin:$PATH

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

# Python deps first (cached layer) then app/TS code.
COPY transcriber/pyproject.toml ./
# Install PyTorch from the cu128 wheel index BEFORE the editable install
# so audio-separator's `torch` dep is satisfied by this CUDA-12.8 build
# rather than the default PyPI wheel (which now ships against CUDA 13.0
# and refuses to run on drivers older than ~580). The torch pin in
# pyproject.toml (>=2.8,<2.12) matches what cu128 carries, so the
# subsequent `pip install -e .` won't try to upgrade torch out of range.
# Use --extra-index-url on the editable install too, so any transitive
# torchaudio resolution also looks at cu128 wheels.
RUN python3.11 -m pip install --upgrade pip setuptools wheel \
    && python3.11 -m pip install --no-cache-dir 'numpy>=1.26' cython \
    && python3.11 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch torchaudio \
    && python3.11 -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -e .

# TS source the bun bridge needs (parser, jot model, etc.).
COPY src/ ./src/
COPY transcriber/tools/ ./tools/
# Synthesise a tsconfig at /app so `bun` resolves the `src/*` path alias
# the bridge script uses. Keeping this inline (rather than COPYing a
# shipped file) avoids the local-vs-Docker baseUrl mismatch.
RUN printf '%s\n' \
    '{' \
    '  "compilerOptions": {' \
    '    "target": "ES2022",' \
    '    "module": "ESNext",' \
    '    "moduleResolution": "bundler",' \
    '    "allowImportingTsExtensions": true,' \
    '    "noEmit": true,' \
    '    "strict": false,' \
    '    "skipLibCheck": true,' \
    '    "esModuleInterop": true,' \
    '    "baseUrl": ".",' \
    '    "paths": { "src/*": ["src/*"] },' \
    '    "types": ["bun"]' \
    '  },' \
    '  "include": ["src", "tools"]' \
    '}' > tsconfig.json

# Python app + prompts + SPEC.
COPY transcriber/app/ ./app/
COPY transcriber/prompts/ ./prompts/
COPY SPEC.md ./SPEC.md

# Benchmark harness code (datasets/ and results/ are intentionally NOT
# copied — they're host-mounted at runtime via docker-compose volumes
# so the user can paste large fixture files in without rebuilding).
COPY transcriber/benchmarks/__init__.py \
     transcriber/benchmarks/README.md \
     transcriber/benchmarks/run_benchmark.py \
     ./benchmarks/
COPY transcriber/benchmarks/core/ ./benchmarks/core/
COPY transcriber/benchmarks/loaders/ ./benchmarks/loaders/

# Pre-resolve the bun tools' TS imports so the first request doesn't pay
# resolution latency. (Bun JIT-compiles; this is mostly a smoke test
# that imports work before we hit /transcribe.) Both bridges share the
# same `src/` tree and tsconfig, so a broken import fails here at build
# time rather than mid-request.
RUN bun --bun run --silent /app/tools/jot_to_onsets.ts < /dev/null \
    || echo "jot_to_onsets bridge smoke test exited non-zero (expected if no DSL on stdin)"
RUN bun --bun run --silent /app/tools/lint_jot.ts < /dev/null \
    || echo "lint_jot bridge smoke test exited non-zero (expected if no DSL on stdin)"

# Models cache lives on a Docker volume so weights persist across rebuilds.
# Pre-create the runtime-writable paths and chown them to the `app` user.
# A named-volume mount onto a directory the image owns at app:app will
# inherit that ownership on FIRST creation — if you already have a
# models-cache volume from a previous root-owned run, you'll need to
# `docker compose down -v` once for the new ownership to stick.
RUN mkdir -p /models /debug /app/benchmarks/datasets /app/benchmarks/results \
    && chown -R app:app /app /models /debug
ENV AUDIO_SEPARATOR_MODEL_FILE_DIR=/models \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    TORCH_HOME=/models/torch \
    JOT_TO_ONSETS_TOOL=/app/tools/jot_to_onsets.ts

USER app:app

EXPOSE 8001

# Healthcheck so docker-compose / orchestrators can detect readiness.
# `start-period` is generous because the first container startup eagerly
# loads both separation models, and may also download them on demand
# (~3 GB) the first time the models volume is empty.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["python3.11", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
