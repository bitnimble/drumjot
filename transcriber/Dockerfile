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
# build-essential + python3-dev + cython are required by madmom's
# C extensions; unzip + curl install bun.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        build-essential \
        ffmpeg libsndfile1 \
        git curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

# Bun for the TS parser bridge (jot_to_onsets.ts). Pinned to keep
# behaviour reproducible across rebuilds.
RUN curl -fsSL https://bun.sh/install | bash -s "bun-v1.3.5"
ENV PATH=/root/.bun/bin:$PATH

WORKDIR /app

# Python deps first (cached layer) then app/TS code.
COPY transcriber/pyproject.toml ./
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir 'numpy<2' cython \
    && python3 -m pip install -e .

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

# Pre-resolve the bun tool's TS imports so first request doesn't pay
# resolution latency. (Bun JIT-compiles; this is mostly a smoke test
# that imports work before we hit /transcribe.)
RUN bun --bun run --silent /app/tools/jot_to_onsets.ts < /dev/null \
    || echo "bun bridge smoke test exited non-zero (expected if no DSL on stdin)"

# Models cache lives on a Docker volume so weights persist across rebuilds.
RUN mkdir -p /models
ENV AUDIO_SEPARATOR_MODEL_FILE_DIR=/models \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    TORCH_HOME=/models/torch \
    JOT_TO_ONSETS_TOOL=/app/tools/jot_to_onsets.ts

EXPOSE 8001

# Healthcheck so docker-compose / orchestrators can detect readiness.
# `start-period` is generous because the first container startup eagerly
# loads both separation models, and may also download them on demand
# (~3 GB) the first time the models volume is empty.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
