#!/usr/bin/env bash
# Container entrypoint: launch the pipeline worker, the api worker, and
# Caddy as siblings. If any of the three dies the script exits, which
# (combined with `restart: unless-stopped` in docker-compose.yml) lets
# the orchestrator restart the whole image cleanly rather than ending up
# in a half-up state where one process is missing.
#
# Layout (all internal except 8001):
#   127.0.0.1:8002   pipeline worker (WORKER_ROLE=pipeline) - loads models
#   127.0.0.1:8003   api worker      (WORKER_ROLE=api)      - no GPU
#   0.0.0.0:8001     Caddy           - routes by method+path (see Caddyfile)
set -euo pipefail

pids=()

cleanup() {
	trap - INT TERM EXIT
	# `kill -0` swallows the "not running" case so we don't error on
	# processes that already exited (which is the normal `wait -n` path).
	for pid in "${pids[@]}"; do
		if kill -0 "$pid" 2>/dev/null; then
			kill -TERM "$pid" 2>/dev/null || true
		fi
	done
	wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

WORKER_ROLE=pipeline python3.11 -m uvicorn app.main:app \
	--host 127.0.0.1 --port 8002 &
pids+=($!)

WORKER_ROLE=api python3.11 -m uvicorn app.main:app \
	--host 127.0.0.1 --port 8003 &
pids+=($!)

caddy run --config /etc/caddy/Caddyfile --adapter caddyfile &
pids+=($!)

# Block until any of the three exits; the trap above tears the rest down.
wait -n
exit $?
