#!/usr/bin/env bash
# Container entrypoint: launch the pipeline worker, the api worker, and
# Caddy as siblings. If any of the three dies the script exits, which
# (combined with `restart: unless-stopped` in docker/docker-compose.*.yml) lets
# the orchestrator restart the whole image cleanly rather than ending up
# in a half-up state where one process is missing.
#
# Layout (all internal except 8001 + optional 5173):
#   127.0.0.1:8002   pipeline worker (WORKER_ROLE=pipeline) - loads models
#   127.0.0.1:8003   api worker      (WORKER_ROLE=api)      - no GPU
#   0.0.0.0:8001     Caddy           - routes by method+path (see Caddyfile)
#   0.0.0.0:5173     Caddy           - bundled SPA + /api proxy
#                                      (omitted when DISABLE_FRONTEND=1)
set -euo pipefail

# Assemble the runtime Caddyfile. The :8001 router is always included;
# the :5173 frontend listener is appended unless DISABLE_FRONTEND=1, so
# the user can run a host-side Vite dev server on :5173 without colliding
# with this container's bundled frontend. /tmp is used because the `app`
# user can't write into /etc/caddy.
CADDY_CONFIG=/tmp/Caddyfile.runtime
cp /etc/caddy/Caddyfile "$CADDY_CONFIG"
if [ "${DISABLE_FRONTEND:-0}" != "1" ]; then
	cat /etc/caddy/Caddyfile.frontend >> "$CADDY_CONFIG"
else
	echo "entrypoint: DISABLE_FRONTEND=1, skipping :5173 frontend listener"
fi

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

caddy run --config "$CADDY_CONFIG" --adapter caddyfile &
pids+=($!)

# Block until any of the three exits; the trap above tears the rest down.
wait -n
exit $?
