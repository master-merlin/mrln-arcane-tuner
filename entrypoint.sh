#!/usr/bin/env bash
# MRLN Arcane Tuner — container entrypoint.
# Wires persistence to a mounted volume, then launches the single uvicorn
# process that serves the API, WebSocket, media, and the built Angular SPA.
set -euo pipefail

# ── Persistence root ─────────────────────────────────────────────────────
# Prefer a mounted volume (e.g. a RunPod network volume at /workspace) so the
# DB, datasets, models, and outputs survive pod restarts and download once.
# Falls back to ephemeral in-container storage (LOST on stop) when absent.
DATA_DIR="${MRLN_DATA_DIR:-/workspace}"
if [ ! -d "$DATA_DIR" ]; then
    echo "[entrypoint] WARNING: '$DATA_DIR' not found — using ephemeral /tmp/mrln-data (lost on stop). Attach a network volume for persistence."
    DATA_DIR="/tmp/mrln-data"
fi
mkdir -p "$DATA_DIR/datasets" "$DATA_DIR/models/upscale" "$DATA_DIR/outputs"

BACKEND_DIR="/app/backend"

# ── Symlink the app's working dirs onto the persistent volume ────────────
link_dir() {
    local target="$1" link="$2"
    rm -rf "$link"
    ln -s "$target" "$link"
}
link_dir "$DATA_DIR/datasets" "$BACKEND_DIR/datasets"
link_dir "$DATA_DIR/models"   "$BACKEND_DIR/models"
link_dir "$DATA_DIR/outputs"  "$BACKEND_DIR/outputs"

# ── Runtime env ──────────────────────────────────────────────────────────
export MRLN_CONTAINER=1
export MRLN_DB_PATH="$DATA_DIR/arcane_tuner.db"
export MRLN_FRONTEND_DIST="${MRLN_FRONTEND_DIST:-/app/frontend/browser}"
PORT="${PORT:-8000}"
export PORT

AUTH_STATE="off"; [ -n "${MRLN_AUTH_TOKEN:-}" ] && AUTH_STATE="on"
echo "[entrypoint] data_dir=$DATA_DIR port=$PORT auth=$AUTH_STATE dist=$MRLN_FRONTEND_DIST"

cd "$BACKEND_DIR"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
