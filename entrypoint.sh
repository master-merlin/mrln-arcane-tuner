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
mkdir -p "$DATA_DIR/datasets" "$DATA_DIR/models/upscale" "$DATA_DIR/outputs" \
         "$DATA_DIR/hf-cache"

# ── Drop privileges (first-boot chown, then run as the app user) ──────────
# The app must NOT run as root: /app is a live git checkout the running
# container can pull into and rebuild, so root would mean "can rewrite the code
# it is about to execute" with nothing between.
#
# But a freshly-mounted volume arrives root-owned, and a non-root process cannot
# chown it — so this script deliberately starts as root, fixes ownership once,
# and only then drops. That ordering is the whole reason the image has no
# `USER` line.
#
# Three cases, all real:
#   1. running as root, app user exists  -> chown, then re-exec as the app user
#   2. already non-root (`docker run --user`) -> cannot chown and must not try;
#      the operator owns the volume's permissions. Proceed as whoever we are.
#   3. root but no app user (an image built before this change, or a derived
#      image that removed it) -> proceed as root rather than fail, but say so
#      loudly. Refusing to boot would turn a hardening regression into an
#      outage, which is the wrong trade for an app that is already running.
APP_UID="${MRLN_APP_UID:-10001}"
APP_GID="${MRLN_APP_GID:-10001}"

if [ "$(id -u)" = "0" ]; then
    if getent passwd "$APP_UID" >/dev/null 2>&1; then
        # Only the directories the app actually writes. Not a blanket -R on the
        # volume: a populated dataset/model tree can be hundreds of GB, and
        # chown -R over it on every boot would add minutes to startup for
        # ownership that is already correct after the first run.
        for d in "$DATA_DIR" "$DATA_DIR/datasets" "$DATA_DIR/models" \
                 "$DATA_DIR/models/upscale" "$DATA_DIR/outputs" "$DATA_DIR/hf-cache"; do
            [ -d "$d" ] && chown "$APP_UID:$APP_GID" "$d" 2>/dev/null || true
        done
        # The DB file itself, when it already exists from a previous boot.
        [ -f "$DATA_DIR/arcane_tuner.db" ] && \
            chown "$APP_UID:$APP_GID" "$DATA_DIR/arcane_tuner.db" 2>/dev/null || true

        if command -v setpriv >/dev/null 2>&1; then
            echo "[entrypoint] dropping root -> uid=$APP_UID gid=$APP_GID"
            # Re-exec THIS script as the app user; it then takes the non-root
            # branch below and never returns here. --init-groups so the app
            # picks up its supplementary groups rather than root's.
            exec setpriv --reuid="$APP_UID" --regid="$APP_GID" --init-groups "$0" "$@"
        fi
        echo "[entrypoint] WARNING: setpriv not found — CONTINUING AS ROOT."
        echo "[entrypoint] WARNING: install util-linux to restore the privilege drop."
    else
        echo "[entrypoint] WARNING: uid $APP_UID does not exist — CONTINUING AS ROOT."
        echo "[entrypoint] WARNING: this image predates the non-root change, or the user was removed."
    fi
fi

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
# App settings (incl. the UI-set Hugging Face token) live on the persistent
# volume, not the ephemeral /app checkout — otherwise a pod restart wipes them.
export MRLN_SETTINGS_PATH="${MRLN_SETTINGS_PATH:-$DATA_DIR/settings.json}"
export MRLN_FRONTEND_DIST="${MRLN_FRONTEND_DIST:-/app/frontend/browser}"
PORT="${PORT:-8000}"
export PORT

# Hugging Face cache → persistent volume so downloaded base models / encoders
# survive pod restarts and download only once. HF_HOME is the umbrella var that
# huggingface_hub, transformers, and diffusers all honour (hub/ lives under it).
# Respect an explicit HF_HOME if the operator set one; otherwise default to the
# data volume.
export HF_HOME="${HF_HOME:-$DATA_DIR/hf-cache}"

# Trainer subprocess uses the same interpreter as the server. Deps are
# installed system-wide in the image (no project venv), so point the trainer
# at it explicitly — avoids a "venv python not found" fallback warning.
export MRLN_TRAINER_PYTHON="${MRLN_TRAINER_PYTHON:-$(command -v python)}"

# Self-update: trust the /app checkout and default git coords so the
# SelfUpdateService can run git against it without "dubious ownership".
# This now runs as the app user (the image chowns /app to it at build time), so
# the config lands in that user's HOME rather than root's — which is what the
# server process will actually read.
git config --global --add safe.directory /app 2>/dev/null || true
export MRLN_APP_DIR="${MRLN_APP_DIR:-/app}"
export MRLN_GIT_BRANCH="${MRLN_GIT_BRANCH:-main}"
export MRLN_GIT_REMOTE="${MRLN_GIT_REMOTE:-}"

AUTH_STATE="off"; [ -n "${MRLN_AUTH_TOKEN:-}" ] && AUTH_STATE="on"
# Whether the pod injected an HF token into the container process. If this
# reads "off" but you set HF_TOKEN in the RunPod template, the variable isn't
# reaching the process — use Server → Models instead. The token VALUE is never
# printed.
HF_STATE="off"; { [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; } && HF_STATE="on"
echo "[entrypoint] data_dir=$DATA_DIR port=$PORT auth=$AUTH_STATE hf_token_env=$HF_STATE dist=$MRLN_FRONTEND_DIST hf_home=$HF_HOME trainer_python=$MRLN_TRAINER_PYTHON"

# --- Ollama sidecar (best-effort; never blocks app startup) ---
# Launched only when the binary is present (the image installs it best-effort).
# A missing or failing Ollama must NEVER prevent the FastAPI app from starting,
# so the whole block is guarded and backgrounded — no failure mode propagates
# to the `exec uvicorn` below. Models live under the data volume so pulls
# survive pod restarts.
if command -v ollama >/dev/null 2>&1; then
    export OLLAMA_MODELS="${DATA_DIR}/models/ollama"
    mkdir -p "$OLLAMA_MODELS" || true
    echo "[entrypoint] starting ollama serve (OLLAMA_MODELS=$OLLAMA_MODELS)"
    ollama serve >/tmp/ollama.log 2>&1 &
else
    echo "[entrypoint] ollama not installed; LLM caption refinement sidecar disabled"
fi

cd "$BACKEND_DIR"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
