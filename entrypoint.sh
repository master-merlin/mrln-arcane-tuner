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
            # setpriv changes the CREDENTIALS and nothing else — it does not
            # touch the environment, so without this the app user inherits
            # root's HOME=/root, which it cannot write. That is not cosmetic:
            #   * numba (via pymatting <- rembg <- the masking service) caches
            #     JIT'd functions next to the source, falls back to a HOME-based
            #     cache when site-packages is read-only — as it is for the app
            #     user — and when THAT is unwritable too it raises
            #     "no locator available", at import time, which took the whole
            #     app down on first boot of the non-root image;
            #   * `git config --global` below silently failed to write
            #     /root/.gitconfig, so safe.directory was never actually set and
            #     the self-updater would later refuse the checkout as
            #     "dubious ownership".
            # Both are one bug: a privilege drop that moved the user but not the
            # environment. Take HOME from the passwd entry rather than assuming
            # /home/<name>, so a rebuilt image with a different APP_UID still
            # lands in that account's real home.
            APP_HOME="$(getent passwd "$APP_UID" | cut -d: -f6)"
            if [ -n "$APP_HOME" ] && [ -d "$APP_HOME" ]; then
                # The volume chown above does not cover HOME (it is in the
                # image, not on the volume), and a derived image could leave it
                # root-owned; make it the app user's or the drop reintroduces
                # the same unwritable-HOME failure.
                chown "$APP_UID:$APP_GID" "$APP_HOME" 2>/dev/null || true
                export HOME="$APP_HOME"
                export USER="$(getent passwd "$APP_UID" | cut -d: -f1)"
                export LOGNAME="$USER"
            else
                echo "[entrypoint] WARNING: no home directory for uid $APP_UID —"
                echo "[entrypoint] WARNING: HOME stays $HOME; numba/git may fail if it is unwritable."
            fi
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

# The checkout is MRLN_APP_DIR (the same variable the self-updater reads,
# backend/app/core/self_update.py) — which is also what lets the supervisor
# loop below be exercised by a test outside a container.
export MRLN_APP_DIR="${MRLN_APP_DIR:-/app}"
BACKEND_DIR="$MRLN_APP_DIR/backend"

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
# The port has ONE producer: backend/port_resolver.py, the same one the local
# launchers call. PORT still wins inside it, and must: Docker's published-port
# mapping lives in the daemon, outside this namespace, so an operator's PORT is
# authoritative here in a way it is not on a desktop (DECISION-11). With PORT
# unset the resolver honours the port saved on the data volume — which is what
# the settings screen shows, and what every launcher used to ignore.
#
# A failure is a REFUSAL, not a fallback: falling back to 8000 would start a
# server on a port the settings screen denies, which is the silent disagreement
# this replaced. MRLN_SETTINGS_PATH is exported above, so the resolver reads the
# volume's settings file rather than the ephemeral checkout's.
#
# Resolved before EVERY launch, not once: a restart must pick up a port the
# user moved on the settings screen while the old server ran (the restart
# contract, backend/app/core/restart_contract.py). The operator's own PORT is
# remembered and handed to the resolver each time, so re-resolving never turns
# the previous answer into an override of the settings file.
OPERATOR_PORT="${PORT:-}"
resolve_port() {
    if ! PORT="$(PORT="$OPERATOR_PORT" python "$BACKEND_DIR/port_resolver.py")"; then
        echo "[entrypoint] refusing to start: the backend port could not be determined." >&2
        echo "[entrypoint] the reason is above; the settings file is $MRLN_SETTINGS_PATH" >&2
        exit 1
    fi
    export PORT
}
resolve_port

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
# to the uvicorn launch below. Models live under the data volume so pulls
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

# A container binds 0.0.0.0 by necessity, not by preference: bind loopback here
# and the port publishing that makes the container reachable at all stops
# working, because the mapped interface is outside the namespace. So the
# safe-by-default trick the local launchers use is not available here.
#
# That is exactly why the app refuses to start when this is reachable and
# MRLN_AUTH_TOKEN is empty (DECISION-3 (a)). BREAKING: a token-less
# 0.7.8-beta container that upgraded into this release will stop with a message
# naming the fix, instead of quietly continuing to serve your datasets, models
# and GPU to anyone who can reach the port. Set MRLN_AUTH_TOKEN.
export MRLN_BIND_HOST="${MRLN_BIND_HOST:-0.0.0.0}"
if [ -z "${MRLN_AUTH_TOKEN:-}" ]; then
    echo "[entrypoint] MRLN_AUTH_TOKEN is empty and the bind is $MRLN_BIND_HOST —"
    echo "[entrypoint] the app will refuse to start. Set MRLN_AUTH_TOKEN in the pod"
    echo "[entrypoint] template, or set MRLN_BIND_HOST=127.0.0.1 for a private run."
fi

# ── Supervise: relaunch in this container on the restart sentinel ────────
# This script is the server's SUPERVISOR (the restart contract lives in
# backend/app/core/restart_contract.py, the one producer of the exit code):
# the server is told it is supervised, and when it exits with code 75 —
# "relaunch me", what a restart from the UI or the self-updater asks for — it
# is started again here, in this container, with MRLN_RESTART=1 and the port
# resolved afresh. Any other exit code is the container's exit code too: a
# crash must not loop.
#
# Not `exec`: with uvicorn as PID 1 its exit tore the namespace down, so a
# restart from the UI simply ended the container (README's "restarts itself"
# was true only from this change on). The shell stays PID 1 and forwards
# TERM/INT to the server, so `docker stop` still reaches uvicorn and the
# container exits with the server's code within the grace period, not 143.
#
# Written FOR `set -euo pipefail`, and every line of it matters:
#   * `wait` on the LEFT of `||` is exempt from errexit, and `$?` on the right
#     is wait's status — a bare `wait "$pid"` returning 75 would abort the
#     script at that line before any comparison;
#   * a TERM delivered during the first wait returns 128+15 BEFORE the child
#     has exited (bash runs the trap first); only the second wait, in the same
#     guarded form, yields the child's real code;
#   * the trap is installed before the first launch and reads `${pid:-}`,
#     because under `set -u` a bare `$pid` in a trap that fires early aborts.
export MRLN_SUPERVISED=1
pid=""
forward_signal() {
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
    fi
}
trap forward_signal TERM INT
while :; do
    python -m uvicorn app.main:app --host "$MRLN_BIND_HOST" --port "$PORT" &
    pid=$!
    code=0; wait "$pid" || code=$?
    code=0; wait "$pid" || code=$?
    if [ "$code" -ne 75 ]; then
        exit "$code"
    fi
    echo "[entrypoint] restart requested - starting again in this container"
    export MRLN_RESTART=1
    resolve_port
done
