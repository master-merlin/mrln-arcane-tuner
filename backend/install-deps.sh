#!/usr/bin/env bash
# Canonical backend Python dependency install.
#
# scenedetect MUST be installed with --no-deps: its declared dependency is the
# GUI build `opencv-python`, which collides with the pinned
# `opencv-python-headless` (both ship the `cv2` module). scenedetect's other
# deps (click, numpy, platformdirs, tqdm) are all pinned in requirements.txt,
# so --no-deps is safe. A plain `pip install -r requirements.txt` would pull
# the GUI opencv and break the headless install.
#
# Used by BOTH the Docker build and the runtime self-update so the two installs
# never diverge. Run from backend/ (or pass the requirements path as $1).
set -eu

REQ="${1:-requirements.txt}"
PIP="python -m pip install --break-system-packages"
TMP_REQ="$(mktemp)"

# 1. Everything except scenedetect, resolved normally (torch, already installed
#    from the CUDA wheel index in the image, is seen as satisfied and skipped).
grep -ivE '^[[:space:]]*scenedetect([[:space:]=<>!~#]|$)' "$REQ" > "$TMP_REQ"
$PIP -r "$TMP_REQ"
rm -f "$TMP_REQ"

# 2. scenedetect (pinned line, comment stripped) without its GUI-opencv dep.
SD="$(grep -iE '^[[:space:]]*scenedetect[[:space:]]*==' "$REQ" | sed -E 's/#.*$//' | tr -d '[:space:]' || true)"
if [ -n "$SD" ]; then
  $PIP --no-deps "$SD"
fi
