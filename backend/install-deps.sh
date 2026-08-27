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
# hpsv2 MUST also be installed with --no-deps: it declares `pytest ==7.2.0`
# and `pytest-split ==0.8.0` as INSTALL requirements — its dev deps leaked into
# its metadata. It never imports either and registers no pytest11 plugin hook,
# so the pins are inert: verified by `entry_points.txt` being absent and by a
# sweep of the package for `import pytest`. Left in the bulk resolve they make
# pip abort against our own modern pytest pin (ResolutionImpossible), which
# would break fresh installs, this image and CI. Every one of hpsv2's 18 REAL
# deps is already pinned in requirements.txt, so --no-deps hand-maintains
# nothing — see test_hpsv2_works_under_a_runner_its_metadata_forbids.
#
# sam3 MUST also be installed with --no-deps: it declares
# `huggingface-hub<1.0,>=0.30.0`, but this repo pins huggingface-hub==1.27.0
# (transformers 5.x requires >=1.5.2 hub-side via hf-xet). sam3's *import*
# works fine under hub 1.x — its declared ceiling is just stale — see
# test_sam3_imports_cleanly_despite_declared_hub_pin. Left in the bulk
# resolve, the `==` pins on both packages make pip's resolver abort instead
# of backtracking. sam3's other deps are all pinned in requirements.txt.
#
# torch/torchvision/torchaudio/triton/triton-windows are EXCLUDED from this
# script entirely (split-stack, see backend/requirements.txt): the Docker
# image bakes its own torch 2.11 trio in the Dockerfile layer, and the local
# dev venv installs its own torch 2.12.1 trio (+ torchaudio 2.11.0 --no-deps)
# manually. requirements.txt's pins on those lines are LOCAL documentation
# only — this script must never install or reinstall them, or it would
# clobber whichever torch is already baked into the image (build time) or
# fight the venv's manually-installed trio (runtime self-update).
#
# Used by BOTH the Docker build and the runtime self-update so the two installs
# never diverge. Run from backend/ (or pass the requirements path as $1).
set -eu

REQ="${1:-requirements.txt}"
PIP="python -m pip install --break-system-packages"
TMP_REQ="$(mktemp)"

# 1. Everything except scenedetect, sam3, hpsv2, and the torch/triton stack,
#    resolved normally (torch itself — already installed from the CUDA wheel
#    index in the image, or manually in the local venv — is seen as
#    satisfied and skipped by anything that merely depends on it).
grep -ivE '^[[:space:]]*(scenedetect|sam3|hpsv2|torch|torchvision|torchaudio|triton|triton-windows)([[:space:]=<>!~#]|$)' "$REQ" > "$TMP_REQ"
$PIP -r "$TMP_REQ"
rm -f "$TMP_REQ"

# 2. scenedetect (pinned line, comment stripped) without its GUI-opencv dep.
SD="$(grep -iE '^[[:space:]]*scenedetect[[:space:]]*==' "$REQ" | sed -E 's/#.*$//' | tr -d '[:space:]' || true)"
if [ -n "$SD" ]; then
  $PIP --no-deps "$SD"
fi

# 3. sam3 (pinned line, comment stripped) without its stale hub ceiling.
S3="$(grep -iE '^[[:space:]]*sam3[[:space:]]*==' "$REQ" | sed -E 's/#.*$//' | tr -d '[:space:]' || true)"
if [ -n "$S3" ]; then
  $PIP --no-deps "$S3"
fi

# 4. hpsv2 (pinned line, comment stripped) without its leaked pytest dev pins.
HP="$(grep -iE '^[[:space:]]*hpsv2[[:space:]]*==' "$REQ" | sed -E 's/#.*$//' | tr -d '[:space:]' || true)"
if [ -n "$HP" ]; then
  $PIP --no-deps "$HP"
fi
