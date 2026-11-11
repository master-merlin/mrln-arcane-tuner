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
# The torch stack -- torch/torchvision/torchaudio/torchao/triton/triton-windows --
# is EXCLUDED from this script entirely: the Docker image bakes its own stack in
# the Dockerfile's torch layer, from the CUDA index that matches its base. The
# local venv never runs this script; its installers install the stack through
# torch_stack.py. requirements.txt's pins on those lines are the LOCAL stack,
# so this script must never install or reinstall them: at build time it would
# clobber the image's baked stack, and in a running container's self-update it
# would pull the local stack's versions into an image built for another torch.
# torchao is in the stack because its wheel ships compiled CUDA extensions: a
# bare `torchao==` pin resolved here would take PyPI's build, made for another
# CUDA than the image's.
#
# Filtering the stack's own lines is not enough: a NON-stack package can require
# another torch in its metadata, and pip would replace the stack to satisfy it.
# So the bulk install runs under constraints written from the INSTALLED stack
# (torch_stack.py constraints, each pinned with its local version): such a
# package fails pip's resolution, which installs nothing, and this script names
# the newer image to pull. The image's own copy of torch_stack.py, baked in
# /opt/torch-stack, is used when present; a self-update never changes it.
#
# Used by the Docker build, the container's runtime self-update and CI, so the
# three never diverge (CI installs its own stack first, from the CPU index).
# Run from backend/ (or pass the requirements path as $1).
set -eu

REQ="${1:-requirements.txt}"
PIP="python -m pip install --break-system-packages"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="mastermerlin/mrln-arcane-tuner"
TMP_REQ="$(mktemp)"
CONSTRAINTS="$(mktemp)"
PIP_ERR="$(mktemp)"
trap 'rm -f "$TMP_REQ" "$CONSTRAINTS" "$PIP_ERR"' EXIT

STACK_PY=/opt/torch-stack/torch_stack.py
[ -f "$STACK_PY" ] || STACK_PY="$HERE/torch_stack.py"
STACK_CFG="$(dirname "$STACK_PY")/torch-stack.cfg"

# 0. The installed stack as constraints. Fail closed: only exit 0 with a torch pin
#    written constrains the install, and only exit 5 -- torch_stack.py's own
#    "confirmed: no torch is installed" -- lets it run unconstrained, as it always
#    did, and never in an image (MRLN_TORCH_SET, or the baked copy), which always
#    has its stack. Anything else -- a crash exits 1, argparse 2 -- is a probe that
#    could not say what is installed, and pip never runs after it.
NO_TORCH=5
PY_EXE="$(python -c 'import sys; print(sys.executable)')"
WITH_C=()
set +e
python "$STACK_PY" constraints --python "$PY_EXE" "$CONSTRAINTS"
RC=$?
set -e
if [ "$RC" -eq 0 ] && grep -qE '^torch==' "$CONSTRAINTS"; then
  WITH_C=(-c "$CONSTRAINTS")
elif [ "$RC" -eq "$NO_TORCH" ] && [ -z "${MRLN_TORCH_SET:-}" ] && [ ! -f /opt/torch-stack/torch_stack.py ]; then
  echo "install-deps.sh: no torch stack is installed, so none is kept; installing unconstrained" >&2
else
  if [ "$RC" -eq "$NO_TORCH" ]; then
    echo "install-deps.sh: this is an image (MRLN_TORCH_SET='${MRLN_TORCH_SET:-}'), yet no installed torch stack was found" >&2
  else
    echo "install-deps.sh: could not read the installed torch stack (torch_stack.py constraints exited $RC; constraints written: $(grep -cE '^torch==' "$CONSTRAINTS" || true) torch pin)" >&2
  fi
  echo "install-deps.sh: nothing was installed." >&2
  [ "$RC" -ne 0 ] && exit "$RC"
  exit 1
fi

# The image that carries a newer stack: :latest for the default set, else :latest-<set>.
newer_image() {
  local default_set=""
  if [ -f "$STACK_CFG" ]; then
    default_set="$(sed -n 's/^DEFAULT=//p' "$STACK_CFG" | tr -d '[:space:]')"
  fi
  local set_name="${MRLN_TORCH_SET:-}"
  if [ -n "$set_name" ] && [ "$set_name" != "$default_set" ]; then
    echo "$IMAGE:latest-$set_name"
  else
    echo "$IMAGE:latest"
  fi
}

# 1. Everything except scenedetect, sam3, hpsv2, and the torch/triton stack,
#    resolved under the constraints (torch itself — already installed from the
#    CUDA wheel index in the image, or manually in the local venv — is seen as
#    satisfied and skipped by anything that merely depends on it).
grep -ivE '^[[:space:]]*(scenedetect|sam3|hpsv2|torch|torchvision|torchaudio|torchao|triton|triton-windows)([[:space:]=<>!~#]|$)' "$REQ" > "$TMP_REQ"
if ! $PIP -r "$TMP_REQ" ${WITH_C[@]+"${WITH_C[@]}"} 2>"$PIP_ERR"; then
  cat "$PIP_ERR" >&2
  if [ "${#WITH_C[@]}" -gt 0 ] && grep -qE 'ResolutionImpossible|conflicting dependencies' "$PIP_ERR"; then
    echo "install-deps.sh: these requirements need a torch stack other than the one installed" >&2
    echo "($(grep -vE '^#' "$CONSTRAINTS" | tr '\n' ' ')), and nothing was installed." >&2
    echo "This image's stack cannot take this update. Pull the newer image: $(newer_image)" >&2
  else
    echo "install-deps.sh: the requirements install failed; see pip's output above." >&2
  fi
  exit 1
fi
cat "$PIP_ERR" >&2

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
