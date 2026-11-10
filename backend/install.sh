#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# MRLN Arcane Tuner — Linux install script
#
# Optionally creates a virtual environment, installs the torch stack this
# machine can run (torch_stack.py reads the driver and every GPU and picks a
# set from torch-stack.cfg), then installs all remaining dependencies. PEP 508
# markers in requirements.txt automatically handle platform-specific packages.
#
#   install.sh [--venv-dir <dir>] [--yes]
#
# --venv-dir  the venv to create or reuse (default: venv next to this script).
# --yes       no questions: reuse or create the venv.
# MRLN_TORCH_INDEX=<set> chooses the torch set (cu132, cu126, cpu); it is
# still refused where that set cannot run.
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Anchored on this script, never on the current directory.
HERE="$(cd "$(dirname "$0")" && pwd)"
REQ="$HERE/requirements.txt"
STACK="$HERE/torch_stack.py"
VENV_DIR="$HERE/venv"
YES=0

while [ $# -gt 0 ]; do
    case "$1" in
        --venv-dir) VENV_DIR="$2"; shift 2 ;;
        --yes) YES=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

confirm() {
    if [ "$YES" = 1 ]; then return 0; fi
    read -rp "$1 [Y/n] " answer
    [[ ! "${answer:-Y}" =~ ^[Nn]$ ]]
}

# ── Virtual environment ──────────────────────────────────────────────────
# Every install below runs this venv's own interpreter by path. Nothing is
# activated, so an already-active venv cannot redirect the install.

if [ -x "$VENV_DIR/bin/python" ]; then
    echo "✅ Virtual environment '$VENV_DIR' already exists."
    if ! confirm "   Install into it?"; then
        echo "Aborted."
        exit 0
    fi
    PY="$VENV_DIR/bin/python"
elif confirm "🐍 Create a virtual environment in '$VENV_DIR'?"; then
    echo "🔧 Creating virtual environment ..."
    python3 -m venv "$VENV_DIR"
    PY="$VENV_DIR/bin/python"
    "$PY" -m pip install --upgrade pip
    echo "✅ Virtual environment created."
else
    echo "⚠️  Skipping venv — installing into the current Python environment."
    PY="$(command -v python3)"
fi

# ── The torch stack ──────────────────────────────────────────────────────
# torch_stack.py chooses the set, refuses (exit 2, with the table) before
# installing anything where no set fits, and installs torch, torchvision,
# torchaudio, torchao and triton from it.

echo ""
"$PY" "$STACK" install --python "$PY"

# ── Remaining dependencies ───────────────────────────────────────────────
# The bulk install never sees the torch stack (torch_stack.py owns it) nor
# scenedetect/sam3/hpsv2 (installed --no-deps below; see install-deps.sh for
# why each needs it).

echo ""
echo "📦 Installing remaining dependencies ..."
TMP_REQ="$(mktemp)"
trap 'rm -f "$TMP_REQ"' EXIT
"$PY" "$STACK" filter "$REQ" "$TMP_REQ" --also scenedetect sam3 hpsv2
"$PY" -m pip install -r "$TMP_REQ"

# scenedetect's declared dependency is the GUI build `opencv-python`, which
# collides with the pinned `opencv-python-headless` (both ship the `cv2`
# module) - install it separately, without its deps.
# sam3's declared ceilings would block the resolve against our pins; its
# import works (test_sam3_imports_cleanly_despite_declared_hub_pin).
# hpsv2 declares its dev deps (`pytest ==7.2.0`, `pytest-split ==0.8.0`) as
# install requirements; it never imports them
# (test_hpsv2_works_under_a_runner_its_metadata_forbids).
for name in scenedetect sam3 hpsv2; do
    PIN="$(grep -iE "^[[:space:]]*${name}[[:space:]]*==" "$REQ" | sed -E 's/#.*$//' | tr -d '[:space:]' || true)"
    if [ -n "$PIN" ]; then
        echo "📦 Installing $PIN (--no-deps) ..."
        "$PY" -m pip install --no-deps "$PIN"
    fi
done

# ── The stack is still the one chosen ────────────────────────────────────

echo ""
"$PY" "$STACK" verify --python "$PY"

echo ""
echo "✅ Done — all dependencies installed."
