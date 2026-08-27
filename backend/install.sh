#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# MRLN Arcane Tuner — Linux install script
#
# Optionally creates a virtual environment, installs PyTorch with CUDA 13.0
# from the official PyTorch wheel index, then installs all remaining
# dependencies. PEP 508 markers in requirements.txt automatically handle
# platform-specific packages.
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV_DIR="venv"

# ── Virtual environment ──────────────────────────────────────────────────

if [ -d "$VENV_DIR" ]; then
    echo "✅ Virtual environment '$VENV_DIR' already exists."
    read -rp "   Activate it and continue? [Y/n] " answer
    answer="${answer:-Y}"
    if [[ "$answer" =~ ^[Nn]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    source "$VENV_DIR/bin/activate"
else
    read -rp "🐍 Create a virtual environment in './$VENV_DIR'? [Y/n] " answer
    answer="${answer:-Y}"
    if [[ "$answer" =~ ^[Nn]$ ]]; then
        echo "⚠️  Skipping venv — installing into current Python environment."
    else
        echo "🔧 Creating virtual environment ..."
        python3 -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip
        echo "✅ Virtual environment created and activated."
    fi
fi

# ── PyTorch (CUDA 13.0, split stack) ─────────────────────────────────────

echo ""
echo "🔧 Installing PyTorch 2.12.1 + torchvision 0.27.1 (CUDA 13.0) ..."
pip install torch==2.12.1 torchvision==0.27.1 \
    --index-url https://download.pytorch.org/whl/cu130

# torchaudio has no 2.12-series wheel yet (maintenance mode) and its own
# metadata pins torch==2.11.0, so it MUST be installed --no-deps or pip would
# downgrade torch back to 2.11.0.
echo "🔧 Installing torchaudio 2.11.0 (--no-deps; declares torch==2.11.0) ..."
pip install torchaudio==2.11.0 --no-deps \
    --index-url https://download.pytorch.org/whl/cu130

# ── Remaining dependencies ───────────────────────────────────────────────
# torch/torchvision/torchaudio (installed above) and scenedetect/sam3 (need
# --no-deps below) are excluded from this bulk install — see install-deps.sh
# for the full rationale (this mirrors its filter minus triton/triton-windows
# — local venvs need those from requirements; only the container filters them
# to protect its baked 2.11-matched copy).

echo ""
echo "📦 Installing remaining dependencies ..."
TMP_REQ="$(mktemp)"
grep -ivE '^[[:space:]]*(scenedetect|sam3|hpsv2|torch|torchvision|torchaudio)([[:space:]=<>!~#]|$)' requirements.txt > "$TMP_REQ"
pip install -r "$TMP_REQ"
rm -f "$TMP_REQ"

# scenedetect's declared dependency is the GUI build `opencv-python`, which
# collides with the pinned `opencv-python-headless` (both ship the `cv2`
# module) — install it separately, without its deps.
SD="$(grep -iE '^[[:space:]]*scenedetect[[:space:]]*==' requirements.txt | sed -E 's/#.*$//' | tr -d '[:space:]' || true)"
if [ -n "$SD" ]; then
    echo "📦 Installing $SD (--no-deps) ..."
    pip install --no-deps "$SD"
fi

# sam3 declares `huggingface-hub<1.0,>=0.30.0`, but this repo pins
# huggingface-hub==1.27.0 (transformers 5.x requires it). The import works
# fine under hub 1.x - its declared ceiling is just stale (see
# test_sam3_imports_cleanly_despite_declared_hub_pin) - so install it
# separately, without its deps, rather than letting it block the resolve.
S3="$(grep -iE '^[[:space:]]*sam3[[:space:]]*==' requirements.txt | sed -E 's/#.*$//' | tr -d '[:space:]' || true)"
if [ -n "$S3" ]; then
    echo "📦 Installing $S3 (--no-deps) ..."
    pip install --no-deps "$S3"
fi

# hpsv2 declares `pytest ==7.2.0` and `pytest-split ==0.8.0` as INSTALL
# requirements - its dev deps leaked into its metadata. It never imports either
# and registers no pytest11 hook, so the pins are inert (see
# test_hpsv2_works_under_a_runner_its_metadata_forbids), but left in the bulk
# resolve they abort it against our own pytest pin. All 18 of its real deps are
# already in requirements.txt, so nothing is lost by skipping its metadata.
HP="$(grep -iE '^[[:space:]]*hpsv2[[:space:]]*==' requirements.txt | sed -E 's/#.*$//' | tr -d '[:space:]' || true)"
if [ -n "$HP" ]; then
    echo "📦 Installing $HP (--no-deps) ..."
    pip install --no-deps "$HP"
fi

echo ""
echo "✅ Done — all dependencies installed."
