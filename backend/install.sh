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

# ── PyTorch (CUDA 13.0) ─────────────────────────────────────────────────

echo ""
echo "🔧 Installing PyTorch 2.10.0 + CUDA 12.6 ..."
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu126

# ── Remaining dependencies ───────────────────────────────────────────────

echo ""
echo "📦 Installing remaining dependencies ..."
pip install -r requirements.txt

echo ""
echo "✅ Done — all dependencies installed."
