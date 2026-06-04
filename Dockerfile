# syntax=docker/dockerfile:1

# ── Stage 1: build the Angular SPA ───────────────────────────────────────
FROM node:20-bookworm AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build -- --configuration production

# ── Stage 2: runtime (CUDA 13.0 + Python 3.12) ──────────────────────────
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MRLN_CONTAINER=1 \
    MRLN_FRONTEND_DIST=/app/frontend/browser \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git ca-certificates \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Install PyTorch (CUDA 13.0) first, then the rest — mirrors backend/install.sh.
COPY backend/requirements.txt ./requirements.txt
RUN python -m pip install --break-system-packages \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu130 \
    && python -m pip install --break-system-packages -r requirements.txt

# Native shared libraries required by OpenCV (cv2) and friends at import time
# — the slim CUDA runtime base lacks them. Placed after the pip layer so
# editing this list never invalidates the cached dependency install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Backend source.
COPY backend/ /app/backend/

# Built SPA from stage 1.
COPY --from=frontend /build/dist/frontend/browser /app/frontend/browser

# Entrypoint.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
