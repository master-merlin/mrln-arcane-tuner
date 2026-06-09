# syntax=docker/dockerfile:1

# ── Stage 1: build the Angular SPA ───────────────────────────────────────
FROM node:24-bookworm AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
# --legacy-peer-deps: @lucide/angular@1.16.0 still pins its peer range to
# @angular/common 17–21, but the project runs Angular 22. This matches the
# local install that produced package-lock.json.
RUN npm ci --legacy-peer-deps
COPY frontend/ ./
RUN npm run build -- --configuration production

# ── Stage 2: runtime (CUDA 12.6 + Python 3.12) ──────────────────────────
# CUDA 12.6 (not 13.0): CUDA 13 needs an R580+ host driver for native support.
# Common cloud GPU hosts (e.g. RunPod) still ship R565-class drivers (CUDA
# 12.7), where CUDA 13 only works via a forward-compat layer that leaves cuBLAS
# broken — every GEMM fails with CUBLAS_STATUS_INVALID_VALUE. 12.6 is supported
# natively by R560+ and stays minor-version compatible across the 12.x fleet.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MRLN_CONTAINER=1 \
    MRLN_FRONTEND_DIST=/app/frontend/browser \
    PORT=8000

# apt-get upgrade patches base-OS CVEs (e.g. gnupg2 CVE-2025-68973) flagged by
# Docker Scout that ship in the CUDA base image.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git ca-certificates \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Install PyTorch (CUDA 12.6) first, then the rest — mirrors backend/install.sh.
COPY backend/requirements.txt ./requirements.txt
RUN python -m pip install --break-system-packages \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu126 \
    # setuptools/wheel ship via apt in the base image (no pip RECORD), so pip
    # can't uninstall them to honor the pinned versions. Install pip-managed
    # copies with --ignore-installed first; requirements.txt then sees the
    # pinned versions already satisfied.
    && python -m pip install --break-system-packages --ignore-installed \
        setuptools==78.1.1 wheel==0.46.2 \
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
