# syntax=docker/dockerfile:1

# Global build arg: must be declared before the first FROM so the runtime stage's
# FROM can interpolate it. See the runtime stage below for the cu128/cu126 rationale.
ARG CUDA_BASE=12.8.1

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

# ── Stage 2: runtime (CUDA + Python 3.12) ───────────────────────────────
# CUDA target is parameterized so one Dockerfile builds two variants:
#   • cu128 (DEFAULT — the `latest` / version tags): ships Blackwell (sm_120)
#     kernels alongside Hopper/Ada/Ampere. Needs an R570+ host driver, but
#     Blackwell cards mandate R570+ anyway, so this natively covers the whole
#     modern GPU fleet. cu126 has NO Blackwell SASS → "no kernel image is
#     available for execution on the device" on RTX/PRO Blackwell cards.
#   • cu126 (the `-cu126` fallback tag): for legacy hosts pinned to R560–R565
#     drivers (no Blackwell). Build with:
#       --build-arg CUDA_BASE=12.6.3 --build-arg TORCH_CUDA=cu126
# Do NOT default to cu130 (CUDA 13): it needs R580+ and the 12→13 forward-compat
# layer breaks cuBLAS on older drivers (every GEMM → CUBLAS_STATUS_INVALID_VALUE).
# The base-image CUDA version also sets NVIDIA_REQUIRE_CUDA, which gates
# container startup on the host driver — keep CUDA_BASE's minor aligned with
# TORCH_CUDA so the gate matches the wheels' real driver floor.
# (CUDA_BASE is declared as a global ARG at the top of this file.)
FROM nvidia/cuda:${CUDA_BASE}-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # pypi.nvidia.com sporadically stalls mid-download on the large CUDA wheels.
    # A 120s socket timeout + retries turns an indefinite hang into an
    # error-and-retry so the build self-heals instead of wedging.
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    MRLN_CONTAINER=1 \
    MRLN_FRONTEND_DIST=/app/frontend/browser \
    PORT=8000

# Self-update coordinates. /app is cloned (below) as a real git checkout so the
# running container can pull + rebuild itself. REPO_URL is overridable at build
# time; the lowercase canonical remote is the default.
ARG REPO_URL=https://github.com/master-merlin/mrln-arcane-tuner.git
ARG GIT_BRANCH=main
ENV MRLN_GIT_REMOTE=${REPO_URL} \
    MRLN_GIT_BRANCH=${GIT_BRANCH} \
    MRLN_APP_DIR=/app

# apt-get upgrade patches base-OS CVEs (e.g. gnupg2 CVE-2025-68973) flagged by
# Docker Scout that ship in the CUDA base image.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git ca-certificates curl \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Node 24 LTS — enables runtime frontend rebuilds during self-update.
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (pinned versions, independent of the repo so this heavy layer
# stays cached across code changes). TORCH_CUDA selects the wheel build (cu128
# default; cu126 for the fallback tag). The app's own deps are installed from the
# cloned requirements.txt after the git checkout below — keeping /app empty until
# then so the clone (which requires an empty target) succeeds.
ARG TORCH_CUDA=cu128
RUN python -m pip install --break-system-packages \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
    # setuptools/wheel ship via apt in the base image (no pip RECORD), so pip
    # can't uninstall them to honor the pinned versions. Install pip-managed
    # copies with --ignore-installed first; requirements.txt then sees the
    # pinned versions already satisfied.
    && python -m pip install --break-system-packages --ignore-installed \
        setuptools==78.1.1 wheel==0.46.2 \
    # --ignore-installed leaves the old apt-managed setuptools/wheel on disk
    # alongside the pinned copies (it can't uninstall them — no pip RECORD), so
    # Docker Scout still flags the vulnerable 68.1.2/0.42.0 files. The pinned
    # copies in /usr/local win on sys.path; delete the stale apt copies and the
    # Debian bundled wheel so the vulnerable files leave the image entirely.
    && rm -rf /usr/lib/python3/dist-packages/setuptools* \
              /usr/lib/python3/dist-packages/pkg_resources* \
              /usr/lib/python3/dist-packages/wheel* \
              /usr/share/python-wheels/setuptools-*.whl

# Native shared libraries required by OpenCV (cv2) and friends at import time
# — the slim CUDA runtime base lacks them. Placed after the pip layer so
# editing this list never invalidates the cached dependency install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# /app is a real git checkout so the app can self-update at runtime. Private
# repo → pass the PAT as a build secret (id=git_token); falls back to an
# unauthenticated clone for public repos. The remote is reset to the clean
# REPO_URL afterward so no token is baked into the image's git config.
RUN --mount=type=secret,id=git_token \
    sh -c 'if [ -f /run/secrets/git_token ]; then \
              AUTH="https://$(cat /run/secrets/git_token)@$(echo "$REPO_URL" | sed -E "s#https?://##")"; \
            else AUTH="$REPO_URL"; fi; \
            git clone --branch "$GIT_BRANCH" "$AUTH" /app && \
            cd /app && git remote set-url origin "$REPO_URL"'
WORKDIR /app/backend
# Install the app's Python deps from the cloned checkout (torch is already in
# place from the cached layer above; pip sees it satisfied and skips it).
RUN python -m pip install --break-system-packages -r requirements.txt

# Built SPA from stage 1 (overwrites the cloned, unbuilt frontend dist path).
COPY --from=frontend /build/dist/frontend/browser /app/frontend/browser

# --- Ollama (local LLM sidecar for caption refinement) ---
# Installed best-effort; the entrypoint only launches it if the binary is present,
# so an install hiccup never blocks the app. zstd is REQUIRED: Ollama's installer
# now ships zstd-compressed tarballs and aborts extraction with "requires zstd"
# (which the best-effort `|| echo` then silently swallows, shipping no binary).
RUN apt-get update && apt-get install -y --no-install-recommends zstd \
    && rm -rf /var/lib/apt/lists/* \
    && ( curl -fsSL https://ollama.com/install.sh | sh \
         || echo "WARN: ollama install failed; sidecar will be skipped at runtime" )

# Entrypoint.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
