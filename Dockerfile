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
#
# Split-stack: the image pins torch 2.11.0 here (cu128/R570+ and cu126/R560+
# both publish it, preserving today's exact host-driver floors for both
# variants), while the LOCAL dev venv runs torch 2.12.1+cu130 instead (newer
# CUDA, no matching Linux wheel index parity yet). requirements.txt's torch/
# torchvision/torchaudio/triton pins document the LOCAL 2.12.1 stack only —
# install-deps.sh filters those lines out of its requirements install so
# neither this image build nor the runtime self-update ever clobbers the
# trio baked into this layer.
ARG TORCH_CUDA=cu128
RUN python -m pip install --break-system-packages \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
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
# ffmpeg: the video-clip split pipeline (LosslessCut cutlist / scene-detect →
# training clips) shells out to the ffmpeg CLI for stream-copy / libx264
# re-encode. PyAV's bundled libs cover decode/probe/encode, but a real ffmpeg
# binary (found first by shutil.which, ahead of the imageio-ffmpeg fallback)
# is the robust path for re-encode + AAC mux.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Ollama (local LLM sidecar for caption refinement) ---
# Installed best-effort; the entrypoint only launches it if the binary is present,
# so an install hiccup never blocks the app. zstd is REQUIRED: Ollama's installer
# now ships zstd-compressed tarballs and aborts extraction with "requires zstd"
# (which the best-effort `|| echo` then silently swallows, shipping no binary).
# Placed BEFORE the git clone (it's independent of app code) so a re-clone for a
# new GIT_SHA never re-downloads Ollama's large tarball.
#
# SUPPLY CHAIN — read before changing:
#   * Set OLLAMA_VERSION *and* OLLAMA_SHA256 and the build fetches that exact
#     release tarball and REFUSES to proceed if the digest does not match. This
#     is the path release builds should use.
#   * Leave them empty (the default) and the build falls back to piping
#     ollama.com/install.sh into a root shell. That is an UNPINNED third party
#     executing arbitrary code at build time. It is deliberately still the
#     default so a plain `docker build` keeps working, and it is deliberately
#     loud about it — but it is a real exposure, not a formality.
#   * INSTALL_OLLAMA=0 skips the layer entirely; the app degrades cleanly
#     because the entrypoint already probes for the binary.
# The asset is `ollama-linux-amd64.tar.zst` (zstd, NOT gzip): Ollama replaced the
# old `ollama-linux-amd64.tgz` with split `.tar.zst` bundles and no longer ships
# the `.tgz` at all — pinning against that stale name 404s on every current
# release, which is why the extract below is `tar --zstd` and why zstd is
# installed above for BOTH paths, not just the unpinned one.
# Getting the digest: take it from the release's own `sha256sum.txt` asset
# (`curl -fsSL https://github.com/ollama/ollama/releases/download/<tag>/sha256sum.txt`)
# and use the `ollama-linux-amd64.tar.zst` line. No digest is hardcoded
# here on purpose — a checksum nobody verified is worse than none, because it
# reads as proof.
ARG INSTALL_OLLAMA=1
ARG OLLAMA_VERSION=
ARG OLLAMA_SHA256=
RUN if [ "$INSTALL_OLLAMA" != "1" ]; then \
        echo "[build] INSTALL_OLLAMA=$INSTALL_OLLAMA — skipping Ollama layer"; \
    else \
      apt-get update && apt-get install -y --no-install-recommends zstd \
      && rm -rf /var/lib/apt/lists/* \
      && if [ -n "$OLLAMA_VERSION" ] && [ -n "$OLLAMA_SHA256" ]; then \
             echo "[build] Ollama ${OLLAMA_VERSION} — pinned, verifying sha256"; \
             curl -fsSL -o /tmp/ollama.tar.zst \
               "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
             && echo "${OLLAMA_SHA256}  /tmp/ollama.tar.zst" | sha256sum -c - \
             && tar -C /usr/local --zstd -xf /tmp/ollama.tar.zst \
             && rm -f /tmp/ollama.tar.zst; \
         elif [ -n "$OLLAMA_VERSION" ] || [ -n "$OLLAMA_SHA256" ]; then \
             echo "ERROR: OLLAMA_VERSION and OLLAMA_SHA256 must be set together." >&2; \
             echo "       One without the other is an unverified download wearing a pin." >&2; \
             exit 1; \
         else \
             echo "WARN: Ollama is being installed UNPINNED from ollama.com/install.sh."; \
             echo "WARN: set OLLAMA_VERSION + OLLAMA_SHA256 for a verified build."; \
             ( curl -fsSL https://ollama.com/install.sh | sh \
               || echo "WARN: ollama install failed; sidecar will be skipped at runtime" ); \
         fi; \
    fi

# /app is a real git checkout so the app can self-update at runtime. Private
# repo → pass the PAT as a build secret (id=git_token); falls back to an
# unauthenticated clone for public repos. The remote is reset to the clean
# REPO_URL afterward so no token is baked into the image's git config.
# GIT_SHA is REQUIRED and is what the image is built from. Building from a bare
# branch name meant two builds of the "same" image could contain different code
# with nothing in the image recording which — not reproducible, and not
# auditable after the fact. The build now fails closed rather than silently
# tracking a moving branch.
#
# It also subsumes the old CACHEBUST arg: this clone (and the pip install below)
# is the only part that depends on app code, and a different GIT_SHA changes
# this RUN's command string, so Docker invalidates the layer on its own. The
# torch, Ollama and apt layers above stay cached. Passing a timestamp to force a
# rebuild is no longer possible, which is the point — the input is the commit.
#
# The branch is still cloned and HEAD stays ON it (reset --hard, not
# `checkout --detach`): the self-update service reports
# `git rev-parse --abbrev-ref HEAD`, which answers the literal string "HEAD" on a
# detached checkout, so detaching would have shown the branch as "HEAD" in the
# UI. Content is pinned; branch identity is preserved; `fetch origin <branch> +
# reset --hard` in SelfUpdateService still works unchanged.
ARG GIT_SHA
RUN --mount=type=secret,id=git_token \
    sh -c 'case "$GIT_SHA" in \
             *[!0-9a-fA-F]*|"") \
               echo "ERROR: --build-arg GIT_SHA=<full 40-hex commit> is required." >&2; \
               echo "       Build from a commit, not a moving branch." >&2; \
               exit 1;; \
           esac; \
           [ ${#GIT_SHA} -eq 40 ] || { \
             echo "ERROR: GIT_SHA must be the FULL 40-character commit sha (got ${#GIT_SHA} chars)." >&2; \
             echo "       A short sha is ambiguous and cannot be verified below." >&2; \
             exit 1; }; \
           if [ -f /run/secrets/git_token ]; then \
             AUTH="https://$(cat /run/secrets/git_token)@$(echo "$REPO_URL" | sed -E "s#https?://##")"; \
           else AUTH="$REPO_URL"; fi; \
           git clone --branch "$GIT_BRANCH" "$AUTH" /app && \
           cd /app && git remote set-url origin "$REPO_URL" && \
           git reset --hard "$GIT_SHA" && \
           [ "$(git rev-parse HEAD)" = "$GIT_SHA" ] || { \
             echo "ERROR: /app is not at $GIT_SHA after reset." >&2; exit 1; }'
WORKDIR /app/backend
# Install the app's Python deps from the cloned checkout. install-deps.sh
# filters the torch/torchvision/torchaudio/triton/triton-windows lines out of
# requirements.txt before installing, so this step never touches (or clobbers)
# the trio already baked into the cached layer above.
# It also installs requirements.txt EXCEPT scenedetect and sam3, then
# installs each of those with --no-deps: scenedetect's declared GUI
# `opencv-python` dep would otherwise clobber the pinned
# `opencv-python-headless`, and sam3's declared `huggingface-hub<1.0` ceiling
# is stale (see test_sam3_imports_cleanly_despite_declared_hub_pin) and would
# otherwise abort the resolve against the pinned hub 1.27.0. The runtime
# self-update reuses the same script, so the build and self-update installs
# never diverge.
RUN bash install-deps.sh

# --- hpsv2's BPE vocabulary, baked in as root ---
# hpsv2 vendors its own open_clip, whose tokenizer resolves this file with a
# HARDCODED package-relative path (`os.path.join(os.path.dirname(__file__),
# "bpe_simple_vocab_16e6.txt.gz")`) and whose wheel does not ship it. There is
# no environment variable that redirects it, so it MUST live inside
# site-packages. `apply_hpsv2_patches()` in backend/app/core/compat.py fetched
# it on first boot, which worked only while the app ran as root: under the
# non-root user the write is EACCES, the failure is swallowed into a warning,
# and HPSv2 scoring is silently dead.
# Doing it here, as root at build time, is the fix. It also makes the runtime
# path a no-op — compat.py returns early when the file is already present — and
# stops every fresh container needing github.com reachable at boot.
# Chowning site-packages instead would "work" and would be wrong: the app must
# not be able to rewrite the libraries it is about to execute.
# The destination is resolved from the INSTALLED DISTRIBUTION, never hardcoded:
# a base-image Python bump moves dist-packages, and a literal python3.12 path
# would silently drop the file where nothing reads it.
# The gzip probe is the integrity check that matters here — the failure this
# guards against is a 200-with-an-HTML-error-page, which `curl -f` does not
# catch and which would sit in the image looking like a vocabulary.
RUN set -eu; \
    dest="$(python -c 'import importlib.metadata as m, pathlib; print(pathlib.Path(str(m.distribution("hpsv2").locate_file(""))) / "hpsv2" / "src" / "open_clip")')"; \
    [ -n "$dest" ] || { echo "ERROR: could not resolve the hpsv2 package directory." >&2; exit 1; }; \
    mkdir -p "$dest"; \
    curl -fsSL -o "$dest/bpe_simple_vocab_16e6.txt.gz" \
      "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"; \
    python -c 'import gzip,sys; gzip.open(sys.argv[1],"rb").read(1)' "$dest/bpe_simple_vocab_16e6.txt.gz"; \
    echo "[build] hpsv2 BPE vocab baked into $dest"

# Built SPA from stage 1 (overwrites the cloned, unbuilt frontend dist path).
COPY --from=frontend /build/dist/frontend/browser /app/frontend/browser

# ── Non-root runtime user ────────────────────────────────────────────────
# The app ran as root, and this image is not a sandbox: /app is a live git
# checkout that the running container can pull into and rebuild (SelfUpdateService),
# so "root" and "can rewrite the code it is about to execute" were the same
# principal. Dropping to a dedicated UID does not make self-update safe on its
# own, but it stops a compromise there from owning the whole container.
#
# UID 10001 is deliberately outside the range distributions hand out: Ubuntu
# 24.04 already ships a user at 1000, and a collision would silently share an
# identity with it. Fixed rather than auto-assigned so ownership on a mounted
# volume stays stable across image rebuilds — an auto-assigned UID that shifts
# makes yesterday's data unreadable.
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --gid "$APP_GID" mrln \
    && useradd --uid "$APP_UID" --gid "$APP_GID" --create-home --shell /bin/bash mrln \
    && chown -R "$APP_UID:$APP_GID" /app

# Entrypoint. Stays root-owned and NOT writable by the app user: it runs as root
# (see below) to fix volume ownership before dropping privileges, so an app-user
# write to it would be a direct path back to root.
COPY entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh && chown root:root /entrypoint.sh

# NOTE: no `USER mrln` here, on purpose. The entrypoint starts as root ONLY to
# chown the freshly-mounted data volume (which arrives root-owned and would
# otherwise be unwritable), then drops to APP_UID via setpriv before exec'ing
# uvicorn — so nothing the app runs holds root. Setting `USER` here instead
# would make that first-boot chown impossible and leave the volume unusable.
# The app user's identity is passed through the environment so the entrypoint
# does not have to hardcode it in two places.
ENV MRLN_APP_UID=${APP_UID} \
    MRLN_APP_GID=${APP_GID}

# STILL DEFERRED, named so the deferral stays a decision rather than an
# oversight: base-image digest pinning (`FROM nvidia/cuda@sha256:...`), SBOM
# generation, build provenance attestation, and image signing. Each is a real
# gap; none is closed here.

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
