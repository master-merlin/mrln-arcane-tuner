# MRLN Arcane Tuner

> **Dataset-first LoRA training studio** — because a great LoRA starts with a great dataset.

`v0.6.8-beta` · PyTorch 2.10 · CUDA 13.0 local / 12.8 container (+cu126 fallback) · Angular 22 · Node 24 · FastAPI

**Author:** [master-merlin](https://github.com/master-merlin) · **Repository:** [github.com/master-merlin/mrln-arcane-tuner](https://github.com/master-merlin/mrln-arcane-tuner)

---

## Why This Exists

Every LoRA trainer will tell you: **the dataset is 90 % of the result**. Yet most training tools treat dataset management as an afterthought — a folder of images you dump somewhere and hope for the best.

MRLN Arcane Tuner started as a personal experiment to fix that. The goal was simple: build a workflow where dataset curation is the **heart and center** of the process, not a chore you skip through to get to training. From smart cropping and image adjustments to duplicate detection and stacked LUT color grading — the dataset pipeline is where most of the R&D effort lives.

The training engine, job management, and LoRA tools grew organically around that core — because once your data is good, training should be straightforward.

---

## Acknowledgments

This project wouldn't exist without the incredible open-source community that pioneered LoRA training for diffusion models. A sincere thank you to:

- **[kohya-ss](https://github.com/kohya-ss/sd-scripts)** — Pioneered the entire LoRA training ecosystem. The Kohya metadata format (`ss_*` keys) is the de-facto standard for LoRA checkpoint interoperability, and MRLN Arcane Tuner writes these keys for full compatibility with ComfyUI, A1111, and other inference tools.
- **[Ostris](https://github.com/ostris/ai-toolkit)** — Timestep sampling strategies for flow-matching models are derived from ai-toolkit (MIT License). Credited in [`flux2/trainer.py`](backend/app/engine/models/families/flux2/trainer.py).
- **[Nerogar](https://github.com/Nerogar/OneTrainer)** — Inspiration for the unified multi-model trainer architecture that supports multiple model families through a single pipeline.
- **[Hugging Face / diffusers](https://github.com/huggingface/diffusers)** — Key mapping logic for PEFT-to-BFL LoRA conversion is derived from `lora_conversion_utils.py`. Credited in [`flux2/saver.py`](backend/app/engine/models/families/flux2/saver.py).
- **[rockerBOO / lora-inspector](https://github.com/rockerBOO/lora-inspector)** — Inspiration for the LoRA inspection tooling (format detection, weight statistics, layer analysis). Credited in [`lora_tools.py`](backend/app/engine/utils/lora_tools.py).
- **[NyxAwroo / IMG-Dataset-Refiner](https://github.com/NyxAwroo/IMG-Dataset-Refiner)** — Inspiration for the model-aware caption workflow — per-model caption variants, architecture-aware token budgets, tag analytics, and LLM-assisted caption refinement.

> **Note:** MRLN Arcane Tuner is a personal experiment and is **not intended to compete** with any of these projects. They are community pillars. This tool simply explores a different angle — dataset quality first.

---

## Installation

### Prerequisites

| Requirement   | Version              | Notes                     |
| ------------- | -------------------- | ------------------------- |
| Python        | 3.12+                | With `venv` support       |
| NVIDIA GPU    | Ampere+ (RTX 30xx)   | CUDA 13.0 (driver R580+); see container note below for cloud hosts |
| Node.js       | 24+ (LTS)            | Required by Angular 22 / TS 6 |
| npm           | 10+                  | Comes with Node.js        |

### Scripted Install (Recommended)

The install scripts create a virtual environment, install PyTorch with CUDA support, and install all Python dependencies.

**Windows:**
```cmd
cd backend
install.bat
```

**Linux / macOS:**
```bash
cd backend
chmod +x install.sh
./install.sh
```

Then install the frontend:
```bash
cd frontend
npm install
```

### Manual Install

```bash
# 1. Create and activate a virtual environment
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Linux:   source venv/bin/activate

# 2. Install PyTorch with CUDA 13.0 (local dev; needs an R580+ driver).
#    The published container ships CUDA 12.8 (cu128, Blackwell-capable) with a
#    cu126 fallback for older host drivers — see "Run as a container" below.
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130

# 3. Install remaining Python dependencies
pip install -r requirements.txt

# 4. Install frontend dependencies
cd ../frontend
npm install
```

### Starting the Application

**Backend:**
```cmd
cd backend
start_backend.bat          # Windows
./start_backend.sh         # Linux / macOS
```

The backend creates required directories (`datasets/`, `models/`, `models/upscale/`, `outputs/`) and a default `settings.json` on first launch.

**Frontend:**
```bash
cd frontend
npm run start
```

Or enable **auto-start** in Server Settings — the backend will launch the frontend and open your browser automatically on first start.

The frontend runs on `http://localhost:4200` by default. Both ports are configurable in Server Settings.

---

## Run as a container on RunPod

The app ships as a single Docker image that serves the API, the WebSocket log
stream, media, and the Angular UI **from one port over HTTPS**. RunPod is the
example provider here, but the same image works on any host with an HTTPS
ingress proxy.

### How it works

- One `uvicorn` process serves everything at one origin: `/` (UI), `/api`,
  `/api/ws`, `/media`. The frontend uses **same-origin** URLs, so it works
  behind RunPod's per-port proxy with zero config.
- **HTTPS is handled by RunPod's proxy** (Cloudflare terminates TLS). The
  container itself only serves plain HTTP on `0.0.0.0` — no certificates to
  manage.
- Persistent data (SQLite DB, datasets, models, outputs) lives on a mounted
  volume so it survives pod restarts.

### 1. Get the image

The published image is on Docker Hub — you can use it directly, no build
required:

```
mastermerlin/mrln-arcane-tuner:latest             # rolling latest (CUDA 12.8 / cu128)
mastermerlin/mrln-arcane-tuner:0.6.8-beta        # pinned version (CUDA 12.8 / cu128)
mastermerlin/mrln-arcane-tuner:0.6.8-beta-cu126  # fallback for legacy R560–R565 drivers
```

The default image bundles **CUDA 12.8 (cu128) · PyTorch 2.10 · Python 3.12**
(runtime) and a **Node 24 / Angular 22** production build of the UI. cu128 ships
**Blackwell (sm_120/sm_100)** kernels plus Hopper/Ada/Ampere, and needs an
**R570+** host driver — which Blackwell cards require anyway, so it covers the
whole modern fleet. The `-cu126` tag is for older hosts pinned to **R560–R565**
drivers (no Blackwell support).

> **"no kernel image is available for execution on the device"** on a Blackwell
> card (e.g. RTX PRO 6000) means you're on an older cu126 image — pull `latest`
> (cu128).

**Building your own** (only needed if you've modified the code). The CUDA target
is parameterized via build args (default cu128):

```bash
# Primary (cu128 — Blackwell + modern fleet). Tag with version and latest.
docker build -t mastermerlin/mrln-arcane-tuner:0.6.8-beta -t mastermerlin/mrln-arcane-tuner:latest .
docker push mastermerlin/mrln-arcane-tuner:0.6.8-beta
docker push mastermerlin/mrln-arcane-tuner:latest

# Fallback (cu126 — legacy R560–R565 drivers).
docker build --build-arg CUDA_BASE=12.6.3 --build-arg TORCH_CUDA=cu126 \
    -t mastermerlin/mrln-arcane-tuner:0.6.8-beta-cu126 .
docker push mastermerlin/mrln-arcane-tuner:0.6.8-beta-cu126
```

### 2. Create the pod on RunPod

- **GPU:** any NVIDIA Ampere+ GPU, **including Blackwell** (RTX 50xx / RTX PRO
  6000 Blackwell). The default image is built for **CUDA 12.8** (cu128) and
  needs an **R570+** host driver — standard on current cloud hosts and mandatory
  for Blackwell anyway. On a legacy host stuck on **R560–R565**, use the
  `:0.6.8-beta-cu126` tag instead (no Blackwell support). Avoid CUDA 13 in the
  container: it needs R580+ and its forward-compat layer breaks cuBLAS on older
  drivers.
- **Container image:** `mastermerlin/mrln-arcane-tuner:latest`
- **Volume (strongly recommended):** attach a **network volume mounted at
  `/workspace`**. The SQLite DB, `datasets/`, `models/`, `outputs/`, **and the
  Hugging Face cache** (`hf-cache/`) are stored there, so base models /
  encoders download only once and your work survives restarts. *Without a
  volume the container runs but all data — including multi-GB model downloads —
  is lost when the pod stops.*
- **Expose HTTP port:** add `8000` to the pod's **Expose HTTP Ports** field.
- **Environment variables:**

  | Variable | Purpose | Default |
  |---|---|---|
  | `MRLN_AUTH_TOKEN` | Require this token to access the app (recommended — the proxy URL is public). | _unset = open_ |
  | `PORT` | Internal port (match the exposed HTTP port). | `8000` |
  | `MRLN_DATA_DIR` | Persistence root (DB, datasets, models, outputs, HF cache). | `/workspace` |
  | `HF_TOKEN` | Hugging Face token — set it if you train/pull **gated** models (e.g. some FLUX weights). | _unset_ |
  | `HF_HOME` | Hugging Face cache location. Auto-set to `$MRLN_DATA_DIR/hf-cache` so downloads persist on the volume — only override to relocate the cache. | `/workspace/hf-cache` |
  | `CUDA_VISIBLE_DEVICES` | Pin a specific GPU on multi-GPU pods. | _all_ |

### 3. Open the app

RunPod exposes the port at:

```
https://[POD_ID]-8000.proxy.runpod.net
```

Open it; if `MRLN_AUTH_TOKEN` is set you'll get a sign-in page — enter the
token once and a cookie keeps you signed in.

### Notes & caveats

- **Proxy timeout:** RunPod's Cloudflare proxy closes any single request that
  takes longer than ~100 seconds. The log WebSocket and normal API calls are
  fine; training runs server-side and is unaffected. Very large single uploads
  through the proxy can hit this limit — for big datasets, place them on the
  `/workspace` volume directly.
- **GPU selection:** the app auto-detects the GPU. To pin one on a multi-GPU
  pod, set `CUDA_VISIBLE_DEVICES`.
- **Other providers:** any platform that exposes a container HTTP port over
  HTTPS works — point its ingress at port `8000` and (optionally) set
  `MRLN_AUTH_TOKEN`.

---

## Architecture

MRLN Arcane Tuner is a full-stack application with a FastAPI backend and an Angular frontend connected via REST API and WebSocket.

```
┌──────────────────────────────────────────────────────┐
│                  Angular 21 SPA                      │
│    43 Standalone Components · Signals · Tailwind     │
└──────────────────┬────────────────┬──────────────────┘
                   │ REST           │ WebSocket
┌──────────────────┴────────────────┴──────────────────┐
│                FastAPI Backend                       │
│    9 Route Domains · Structured Logging · Middleware │
├─────────────┬────────────┬───────────┬───────────────┤
│ Dataset     │ Training   │ AI        │ System        │
│ Manager     │ Engine     │ Services  │ Settings      │
├─────────────┴────────────┴───────────┴───────────────┤
│          PyTorch · Diffusers · PEFT · SQLite         │
└──────────────────────────────────────────────────────┘
```

For the full component and API route inventory, see [`documentation/ARCHITECTURE.md`](documentation/ARCHITECTURE.md).

---

## Features

### 🎯 Dataset Management — The Heart of the Tool

The dataset pipeline is designed to get your images from raw collection to training-ready with maximum control.

#### Multi-Dataset Scanning
Automatic directory scanning with image–caption pairing. Supports `.png`, `.jpg`, `.webp` images with paired `.txt` caption files. Incremental and full rescan modes.

#### Image Manipulation
A 9-stage non-destructive adjustment pipeline that processes images before training:

| Stage                              | Controls                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------- |
| Brightness, Contrast, Saturation   | Standard exposure adjustments                                             |
| Hue                                | Global color rotation                                                     |
| Curves                             | Per-channel Bézier curves (RGB + individual R/G/B) with dropdown presets  |
| Stacked LUT                        | Apply `.cube` LUT files with adjustable strength blending                 |
| HSL                                | Selective hue/saturation/lightness control per color range                |
| Sharpness                          | Detail enhancement                                                        |
| Noise                              | Grain control                                                             |

All adjustments include a real-time canvas preview with live histogram visualization.

#### Smart Cropping
Resolution-aware aspect-ratio bucketing with visual crop preview. Images are automatically grouped into optimal width×height buckets (divisible by 32) matching your target training resolutions.

**Bucketing modes:**
- **Kohya**: Each image appears in one bucket (closest aspect ratio match)
- **Multi**: Each image appears in every qualifying bucket for maximum latent diversity

#### Dataset Analysis
- **Harmonization analysis** — evaluate color and exposure consistency across the dataset
- **Duplicate detection** — perceptual hash similarity scoring to find near-duplicate images
- **Per-image enable/disable** — toggle individual images in or out of training without deleting them

#### Versioning & Caching
Dataset version bumping invalidates latent and text embedding caches, ensuring training always uses current image state. Cache admin UI lets you inspect and purge cached data.

#### Neural Upscaling
Tiled neural upscaling using ESRGAN and SwinIR models for images that need higher resolution before training.

---

### 🤖 AI Services

Integrated AI models for automated dataset annotation, running as GPU-backed batch services:

#### Auto-Captioning
| Model               | Specialty                                                                           |
| ------------------- | ----------------------------------------------------------------------------------- |
| **Florence-2**      | Fast, reliable descriptions. Multiple detail levels.                                |
| **JoyCaption Beta** | 12 caption types (descriptive, prompt-style, tag lists). Extensive control options. |
| **Qwen3-VL**        | Large vision-language model for nuanced descriptions. Configurable variant (4B/8B). |
| **Youtu-VL**        | Tencent's vision-language model with fine-grained parameter control.                |

All models support batch processing with real-time progress, custom system prompts, and per-model template management.

#### Auto-Masking
| Model       | Approach                                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------- |
| **SAM 3**   | Text-prompted segmentation (Meta's Segment Anything). Multi-mask output.                          |
| **RemBG**   | Background removal with 15+ model variants (BiRefNet, ISNet, U2Net, BRIA). Alpha matting support. |

Supports batch mass-apply across entire datasets.

---

### ⚙️ Training Configuration

Training is configured through a **dynamic JSON Schema-driven UI** — the form auto-generates from model-family definitions, so new fields appear automatically without frontend code changes.

#### Supported Model Families

| Family               | Architecture                  | Text Encoder            | Notes                               |
| -------------------- | ----------------------------- | ----------------------- | ----------------------------------- |
| **SDXL**             | UNet + DDPMScheduler          | Dual CLIP (TE1 + TE2)   | Epsilon prediction, Min-SNR gamma   |
| **Flux.1**           | Transformer + Flow Matching   | Qwen3                   | BFL-format export for ComfyUI       |
| **Flux.2 (Klein)**   | Transformer + Flow Matching   | Qwen3                   | No guidance embed, packed latents   |

#### 🧪 Experimental: Edit (Paired) & Video Training

> ⚠️ **Available but experimental.** These features ship and pass the test
> suite, but are still being validated on real end-to-end training runs. Treat
> results as beta-quality and expect rough edges.

- **Edit / paired-image training** — two-image (control → target) edit datasets
  and captioning for instruction-edit models (Flux.1 Kontext, Qwen-Image-Edit):
  paired-pair production, two-image VLM captioning, and edit-aware training.
- **Video training** — LoRA training for video diffusion models: **WAN 2.1**
  (T2V / I2V), **WAN 2.2** (dual high/low-noise experts, single-run
  auto-switch, dual LoRA output), and **LTX 2.3** (T2V / I2V, optional joint
  audio). Backed by a video dataset-curation layer: lazy clip preview,
  LosslessCut cutlist import, scene-detect auto-split, non-destructive in-app
  trim, per-clip health checks, and multi-frame auto-captioning.

#### Optimizers

MRLN Arcane Tuner supports a wide range of optimizers, from proven defaults to cutting-edge research:

**Standard (Adam-family):**
- **AdamW** / **AdamW8bit** — Reliable baselines. 8bit variant uses ~50% less optimizer VRAM.
- **RAdam** — Automatic variance-based warmup, no manual warmup steps needed.
- **StableAdamW** — RMS-based gradient scaling eliminates need for gradient clipping.

**Adaptive Learning Rate:**
- **Prodigy** — Automatically discovers optimal LR. Set learning rate to `1.0`.
- **ProdigyPlusSF** — Prodigy + Schedule-Free + factored second moments. Features cautious updates, OrthoGrad, FOCUS, SPEED, and per-group step size adaptation. Lowest memory overhead of the adaptive optimizers.

**Memory-Efficient:**
- **Lion** — Sign-based, ~50% less state memory than AdamW.
- **Adafactor** — Factored second moments for extremely low memory on large models. Supports relative step scaling.

**Second-Order:**
- **SophiaH / SophiaG** — Hutchinson trace / Gauss-Newton Hessian approximation. Faster convergence on some tasks. ⚠️ High VRAM — may OOM on 9B+ models.
- **Shampoo** — Kronecker-factored preconditioned gradients. ⚠️ High VRAM.

**Advanced:**
- **AdEMAMix** — Dual EMA (fast β1 + slow β3) for long-horizon convergence.

#### Timestep Sampling Strategies

| Strategy           | Best For                                                                                                             |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| **logit_normal**   | Default for flow matching (Flux). Mid-range focus.                                                                   |
| **uniform**        | Equal probability baseline.                                                                                          |
| **sigmoid**        | Simplified logit-normal with fixed parameters.                                                                       |
| **cosmap**         | Cosine-mapped smooth distribution.                                                                                   |
| **mode**           | Configurable mid-range emphasis.                                                                                     |
| **flux_shift**     | Resolution-dependent shifting. Original Flux recipe.                                                                 |
| **radc**           | Resolution-Aware Dynamic Curriculum. Progressive coarse-to-fine learning. Best for multi-phase curriculum training.  |

**RADC** is a standout feature — it progressively shifts the noise focus from high-noise (learning structure and composition) to low-noise (refining details and textures) over the course of training, with optional resolution-aware weighting for multi-resolution datasets.

#### LoRA Parameters
- **Network rank** (dim): 4–128+. Controls adapter capacity. Rank 16 is a solid default.
- **Network alpha**: Scaling factor for LoRA influence. Start with `alpha = rank / 2`.
- **Targeted layers**: Select specific transformer blocks to receive LoRA adapters — train only the layers that matter for your concept.

#### VRAM Management
- **Model quantization**: FP8, NF4, INT8, INT4 for the frozen base model
- **Text encoder quantization**: Independent quantization for TEs
- **Gradient checkpointing**: ~30–50% VRAM reduction at ~20–30% speed cost
- **Block swapping**: Granular per-block CPU offloading with adjustable percentage
- **VAE / TE offloading**: Move inactive components to CPU after caching
- **Latent caching**: Pre-encode images through VAE, cache to disk
- **Text embedding caching**: Cache TE outputs for all captions, unload TE from VRAM

#### Template System
Save, load, and manage training configurations as templates per model family. Auto-save on job creation.

#### Checkpointing & Resume
Full checkpoint support including LoRA weights, optimizer state, scheduler, GradScaler, EMA shadow weights, and latent/embedding cache manifests. Resume training with selective cache re-use.

---

### 📊 Job Queue & Monitoring

#### Multi-Job Queue
- Create, start, stop, pause, resume, and soft-stop training jobs
- Soft-stop: finish the current step, save checkpoint, then stop cleanly
- Restart failed or stopped jobs from last checkpoint

#### Real-Time Monitoring
- **Live loss chart** — interactive uPlot chart with loss and learning rate curves
- **Sample images** — periodic generation previews at configurable intervals
- **Structured logs** — real-time WebSocket streaming of JSON-formatted training events
- **Live terminal** — xterm.js terminal for raw log output

#### Job History
SQLite-backed persistent tracking of all training runs:
- Full training configuration snapshot
- Step-by-step metrics (loss, LR, ETA)
- Checkpoint locations and metadata
- Final LoRA output paths

---

### 🔍 LoRA Tools

#### Inspect
Analyze any `.safetensors` LoRA file without loading a model:

- **Format detection** — Kohya, ai-toolkit (Ostris), PEFT
- **Rank & alpha extraction** — from metadata or weight shapes
- **Per-layer analysis** — Frobenius norms, effective delta W=B@A, magnitude, strength
- **Layer relevance scoring** — identify which layers carry the most learned information
- **Weight statistics** — per-component (UNet, TE) average magnitude and strength
- **Training metadata** — parsed Kohya-style `ss_*` keys (optimizer, LR, schedule, resolution, etc.)
- **Tag frequency** — from `ss_tag_frequency` metadata
- **Block config** — variable per-block rank/alpha detection (DyLoRA-like)

#### Resize
SVD-based rank change (up or down) with proportional alpha scaling. Reconstructs the effective weight delta `W = B @ A`, decomposes via truncated SVD, and re-factors to the target rank.

#### Targeted Layer Training Workflow
Combine **Inspect** results with **Targeted Layer Training** — inspect a reference LoRA to identify which layers learned most, then configure your next training run to target only those layers for more efficient, focused learning.

---

### 🖥️ Server Settings

- **Backend & frontend port configuration** — dynamic runtime config with automatic frontend discovery
- **Log level control** — adjust structured logging verbosity at runtime
- **Frontend auto-start** — optionally launch the Angular dev server and open a browser on backend startup
- **System restart** — restart the backend from the UI
- **GPU monitoring** — real-time GPU utilization and VRAM usage display

---

## Documentation

Detailed architecture documentation, including full API route inventory and component listing:

- [**ARCHITECTURE.md**](documentation/ARCHITECTURE.md) — System architecture, API routes, frontend components, conventions

---

## Author

Created and maintained by **[master-merlin](https://github.com/master-merlin)**.

Repository: **[github.com/master-merlin/mrln-arcane-tuner](https://github.com/master-merlin/mrln-arcane-tuner)** — issues, contributions, and discussion are welcome there.

---

## License

*License information to be added.*
