# HiDream-O1 Spike Notes

This document captures findings from PR A (the spike). Each section is filled
in by the corresponding spike task. PR B references these notes directly.

## Task 2 — Vendored upstream files

**Upstream repo URL:** `https://github.com/HiDream-ai/HiDream-O1-Image`
- The implementation plan assumed `HiDream-ai/HiDream-O1`, but that repo does not exist (GitHub 404). The correct repo is `HiDream-ai/HiDream-O1-Image`. Both the HuggingFace model card README and GitHub search confirm this. The `_refresh.py` script's `UPSTREAM_REPO` was updated accordingly.

**Pinned revision:** `4e56686b857587e6723eb542f73bcab48b19c9ee` (latest commit on `main` as of 2026-05-24, message: "Update model links to include ModelScope repository").

**Files copied:**
- `models/pipeline.py` → `vendor/pipeline.py` (437 lines). Pure inference pipeline: `generate_image()` function + helpers. No model class definition.

**Upstream `models/` directory also contained** (not vendored — inference helpers, would bloat the vendor package and are not needed for training):
- `__init__.py`, `flash_scheduler.py`, `fm_solvers_unipc.py`, `qwen3_vl_transformers.py`, `utils.py`

**Patches applied:**

- **Patch 1 (flash-attn flag):** Added `use_flash_attn: bool = True` parameter to `generate_image()` signature and plumbed it through to the `forward_once` inner function (replacing hardcoded `True` at what is now line ~345). Upstream HF card says line 341 — confirmed the line in the actual file.

- **Patch 2 (torch_dtype threading):** Added `dtype: torch.dtype | None = None` parameter to `generate_image()`. The upstream code hardcoded `dtype = torch.bfloat16` inside the function body; replaced with `dtype = dtype or torch.bfloat16` so callers can override (e.g., pass `torch.float32` for training without flash-attn).

- **Patch 3 (gradient checkpointing):** **SKIPPED.** `pipeline.py` contains no model class — it is a pure inference pipeline. The model class (`Qwen3VLPreTrainedModel` in `models/qwen3_vl_transformers.py`) already inherits from `transformers.PreTrainedModel` with `supports_gradient_checkpointing = True` and blocks inherit from `GradientCheckpointingLayer`. No stub needed.

**Surprises:**
- The repo name divergence (plan said `HiDream-O1`, actual is `HiDream-O1-Image`) was the only significant surprise.
- `pipeline.py` is a flat inference script, not an OO pipeline class — no `__init__` or model constructor to patch for dtype; the dtype param was added to the `generate_image` function instead.
- The upstream code uses `dtype = torch.bfloat16` (not `torch.float32` as the plan assumed). Default preserved as bfloat16.

## Task 3 — Model loading & training API

(Filled in by Task 3.)

## Task 4 — Recipe convergence

(Filled in by Task 4.)

## Task 5 — VRAM measurements

(Filled in by Task 5.)
